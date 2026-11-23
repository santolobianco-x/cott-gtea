"""
Baseline di prompting diretto (Figura 1 del paper CoTT).

  - slide : il video viene diviso in clip affiancate (NON sovrapposte) da WINDOW_SIZE frame
            a 2 fps (6 s, come lo Stage 1). Per ogni clip si chiede al modello di elencare
            TUTTE le interazioni con i loro timestamp. Le clip sono indipendenti: le
            predizioni vengono solo concatenate, senza merge ne' correzioni.
  - whole : tutti i frame del video in una sola chiamata, chiedendo di elencare tutte
            le interazioni. Il paper usa 2 fps; sul Mac si puo' ridurre fps/risoluzione.

Come nel paper, il prompt e' quello dello Stage 3 in cui "trova l'interazione piu'
evidente" e' sostituito da "elenca tutte le interazioni", con lo stesso formato di uscita.

  - grid  : CONTROLLO "cieco" (non nel paper): griglia fissa di segmenti lunghi GRID_LEN
            secondi, uno ogni GRID_PERIOD secondi, SENZA guardare il video ne' usare il modello.
            Misura quanto F1 si ottiene per puro caso: un metodo che non lo supera
            non sta usando l'informazione visiva. Default 2 s ogni 3 s, lo schema
            prodotto dal modello nei loop della baseline whole-video.

Output: outputs/<video>/baseline_<tipo>/predictions.json  ({"actions": [...]})
Modulo richiamato da main.py (opzione --baseline).
"""

import json
import re
import time
from pathlib import Path

import requests

from src.generate_transition_captions import API_URL, MODEL_NAME, encode_image, frame_number
from src.refine_action_instances import parse_ts

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FPS = 2
WINDOW_SIZE = 12            # come lo Stage 1 (6 s a 2 fps)
DEFAULT_IMAGE_SIDE = 336
RETRY_TEMPERATURES = [0.1, 0.4, 0.7]
PAUSE_SECONDS = 0.2
PROMPT_VERSION = "baseline_all_v1"

MAX_TOKENS = {"slide": 512, "whole": 2048}
TIMEOUT = {"slide": 600, "whole": 3600}

SYSTEM_PROMPT = """You are an expert video analyzer, specialized in detecting **specific physical interactions**.
Given {nframes} consecutive frames from a {duration}-second video segment, your task is to:
1. Provide ALL physical interactions performed by the camera wearer's hands in the given video, in chronological order.
2. For each interaction, precisely localize its **start** and **end** using the exact timestamps provided with the frames.
3. Provide a concrete description for each interaction.

### Key Instructions
1. Do NOT label the entire segment as a single action. List every distinct interaction separately.
2. Never use Double Quotation Marks or Single Quotation Marks inside the descriptions.
3. Each description must contain exactly ONE concrete low-level verb (e.g., grasp, put into, take out, open, close, pour, stir).
4. Interactions must not overlap in time.
5. If no physical interaction is visible, return an empty list.
6. Return ONLY a fenced JSON code block starting with ```json and ending with ```.
7. Each description string MUST begin with <start> and end with <end>.

```json
{
"intervals": [
    {"start": <start time>, "end": <end time>, "description": "<start><action description><end>"},
    <possibly multiple elements>
]
}
```"""


# ============================================================
# UTILITA'
# ============================================================
def load_frames(video_id):
    frames_dir = PROJECT_ROOT / "outputs" / "frames" / video_id
    files = sorted(frames_dir.glob("frame_*.jpg"), key=frame_number)
    if not files:
        raise FileNotFoundError(f"Nessun frame in {frames_dir}: esegui prima l'estrazione dei frame.")
    return files


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def parse_intervals(raw_text):
    """Estrae la lista di intervalli dalla risposta. Tollera JSON troncati:
    se il JSON completo non e' valido, recupera i singoli oggetti {...} completi."""
    text = re.sub(r"```(?:json)?", "", raw_text).strip()
    text = re.sub(r",\s*([\]}])", r"\1", text)

    data = None
    for opener, closer in (("{", "}"), ("[", "]")):
        s, e = text.find(opener), text.rfind(closer)
        if s != -1 and e > s:
            try:
                data = json.loads(text[s:e + 1])
                break
            except json.JSONDecodeError:
                continue

    if isinstance(data, dict):
        items = data.get("intervals") or data.get("actions") or []
    elif isinstance(data, list):
        items = data
    else:
        items = []
        for chunk in re.findall(r"\{[^{}]*\}", text):
            try:
                items.append(json.loads(chunk))
            except json.JSONDecodeError:
                continue
        if not items and not re.search(r"\[\s*\]", text):
            raise ValueError("Nessun intervallo leggibile nella risposta.")
    return [it for it in items if isinstance(it, dict)]


def to_actions(items, t_min, t_max):
    """Converte gli intervalli in azioni valide, limitate all'intervallo [t_min, t_max]."""
    actions = []
    for it in items:
        s, e = parse_ts(it.get("start")), parse_ts(it.get("end"))
        if s is None or e is None:
            continue
        s, e = max(t_min, s), min(t_max, e)
        if e <= s:
            continue
        desc = re.sub(r"</?(start|end)>", "", str(it.get("description", ""))).strip()
        actions.append({"start_time": round(s, 3), "end_time": round(e, 3), "description": desc})
    return actions


def call_model(frame_paths, image_side, mode, temperature):
    first, last = frame_number(frame_paths[0]), frame_number(frame_paths[-1])
    duration = round((last - first + 1) / FPS, 2)
    sys_prompt = SYSTEM_PROMPT.replace("{nframes}", str(len(frame_paths))).replace("{duration}", str(duration))

    content = []
    for path in frame_paths:
        content.append({"type": "text", "text": f"Timestamp {frame_number(path) / FPS:.1f}s:"})
        content.append({"type": "image_url", "image_url": {"url": encode_image(path, image_side)}})

    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "system", "content": sys_prompt}, {"role": "user", "content": content}],
        "temperature": temperature,
        "max_tokens": MAX_TOKENS[mode],
    }
    response = requests.post(API_URL, json=payload, timeout=TIMEOUT[mode])
    if not response.ok:
        print(f"  [Server {response.status_code}] {response.text[:300]}")
        response.raise_for_status()
    choice = response.json()["choices"][0]
    return choice["message"]["content"] or "", choice.get("finish_reason")


def process_chunk(frame_paths, image_side, mode, raw_path, t_max=None):
    """Una chiamata al modello (con retry) su un gruppo di frame -> lista di azioni."""
    t_min = frame_number(frame_paths[0]) / FPS
    if t_max is None:
        t_max = (frame_number(frame_paths[-1]) + 1) / FPS
    last_error = None
    for attempt, temp in enumerate(RETRY_TEMPERATURES):
        text, finish = call_model(frame_paths, image_side, mode, temp)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.with_name(f"{raw_path.stem}_try{attempt}.txt").write_text(text, encoding="utf-8")
        try:
            actions = to_actions(parse_intervals(text), t_min, t_max)
            if finish == "length":
                print(f"  [Avviso] risposta troncata (max_tokens): recuperate {len(actions)} azioni complete")
            return {"status": "success", "attempts": attempt + 1, "finish_reason": finish, "actions": actions}
        except ValueError as err:
            last_error = str(err)
            print(f"  tentativo {attempt + 1} (T={temp}) scartato: {last_error}")
    return {"status": "error", "attempts": len(RETRY_TEMPERATURES), "error": last_error, "actions": []}


def write_predictions(out_dir, video_id, mode, config, actions):
    actions = sorted(actions, key=lambda a: (a["start_time"], a["end_time"]))
    for i, a in enumerate(actions):
        a["action_id"] = i
    save_json(out_dir / "predictions.json",
              {"video_id": video_id, "baseline": mode, "config": config, "actions": actions})


# ============================================================
# BASELINE SLIDING-WINDOW
# ============================================================
def run_slide(video_id, force=False, image_side=DEFAULT_IMAGE_SIDE):
    out_dir = PROJECT_ROOT / "outputs" / video_id / "baseline_slide"
    windows_file = out_dir / "windows.json"
    config = {"prompt_version": PROMPT_VERSION, "model": MODEL_NAME, "window_size": WINDOW_SIZE,
              "stride": WINDOW_SIZE, "fps": FPS, "image_side": image_side}

    frames = load_frames(video_id)
    # clip affiancate senza sovrapposizione; l'ultima puo' essere piu' corta (min 2 frame)
    chunks = [frames[i:i + WINDOW_SIZE] for i in range(0, len(frames), WINDOW_SIZE)]
    chunks = [c for c in chunks if len(c) >= 2]

    results = {}
    if not force and windows_file.exists():
        saved = json.loads(windows_file.read_text(encoding="utf-8"))
        if saved.get("config") == config:
            results = {w["window_id"]: w for w in saved["windows"] if w["status"] == "success"}

    print(f"\n[Baseline slide] {video_id}: {len(chunks)} clip | gia' fatte: {len(results)}")
    for wid, chunk in enumerate(chunks):
        if wid in results:
            continue
        print(f"  clip {wid + 1}/{len(chunks)} ({frame_number(chunk[0]) / FPS:.1f}s - "
              f"{(frame_number(chunk[-1]) + 1) / FPS:.1f}s)")
        try:
            res = process_chunk(chunk, image_side, "slide", out_dir / "raw_responses" / f"window_{wid:03d}")
        except Exception as err:
            res = {"status": "error", "error": f"{type(err).__name__}: {err}", "actions": []}
            print(f"  ERRORE: {res['error']}")
        results[wid] = {"window_id": wid, **res}
        save_json(windows_file, {"config": config, "windows": [results[k] for k in sorted(results)]})
        time.sleep(PAUSE_SECONDS)

    failed = [k for k, w in results.items() if w["status"] != "success"]
    all_actions = [a for k in sorted(results) for a in results[k]["actions"]]
    write_predictions(out_dir, video_id, "slide", config, all_actions)
    print(f"[Baseline slide] {video_id}: {len(all_actions)} azioni | clip fallite: {failed}")


# ============================================================
# BASELINE WHOLE-VIDEO
# ============================================================
def run_whole(video_id, force=False, image_side=DEFAULT_IMAGE_SIDE, fps=FPS):
    out_dir = PROJECT_ROOT / "outputs" / video_id / "baseline_whole"
    pred_file = out_dir / "predictions.json"
    config = {"prompt_version": PROMPT_VERSION, "model": MODEL_NAME, "fps": fps, "image_side": image_side}

    if not force and pred_file.exists():
        saved = json.loads(pred_file.read_text(encoding="utf-8"))
        if saved.get("config") == config and saved.get("status") == "success":
            print(f"[Baseline whole] {video_id}: gia' fatto, salto.")
            return

    frames = load_frames(video_id)
    step = max(1, round(FPS / fps))           # es. fps=1 -> un frame ogni 2
    selected = frames[::step]
    print(f"\n[Baseline whole] {video_id}: {len(selected)} frame in una sola chiamata "
          f"({fps} fps, lato {image_side}px)")

    t_max = (frame_number(frames[-1]) + 1) / FPS   # fine reale del video
    try:
        res = process_chunk(selected, image_side, "whole", out_dir / "raw_responses" / "whole", t_max=t_max)
    except Exception as err:
        res = {"status": "error", "error": f"{type(err).__name__}: {err}", "actions": []}
        print(f"  ERRORE: {res['error']}")

    write_predictions(out_dir, video_id, "whole", config, res["actions"])
    data = json.loads(pred_file.read_text(encoding="utf-8"))
    data["status"] = res["status"]
    data["error"] = res.get("error")
    save_json(pred_file, data)
    print(f"[Baseline whole] {video_id}: {len(res['actions'])} azioni ({res['status']})")


# ============================================================
# CONTROLLO CIECO: GRIGLIA FISSA (non usa ne' il video ne' il modello)
# ============================================================
GRID_LEN = 2.0
GRID_PERIOD = 3.0


def run_grid(video_id, seg_len=GRID_LEN, period=GRID_PERIOD):
    out_dir = PROJECT_ROOT / "outputs" / video_id / "baseline_grid"
    duration = len(load_frames(video_id)) / FPS      # unica informazione usata: la durata
    actions, t = [], 0.0
    while t < duration:
        end = min(t + seg_len, duration)
        if end > t:
            actions.append({"start_time": round(t, 3), "end_time": round(end, 3), "description": "grid"})
        t += period
    config = {"seg_len": seg_len, "period": period}
    write_predictions(out_dir, video_id, "grid", config, actions)
    print(f"[Baseline grid] {video_id}: {len(actions)} segmenti da {seg_len}s ogni {period}s")


def run(video_id, mode, force=False, image_side=DEFAULT_IMAGE_SIDE, whole_fps=FPS,
        grid_len=GRID_LEN, grid_period=GRID_PERIOD):
    if mode == "grid":
        run_grid(video_id, seg_len=grid_len, period=grid_period)
    elif mode == "slide":
        run_slide(video_id, force=force, image_side=image_side)
    elif mode == "whole":
        run_whole(video_id, force=force, image_side=image_side, fps=whole_fps)
    else:
        raise ValueError(f"Baseline sconosciuta: {mode}")