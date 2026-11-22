import base64
import json
import re
import time
from pathlib import Path

import cv2
import requests

# ============================================================
# CONFIGURAZIONE
# ============================================================
FPS = 2
PADDING_SECONDS = 1.0
MAX_FRAMES = 12
API_URL = "http://127.0.0.1:9931/v1/chat/completions"
MODEL_NAME = "Qwen2.5-VL-3B-Instruct"
MAX_TOKENS = 256
TEMPERATURE = 0.1
DEFAULT_IMAGE_SIDE = 336
JPEG_QUALITY = 90
PAUSE_SECONDS = 0.3

# Versione del prompt: se cambia, le azioni gia' rifinite con una versione diversa
# vengono rielaborate (e il vecchio file viene salvato come backup).
# v2: aggiunto il punteggio di confidenza (0-1) richiesto dal prompt originale del paper.
PROMPT_VERSION = "stage3_conf_v2"

SYSTEM_PROMPT = """You are an expert video analyzer, specialized in detecting **specific physical interactions**.
Given {nframes} consecutive frames from a {duration}-second video segment, your task is to:
1. Precisely localize the **start** and **end** of the most prominent physical interaction using the exact timestamps provided with the frames.
2. Provide a refined and concrete description.
3. Assign a confidence score (0-1) that the interaction is really visible in the frames.

You will be given an initial hypothesized interaction: "{description}".
WARNING: This hypothesis may be inaccurate. Always rely on the actual visual evidence.

### Key Instructions
1. Do NOT label the entire segment as the action. The segment contains padding; find the exact timestamps where the action begins and ends.
2. Never use Double Quotation Marks or Single Quotation Marks inside the description.
3. Each description must contain exactly ONE concrete low-level verb (e.g., grasp, put into, take out, open, close, pour, stir).
4. The confidence must be a number between 0 and 1. Use a LOW value if the hands are not clearly interacting with an object or if no interaction is visible.
5. Return ONLY a fenced JSON code block starting with ```json and ending with ```.
6. The description string MUST begin with <start> and end with <end>.

```json
{
"start": <start time>,
"end": <end time>,
"description": "<start><refined action description><end>",
"confidence": <confidence between 0 and 1>
}
```"""


def encode_image(image_path, max_side):
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Impossibile leggere l'immagine: {image_path}")
    height, width = image.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    if scale < 1.0:
        image = cv2.resize(
            image,
            (int(width * scale), int(height * scale)),
            interpolation=cv2.INTER_AREA,
        )
    ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    if not ok:
        raise ValueError("Errore nella codifica")
    return f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"


def extract_json(raw_text):
    text_clean = re.sub(r"```json\s*", "", raw_text)
    text_clean = re.sub(r"```\s*", "", text_clean)
    start = text_clean.find("{")
    end = text_clean.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("JSON non trovato nella risposta.")
    json_str = re.sub(r",\s*([\]}])", r"\1", text_clean[start : end + 1])
    return json.loads(json_str)


def parse_ts(val):
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        try:
            return float(val.strip().rstrip("sS").strip())
        except ValueError:
            return None
    return None


def parse_confidence(val):
    """Converte la confidenza in un float in [0, 1]; None se assente o non valida."""
    if isinstance(val, str):
        val = val.strip().rstrip("%")
        try:
            val = float(val)
        except ValueError:
            return None
    if not isinstance(val, (int, float)):
        return None
    val = float(val)
    if 1.0 < val <= 100.0:   # alcuni modelli rispondono in percentuale
        val /= 100.0
    return val if 0.0 <= val <= 1.0 else None


def process_action(action, frames_dir, raw_dir, max_side):
    padded_start = max(0.0, action["start_time"] - PADDING_SECONDS)
    padded_end = action["end_time"] + PADDING_SECONDS
    duration = round(padded_end - padded_start, 2)

    start_frame_idx = int(padded_start * FPS)
    end_frame_idx = int(padded_end * FPS)

    frames_to_load = []
    for f in range(start_frame_idx, end_frame_idx + 1):
        path = frames_dir / f"frame_{f:05d}.jpg"
        if path.exists():
            frames_to_load.append((f, path))

    if not frames_to_load:
        raise ValueError("Nessun frame trovato per questo intervallo.")

    if len(frames_to_load) > MAX_FRAMES:
        step = len(frames_to_load) / MAX_FRAMES
        frames_to_load = [frames_to_load[int(i * step)] for i in range(MAX_FRAMES)]

    sys_prompt = (
        SYSTEM_PROMPT.replace("{nframes}", str(len(frames_to_load)))
        .replace("{duration}", str(duration))
        .replace("{description}", action["action_label"])
    )

    content = []
    for f_idx, path in frames_to_load:
        timestamp = f_idx / FPS
        content.append({"type": "text", "text": f"Timestamp {timestamp:.1f}s:"})
        content.append({"type": "image_url", "image_url": {"url": encode_image(path, max_side)}})

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": content},
        ],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
    }

    response = requests.post(API_URL, json=payload, timeout=600)
    if not response.ok:
        print(f"\n[Dettaglio Server {response.status_code}] {response.text[:500]}")
        response.raise_for_status()

    raw_text = response.json()["choices"][0]["message"]["content"]
    raw_path = raw_dir / f"action_{action['action_id']:03d}.txt"
    raw_path.write_text(raw_text, encoding="utf-8")

    parsed = extract_json(raw_text)
    s = parse_ts(parsed.get("start"))
    e = parse_ts(parsed.get("end"))
    confidence = parse_confidence(parsed.get("confidence"))
    desc = re.sub(r"</?(start|end)>", "", str(parsed.get("description", ""))).strip()

    valid_time = (
        s is not None
        and e is not None
        and padded_start <= s < e <= padded_end
    )

    if not valid_time:
        print(
            f"  [Avviso] Timestamp non validi ({parsed.get('start')} - {parsed.get('end')}) "
            f"fuori da [{padded_start}, {padded_end}]. Fallback a Stage 2."
        )
        s, e, refined_ok = action["start_time"], action["end_time"], False
    else:
        refined_ok = True

    if not desc or "<refined" in desc.lower():
        desc = action["action_label"]

    return {
        "action_id": action["action_id"],
        "start_time": float(s),
        "end_time": float(e),
        "description": desc,
        "confidence": confidence,
        "refined": refined_ok,
        "raw_stage2_label": action["action_label"],
        "prompt_version": PROMPT_VERSION,
    }


def fallback_result(act):
    return {
        "action_id": act["action_id"],
        "start_time": act["start_time"],
        "end_time": act["end_time"],
        "description": act["action_label"],
        "confidence": None,
        "refined": False,
        "raw_stage2_label": act["action_label"],
        "prompt_version": PROMPT_VERSION,
    }


def load_existing(output_file, force):
    """Carica le azioni gia' rifinite con la versione di prompt corrente.
    Se il file contiene risultati di una versione diversa, ne salva un backup."""
    if force or not output_file.exists():
        return {}

    items = json.loads(output_file.read_text(encoding="utf-8"))
    current = {it["action_id"]: it for it in items if it.get("prompt_version") == PROMPT_VERSION}

    if len(current) < len(items):
        old_versions = {it.get("prompt_version", "v1") for it in items} - {PROMPT_VERSION}
        tag = "_".join(sorted(str(v) for v in old_versions))
        backup = output_file.with_name(f"refined_actions_{tag}.json")
        if not backup.exists():
            backup.write_text(json.dumps(items, indent=4, ensure_ascii=False), encoding="utf-8")
            print(f"Backup dei risultati della versione precedente: {backup.name}")
    return current


def run(video_id: str, image_side: int = DEFAULT_IMAGE_SIDE, force: bool = False):
    project_root = Path(__file__).resolve().parents[1]

    input_file = project_root / "outputs" / video_id / "stage2" / "action_grouping_stage2.json"
    frames_dir = project_root / "outputs" / "frames" / video_id
    output_dir = project_root / "outputs" / video_id / "stage3"
    raw_dir = output_dir / "raw_responses"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "refined_actions.json"

    if not input_file.exists():
        raise FileNotFoundError(f"File JSON di input non trovato: {input_file}")

    with open(input_file, "r", encoding="utf-8") as f:
        actions = json.load(f)["actions"]

    refined_map = load_existing(output_file, force)

    print("\n" + "=" * 55)
    print(f"STAGE 3 - ACTION REFINEMENT ({PROMPT_VERSION})")
    print("=" * 55)
    print(f"Azioni totali: {len(actions)} | Gia' elaborate con questa versione: {len(refined_map)}")

    for idx, act in enumerate(actions, 1):
        aid = act["action_id"]
        if aid in refined_map:
            continue

        print(
            f"[{idx}/{len(actions)}] Rifinitura azione {aid}: "
            f"'{act['action_label']}' ({act['start_time']}s - {act['end_time']}s)"
        )
        try:
            refined = process_action(act, frames_dir, raw_dir, image_side)
            refined_map[aid] = refined
            status_tag = "OK" if refined["refined"] else "FALLBACK"
            conf = refined["confidence"]
            conf_str = f"{conf:.2f}" if conf is not None else "n/d"
            print(f"  {status_tag}: {refined['start_time']}s - {refined['end_time']}s | "
                  f"'{refined['description']}' | conf={conf_str}")
        except Exception as e:
            print(f"  ERRORE ({e}) -> Fallback ai confini dello Stage 2.")
            refined_map[aid] = fallback_result(act)

        ordered = [refined_map[k] for k in sorted(refined_map)]
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(ordered, f, indent=4, ensure_ascii=False)

        time.sleep(PAUSE_SECONDS)

    # Rimuove eventuali azioni rimaste da una versione precedente che non esistono piu' nello Stage 2
    valid_ids = {a["action_id"] for a in actions}
    ordered = [refined_map[k] for k in sorted(refined_map) if k in valid_ids]
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(ordered, f, indent=4, ensure_ascii=False)

    n_conf = sum(1 for a in ordered if a.get("confidence") is not None)
    print("\n" + "=" * 55)
    print(f"STAGE 3 COMPLETATO: {len(ordered)} azioni salvate in {output_file}")
    print(f"Azioni con confidenza valida: {n_conf}/{len(ordered)}")
    print("=" * 55)