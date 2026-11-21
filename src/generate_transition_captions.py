"""
CoTT - Stage 1: Transition Captioning (versione per modelli piccoli)
"""

import base64
import json
import re
import time
from pathlib import Path

import cv2
import requests

# ============================================================
# CONFIGURAZIONE COSTANTI
# ============================================================
FPS = 2
WINDOW_SIZE = 12
STRIDE = 6
API_URL = "http://127.0.0.1:9931/v1/chat/completions"
MODEL_NAME = "Qwen2.5-VL-3B-Instruct"
MAX_TOKENS = 384
TEMPERATURE = 0.1
TIMEOUT_SECONDS = 600
DEFAULT_IMAGE_SIDE = 336
JPEG_QUALITY = 90
PAUSE_SECONDS = 0.2
PROMPT_VERSION = "stage1_small_v4"
OUTPUT_FORMAT = "numbered_lines_v1"

STATUS_ALIASES = {
    "start": "start",
    "starting": "start",
    "continue": "continue",
    "continuing": "continue",
    "end": "end",
    "ending": "end",
    "none": "none",
}

SYSTEM_PROMPT = """
You analyze egocentric video frames to detect physical interactions
between the camera wearer's hands and objects.

Describe only actions supported by visible evidence.
Never guess, infer unseen actions or invent object movements.
Focus on one low-level physical action per description.
Use short, concrete phrases.

Track interactions across consecutive frames:
- start: a new interaction begins.
- continue: the same interaction is still ongoing.
- end: the interaction visibly finishes.
- none: no relevant hand-object interaction is visible.

Do not assign continue automatically.
Do not restart an interaction unless a new action is visible.
Do not repeat actions or invent sequences.
Be conservative when the action is unclear.
""".strip()

def build_user_prompt(n_frames):
    n_trans = n_frames - 1
    return f"""
Analyze {n_frames} consecutive egocentric video frames,
in chronological order.

Describe each of the {n_trans} transitions between adjacent frames.
Transition k goes from Frame k to Frame k+1.

For each transition:
- Describe the visible hand-object action.
- Use the full sequence to track ongoing interactions.
- Report only actions supported by the two frames.
- Do not infer unseen actions or repeat completed actions.
- Use none if no interaction is visible.

Status:
start = new interaction begins
continue = same interaction continues
end = interaction finishes
none = no visible interaction

Return exactly {n_trans} lines:

<k> | <status> | <description>

Number lines from 1 to {n_trans}.
Use only start, continue, end, none.
One short action per line.
No timestamps or explanations.
""".strip()

def frame_number(path):
    match = re.search(r"(\d+)", path.stem)
    if not match:
        raise ValueError(f"Nome frame non riconosciuto: {path.name}")
    return int(match.group(1))

def load_frames(frames_dir):
    if not frames_dir.exists():
        raise FileNotFoundError(f"Cartella frame non trovata: {frames_dir}\nEsegui prima l'estrazione dei frame.")
    files = []
    for pattern in ("*.jpg", "*.jpeg", "*.png"):
        files.extend(frames_dir.glob(pattern))
    files.sort(key=frame_number)
    
    if len(files) < WINDOW_SIZE:
        raise ValueError(f"Frame insufficienti: {len(files)}, servono almeno {WINDOW_SIZE}.")
    
    indices = [frame_number(path) for path in files]
    for previous, current in zip(indices, indices[1:]):
        if current != previous + 1:
            raise ValueError(f"Sequenza frame non consecutiva: {previous} -> {current}")
    return files

def create_windows(frame_files):
    windows = []
    for start in range(0, len(frame_files) - WINDOW_SIZE + 1, STRIDE):
        selected = frame_files[start:start + WINDOW_SIZE]
        windows.append({
            "window_id": len(windows),
            "frame_paths": selected,
            "frame_indices": [frame_number(path) for path in selected],
        })
    return windows

def encode_image(image_path, max_side):
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Impossibile leggere l'immagine: {image_path}")
    height, width = image.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    if scale < 1.0:
        image = cv2.resize(image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
    ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    if not ok:
        raise ValueError(f"Errore nella codifica: {image_path}")
    encoded = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"

def build_user_content(window, max_side):
    paths = window["frame_paths"]
    content = [{"type": "text", "text": build_user_prompt(len(paths))}]
    for position, path in enumerate(paths, start=1):
        content.append({"type": "text", "text": f"Frame {position}:"})
        content.append({"type": "image_url", "image_url": {"url": encode_image(path, max_side)}})
    return content

def call_model(window, max_side):
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_content(window, max_side)},
        ],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
    }
    response = requests.post(API_URL, json=payload, timeout=TIMEOUT_SECONDS)
    response.raise_for_status()
    result = response.json()
    if not result.get("choices"):
        raise ValueError(f"Risposta API senza choices: {result}")
    choice = result["choices"][0]
    return {
        "text": choice["message"]["content"] or "",
        "finish_reason": choice.get("finish_reason"),
        "usage": result.get("usage"),
    }

LINE_RE = re.compile(r"^\s*(\d+)\s*\|\s*([A-Za-z]+)\s*\|\s*(.*?)\s*$")

def parse_transitions(text, expected):
    found = {}
    for raw_line in text.splitlines():
        line = raw_line.strip().lstrip("-*• ").strip()
        if not line:
            continue
        match = LINE_RE.match(line)
        if not match:
            continue
        k = int(match.group(1))
        if k < 1 or k > expected:
            print(f"  [Avviso] Ignorata riga in eccesso dal modello: id {k}")
            continue
        if k in found:
            raise ValueError(f"Identificativo duplicato: {k}")
        
        status_raw = match.group(2).lower()
        status = STATUS_ALIASES.get(status_raw)
        if status is None:
            raise ValueError(f"Stato non valido alla riga {k}: {status_raw!r}")
        
        description = match.group(3).replace('"', "").replace("'", "").strip()
        if not description:
            raise ValueError(f"Descrizione vuota alla riga {k}.")
        if status == "none":
            description = "no interaction"
        found[k] = (status, description)

    missing = [k for k in range(1, expected + 1) if k not in found]
    if missing:
        raise ValueError(f"Righe mancanti o non valide: {missing}")
    if len(found) != expected:
        raise ValueError(f"Numero di righe non valido: {len(found)} invece di {expected}")
    return [found[k] for k in range(1, expected + 1)]

def quality_warnings(parsed):
    warnings = []
    descriptions = [description.lower() for _, description in parsed]
    statuses = [status for status, _ in parsed]
    if len(set(descriptions)) == 1 and descriptions[0] != "no interaction":
        warnings.append("repetitive")
    if all(status == "none" for status in statuses):
        warnings.append("all_none")
    if len(set(statuses)) == 1 and statuses[0] in ("start", "end"):
        warnings.append("single_status")
    return warnings

def build_transitions(window, parsed):
    indices = window["frame_indices"]
    transitions = []
    for k, (status, description) in enumerate(parsed, start=1):
        frame_a = indices[k - 1]
        frame_b = indices[k]
        transitions.append({
            "window_id": window["window_id"],
            "transition_id": frame_a,
            "frame_start": frame_a,
            "frame_end": frame_b,
            "start_time": frame_a / FPS,
            "end_time": frame_b / FPS,
            "description": description,
            "status": status,
        })
    return transitions

def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)

def load_existing(windows_file, config, force):
    if force or not windows_file.exists():
        return {}
    data = json.loads(windows_file.read_text(encoding="utf-8"))
    if data.get("config") != config:
        raise SystemExit(f"La configurazione e' cambiata. Rilancia con --force per ricominciare.")
    return {window["window_id"]: window for window in data.get("windows", [])}

def write_outputs(video_id, windows_file, flat_file, config, results):
    windows = [results[index] for index in sorted(results)]
    save_json(windows_file, {
        "video_id": video_id,
        "stage": "transition_captioning",
        "config": config,
        "windows": windows,
    })
    flat = []
    for window in windows:
        if window["status"] == "success":
            flat.extend(window["transitions"])
    flat.sort(key=lambda transition: (transition["start_time"], transition["window_id"]))
    save_json(flat_file, flat)

def process_window(window, raw_dir, max_side, retries):
    wid = window["window_id"]
    expected = WINDOW_SIZE - 1
    last_error = None
    
    for attempt in range(retries + 1):
        result = call_model(window, max_side)
        raw_path = raw_dir / f"window_{wid:03d}_try{attempt}.txt"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(result["text"], encoding="utf-8")
        
        completion_tokens = (result["usage"] or {}).get("completion_tokens")
        print(f"  tentativo {attempt + 1}: finish={result['finish_reason']} completion_tokens={completion_tokens}")
        
        try:
            if result["finish_reason"] != "stop":
                raise ValueError(f"Risposta non completata (finish_reason={result['finish_reason']})")
            parsed = parse_transitions(result["text"], expected)
        except ValueError as error:
            last_error = str(error)
            print(f"  scartato: {last_error}")
            continue
            
        warnings = quality_warnings(parsed)
        return {
            "window_id": wid,
            "frame_start": window["frame_indices"][0],
            "frame_end": window["frame_indices"][-1],
            "status": "success",
            "attempts": attempt + 1,
            "warnings": warnings,
            "finish_reason": result["finish_reason"],
            "usage": result["usage"],
            "transitions": build_transitions(window, parsed),
        }
        
    return {
        "window_id": wid,
        "frame_start": window["frame_indices"][0],
        "frame_end": window["frame_indices"][-1],
        "status": "error",
        "attempts": retries + 1,
        "error": last_error,
        "transitions": [],
    }

def run(video_id: str, limit: int = None, only: list = None, force: bool = False, image_side: int = DEFAULT_IMAGE_SIDE, retries: int = 2):
    """
    Esegue lo Stage 1: Transition Captioning per il video specificato.
    """
    project_root = Path(__file__).resolve().parents[1]
    
    frames_dir = project_root / "outputs" / "frames" / video_id
    output_dir = project_root / "outputs" / video_id / "stage1"
    raw_dir = output_dir / "raw_responses"
    windows_file = output_dir / "stage1_windows.json"
    flat_file = output_dir / "transitions_all.json"

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "fps": FPS,
        "window_size": WINDOW_SIZE,
        "stride": STRIDE,
        "image_side": image_side,
        "jpeg_quality": JPEG_QUALITY,
        "model": MODEL_NAME,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "prompt_version": PROMPT_VERSION,
        "output_format": OUTPUT_FORMAT,
    }

    frame_files = load_frames(frames_dir)
    windows = create_windows(frame_files)

    if only is not None:
        requested_ids = set(only)
        windows = [w for w in windows if w["window_id"] in requested_ids]

    if limit is not None:
        windows = windows[:limit]

    results = load_existing(windows_file, config, force)
    to_do = [w for w in windows if results.get(w["window_id"], {}).get("status") != "success"]

    print("=" * 60)
    print("COTT - STAGE 1: TRANSITION CAPTIONING")
    print("=" * 60)
    print(f"Video: {video_id} | frame: {len(frame_files)}")
    print(f"Finestre selezionate: {len(windows)} | da elaborare: {len(to_do)}")

    for position, window in enumerate(to_do, start=1):
        indices = window["frame_indices"]
        print(f"\n[{position}/{len(to_do)}] Finestra {window['window_id']}: frame {indices[0]}-{indices[-1]}")
        try:
            result = process_window(window, raw_dir, image_side, retries)
        except Exception as error:
            print(f"  ERRORE: {type(error).__name__}: {error}")
            result = {
                "window_id": window["window_id"],
                "frame_start": indices[0],
                "frame_end": indices[-1],
                "status": "error",
                "error": f"{type(error).__name__}: {error}",
                "transitions": [],
            }

        if result["status"] == "success":
            tag = f" ATTENZIONE: {result['warnings']}" if result.get("warnings") else ""
            print(f"  OK ({len(result['transitions'])} transizioni){tag}")
        else:
            print(f"  FALLITA: {result['error']}")

        results[result["window_id"]] = result
        write_outputs(video_id, windows_file, flat_file, config, results)
        time.sleep(PAUSE_SECONDS)

    print("\n" + "=" * 60)
    print(f"Completato Stage 1 per: {video_id}")