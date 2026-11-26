import base64
import json
import re
import time
import cv2
import requests


from src.config import (
    API_URL, FPS, FRAMES_DIR, IMAGE_SIDE, JPEG_QUALITY, MODEL_NAME, OUTPUTS_DIR,
    S1_MAX_TOKENS, S1_PAUSE, S1_RETRIES, S1_STRIDE, S1_TEMPERATURE, S1_TIMEOUT, S1_WINDOW_SIZE,
)



"""
CoTT - Stage 1: Transition Captioning (sottotitolazione delle transizioni)

    --> invece di chiedere quali azioni ci sono, si chiede di osservare tra una coppia di
        frame consecutivi, con uno stato:
            start    -> inizia una nuova iterazione mano-oggetto
            continue -> la stessa iterazione prosegue
            end      -> l'interazione finisce
            none     -> nessuna iterazione visibile

    
"""

PROMPT_VERSION = "stage1_small_v4"
OUTPUT_FORMAT = "numbered_lines_v1"


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
# il prompt è in inglese dato che è più compatta come lingua(ergo meno token utilizzati)


def build_user_prompt(n_frames):
    """Istruzioni per una finestra di n_frame e n-1 transizioni."""
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
    """Si ricava il numero di un frame dal nome del suo file"""
    match = re.search(r"(\d+)", path.stem)
    if not match:
        raise ValueError(f"Nome frame non riconosciuto: {path.name}")
    return int(match.group(1))




def load_frames(frames_dir):
    """Elenco ordinato dei frame del video, con un controllo che non ne manchi nessuno."""

    if not frames_dir.exists():
        raise FileNotFoundError(f"Cartella frame non trovata: {frames_dir}\n"
                                f"Esegui prima l'estrazione dei frame.")

    files = []

    for pattern in ("*.jpg", "*.jpeg", "*.png"):
        files.extend(frames_dir.glob(pattern))

    files.sort(key=frame_number)#ordine numerico

    if len(files) < S1_WINDOW_SIZE:
        raise ValueError(f"Frame insufficienti: {len(files)}, servono almeno {S1_WINDOW_SIZE}.")

    indices = [frame_number(path) for path in files]

    # verifica che i frame siano tutti presenti
    for previous, current in zip(indices, indices[1:]):
        if current != previous + 1:
            raise ValueError(f"Sequenza frame non consecutiva: {previous} -> {current}")
    return files



def create_windows(frame_files):
    """Finestre da S1_WINDOW_SIZE frame, una ogni S1_STRIDE frame.
    Con 12 e 6: finestra 0 = frame 0-11, finestra 1 = frame 6-17, ...
    """

    windows = []
    for start in range(0, len(frame_files) - S1_WINDOW_SIZE + 1, S1_STRIDE):
        selected = frame_files[start:start + S1_WINDOW_SIZE]
        windows.append({
            "window_id": len(windows),
            "frame_paths": selected,
            "frame_indices": [frame_number(path) for path in selected],
        })
        
    return windows



def encode_image(image_path, max_side):

    """Ridimensiona l'immagine (lato massimo max_side px) e la converte in testo base64,
    il formato con cui le immagini viaggiano dentro una richiesta HTTP."""
    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Impossibile leggere l'immagine: {image_path}")
    
    height, width = image.shape[:2]

    scale = min(1.0, max_side / max(height, width)) # vengono rimpicciolite

    if scale < 1.0:
        image = cv2.resize(image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
    
    ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])

    if not ok:
        raise ValueError(f"Errore nella codifica: {image_path}")
    
    encoded = base64.b64encode(buffer).decode("utf-8")

    return f"data:image/jpeg;base64,{encoded}"



def build_user_content(window, max_side):
    """Messaggio utente: istruzioni, poi 'Frame 1:' + immagine, 'Frame 2:' + immagine, ..."""

    paths = window["frame_paths"]
    content = [{"type": "text", "text": build_user_prompt(len(paths))}]

    for position, path in enumerate(paths, start=1):
        content.append({"type": "text", "text": f"Frame {position}:"})
        content.append({"type": "image_url", "image_url": {"url": encode_image(path, max_side)}})
    return content



def call_model(window, max_side):
    """Invia la finestra al server del modello e restituisce testo, motivo di fine e token usati."""

    # pacco da spedire al modello
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_content(window, max_side)},
        ],
        "temperature": S1_TEMPERATURE, # la temperatura influza la scelta delle parole più probabili
        # temperatura vicino lo 0 permette di non avere risposte più varie, ma più probabili
        "max_tokens": S1_MAX_TOKENS,
    }

    response = requests.post(API_URL, json=payload, timeout=S1_TIMEOUT)
    response.raise_for_status()  # errore HTTP (es. server spento) -> eccezione
    result = response.json()

    if not result.get("choices"):
        raise ValueError(f"Risposta API senza choices: {result}")
    
    choice = result["choices"][0]# si chiede una sola risposta 
    return {
        "text": choice["message"]["content"] or "",
        "finish_reason": choice.get("finish_reason"),   # "stop" = risposta completa, "length" = troncata
        "usage": result.get("usage"),
    }



LINE_RE = re.compile(r"^\s*(\d+)\s*\|\s*([A-Za-z]+)\s*\|\s*(.*?)\s*$")

# Il modello a volte scrive "starting" invece di "start": accettiamo entrambe le forme.
STATUS_ALIASES = {
    "start": "start", "starting": "start",
    "continue": "continue", "continuing": "continue",
    "end": "end", "ending": "end",
    "none": "none",
}


def parse_transitions(text, expected):
    """Legge le righe "k | stato | descrizione" e verifica che ci siano tutte e sole le
    righe 1..expected. 
    Se il formato non e' rispettato solleva ValueError (-> nuovo tentativo)."""

    found = {}
    for raw_line in text.splitlines():
        line = raw_line.strip().lstrip("-*• ").strip()  
        if not line:
            continue

        match = LINE_RE.match(line)
        if not match:
            continue
        k = int(match.group(1))

        # vengono ignorate eventuali righe in eccesso
        if k < 1 or k > expected:
            print(f"  [Avviso] Ignorata riga in eccesso dal modello: id {k}")
            continue
        if k in found:
            raise ValueError(f"Identificativo duplicato: {k}") # viene scartata tutta la risposta
        # se ci sono due righecon lo stesso numero, non si può sapere quale sia giusta


        status_raw = match.group(2).lower()
        status = STATUS_ALIASES.get(status_raw)
        if status is None:
            raise ValueError(f"Stato non valido alla riga {k}: {status_raw!r}")


        description = match.group(3).replace('"', "").replace("'", "").strip()

        if not description:
            raise ValueError(f"Descrizione vuota alla riga {k}.")
        
        if status == "none":
            description = "no interaction"               # descrizione uniforme per i "none"
        found[k] = (status, description)

    missing = [k for k in range(1, expected + 1) if k not in found] # controlla che righe siano tutte
    if missing:
        raise ValueError(f"Righe mancanti o non valide: {missing}")
    return [found[k] for k in range(1, expected + 1)]


def quality_warnings(parsed):
    """Segnala risposte formalmente valide ma sospette:
    - repetitive, all_none, single_status(tutte start o tutte end)
    """

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
    """Riga k della risposta -> transizione dal frame k al frame k+1 della finestra, con i tempi.
    transition_id = indice del primo frame: identifica la transizione in tutto il video."""
    indices = window["frame_indices"]
    transitions = []
    for k, (status, description) in enumerate(parsed, start=1):
        frame_a, frame_b = indices[k - 1], indices[k]
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
    """Scrittura atomica: prima su un file temporaneo, poi rinomina.
    Se il programma viene interrotto a meta', il file vecchio resta integro."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def load_existing(windows_file, config, force):
    """Risultati gia' salvati, per riprendere da dove ci si era fermati.
    Se nel frattempo e' cambiata la configurazione, i vecchi risultati non sono validi."""

    if force or not windows_file.exists():
        return {}
    
    data = json.loads(windows_file.read_text(encoding="utf-8"))


    if data.get("config") != config:
        raise SystemExit("La configurazione dello Stage 1 e' cambiata. Rilancia con --force per ricominciare.")
    return {window["window_id"]: window for window in data.get("windows", [])}


def write_outputs(video_id, windows_file, flat_file, config, results):
    """Salva l'esito di tutte le finestre e l'elenco piatto delle transizioni riuscite."""

    windows = [results[index] for index in sorted(results)]

    save_json(windows_file, {
        "video_id": video_id,
        "stage": "transition_captioning",
        "config": config,
        "windows": windows,
    })

    flat = [t for window in windows if window["status"] == "success" for t in window["transitions"]]

    flat.sort(key=lambda transition: (transition["start_time"], transition["window_id"]))

    save_json(flat_file, flat)



def error_result(window, error, attempts=None):
    """Costruisce il verbale di una finestra fallita"""
    indices = window["frame_indices"]
    return {
        "window_id": window["window_id"],
        "frame_start": indices[0],
        "frame_end": indices[-1],
        "status": "error",
        "attempts": attempts,
        "error": error,
        "transitions": [],
    }

def process_window(window, raw_dir, max_side, retries):
    """Elabora una finestra: chiama il modello finche' la risposta e' valida"""
    wid = window["window_id"]
    expected = S1_WINDOW_SIZE - 1           # 12 frame -> 11 transizioni (righe attese)
    last_error = None # motivo dell'ultimo scarto, da riportare se falliscono tutti

    for attempt in range(retries + 1):    
        result = call_model(window, max_side)

        
        (raw_dir / f"window_{wid:03d}_try{attempt}.txt").write_text(result["text"], encoding="utf-8")# Salva SUBITO la risposta grezza, anche se poi verra' scartata: serve per il debug.

        # (usage or {}) evita un errore se il server non ha mandato le statistiche dei token.
        completion_tokens = (result["usage"] or {}).get("completion_tokens")
        print(f"  tentativo {attempt + 1}: finish={result['finish_reason']} completion_tokens={completion_tokens}")

       
        try:
            if result["finish_reason"] != "stop":
                raise ValueError(f"Risposta non completata (finish_reason={result['finish_reason']})")
            parsed = parse_transitions(result["text"], expected)
        except ValueError as error:
            last_error = str(error)# ricorda il motivo
            print(f"  scartato: {last_error}")
            continue # passa al tentativo successivo

        #  la risposta e' valida: return esce subito, senza tentativi inutili.
        return {
            "window_id": wid,
            "frame_start": window["frame_indices"][0],     
            "frame_end": window["frame_indices"][-1],      
            "status": "success",
            "attempts": attempt + 1,                       
            "warnings": quality_warnings(parsed),          
            "finish_reason": result["finish_reason"],
            "usage": result["usage"],                      
            "transitions": build_transitions(window, parsed),
        }

    # Si arriva qui solo se il ciclo e' finito senza return: tutti i tentativi scartati.
    return error_result(window, last_error, attempts=retries + 1)


def run(video_id, force=False, image_side=IMAGE_SIDE, retries=S1_RETRIES):
    """Esegue lo Stage 1 su un video. Riprende dalle finestre mancanti, salvo force=True."""

    output_dir = OUTPUTS_DIR / video_id / "stage1"
    raw_dir = output_dir / "raw_responses"
    windows_file = output_dir / "stage1_windows.json"
    flat_file = output_dir / "transitions_all.json"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # "Impronta" della configurazione: salvata con i risultati, serve a capire se sono ancora validi.
    config = {
        "fps": FPS,
        "window_size": S1_WINDOW_SIZE,
        "stride": S1_STRIDE,
        "image_side": image_side,
        "jpeg_quality": JPEG_QUALITY,
        "model": MODEL_NAME,
        "max_tokens": S1_MAX_TOKENS,
        "temperature": S1_TEMPERATURE,
        "prompt_version": PROMPT_VERSION,
        "output_format": OUTPUT_FORMAT,
    }

    frame_files = load_frames(FRAMES_DIR / video_id)
    windows = create_windows(frame_files)
    results = load_existing(windows_file, config, force)
    to_do = [w for w in windows if results.get(w["window_id"], {}).get("status") != "success"]

    print("=" * 60)
    print("COTT - STAGE 1: TRANSITION CAPTIONING")
    print("=" * 60)
    print(f"Video: {video_id} | frame: {len(frame_files)}")
    print(f"Finestre: {len(windows)} | da elaborare: {len(to_do)}")

    for position, window in enumerate(to_do, start=1):
        indices = window["frame_indices"]
        print(f"\n[{position}/{len(to_do)}] Finestra {window['window_id']}: frame {indices[0]}-{indices[-1]}")
        try:
            result = process_window(window, raw_dir, image_side, retries)
        except Exception as error:                 # es. server non raggiungibile
            print(f"  ERRORE: {type(error).__name__}: {error}")
            result = error_result(window, f"{type(error).__name__}: {error}")

        if result["status"] == "success":
            tag = f" ATTENZIONE: {result['warnings']}" if result["warnings"] else ""
            print(f"  OK ({len(result['transitions'])} transizioni){tag}")
        else:
            print(f"  FALLITA: {result['error']}")

        results[result["window_id"]] = result
        write_outputs(video_id, windows_file, flat_file, config, results)   # salva dopo ogni finestra
        time.sleep(S1_PAUSE)

    # riscrive comunque gli output, cosi' transitions_all.json esiste anche se non c'era niente da fare.
    write_outputs(video_id, windows_file, flat_file, config, results)

    print("\n" + "=" * 60)
    print(f"Completato Stage 1 per: {video_id}")