import json
from src.config import OUTPUTS_DIR


"""
CoTT - Stage 1: una sola versione per ogni transizione.

Le finestre dello Stage 1 si sovrappongono a meta' (12 frame, passo 6), quindi le
transizioni interne vengono descritte due volte, da due finestre diverse.

Questo passo tiene una versione per ogni transizione e controlla che i dati siano coerenti.
"""

# campi che ogni transizione deve avere
REQUIRED_FIELDS = [
    "transition_id",
    "start_time",
    "end_time",
    "frame_start",
    "frame_end",
    "description",
    "status",
]


def clean_text(value, transition_id, field_name):
    """Ripulisce un campo di testo: toglie gli spazi ai bordi e trasforma in None
    i valori vuoti o la parola "null" """

    if value is None:
        return None
    
    if not isinstance(value, str): # verifica se il valore è stringa di testo
        raise ValueError(f"Transizione {transition_id}: il campo '{field_name}' deve essere una stringa o null.")
    value = value.strip()
    
    if not value or value.lower() == "null":
        return None
    
    return value


def clean_transition(transition):
    """Controlla una transizione e ne restituisce una copia con solo i campi utili
    (toglie ad esempio window_id, che dopo la deduplicazione non serve piu')."""

    if not isinstance(transition, dict): # verifica se è un'istanza di un dizionario
        raise ValueError("Ogni transizione deve essere un oggetto JSON.")

    
    for field in REQUIRED_FIELDS: # verifica se i campi ci sono tutti
        if field not in transition:
            raise ValueError(f"Campo obbligatorio mancante: {field}")

    tid = transition["transition_id"]

    if transition["end_time"] <= transition["start_time"]: # una transizione deve avere durata > 0
        raise ValueError(f"Transizione {tid}: end_time deve essere > start_time.")

    return {
        "transition_id": tid,
        "start_time": transition["start_time"],
        "end_time": transition["end_time"],
        "frame_start": transition["frame_start"],
        "frame_end": transition["frame_end"],
        "description": clean_text(transition["description"], tid, "description"),
        "status": clean_text(transition["status"], tid, "status"),
    }


def save_json(path, data):
    """Scrittura atomica(file temporaneo + rinomina)"""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")

    tmp.write_text(json.dumps(data, indent=4, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def clean_captions(input_file, output_file):
    """Legge tutte le transizioni, tiene la prima occorrenza di ognuna e salva il risultato."""

    if not input_file.exists():
        raise FileNotFoundError(f"File di input non trovato: {input_file}")

    transitions = json.loads(input_file.read_text(encoding="utf-8"))

    # Deduplicazione. 
    # transitions_all.json e' ordinato per (tempo, finestra): per ogni
    # transizione la PRIMA occorrenza e' quindi quella della finestra precedente.
    cleaned_transitions = []
    seen_ids = set() # transition_id gia' presi (insieme: ricerca veloce)
    for transition in transitions:
        cleaned = clean_transition(transition)
        if cleaned["transition_id"] not in seen_ids:
            seen_ids.add(cleaned["transition_id"])
            cleaned_transitions.append(cleaned)


    cleaned_transitions.sort(key=lambda t: t["start_time"])

    save_json(output_file, cleaned_transitions)

    print("\n" + "=" * 55)
    print("PULIZIA TRANSITION CAPTIONS COMPLETATA")
    print("=" * 55)
    print(f"Transizioni lette (con i doppioni delle finestre sovrapposte): {len(transitions)}")
    print(f"Transizioni uniche salvate: {len(cleaned_transitions)}")
    print(f"File salvato in: {output_file}")


def run(video_id):
    """Punto di ingresso usato da main.py: pulisce le transizioni dello Stage 1 di un video."""
    stage1_dir = OUTPUTS_DIR / video_id / "stage1"
    clean_captions(stage1_dir / "transitions_all.json", stage1_dir / "transitions_cleaned.json")