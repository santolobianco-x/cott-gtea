import json
from pathlib import Path

REQUIRED_METADATA = [
    "transition_id",
    "start_time",
    "end_time",
    "frame_start",
    "frame_end",
    "description",
    "status"
]

def clean_text(value, transition_id, field_name):
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"Transizione {transition_id}: il campo '{field_name}' deve essere una stringa o null.")
    value = value.strip()
    if not value or value.lower() == "null":
        return None
    return value

def clean_transition(transition):
    if not isinstance(transition, dict):
        raise ValueError("Ogni transizione deve essere un oggetto JSON.")
    for field in REQUIRED_METADATA:
        if field not in transition:
            raise ValueError(f"Campo obbligatorio mancante: {field}")
            
    tid = transition["transition_id"]
    if transition["end_time"] <= transition["start_time"]:
        raise ValueError(f"Transizione {tid}: end_time deve essere > start_time.")

    cleaned = {
        "transition_id": tid,
        "start_time": transition["start_time"],
        "end_time": transition["end_time"],
        "frame_start": transition["frame_start"],
        "frame_end": transition["frame_end"],
        "description": clean_text(transition["description"], tid, "description"),
        "status": clean_text(transition["status"], tid, "status")
    }
    return cleaned

def clean_captions(input_file, output_file):
    if not input_file.exists():
        raise FileNotFoundError(f"File di input non trovato: {input_file}")

    with open(input_file, "r", encoding="utf-8") as file:
        transitions = json.load(file)

    cleaned_transitions = []
    seen_ids = set()

    for transition in transitions:
        cleaned = clean_transition(transition)
        tid = cleaned["transition_id"]
        
        # Deduplica: siccome usiamo sliding windows con stride, la stessa transizione
        # comparirà più volte. Teniamo la prima occorrenza valida.
        if tid not in seen_ids:
            seen_ids.add(tid)
            cleaned_transitions.append(cleaned)

    cleaned_transitions.sort(key=lambda x: x["start_time"])

    for i in range(1, len(cleaned_transitions)):
        if cleaned_transitions[i]["start_time"] < cleaned_transitions[i - 1]["start_time"]:
            raise ValueError("Le transizioni non sono ordinate temporalmente.")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(cleaned_transitions, file, indent=4, ensure_ascii=False)

    print("\n" + "=" * 55)
    print("PULIZIA TRANSITION CAPTIONS COMPLETATA")
    print("=" * 55)
    print(f"Transizioni elaborate (incluse overlap): {len(transitions)}")
    print(f"Transizioni uniche salvate: {len(cleaned_transitions)}")
    print(f"File salvato in: {output_file}")

def run(video_id: str):
    """
    Pulisce e deduplica le Transition Captions (Stage 1).
    """
    project_root = Path(__file__).resolve().parents[1]
    input_file = project_root / "outputs" / video_id / "stage1" / "transitions_all.json"
    output_file = project_root / "outputs" / video_id / "stage1" / "transitions_cleaned.json"

    clean_captions(input_file, output_file)