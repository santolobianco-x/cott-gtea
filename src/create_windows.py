import json
from pathlib import Path

# ============================================================
# CONFIGURAZIONE
# ============================================================
WINDOW_SIZE = 12
STRIDE = 6
FPS = 2

def run(video_id: str):
    """
    Crea le finestre di frame (sliding windows) per il video specificato.
    """
    project_root = Path(__file__).resolve().parents[1]
    
    frame_dir = project_root / "outputs" / "frames" / video_id
    output_dir = project_root / "outputs" / "windows" / video_id

    output_dir.mkdir(parents=True, exist_ok=True)

    if not frame_dir.exists():
        raise FileNotFoundError(f"Cartella frame non trovata: {frame_dir}\nAssicurati di aver prima estratto i frame.")

    frame_files = sorted(frame_dir.glob("*.jpg"))

    windows = []

    for start in range(0, len(frame_files) - WINDOW_SIZE + 1, STRIDE):
        selected_frames = frame_files[start:start+WINDOW_SIZE]
        window = {
            "window_id": len(windows),
            "start_time": start / FPS,
            "end_time": (start + WINDOW_SIZE) / FPS,
            "frames": [str(path.name) for path in selected_frames]  # Salvo solo il nome per pulizia
        }

        windows.append(window)
        
    output_file = output_dir / "windows.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(windows, f, indent=4, ensure_ascii=False)

    print("\n" + "=" * 55)
    print("CREAZIONE FINESTRE (SLIDING WINDOWS) COMPLETATA")
    print("=" * 55)
    print(f"Video: {video_id}")
    print(f"Frame totali: {len(frame_files)}")
    print(f"Finestre create: {len(windows)}")
    print(f"File salvato in: {output_file}")