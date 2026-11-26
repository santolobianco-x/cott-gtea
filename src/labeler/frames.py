import cv2
from src.config import FPS, FRAMES_DIR, VIDEOS_DIR


"""
MODULO INCARICATO ALL'ESTRAZIONE DEI FRAME

--> dal video orignale di GTEA salva un frame ogni 1/FPS(FPS =2 come nel paper)

"""


def count_expected_frames(original_fps, total_frames, target_fps):
    """Conta quanti frame verranno estratti."""
    count = 0
    while round((count / target_fps) * original_fps) <total_frames:
        count += 1
    return count


def extract_frames(video_path, output_dir, target_fps=FPS, force=False):
    """Estrae un frame ogni 1/target_fps secondi e lo salva come JPEG in output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path)) # crea un lettore associato al video
    if not cap.isOpened():
        raise RuntimeError(f"Impossibile aprire il video: {video_path}")

    original_fps = cap.get(cv2.CAP_PROP_FPS) # assegnazione degli fps originali
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


    if original_fps == 0:
        cap.release()
        raise ValueError("Non e' stato possibile leggere gli FPS del video.")


    expected = count_expected_frames(original_fps, total_frames, target_fps)

    # si verifica se i frame sono già stati estratti
    # nel caso in cui siano stati già tutti estratti allora si salta(a meno del caso force)
    if not force and len(list(output_dir.glob("frame_*.jpg"))) == expected:
        cap.release()
        print(f"Frame gia' estratti per {video_path.name} ({expected}). Salto.")
        return

    print(f"\nESTRAZIONE FRAME: {video_path.name} | {original_fps:.0f} fps -> {target_fps} fps "
          f"| durata {total_frames / original_fps:.1f} s")

    k = 0
    while True:
        
        source_frame = round((k / target_fps) * original_fps)# istante k/target_fps secondi -> frame piu' vicino nel video originale.
        if source_frame >= total_frames:
            break

        cap.set(cv2.CAP_PROP_POS_FRAMES, source_frame)# cambia l'etichetta della posizione in frame
        ok, frame = cap.read()
        if not ok:
            break

        cv2.imwrite(str(output_dir / f"frame_{k:05d}.jpg"), frame)
        k += 1

    cap.release()
    print(f"Frame estratti: {k} -> {output_dir}")


def run(video_id, target_fps=FPS, force=False):
    """Punto di ingresso usato da main.py: estrae i frame del video indicato."""
    video_path = VIDEOS_DIR / f"{video_id}.mp4"
    if not video_path.exists():
        raise FileNotFoundError(f"Video non trovato: {video_path}")
    extract_frames(video_path, FRAMES_DIR / video_id, target_fps=target_fps, force=force)