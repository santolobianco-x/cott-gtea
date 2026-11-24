import cv2
from pathlib import Path


def extract_frames(video_path, output_dir, target_fps=2, force=False):
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Impossibile aprire il video: {video_path}")

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if original_fps == 0:
        cap.release()
        raise ValueError("Non è stato possibile calcolare gli FPS dal video.")

    duration = total_frames / original_fps
    # Calcola il numero esatto di frame attesi dal ciclo
    expected_count = 0
    while round((expected_count / target_fps) * original_fps) < total_frames:
        expected_count += 1

    existing_frames = list(output_dir.glob("frame_*.jpg"))
    if not force and len(existing_frames) == expected_count:
        cap.release()
        print(f"⏩ Estrazione frame già completa per {video_path.name} ({expected_count}/{expected_count} frame). Salto.")
        return

    print("\n" + "=" * 55)
    print("ESTRAZIONE FRAME")
    print("=" * 55)
    print(f"Video: {video_path.name} | FPS: {original_fps:.2f} -> {target_fps} | Durata: {duration:.2f}s")

    extracted_count = 0
    while True:
        timestamp = extracted_count / target_fps
        target_frame = round(timestamp * original_fps)
        if target_frame >= total_frames:
            break

        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = cap.read()
        if not ret:
            break

        filename = output_dir / f"frame_{extracted_count:05d}.jpg"
        cv2.imwrite(str(filename), frame)
        extracted_count += 1

    cap.release()
    print(f"Frame estratti: {extracted_count} | Salvati in: {output_dir}")


def run(video_id: str, target_fps: int = 2, force: bool = False):
    project_root = Path(__file__).resolve().parents[1]
    video_path = project_root / "data" / "Videos" / f"{video_id}.mp4"
    output_dir = project_root / "outputs" / "frames" / video_id

    if not video_path.exists():
        raise FileNotFoundError(f"File video non trovato: {video_path}")

    extract_frames(video_path, output_dir, target_fps=target_fps, force=force)