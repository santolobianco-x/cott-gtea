import json
import re
from pathlib import Path


def format_timestamp(seconds_float):
    total_ms = max(0, round(float(seconds_float) * 1000))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    seconds, ms = divmod(rem, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{ms:03d}"


def run(video_id: str):
    project_root = Path(__file__).resolve().parents[1]
    input_json = project_root / "outputs" / video_id / "stage3" / "refined_actions.json"
    output_srt = project_root / "outputs" / video_id / f"{video_id}.srt"

    if not input_json.exists():
        raise FileNotFoundError(f"File JSON non trovato: {input_json}")

    with open(input_json, "r", encoding="utf-8") as f:
        actions = json.load(f)

    actions.sort(key=lambda a: (float(a["start_time"]), float(a["end_time"])))

    with open(output_srt, "w", encoding="utf-8") as srt_file:
        for i, action in enumerate(actions, start=1):
            start_str = format_timestamp(action["start_time"])
            end_str = format_timestamp(action["end_time"])
            desc = re.sub(r"</?(start|end)>", "", str(action["description"])).strip()
            tag = "" if action.get("refined", True) else " [Stage2]"

            srt_file.write(f"{i}\n")
            srt_file.write(f"{start_str} --> {end_str}\n")
            srt_file.write(f"{desc}{tag}\n\n")

    print(f"\nSottotitoli salvati in: {output_srt}")