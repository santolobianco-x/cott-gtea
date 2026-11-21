import argparse
import json
import sys
import traceback
from pathlib import Path

from src.video_processing import run as run_video_processing
from src.generate_transition_captions import run as run_stage1
from src.clean_transition_captions import run as run_clean_captions
from src.action_grouping import run as run_stage2
from src.refine_action_instances import run as run_stage3
from src.generate_srt import run as run_generate_srt
from src.evaluate import run as run_evaluate, summarize_batch


def check_stage1_health(video_id: str, max_fail_ratio: float = 0.5):
    windows_file = Path(__file__).resolve().parent / "outputs" / video_id / "stage1" / "stage1_windows.json"
    if not windows_file.exists():
        return True
    data = json.loads(windows_file.read_text(encoding="utf-8"))
    windows = data.get("windows", [])
    if not windows:
        return True
    failed = sum(1 for w in windows if w.get("status") != "success")
    ratio = failed / len(windows)
    if ratio > max_fail_ratio:
        print(f"\n⚠️ ATTENZIONE: Stage 1 ha fallito su {failed}/{len(windows)} finestre ({ratio*100:.1f}%).")
        return False
    return True


def process_video(vid: str, force: bool):
    print(f"\n🚀 Avvio pipeline CoTT per: {vid}\n")
    run_video_processing(vid, force=force)
    run_stage1(vid, force=force)
    if not check_stage1_health(vid):
        return None
    run_clean_captions(vid)
    run_stage2(vid, force=force)
    run_stage3(vid, force=force)
    run_generate_srt(vid)
    return run_evaluate(vid)


def main():
    parser = argparse.ArgumentParser(description="Main Pipeline CoTT-Labeler su GTEA")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video_id", type=str, help="Singolo video (es. S1_Coffee_C1)")
    group.add_argument("--all", action="store_true", help="Elabora tutti i video in data/Videos/")
    parser.add_argument("--limit-videos", type=int, default=None, help="Con --all, elabora solo i primi N video")
    parser.add_argument("--force", action="store_true", help="Forza ricalcolo ignorando file/checkpoint esistenti")
    args = parser.parse_args()

    if args.video_id:
        video_ids = [args.video_id]
    else:
        videos_dir = Path(__file__).resolve().parent / "data" / "Videos"
        video_ids = sorted(p.stem for p in videos_dir.glob("*.mp4"))
        if args.limit_videos:
            video_ids = video_ids[:args.limit_videos]

    all_metrics = []
    for idx, vid in enumerate(video_ids, 1):
        print(f"\n{'#'*60}\nVIDEO {idx}/{len(video_ids)}: {vid}\n{'#'*60}")
        try:
            res = process_video(vid, args.force)
            if res:
                all_metrics.append(res)
        except Exception:
            print(f"\n❌ ERRORE su {vid}:")
            traceback.print_exc()
            if len(video_ids) == 1:
                sys.exit(1)

    if len(all_metrics) > 1:
        summarize_batch(all_metrics)


if __name__ == "__main__":
    main()