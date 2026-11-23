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
from src.evaluate import run as run_evaluate, summarize_batch, evaluate_existing
from src.analyze_errors import run_analysis
from src.baselines import run as run_baseline, DEFAULT_IMAGE_SIDE as BASELINE_IMAGE_SIDE, GRID_LEN, GRID_PERIOD
from src.postprocess import run as run_dedup, DEDUP_TIOU, DEDUP_STRATEGY, STRATEGIES, MIN_CONF

PROJECT_ROOT = Path(__file__).resolve().parent


def check_stage1_health(video_id: str, max_fail_ratio: float = 0.5):
    windows_file = PROJECT_ROOT / "outputs" / video_id / "stage1" / "stage1_windows.json"
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


def process_video(vid: str, force: bool, dedup_thr: float, dedup_strategy: str, min_conf: float,
                  from_stage: int = 1):
    print(f"\n🚀 Avvio pipeline CoTT per: {vid} (da Stage {from_stage})\n")
    if from_stage <= 1:
        run_video_processing(vid, force=force)
        run_stage1(vid, force=force)
        if not check_stage1_health(vid):
            return None
        run_clean_captions(vid)
    if from_stage <= 2:
        run_stage2(vid, force=force)
    run_stage3(vid, force=force)
    run_dedup(vid, thr=dedup_thr, strategy=dedup_strategy, min_conf=min_conf)  # aggiunta, NON nel paper
    run_generate_srt(vid)
    return run_evaluate(vid)


def main():
    parser = argparse.ArgumentParser(description="Main Pipeline CoTT-Labeler su GTEA")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video_id", type=str, help="Singolo video (es. S1_Coffee_C1)")
    group.add_argument("--all", action="store_true", help="Elabora tutti i video in data/Videos/")
    parser.add_argument("--limit-videos", type=int, default=None, help="Con --all, elabora solo i primi N video")
    parser.add_argument("--force", action="store_true", help="Forza ricalcolo ignorando file/checkpoint esistenti")
    parser.add_argument("--eval-only", action="store_true",
                        help="Non esegue il modello: valuta solo gli output gia' presenti in outputs/")
    parser.add_argument("--analyze", action="store_true",
                        help="Non esegue il modello: analizza gli errori degli output gia' presenti")
    parser.add_argument("--from-stage", type=int, choices=(1, 2, 3), default=1,
                        help="Riparte da questo stage riusando gli output precedenti (es. 3 = solo refinement)")
    parser.add_argument("--dedup-thr", type=float, default=DEDUP_TIOU,
                        help="Soglia tIoU per considerare duplicate due predizioni dello Stage 3")
    parser.add_argument("--dedup-strategy", choices=STRATEGIES, default=DEDUP_STRATEGY,
                        help="Come fondere i duplicati: mean, union o keep")
    parser.add_argument("--min-conf", type=float, default=MIN_CONF,
                        help="Scarta le predizioni dello Stage 3 con confidenza inferiore (0 = filtro disattivato)")
    parser.add_argument("--baseline", choices=("slide", "whole", "grid"), default=None,
                        help="Esegue una baseline (slide/whole: Figura 1; grid: controllo cieco senza modello)")
    parser.add_argument("--grid-len", type=float, default=GRID_LEN,
                        help="Baseline grid: durata di ogni segmento in secondi")
    parser.add_argument("--grid-period", type=float, default=GRID_PERIOD,
                        help="Baseline grid: distanza tra l'inizio di due segmenti in secondi")
    parser.add_argument("--whole-fps", type=float, default=2,
                        help="Baseline whole: fps dei frame inviati al modello (paper: 2)")
    parser.add_argument("--image-side", type=int, default=BASELINE_IMAGE_SIDE,
                        help="Baseline: lato massimo delle immagini in pixel")
    parser.add_argument("--tag", type=str, default=None,
                        help="Nome dell'esperimento salvato nel CSV dei risultati (es. qwen3b_baseline)")
    args = parser.parse_args()

    if args.video_id:
        video_ids = [args.video_id]
    else:
        videos_dir = PROJECT_ROOT / "data" / "Videos"
        video_ids = sorted(p.stem for p in videos_dir.glob("*.mp4"))
        if args.limit_videos:
            video_ids = video_ids[:args.limit_videos]

    if args.eval_only or args.analyze:
        # La deduplicazione e' deterministica e istantanea: la ricalcoliamo sempre
        # dagli output dello Stage 3, cosi' si possono provare soglie e strategie diverse.
        for vid in video_ids:
            run_dedup(vid, thr=args.dedup_thr, strategy=args.dedup_strategy,
                      min_conf=args.min_conf, verbose=False)
        if args.eval_only:
            evaluate_existing(video_ids, tag=args.tag)
        if args.analyze:
            run_analysis(video_ids, tag=args.tag)
        return

    if args.baseline:
        all_metrics = []
        for idx, vid in enumerate(video_ids, 1):
            print(f"\n{'#'*60}\nBASELINE {args.baseline.upper()} - VIDEO {idx}/{len(video_ids)}: {vid}\n{'#'*60}")
            try:
                run_video_processing(vid, force=False)      # servono i frame a 2 fps
                run_baseline(vid, args.baseline, force=args.force,
                             image_side=args.image_side, whole_fps=args.whole_fps,
                             grid_len=args.grid_len, grid_period=args.grid_period)
                all_metrics.append(run_evaluate(vid, verbose=False))
            except Exception:
                print(f"\n❌ ERRORE su {vid}:")
                traceback.print_exc()
        if all_metrics:
            summarize_batch(all_metrics, tag=args.tag)
        return

    all_metrics = []
    for idx, vid in enumerate(video_ids, 1):
        print(f"\n{'#'*60}\nVIDEO {idx}/{len(video_ids)}: {vid}\n{'#'*60}")
        try:
            res = process_video(vid, args.force, args.dedup_thr, args.dedup_strategy, args.min_conf,
                                args.from_stage)
            if res:
                all_metrics.append(res)
        except Exception:
            print(f"\n❌ ERRORE su {vid}:")
            traceback.print_exc()
            if len(video_ids) == 1:
                sys.exit(1)

    if len(all_metrics) > 1:
        summarize_batch(all_metrics, tag=args.tag)


if __name__ == "__main__":
    main()