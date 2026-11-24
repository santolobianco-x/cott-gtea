"""
Punto di ingresso di CoTT-Lite (il modello online leggero del paper).

Uso:
    python main_lite.py features                     # 1) estrae le feature CLIP di tutti i video
    python main_lite.py train --labels stage3        # 2) CoTT-Lite sulle pseudo-label di CoTT (paper)
    python main_lite.py train --labels gt            #    controllo: limite superiore
    python main_lite.py train --labels grid          #    controllo: livello del caso
    python main_lite.py train --labels stage3_dedup  #    variante con la nostra deduplicazione

"train" esegue la cross-validation per soggetto (4 fold), l'inferenza in streaming
sui video di test e la valutazione sul ground truth.
"""

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def all_video_ids():
    return sorted(p.stem for p in (PROJECT_ROOT / "data" / "Videos").glob("*.mp4"))


def cmd_features(args):
    from src.lite.features import run, DEFAULT_MODEL
    video_ids = [args.video_id] if args.video_id else all_video_ids()
    run(video_ids, model_name=args.model or DEFAULT_MODEL, batch_size=args.batch_size, force=args.force)


def cmd_train(args):
    from src.lite.features import DEFAULT_MODEL
    from src.lite.train import run_cv
    video_ids = all_video_ids()
    run_cv(video_ids, args.labels, epochs=args.epochs, seed=args.seed, device_name=args.device,
           feature_model=args.model or DEFAULT_MODEL, tag=args.tag)


def main():
    parser = argparse.ArgumentParser(description="CoTT-Lite su GTEA")
    sub = parser.add_subparsers(dest="command", required=True)

    p_feat = sub.add_parser("features", help="Estrae le feature visive (CLIP) dai frame a 2 fps")
    p_feat.add_argument("--video_id", type=str, default=None, help="Un solo video (default: tutti)")
    p_feat.add_argument("--model", type=str, default=None,
                        help="Modello CLIP di Hugging Face (default: openai/clip-vit-base-patch16)")
    p_feat.add_argument("--batch-size", type=int, default=32)
    p_feat.add_argument("--force", action="store_true", help="Ricalcola anche le feature gia' presenti")
    p_feat.set_defaults(func=cmd_features)

    p_train = sub.add_parser("train", help="Cross-validation per soggetto + streaming + valutazione")
    p_train.add_argument("--labels", required=True, choices=("stage3", "stage3_dedup", "gt", "grid"),
                         help="Sorgente delle label di training")
    p_train.add_argument("--epochs", type=int, default=15)
    p_train.add_argument("--seed", type=int, default=0)
    p_train.add_argument("--device", type=str, default="cpu",
                         help="cpu (default, il piu' affidabile per le GRU), mps o auto")
    p_train.add_argument("--model", type=str, default=None, help="Modello delle feature (default: CLIP ViT-B/16)")
    p_train.add_argument("--tag", type=str, default=None)
    p_train.set_defaults(func=cmd_train)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()