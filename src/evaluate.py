"""
Valutazione class-agnostic su GTEA (Hungarian F1 a diverse soglie di tIoU).

Riporta due tipi di metrica:
  - pooled (micro): TP, n_pred e n_gt sommati su tutti i video, poi F1.
                    E' il modo standard di riportare F1 nei paper di On-TAL.
  - mean (macro):   media degli F1 calcolati video per video.

Ogni valutazione viene salvata in CSV, cosi' ogni esperimento diventa una riga
della tabella dei risultati.

Modulo richiamato da main.py (non si esegue da solo).
"""

import csv
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GT_DIR = PROJECT_ROOT / "data" / "gtea_labels_71" / "labels"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
RESULTS_DIR = PROJECT_ROOT / "results"

GTEA_FPS = 15.0
THRESHOLDS = (0.3, 0.5, 0.7)
STAGES = {
    "stage2": ("Stage 2 (Grouping)", Path("stage2") / "action_grouping_stage2.json"),
    "stage3": ("Stage 3 (Refined)", Path("stage3") / "refined_actions.json"),
    # Post-processing aggiunto (NON nel paper): vedi src/postprocess.py
    "stage3_dedup": ("Stage 3 + Conf/Dedup (aggiunta)", Path("stage3") / "refined_actions_dedup.json"),
    # Baseline di prompting diretto (Figura 1 del paper): vedi src/baselines.py
    "baseline_slide": ("Baseline sliding-window", Path("baseline_slide") / "predictions.json"),
    "baseline_whole": ("Baseline whole-video", Path("baseline_whole") / "predictions.json"),
    "baseline_grid": ("Controllo cieco: griglia fissa", Path("baseline_grid") / "predictions.json"),
    # CoTT-Lite (src/lite/): modello online addestrato su label diverse, valutato sempre sul GT
    "lite_stage3": ("CoTT-Lite (label: Stage 3)", Path("lite_stage3") / "predictions.json"),
    "lite_stage3_dedup": ("CoTT-Lite (label: Stage 3 + Dedup)", Path("lite_stage3_dedup") / "predictions.json"),
    "lite_gt": ("CoTT-Lite (label: GT, limite sup.)", Path("lite_gt") / "predictions.json"),
    "lite_grid": ("CoTT-Lite (label: griglia, caso)", Path("lite_grid") / "predictions.json"),
}

# Cattura SOLO le azioni: <verb><noun1, noun2> (start-end) [0/1]
# Ignora le righe di presenza oggetti a singolo tag (<cup> (30-64))
ACTION_LINE_RE = re.compile(r"^<[^>]+><[^>]+>\s*\((\d+)-(\d+)\)")


# ============================================================
# CARICAMENTO DATI
# ============================================================
def load_gtea_gt(label_path: Path):
    if not label_path.exists():
        raise FileNotFoundError(f"File ground truth non trovato: {label_path}")

    segments = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        match = ACTION_LINE_RE.match(line.strip())
        if match:
            start_frame, end_frame = int(match.group(1)), int(match.group(2))
            segments.append((start_frame / GTEA_FPS, end_frame / GTEA_FPS))
    return segments


def load_predictions(json_path: Path):
    if not json_path.exists():
        return None
    data = json.loads(json_path.read_text(encoding="utf-8"))
    actions = data["actions"] if isinstance(data, dict) else data
    return [(float(a["start_time"]), float(a["end_time"])) for a in actions]


def subject_of(video_id: str) -> str:
    """S1_Coffee_C1 -> S1"""
    return video_id.split("_")[0]


# ============================================================
# METRICHE
# ============================================================
def compute_tiou(seg_a, seg_b):
    inter = max(0.0, min(seg_a[1], seg_b[1]) - max(seg_a[0], seg_b[0]))
    union = max(seg_a[1], seg_b[1]) - min(seg_a[0], seg_b[0])
    return inter / union if union > 0 else 0.0


def f1_from_counts(tp, n_pred, n_gt):
    precision = tp / n_pred if n_pred else 0.0
    recall = tp / n_gt if n_gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"f1": f1 * 100, "precision": precision * 100, "recall": recall * 100}


def hungarian_f1(preds, gts, tiou_thresh=0.5):
    n_pred, n_gt = len(preds), len(gts)
    tp = 0
    if n_pred and n_gt:
        iou = np.array([[compute_tiou(p, g) for g in gts] for p in preds])
        rows, cols = linear_sum_assignment(-iou)
        tp = int(np.sum(iou[rows, cols] >= tiou_thresh))
    return {**f1_from_counts(tp, n_pred, n_gt), "tp": tp, "n_pred": n_pred, "n_gt": n_gt}


# ============================================================
# VALUTAZIONE SINGOLO VIDEO
# ============================================================
def run(video_id: str, thresholds=THRESHOLDS, verbose=True):
    gts = load_gtea_gt(GT_DIR / f"{video_id}.txt")
    metrics = {"video_id": video_id, "n_gt": len(gts)}

    if verbose:
        print("\n" + "=" * 60)
        print(f"VALUTAZIONE HUNGARIAN F1 - {video_id} (GT actions: {len(gts)})")
        print("=" * 60)

    for key, (label, rel_path) in STAGES.items():
        metrics[key] = {}
        preds = load_predictions(OUTPUTS_DIR / video_id / rel_path)
        if preds is None:
            if verbose:
                print(f"\n[{label}] File non trovato.")
            continue
        if verbose:
            print(f"\n[{label}] Predizioni: {len(preds)}")
        for th in thresholds:
            res = hungarian_f1(preds, gts, tiou_thresh=th)
            metrics[key][th] = res
            if verbose:
                print(
                    f"  tIoU={th:.1f} | F1: {res['f1']:5.1f}% | "
                    f"Prec: {res['precision']:5.1f}% | Rec: {res['recall']:5.1f}% "
                    f"(TP={res['tp']}/{res['n_gt']})"
                )
    return metrics


# ============================================================
# AGGREGAZIONE SU PIU' VIDEO
# ============================================================
def aggregate(results_list, stage, th):
    vals = [r[stage][th] for r in results_list if th in r.get(stage, {})]
    if not vals:
        return None
    tp = sum(v["tp"] for v in vals)
    n_pred = sum(v["n_pred"] for v in vals)
    n_gt = sum(v["n_gt"] for v in vals)
    pooled = f1_from_counts(tp, n_pred, n_gt)
    return {
        "n_videos": len(vals),
        "tp": tp, "n_pred": n_pred, "n_gt": n_gt,
        "pooled_f1": pooled["f1"], "pooled_prec": pooled["precision"], "pooled_rec": pooled["recall"],
        "mean_f1": float(np.mean([v["f1"] for v in vals])),
        "mean_prec": float(np.mean([v["precision"] for v in vals])),
        "mean_rec": float(np.mean([v["recall"] for v in vals])),
    }


def subsets(results_list):
    """Tutti i video + i 4 split leave-one-subject-out (test = un soggetto)."""
    yield "all", results_list
    for s in sorted({subject_of(r["video_id"]) for r in results_list}):
        yield f"test_{s}", [r for r in results_list if subject_of(r["video_id"]) == s]


def summarize_batch(results_list, thresholds=THRESHOLDS, tag=None, save_csv=True):
    if not results_list:
        return []

    tag = tag or "untagged"
    rows = []
    print("\n" + "#" * 72)
    print(f"RIEPILOGO [{tag}] su {len(results_list)} video")
    print("#" * 72)

    for stage, (label, _) in STAGES.items():
        if aggregate(results_list, stage, thresholds[0]) is None:
            continue  # nessun output per questo stage/baseline
        print(f"\n[{label}]")
        print(f"  {'subset':<9} {'tIoU':>4} | {'pooled F1':>9} {'P':>6} {'R':>6} | {'mean F1':>8} | {'#pred':>6} {'#gt':>5}")
        for subset_name, subset in subsets(results_list):
            for th in thresholds:
                agg = aggregate(subset, stage, th)
                if agg is None:
                    continue
                print(
                    f"  {subset_name:<9} {th:>4.1f} | {agg['pooled_f1']:9.1f} {agg['pooled_prec']:6.1f} "
                    f"{agg['pooled_rec']:6.1f} | {agg['mean_f1']:8.1f} | {agg['n_pred']:6d} {agg['n_gt']:5d}"
                )
                rows.append({"tag": tag, "stage": stage, "subset": subset_name, "tiou": th, **agg})

    if save_csv:
        save_results(results_list, rows, tag)
    return rows


def save_results(results_list, summary_rows, tag):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1) Riepilogo: una riga per (stage, subset, tIoU), appesa a un unico file cumulativo
    summary_file = RESULTS_DIR / "summary.csv"
    write_header = not summary_file.exists()
    with open(summary_file, "a", newline="", encoding="utf-8") as f:
        fields = ["timestamp", *summary_rows[0].keys()]
        writer = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            writer.writeheader()
        for row in summary_rows:
            writer.writerow({"timestamp": stamp, **{k: round(v, 2) if isinstance(v, float) else v for k, v in row.items()}})

    # 2) Dettaglio per video di questo esperimento
    per_video_file = RESULTS_DIR / f"per_video_{tag}_{stamp}.csv"
    with open(per_video_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video_id", "stage", "tiou", "f1", "precision", "recall", "tp", "n_pred", "n_gt"])
        for r in results_list:
            for stage in STAGES:
                for th, m in sorted(r.get(stage, {}).items()):
                    writer.writerow([r["video_id"], stage, th, round(m["f1"], 2), round(m["precision"], 2),
                                     round(m["recall"], 2), m["tp"], m["n_pred"], m["n_gt"]])

    print(f"\nRisultati salvati in: {summary_file} e {per_video_file}")


# ============================================================
# VALUTAZIONE DI OUTPUT GIA' ESISTENTI (senza rieseguire il modello)
# ============================================================
def evaluate_existing(video_ids, tag=None):
    """Valuta gli output gia' presenti in outputs/ per i video indicati."""
    video_ids = [v for v in video_ids if (OUTPUTS_DIR / v).is_dir()]
    if not video_ids:
        print("Nessun output trovato da valutare.")
        return []
    results = [run(v, verbose=len(video_ids) == 1) for v in video_ids]
    if len(results) > 1:
        summarize_batch(results, tag=tag)
    return results