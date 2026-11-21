import argparse
import json
import re
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

GTEA_FPS = 15.0
# Cattura SOLO le azioni: <verb><noun1, noun2> (start-end) [0/1]
# Ignora le righe di presenza oggetti a singolo tag (<cup> (30-64))
ACTION_LINE_RE = re.compile(r"^<[^>]+><[^>]+>\s*\((\d+)-(\d+)\)")


def load_gtea_gt(label_path: Path):
    if not label_path.exists():
        raise FileNotFoundError(f"File ground truth non trovato: {label_path}")

    segments = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        match = ACTION_LINE_RE.match(line)
        if match:
            start_frame = int(match.group(1))
            end_frame = int(match.group(2))
            segments.append((start_frame / GTEA_FPS, end_frame / GTEA_FPS))
    return segments


def load_predictions(json_path: Path):
    if not json_path.exists():
        return None
    data = json.loads(json_path.read_text(encoding="utf-8"))
    actions = data["actions"] if isinstance(data, dict) else data
    return [(float(a["start_time"]), float(a["end_time"])) for a in actions]


def compute_tiou(seg_a, seg_b):
    inter = max(0.0, min(seg_a[1], seg_b[1]) - max(seg_a[0], seg_b[0]))
    union = max(seg_a[1], seg_b[1]) - min(seg_a[0], seg_b[0])
    return inter / union if union > 0 else 0.0


def hungarian_f1(preds, gts, tiou_thresh=0.5):
    n_pred, n_gt = len(preds), len(gts)
    if n_pred == 0 or n_gt == 0:
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0, "tp": 0, "n_pred": n_pred, "n_gt": n_gt}

    iou_matrix = np.zeros((n_pred, n_gt), dtype=np.float64)
    for i, p in enumerate(preds):
        for j, g in enumerate(gts):
            iou_matrix[i, j] = compute_tiou(p, g)

    row_ind, col_ind = linear_sum_assignment(-iou_matrix)
    tp = int(sum(1 for r, c in zip(row_ind, col_ind) if iou_matrix[r, c] >= tiou_thresh))

    precision = tp / n_pred
    recall = tp / n_gt
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "f1": f1 * 100,
        "precision": precision * 100,
        "recall": recall * 100,
        "tp": tp,
        "n_pred": n_pred,
        "n_gt": n_gt,
    }


def run(video_id: str, thresholds=(0.3, 0.5, 0.7)):
    project_root = Path(__file__).resolve().parents[1]
    gt_path = project_root / "data" / "gtea_labels_71" / "labels" / f"{video_id}.txt"
    stage2_path = project_root / "outputs" / video_id / "stage2" / "action_grouping_stage2.json"
    stage3_path = project_root / "outputs" / video_id / "stage3" / "refined_actions.json"

    gts = load_gtea_gt(gt_path)
    print("\n" + "=" * 60)
    print(f"VALUTAZIONE HUNGARIAN F1 - {video_id} (GT actions: {len(gts)})")
    print("=" * 60)

    for label, path in [("Stage 2 (Grouping)", stage2_path), ("Stage 3 (Refined)", stage3_path)]:
        preds = load_predictions(path)
        if preds is None:
            print(f"\n[{label}] File non trovato: {path}")
            continue

        print(f"\n[{label}] Predizioni: {len(preds)}")
        for th in thresholds:
            res = hungarian_f1(preds, gts, tiou_thresh=th)
            print(
                f"  tIoU={th:.1f} | F1: {res['f1']:5.1f}% | "
                f"Prec: {res['precision']:5.1f}% | Rec: {res['recall']:5.1f}% "
                f"(TP={res['tp']}/{res['n_gt']})"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Valutazione Hungarian F1 su GTEA")
    parser.add_argument("--video_id", type=str, required=True)
    args = parser.parse_args()
    run(args.video_id)


def run(video_id: str, thresholds=(0.3, 0.5, 0.7)):
    project_root = Path(__file__).resolve().parents[1]
    gt_path = project_root / "data" / "gtea_labels_71" / "labels" / f"{video_id}.txt"
    stage2_path = project_root / "outputs" / video_id / "stage2" / "action_grouping_stage2.json"
    stage3_path = project_root / "outputs" / video_id / "stage3" / "refined_actions.json"

    gts = load_gtea_gt(gt_path)
    print("\n" + "=" * 60)
    print(f"VALUTAZIONE HUNGARIAN F1 - {video_id} (GT actions: {len(gts)})")
    print("=" * 60)

    metrics = {"video_id": video_id, "n_gt": len(gts), "stage2": {}, "stage3": {}}

    for key, label, path in [
        ("stage2", "Stage 2 (Grouping)", stage2_path),
        ("stage3", "Stage 3 (Refined)", stage3_path),
    ]:
        preds = load_predictions(path)
        if preds is None:
            print(f"\n[{label}] File non trovato: {path}")
            continue

        print(f"\n[{label}] Predizioni: {len(preds)}")
        for th in thresholds:
            res = hungarian_f1(preds, gts, tiou_thresh=th)
            metrics[key][th] = res
            print(
                f"  tIoU={th:.1f} | F1: {res['f1']:5.1f}% | "
                f"Prec: {res['precision']:5.1f}% | Rec: {res['recall']:5.1f}% "
                f"(TP={res['tp']}/{res['n_gt']})"
            )

    return metrics


def summarize_batch(results_list, thresholds=(0.3, 0.5, 0.7)):
    if not results_list:
        return
    print("\n" + "#" * 60)
    print(f"RIEPILOGO MEDIO SU {len(results_list)} VIDEO")
    print("#" * 60)

    for key, label in [("stage2", "Stage 2 (Grouping)"), ("stage3", "Stage 3 (Refined)")]:
        print(f"\n[{label}]")
        for th in thresholds:
            vals = [r[key][th] for r in results_list if th in r[key]]
            if not vals:
                continue
            avg_f1 = np.mean([v["f1"] for v in vals])
            avg_p = np.mean([v["precision"] for v in vals])
            avg_r = np.mean([v["recall"] for v in vals])
            print(f"  tIoU={th:.1f} | Mean F1: {avg_f1:5.1f}% | Mean Prec: {avg_p:5.1f}% | Mean Rec: {avg_r:5.1f}%")