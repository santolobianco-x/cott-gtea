"""
Analisi degli errori delle predizioni CoTT-Labeler su GTEA.

Non chiama il modello: legge gli output gia' presenti in outputs/ e li confronta
con il ground truth per capire PERCHE' le predizioni sbagliano.

Analisi prodotte (per Stage 2 e Stage 3):
  1. Statistiche di granularita': numero di segmenti, durata, copertura del video.
  2. Errori di localizzazione: di quanto sbagliano inizio e fine le predizioni
     che "colpiscono" un'azione (bias sistematico in anticipo/ritardo).
  3. Categorie di falsi positivi (predizioni non accoppiate a tIoU=0.5):
       - duplicate      : si sovrappone a un'azione GT gia' trovata da un'altra predizione
       - localization   : si sovrappone a un'azione GT ma con tIoU < 0.5
       - background     : non si sovrappone (quasi) a nessuna azione GT
  4. Categorie di falsi negativi (azioni GT non trovate a tIoU=0.5):
       - fragmented     : coperta da 2+ predizioni (l'azione e' stata spezzata)
       - partial        : coperta da una predizione con tIoU < 0.5
       - missed         : nessuna predizione si sovrappone
  5. Frammentazione: predizioni adiacenti con etichetta simile (stessa azione spezzata).
  6. What-if: F1 se si uniscono le predizioni adiacenti con etichetta simile
     (regola fissa, NON ottimizzata sul GT: serve solo come diagnosi).

Modulo richiamato da main.py (opzione --analyze).
"""

import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.evaluate import (
    GT_DIR, OUTPUTS_DIR, RESULTS_DIR, STAGES,
    compute_tiou, f1_from_counts, hungarian_f1, load_gtea_gt,
)

FPS_FRAMES = 2
TIOU_EVAL = 0.5        # soglia usata per definire TP / FP / FN
TIOU_TOUCH = 0.1       # sotto questa soglia due segmenti si considerano "non sovrapposti"
MERGE_MAX_GAP = 0.5    # secondi: gap massimo per considerare due predizioni adiacenti
LABEL_SIM_MIN = 0.5    # similarita' minima (Jaccard sulle parole) per "etichetta simile"

STOPWORDS = {"the", "a", "an", "of", "to", "into", "in", "on", "from", "with", "and", "it"}


# ============================================================
# CARICAMENTO
# ============================================================
def load_predictions_with_labels(json_path: Path):
    """Ritorna lista di (start, end, label) ordinata per inizio, oppure None."""
    if not json_path.exists():
        return None
    data = json.loads(json_path.read_text(encoding="utf-8"))
    actions = data["actions"] if isinstance(data, dict) else data
    preds = []
    for a in actions:
        label = a.get("description") or a.get("action_label") or ""
        label = re.sub(r"</?(start|end)>", "", str(label)).strip()
        preds.append((float(a["start_time"]), float(a["end_time"]), label))
    return sorted(preds, key=lambda p: (p[0], p[1]))


def video_duration(video_id: str, gts, preds):
    frames_dir = OUTPUTS_DIR / "frames" / video_id
    n_frames = len(list(frames_dir.glob("*.jpg"))) if frames_dir.exists() else 0
    if n_frames:
        return n_frames / FPS_FRAMES
    ends = [g[1] for g in gts] + [p[1] for p in preds]
    return max(ends) if ends else 0.0


# ============================================================
# UTILITA'
# ============================================================
def label_tokens(label: str):
    words = re.findall(r"[a-z]+", label.lower())
    return {w for w in words if w not in STOPWORDS}


def label_similarity(a: str, b: str):
    ta, tb = label_tokens(a), label_tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def union_length(segments):
    """Durata totale dell'unione di segmenti (senza contare due volte le sovrapposizioni)."""
    total, cur_s, cur_e = 0.0, None, None
    for s, e in sorted((s, e) for s, e, *_ in segments):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += cur_e - cur_s
    return total


def iou_matrix(preds, gts):
    return np.array([[compute_tiou(p[:2], g) for g in gts] for p in preds]) if preds and gts else np.zeros((len(preds), len(gts)))


def merge_similar_adjacent(preds):
    """Unisce predizioni consecutive vicine (gap <= MERGE_MAX_GAP) con etichetta simile."""
    if not preds:
        return []
    merged = [list(preds[0])]
    for s, e, lab in preds[1:]:
        ps, pe, plab = merged[-1]
        if s - pe <= MERGE_MAX_GAP and label_similarity(lab, plab) >= LABEL_SIM_MIN:
            merged[-1][1] = max(pe, e)
        else:
            merged.append([s, e, lab])
    return [tuple(m) for m in merged]


# ============================================================
# ANALISI DI UN VIDEO PER UNO STAGE
# ============================================================
def analyze_video(preds, gts, duration):
    out = {
        "n_pred": len(preds), "n_gt": len(gts),
        "pred_durations": [e - s for s, e, _ in preds],
        "gt_durations": [e - s for s, e in gts],
        "pred_coverage": union_length(preds), "gt_coverage": union_length(gts), "duration": duration,
        "fp": Counter(), "fn": Counter(),
        "start_errors": [], "end_errors": [],
        "fragment_pairs": 0, "adjacent_pairs": 0,
    }

    iou = iou_matrix(preds, gts)

    # --- Matching ungherese (come in evaluate) ---
    matched_pred, matched_gt = {}, {}
    if preds and gts:
        rows, cols = linear_sum_assignment(-iou)
        for r, c in zip(rows, cols):
            if iou[r, c] >= TIOU_EVAL:
                matched_pred[r], matched_gt[c] = c, r

            # errori sui confini per le coppie che "si toccano"
            if iou[r, c] >= TIOU_TOUCH:
                out["start_errors"].append(preds[r][0] - gts[c][0])
                out["end_errors"].append(preds[r][1] - gts[c][1])

    # --- Falsi positivi ---
    for i in range(len(preds)):
        if i in matched_pred:
            continue
        if not gts or iou[i].max() < TIOU_TOUCH:
            out["fp"]["background"] += 1
        elif int(iou[i].argmax()) in matched_gt:
            out["fp"]["duplicate"] += 1
        else:
            out["fp"]["localization"] += 1

    # --- Falsi negativi ---
    for j, (gs, ge) in enumerate(gts):
        if j in matched_gt:
            continue
        # predizioni che cadono per almeno meta' dentro questa azione GT
        inside = [
            i for i, (ps, pe, _) in enumerate(preds)
            if pe > ps and (min(pe, ge) - max(ps, gs)) / (pe - ps) >= 0.5
        ]
        if len(inside) >= 2:
            out["fn"]["fragmented"] += 1
        elif preds and iou[:, j].max() >= TIOU_TOUCH:
            out["fn"]["partial"] += 1
        else:
            out["fn"]["missed"] += 1

    # --- Frammentazione: coppie adiacenti con etichetta simile ---
    for (s1, e1, l1), (s2, e2, l2) in zip(preds, preds[1:]):
        if s2 - e1 <= MERGE_MAX_GAP:
            out["adjacent_pairs"] += 1
            if label_similarity(l1, l2) >= LABEL_SIM_MIN:
                out["fragment_pairs"] += 1

    return out


# ============================================================
# STAMPA E SALVATAGGIO
# ============================================================
def pct(a, b):
    return 100.0 * a / b if b else 0.0


def describe(values):
    if not values:
        return {"mean": 0.0, "median": 0.0, "p10": 0.0, "p90": 0.0}
    v = np.array(values)
    return {"mean": float(v.mean()), "median": float(np.median(v)),
            "p10": float(np.percentile(v, 10)), "p90": float(np.percentile(v, 90))}


def summarize_stage(per_video):
    tot = lambda k: sum(v[k] for v in per_video)
    cat = lambda k: sum((v[k] for v in per_video), Counter())
    allv = lambda k: [x for v in per_video for x in v[k]]

    n_pred, n_gt = tot("n_pred"), tot("n_gt")
    fp, fn = cat("fp"), cat("fn")
    return {
        "n_videos": len(per_video),
        "n_pred": n_pred, "n_gt": n_gt,
        "pred_per_video": n_pred / len(per_video), "gt_per_video": n_gt / len(per_video),
        "pred_duration": describe(allv("pred_durations")),
        "gt_duration": describe(allv("gt_durations")),
        "pred_coverage_pct": pct(tot("pred_coverage"), tot("duration")),
        "gt_coverage_pct": pct(tot("gt_coverage"), tot("duration")),
        "start_error": describe(allv("start_errors")),
        "end_error": describe(allv("end_errors")),
        "start_abs_error_mean": float(np.mean(np.abs(allv("start_errors")))) if allv("start_errors") else 0.0,
        "end_abs_error_mean": float(np.mean(np.abs(allv("end_errors")))) if allv("end_errors") else 0.0,
        "false_positives": dict(fp), "false_negatives": dict(fn),
        "adjacent_pairs": tot("adjacent_pairs"), "fragment_pairs": tot("fragment_pairs"),
    }


def print_stage(label, s, whatif):
    print(f"\n{'=' * 72}\n{label}\n{'=' * 72}")

    print("\n1) GRANULARITA'")
    print(f"   Segmenti per video   : pred {s['pred_per_video']:5.1f}   vs  GT {s['gt_per_video']:5.1f}")
    pdur, gdur = s["pred_duration"], s["gt_duration"]
    print(f"   Durata media (s)     : pred {pdur['mean']:5.2f}   vs  GT {gdur['mean']:5.2f}")
    print(f"   Durata mediana (s)   : pred {pdur['median']:5.2f}   vs  GT {gdur['median']:5.2f}")
    print(f"   Durata p10-p90 (s)   : pred {pdur['p10']:.2f}-{pdur['p90']:.2f}   vs  GT {gdur['p10']:.2f}-{gdur['p90']:.2f}")
    print(f"   Copertura del video  : pred {s['pred_coverage_pct']:5.1f}%  vs  GT {s['gt_coverage_pct']:5.1f}%")

    print(f"\n2) ERRORI SUI CONFINI (coppie con tIoU >= {TIOU_TOUCH}; positivo = in ritardo)")
    se, ee = s["start_error"], s["end_error"]
    print(f"   Inizio: medio {se['mean']:+.2f}s  mediano {se['median']:+.2f}s  |errore| medio {s['start_abs_error_mean']:.2f}s")
    print(f"   Fine  : medio {ee['mean']:+.2f}s  mediano {ee['median']:+.2f}s  |errore| medio {s['end_abs_error_mean']:.2f}s")

    fp, fn = s["false_positives"], s["false_negatives"]
    n_fp, n_fn = sum(fp.values()), sum(fn.values())
    print(f"\n3) FALSI POSITIVI a tIoU={TIOU_EVAL}: {n_fp} su {s['n_pred']} predizioni")
    for k, desc in [("background", "nessuna azione GT sotto"), ("localization", "azione GT giusta, confini sbagliati"),
                    ("duplicate", "azione GT gia' trovata da altra predizione")]:
        print(f"   {k:<13}: {fp.get(k, 0):4d}  ({pct(fp.get(k, 0), n_fp):5.1f}%)  {desc}")

    print(f"\n4) FALSI NEGATIVI a tIoU={TIOU_EVAL}: {n_fn} su {s['n_gt']} azioni GT")
    for k, desc in [("fragmented", "spezzata in 2+ predizioni"), ("partial", "trovata ma confini sbagliati"),
                    ("missed", "nessuna predizione")]:
        print(f"   {k:<13}: {fn.get(k, 0):4d}  ({pct(fn.get(k, 0), n_fn):5.1f}%)  {desc}")

    print("\n5) FRAMMENTAZIONE")
    print(f"   Coppie di predizioni adiacenti (gap <= {MERGE_MAX_GAP}s): {s['adjacent_pairs']}")
    print(f"   ...di cui con etichetta simile               : {s['fragment_pairs']} "
          f"({pct(s['fragment_pairs'], s['adjacent_pairs']):.1f}%)")

    print(f"\n6) WHAT-IF: unione predizioni adiacenti con etichetta simile (regola fissa, solo diagnosi)")
    print(f"   Predizioni: {whatif['before']['n_pred']} -> {whatif['after']['n_pred']}")
    print(f"   F1@{TIOU_EVAL}: {whatif['before']['f1']:.1f} -> {whatif['after']['f1']:.1f}  "
          f"(P {whatif['before']['precision']:.1f} -> {whatif['after']['precision']:.1f}, "
          f"R {whatif['before']['recall']:.1f} -> {whatif['after']['recall']:.1f})")


def pooled_f1(pairs):
    tp = n_pred = n_gt = 0
    for preds, gts in pairs:
        r = hungarian_f1([p[:2] for p in preds], gts, TIOU_EVAL)
        tp, n_pred, n_gt = tp + r["tp"], n_pred + r["n_pred"], n_gt + r["n_gt"]
    return {**f1_from_counts(tp, n_pred, n_gt), "n_pred": n_pred}


# ============================================================
# ENTRY POINT
# ============================================================
def run_analysis(video_ids, tag=None):
    tag = tag or "untagged"
    report = {"tag": tag, "params": {
        "tiou_eval": TIOU_EVAL, "tiou_touch": TIOU_TOUCH,
        "merge_max_gap": MERGE_MAX_GAP, "label_sim_min": LABEL_SIM_MIN}}

    print("\n" + "#" * 72)
    print(f"ANALISI ERRORI [{tag}]")
    print("#" * 72)

    for stage, (label, rel_path) in STAGES.items():
        per_video, pairs, merged_pairs = [], [], []
        for vid in video_ids:
            gt_path = GT_DIR / f"{vid}.txt"
            preds = load_predictions_with_labels(OUTPUTS_DIR / vid / rel_path)
            if preds is None or not gt_path.exists():
                continue
            gts = load_gtea_gt(gt_path)
            per_video.append(analyze_video(preds, gts, video_duration(vid, gts, preds)))
            pairs.append((preds, gts))
            merged_pairs.append((merge_similar_adjacent(preds), gts))

        if not per_video:
            print(f"\n[{label}] nessun output trovato.")
            continue

        summary = summarize_stage(per_video)
        whatif = {"before": pooled_f1(pairs), "after": pooled_f1(merged_pairs)}
        print_stage(f"{label} - {summary['n_videos']} video", summary, whatif)
        report[stage] = {**summary, "whatif_merge": whatif}

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = RESULTS_DIR / f"analysis_{tag}_{stamp}.json"
    out_file.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nReport salvato in: {out_file}")
    return report
