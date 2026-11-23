"""
Confronto statistico tra metodi sui video di GTEA.

Per ogni metodo confrontato con un metodo di riferimento (default: Stage 3 di CoTT):
  - differenza di F1 aggregato (pooled) con intervallo di confidenza al 95%
    ottenuto con bootstrap sui video (si ricampionano i video con reinserimento);
  - test di Wilcoxon appaiato sugli F1 dei singoli video;
  - in quanti video il riferimento vince / pareggia / perde.
I p-value sono corretti per confronti multipli con il metodo di Holm.

Lettura: una differenza e' "affidabile" se l'intervallo di confidenza non contiene 0
e il p-value corretto e' < 0.05.

Modulo richiamato da main.py (opzione --compare).
"""

import csv
import json
from datetime import datetime

import numpy as np
from scipy.stats import wilcoxon

from src.evaluate import OUTPUTS_DIR, RESULTS_DIR, STAGES, f1_from_counts, run as evaluate_video

THRESHOLDS = (0.3, 0.5, 0.7)
N_BOOTSTRAP = 10000
SEED = 0


def pooled_f1(rows):
    tp = sum(r["tp"] for r in rows)
    n_pred = sum(r["n_pred"] for r in rows)
    n_gt = sum(r["n_gt"] for r in rows)
    return f1_from_counts(tp, n_pred, n_gt)["f1"]


def bootstrap_diff(ref_rows, oth_rows, rng):
    """IC 95% della differenza di F1 pooled (rif - altro), ricampionando i video."""
    n = len(ref_rows)
    ref = np.array([[r["tp"], r["n_pred"], r["n_gt"]] for r in ref_rows], dtype=float)
    oth = np.array([[r["tp"], r["n_pred"], r["n_gt"]] for r in oth_rows], dtype=float)
    idx = rng.integers(0, n, size=(N_BOOTSTRAP, n))

    def f1(arr):
        s = arr[idx].sum(axis=1)                         # (B, 3)
        p = np.divide(s[:, 0], s[:, 1], out=np.zeros(len(s)), where=s[:, 1] > 0)
        r = np.divide(s[:, 0], s[:, 2], out=np.zeros(len(s)), where=s[:, 2] > 0)
        return np.divide(2 * p * r, p + r, out=np.zeros(len(s)), where=(p + r) > 0) * 100

    diffs = f1(ref) - f1(oth)
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def holm(pvalues):
    """Correzione di Holm-Bonferroni (None resta None)."""
    valid = sorted((p, i) for i, p in enumerate(pvalues) if p is not None)
    m, adjusted, running = len(valid), [None] * len(pvalues), 0.0
    for rank, (p, i) in enumerate(valid):
        running = max(running, min(1.0, (m - rank) * p))
        adjusted[i] = running
    return adjusted


def describe_grid():
    """Legge la configurazione della griglia attualmente salvata (puo' cambiare tra run)."""
    for path in sorted(OUTPUTS_DIR.glob("*/baseline_grid/predictions.json")):
        cfg = json.loads(path.read_text(encoding="utf-8")).get("config", {})
        return f"{cfg.get('seg_len')}s ogni {cfg.get('period')}s"
    return None


def run_compare(video_ids, reference="stage3", tag=None):
    tag = tag or "untagged"
    if reference not in STAGES:
        raise ValueError(f"Riferimento sconosciuto: {reference}. Scegli tra {list(STAGES)}")

    results = [evaluate_video(v, verbose=False) for v in video_ids if (OUTPUTS_DIR / v).is_dir()]
    rng = np.random.default_rng(SEED)
    ref_label = STAGES[reference][0]
    grid = describe_grid()

    print("\n" + "#" * 78)
    print(f"CONFRONTO STATISTICO [{tag}]  riferimento: {ref_label}")
    if grid:
        print(f"(griglia cieca attualmente salvata: {grid})")
    print("#" * 78)

    rows_out = []
    for th in THRESHOLDS:
        comparisons = []
        for other in STAGES:
            if other == reference:
                continue
            # solo i video in cui entrambi i metodi hanno un output
            pairs = [(r[reference][th], r[other][th]) for r in results
                     if th in r.get(reference, {}) and th in r.get(other, {})]
            if len(pairs) < 5:
                continue
            ref_rows, oth_rows = [p[0] for p in pairs], [p[1] for p in pairs]
            d = np.array([a["f1"] - b["f1"] for a, b in pairs])
            try:
                p = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
            except ValueError:
                p = None
            lo, hi = bootstrap_diff(ref_rows, oth_rows, rng)
            comparisons.append({
                "tiou": th, "reference": reference, "other": other, "n_videos": len(pairs),
                "ref_pooled_f1": pooled_f1(ref_rows), "other_pooled_f1": pooled_f1(oth_rows),
                "diff": pooled_f1(ref_rows) - pooled_f1(oth_rows), "ci_low": lo, "ci_high": hi,
                "wins": int(np.sum(d > 0)), "ties": int(np.sum(d == 0)), "losses": int(np.sum(d < 0)),
                "p_wilcoxon": p,
            })

        for c, p_adj in zip(comparisons, holm([c["p_wilcoxon"] for c in comparisons])):
            c["p_holm"] = p_adj
            c["reliable"] = (c["ci_low"] > 0 or c["ci_high"] < 0) and p_adj is not None and p_adj < 0.05

        print(f"\n--- tIoU = {th} ---")
        print(f"  {'confronto con':<34} {'rif':>5} {'altro':>6} {'diff':>6}  {'IC 95%':>15}  "
              f"{'V/P/S':>8}  {'p (Holm)':>8}")
        for c in comparisons:
            p_str = f"{c['p_holm']:.4f}" if c["p_holm"] is not None else "  n/d"
            mark = "  *" if c["reliable"] else ""
            print(f"  {STAGES[c['other']][0]:<34} {c['ref_pooled_f1']:5.1f} {c['other_pooled_f1']:6.1f} "
                  f"{c['diff']:+6.1f}  [{c['ci_low']:+6.1f}, {c['ci_high']:+6.1f}]  "
                  f"{c['wins']:2d}/{c['ties']:1d}/{c['losses']:2d}  {p_str:>8}{mark}")
        rows_out.extend(comparisons)

    print("\n  diff = F1 pooled del riferimento meno quello dell'altro metodo (positivo = vince il riferimento)")
    print("  V/P/S = video in cui il riferimento vince / pareggia / perde")
    print("  * = differenza affidabile (IC 95% senza lo 0 e p di Holm < 0.05)")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = RESULTS_DIR / f"compare_{tag}_{stamp}.csv"
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[*rows_out[0].keys(), "grid_config"])
        writer.writeheader()
        for row in rows_out:
            writer.writerow({**{k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()},
                             "grid_config": grid})
    print(f"\nRisultati salvati in: {out_file}")
    return rows_out
