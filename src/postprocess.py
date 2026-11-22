"""
Post-processing delle predizioni dello Stage 3: rimozione dei duplicati.

NOTA METODOLOGICA: questo passaggio NON fa parte del paper CoTT.
E' una modifica aggiunta dopo l'analisi degli errori, che ha mostrato come lo
Stage 3 faccia convergere proposte vicine sulla stessa azione (duplicati).
I risultati vanno quindi sempre riportati sia con sia senza deduplicazione.

Regola: due predizioni sono "duplicate" se il loro tIoU e' >= DEDUP_TIOU.
I gruppi di duplicati vengono sostituiti da una sola predizione, secondo la strategia:
  - "mean"  : inizio e fine = media dei confini del gruppo   (default)
  - "union" : dall'inizio piu' precoce alla fine piu' tarda
  - "keep"  : tiene una sola predizione del gruppo (preferisce quelle rifinite
              con successo dallo Stage 3, poi la piu' lunga)

La soglia e la strategia sono fissate a priori, NON ottimizzate sul ground truth.

Modulo richiamato da main.py.
"""

import json
from pathlib import Path

from src.evaluate import OUTPUTS_DIR, compute_tiou

DEDUP_TIOU = 0.5
DEDUP_STRATEGY = "keep"
STRATEGIES = ("mean", "union", "keep")

INPUT_NAME = "refined_actions.json"
OUTPUT_NAME = "refined_actions_dedup.json"


def cluster_duplicates(actions, thr):
    """Raggruppa le azioni che hanno tIoU >= thr con almeno un membro del gruppo."""
    actions = sorted(actions, key=lambda a: (float(a["start_time"]), float(a["end_time"])))
    clusters = []
    for act in actions:
        seg = (float(act["start_time"]), float(act["end_time"]))
        target = None
        for cluster in clusters:
            if any(compute_tiou(seg, (float(m["start_time"]), float(m["end_time"]))) >= thr for m in cluster):
                target = cluster
                break
        if target is None:
            clusters.append([act])
        else:
            target.append(act)
    return clusters


def collapse(cluster, strategy):
    if len(cluster) == 1:
        return {**cluster[0], "merged_from": [cluster[0]["action_id"]]}

    starts = [float(a["start_time"]) for a in cluster]
    ends = [float(a["end_time"]) for a in cluster]
    # rappresentante: prima le rifinite con successo, poi la piu' lunga
    rep = max(cluster, key=lambda a: (a.get("refined", True), float(a["end_time"]) - float(a["start_time"])))

    if strategy == "mean":
        start, end = sum(starts) / len(starts), sum(ends) / len(ends)
    elif strategy == "union":
        start, end = min(starts), max(ends)
    elif strategy == "keep":
        start, end = float(rep["start_time"]), float(rep["end_time"])
    else:
        raise ValueError(f"Strategia sconosciuta: {strategy} (ammesse: {STRATEGIES})")

    return {
        **rep,
        "start_time": round(start, 3),
        "end_time": round(end, 3),
        "merged_from": [a["action_id"] for a in cluster],
    }


def dedup_actions(actions, thr=DEDUP_TIOU, strategy=DEDUP_STRATEGY):
    actions = [{"action_id": i, **a} for i, a in enumerate(actions)]  # garantisce un id
    merged = [collapse(c, strategy) for c in cluster_duplicates(actions, thr)]
    merged.sort(key=lambda a: (a["start_time"], a["end_time"]))
    for idx, act in enumerate(merged):
        act["action_id"] = idx
    return merged


def run(video_id: str, thr: float = DEDUP_TIOU, strategy: str = DEDUP_STRATEGY, verbose=True):
    stage3_dir = OUTPUTS_DIR / video_id / "stage3"
    input_file = stage3_dir / INPUT_NAME
    if not input_file.exists():
        if verbose:
            print(f"[Dedup] File non trovato: {input_file}")
        return None

    actions = json.loads(input_file.read_text(encoding="utf-8"))
    deduped = dedup_actions(actions, thr, strategy)

    output = {
        "video_id": video_id,
        "dedup_tiou": thr,
        "dedup_strategy": strategy,
        "n_before": len(actions),
        "n_after": len(deduped),
        "actions": deduped,
    }
    (stage3_dir / OUTPUT_NAME).write_text(json.dumps(output, indent=4, ensure_ascii=False), encoding="utf-8")

    if verbose:
        print(f"[Dedup] {video_id}: {len(actions)} -> {len(deduped)} azioni "
              f"(tIoU>={thr}, strategia={strategy})")
    return output

