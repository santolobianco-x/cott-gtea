"""
Post-processing delle predizioni dello Stage 3: filtro per confidenza e rimozione dei duplicati.

NOTA METODOLOGICA: questo passaggio NON fa parte del paper CoTT.
E' una modifica aggiunta dopo l'analisi degli errori, che ha mostrato come lo
Stage 3 faccia convergere proposte vicine sulla stessa azione (duplicati).
I risultati vanno quindi sempre riportati sia con sia senza deduplicazione.

Regola: due predizioni sono "duplicate" se il loro tIoU e' >= DEDUP_TIOU.
I gruppi di duplicati vengono sostituiti da una sola predizione, secondo la strategia:
  - "mean"  : inizio e fine = media dei confini del gruppo   (default)
  - "union" : dall'inizio piu' precoce alla fine piu' tarda
  - "keep"  : tiene una sola predizione del gruppo, come la Non-Maximum Suppression
              (preferisce la confidenza piu' alta, poi le rifinite con successo,
              poi la piu' lunga)                                  (default)

Prima della deduplicazione si possono scartare le predizioni con confidenza < MIN_CONF
(confidenza restituita dallo Stage 3, come nel prompt originale del paper).
Le predizioni senza confidenza (es. fallback dello Stage 3) NON vengono scartate.
Con MIN_CONF = 0 il filtro e' disattivato.

Soglie e strategia vanno fissate a priori, NON ottimizzate sul ground truth.

Modulo richiamato da main.py.
"""

import json
from pathlib import Path

from src.evaluate import OUTPUTS_DIR, compute_tiou

DEDUP_TIOU = 0.5
DEDUP_STRATEGY = "keep"
MIN_CONF = 0.0
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


def confidence_of(action):
    c = action.get("confidence")
    return float(c) if isinstance(c, (int, float)) else None


def rank_key(action):
    """Ordine di preferenza: confidenza piu' alta, poi rifinita con successo, poi piu' lunga."""
    conf = confidence_of(action)
    return (
        conf if conf is not None else -1.0,
        bool(action.get("refined", True)),
        float(action["end_time"]) - float(action["start_time"]),
    )


def filter_by_confidence(actions, min_conf):
    if min_conf <= 0:
        return actions
    return [a for a in actions if confidence_of(a) is None or confidence_of(a) >= min_conf]


def collapse(cluster, strategy):
    if len(cluster) == 1:
        return {**cluster[0], "merged_from": [cluster[0]["action_id"]]}

    starts = [float(a["start_time"]) for a in cluster]
    ends = [float(a["end_time"]) for a in cluster]
    rep = max(cluster, key=rank_key)

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


def dedup_actions(actions, thr=DEDUP_TIOU, strategy=DEDUP_STRATEGY, min_conf=MIN_CONF):
    actions = [{"action_id": i, **a} for i, a in enumerate(actions)]  # garantisce un id
    actions = filter_by_confidence(actions, min_conf)
    merged = [collapse(c, strategy) for c in cluster_duplicates(actions, thr)]
    merged.sort(key=lambda a: (a["start_time"], a["end_time"]))
    for idx, act in enumerate(merged):
        act["action_id"] = idx
    return merged


def run(video_id: str, thr: float = DEDUP_TIOU, strategy: str = DEDUP_STRATEGY,
        min_conf: float = MIN_CONF, verbose=True):
    stage3_dir = OUTPUTS_DIR / video_id / "stage3"
    input_file = stage3_dir / INPUT_NAME
    if not input_file.exists():
        if verbose:
            print(f"[Dedup] File non trovato: {input_file}")
        return None

    actions = json.loads(input_file.read_text(encoding="utf-8"))
    deduped = dedup_actions(actions, thr, strategy, min_conf)

    output = {
        "video_id": video_id,
        "dedup_tiou": thr,
        "dedup_strategy": strategy,
        "min_conf": min_conf,
        "n_before": len(actions),
        "n_after": len(deduped),
        "actions": deduped,
    }
    (stage3_dir / OUTPUT_NAME).write_text(json.dumps(output, indent=4, ensure_ascii=False), encoding="utf-8")

    if verbose:
        print(f"[Dedup] {video_id}: {len(actions)} -> {len(deduped)} azioni "
              f"(conf>={min_conf}, tIoU>={thr}, strategia={strategy})")
    return output