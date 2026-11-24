"""
CoTT-Lite - dalle pseudo-label ai target per frame.

Sorgenti di label (stesso formato a segmenti [start, end] in secondi):
  - stage3       : pseudo-label del CoTT-Labeler (Stage 3), come nel paper
  - stage3_dedup : Stage 3 + deduplicazione (aggiunta nostra)
  - gt           : ground truth di GTEA       -> controllo: limite superiore
  - grid         : griglia cieca (baseline)    -> controllo: livello del caso

Target per ogni frame k (a 2 fps, centro del frame = (k + 0.5) / 2 secondi):
  - actionness  : 1 se il centro del frame cade dentro un segmento, altrimenti 0
  - progression : a che punto del segmento si trova il frame (0 = inizio, 1 = fine);
                  definita solo sui frame di azione (sugli altri la loss e' mascherata)
Se due segmenti si sovrappongono, il frame viene assegnato a quello iniziato per ultimo.
"""

import json
from pathlib import Path

import numpy as np

from src.evaluate import GT_DIR, OUTPUTS_DIR, load_gtea_gt

FPS = 2
LABEL_SOURCES = {
    "stage3": Path("stage3") / "refined_actions.json",
    "stage3_dedup": Path("stage3") / "refined_actions_dedup.json",
    "grid": Path("baseline_grid") / "predictions.json",
    "gt": None,
}


def subject_of(video_id):
    return video_id.split("_")[0]


def load_segments(video_id, source):
    """Segmenti [(start, end)] in secondi per il video, dalla sorgente indicata."""
    if source not in LABEL_SOURCES:
        raise ValueError(f"Sorgente sconosciuta: {source}. Scegli tra {list(LABEL_SOURCES)}")
    if source == "gt":
        return load_gtea_gt(GT_DIR / f"{video_id}.txt")
    path = OUTPUTS_DIR / video_id / LABEL_SOURCES[source]
    if not path.exists():
        raise FileNotFoundError(f"Label '{source}' non trovate per {video_id}: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    actions = data["actions"] if isinstance(data, dict) else data
    return [(float(a["start_time"]), float(a["end_time"])) for a in actions]


def frame_targets(segments, n_frames, fps=FPS):
    """Ritorna (actionness[T] int, progression[T] float, mask[T] bool)."""
    centers = (np.arange(n_frames) + 0.5) / fps
    actionness = np.zeros(n_frames, dtype=np.int64)
    progression = np.zeros(n_frames, dtype=np.float32)
    for start, end in sorted(segments):            # i segmenti iniziati dopo sovrascrivono
        if end <= start:
            continue
        inside = (centers >= start) & (centers < end)
        actionness[inside] = 1
        progression[inside] = np.clip((centers[inside] - start) / (end - start), 0.0, 1.0)
    return actionness, progression, actionness.astype(bool)


def load_video(video_id, source, features_loader):
    """Feature [T, D] + target per un video."""
    feats = features_loader(video_id)
    a, p, m = frame_targets(load_segments(video_id, source), feats.shape[0])
    return {"video_id": video_id, "features": feats, "actionness": a, "progression": p, "mask": m}


def make_windows(n_frames, history, stride):
    """Inizi delle finestre di training lunghe 'history', una ogni 'stride' frame.
    Le finestre all'inizio del video iniziano prima di 0 e vengono riempite a sinistra
    con zeri (padding), esattamente come avviene in inferenza per i primi frame."""
    ends = list(range(min(stride, n_frames), n_frames + 1, stride))
    if not ends or ends[-1] != n_frames:
        ends.append(n_frames)
    return [e - history for e in ends]
