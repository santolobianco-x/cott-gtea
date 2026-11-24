"""
CoTT-Lite - inferenza online (streaming) e conversione in segmenti.

Il video viene processato un frame alla volta: al frame t il modello vede solo gli ultimi
HISTORY frame (t-63 ... t), mai il futuro; all'inizio del video la finestra e' riempita
a sinistra con zeri, come nel training. Dalle due uscite si costruiscono i segmenti
con una regola causale, con soglie FISSATE A PRIORI (non ottimizzate sul GT):

  - INIZIO: quando la probabilita' di azione supera ACT_THR;
  - FINE:   quando la probabilita' di azione scende sotto ACT_THR, oppure quando la
            progression "cade" (era > PROG_HIGH e scende di almeno PROG_DROP): significa
            che un'azione e' finita e ne e' iniziata subito un'altra (azioni attaccate).

Un segmento viene emesso nel momento in cui se ne rileva la fine, come nel paper.
"""

import json
import time

import torch

from src.evaluate import OUTPUTS_DIR

FPS = 2
ACT_THR = 0.5
PROG_HIGH = 0.5
PROG_DROP = 0.3


def output_path(video_id, source):
    return OUTPUTS_DIR / video_id / f"lite_{source}" / "predictions.json"


@torch.no_grad()
def stream_video(model, features, history, device):
    """Processa le feature [T, D] frame per frame. Ritorna p_action[T], progression[T], secondi."""
    model.eval()
    feats = features.to(device)
    p_action, progression = [], []
    t0 = time.perf_counter()
    window = torch.zeros(1, history, feats.shape[1], device=device)     # padding a sinistra, come nel training
    for t in range(feats.shape[0]):
        window = torch.cat([window[:, 1:], feats[t].view(1, 1, -1)], dim=1)  # scorre di un frame
        act, prog = model(window)
        pa, pr = model.decode(act[:, -1], prog[:, -1])
        p_action.append(float(pa))
        progression.append(float(pr))
    if device.type == "mps":
        torch.mps.synchronize()
    return p_action, progression, time.perf_counter() - t0


def to_segments(p_action, progression, fps=FPS):
    """Regola causale: dalle curve per frame ai segmenti [start, end] in secondi."""
    segments, start, last_p = [], None, 0.0
    for t, (pa, pr) in enumerate(zip(p_action, progression)):
        active = pa >= ACT_THR
        if start is None:
            if active:
                start, last_p = t, pr
            continue
        if not active:                                        # fine per actionness bassa
            segments.append((start / fps, t / fps))
            start = None
        elif last_p > PROG_HIGH and pr < last_p - PROG_DROP:  # fine per caduta di progression
            segments.append((start / fps, t / fps))
            start = t
        last_p = pr
    if start is not None:
        segments.append((start / fps, len(p_action) / fps))
    return segments


def predict_and_save(model, video_id, features, source, history, device, fold):
    p_action, progression, seconds = stream_video(model, features, history, device)
    segments = to_segments(p_action, progression)
    out = output_path(video_id, source)
    out.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "video_id": video_id, "label_source": source, "test_fold": fold,
        "rule": {"act_thr": ACT_THR, "prog_high": PROG_HIGH, "prog_drop": PROG_DROP},
        "n_frames": len(p_action), "stream_seconds": round(seconds, 4),
        "fps_model": round(len(p_action) / max(seconds, 1e-9), 1),
        "curves": {"p_action": [round(x, 4) for x in p_action],
                   "progression": [round(x, 4) for x in progression]},
        "actions": [{"action_id": i, "start_time": s, "end_time": e} for i, (s, e) in enumerate(segments)],
    }
    out.write_text(json.dumps(data, indent=1), encoding="utf-8")
    return len(segments), len(p_action), seconds
