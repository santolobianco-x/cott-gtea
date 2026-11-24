"""
CoTT-Lite - training con cross-validation per soggetto (leave-one-subject-out, 4 fold).

Per ogni fold: si addestra su 3 soggetti (21 video) con le label della sorgente scelta,
poi si esegue l'inferenza in streaming sui 7 video del soggetto escluso.
Alla fine ogni video ha una predizione fatta da un modello che non l'ha mai visto,
e la valutazione usa sempre il ground truth (src/evaluate.py).

Iperparametri del paper (App. C): AdamW, lr 3e-4, weight decay 0.01, history 64, 15 epoche.
"""

import json
import random
import time

import numpy as np
import torch

from src.evaluate import OUTPUTS_DIR, RESULTS_DIR, run as evaluate_video, summarize_batch
from src.lite.dataset import load_video, make_windows, subject_of
from src.lite.features import DEFAULT_MODEL, features_path
from src.lite.model import CoTTLite, cott_lite_loss
from src.lite.stream import predict_and_save

HISTORY = 64
EPOCHS = 15
LR = 3e-4
WEIGHT_DECAY = 0.01
BATCH_SIZE = 16
WINDOW_STRIDE = 8


def pick_device(name="auto"):
    if name != "auto":
        return torch.device(name)
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_features_loader(model_name):
    def load(video_id):
        feats = torch.load(features_path(video_id, model_name))["features"].float()
        return torch.nn.functional.normalize(feats, dim=-1)          # embedding CLIP normalizzati
    return load


def build_batches(videos, history, stride, batch_size):
    """Finestre di 'history' frame (con padding a sinistra) mescolate in batch."""
    items = [(v, s) for v in videos for s in make_windows(len(v["actionness"]), history, stride)]
    random.shuffle(items)
    for i in range(0, len(items), batch_size):
        chunk = items[i:i + batch_size]
        dim = chunk[0][0]["features"].shape[1]
        x = torch.zeros(len(chunk), history, dim)
        act = torch.zeros(len(chunk), history, dtype=torch.long)
        prog = torch.zeros(len(chunk), history)
        amask = torch.zeros(len(chunk), history, dtype=torch.bool)
        valid = torch.zeros(len(chunk), history, dtype=torch.bool)
        for b, (v, s) in enumerate(chunk):
            lo = max(0, s)
            n = s + history - lo                      # frame reali nella finestra
            sl = slice(history - n, history)          # padding a sinistra
            x[b, sl] = v["features"][lo:lo + n]
            act[b, sl] = torch.from_numpy(v["actionness"][lo:lo + n])
            prog[b, sl] = torch.from_numpy(v["progression"][lo:lo + n])
            amask[b, sl] = torch.from_numpy(v["mask"][lo:lo + n])
            valid[b, sl] = True
        yield x, act, prog, amask, valid


def train_fold(train_videos, device, epochs, seed):
    set_seed(seed)
    model = CoTTLite(in_dim=train_videos[0]["features"].shape[1]).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    for epoch in range(1, epochs + 1):
        model.train()
        tot = ce_sum = hl_sum = n = 0.0
        for x, act, prog, amask, valid in build_batches(train_videos, HISTORY, WINDOW_STRIDE, BATCH_SIZE):
            x, act, prog, amask, valid = (t.to(device) for t in (x, act, prog, amask, valid))
            act_logits, prog_logits = model(x)
            loss, ce, hl = cott_lite_loss(act_logits, prog_logits, act, prog, amask, valid)
            optim.zero_grad()
            loss.backward()
            optim.step()
            tot, ce_sum, hl_sum, n = tot + loss.item(), ce_sum + ce.item(), hl_sum + hl.item(), n + 1
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f"    epoca {epoch:2d}/{epochs}  loss {tot / n:.3f}  (actionness {ce_sum / n:.3f}, "
                  f"progression {hl_sum / n:.3f})")
    return model


def run_cv(video_ids, source, epochs=EPOCHS, seed=0, device_name="auto", feature_model=DEFAULT_MODEL, tag=None):
    device = pick_device(device_name)
    load_features = make_features_loader(feature_model)
    subjects = sorted({subject_of(v) for v in video_ids})
    print(f"\nCoTT-Lite | label: {source} | {len(video_ids)} video | fold per soggetto: {subjects} | "
          f"device: {device} | seed {seed}")

    timing = {"frames": 0, "seconds": 0.0}
    t_start = time.time()
    for test_subject in subjects:
        train_ids = [v for v in video_ids if subject_of(v) != test_subject]
        test_ids = [v for v in video_ids if subject_of(v) == test_subject]
        print(f"\n  Fold test={test_subject}: train {len(train_ids)} video ({source}), test {len(test_ids)} video")
        train_videos = [load_video(v, source, load_features) for v in train_ids]
        model = train_fold(train_videos, device, epochs, seed)

        ckpt = OUTPUTS_DIR / "lite_models" / source / f"fold_{test_subject}_seed{seed}.pt"
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ckpt)

        for vid in test_ids:
            n_seg, n_frames, secs = predict_and_save(model, vid, load_features(vid), source, HISTORY,
                                                     device, test_subject)
            timing["frames"] += n_frames
            timing["seconds"] += secs
        print(f"  Fold {test_subject}: predizioni salvate per {len(test_ids)} video")

    fps = timing["frames"] / max(timing["seconds"], 1e-9)
    print(f"\nTempo totale: {time.time() - t_start:.0f}s | streaming del modello: {fps:.0f} frame/s "
          f"({1000 / fps:.1f} ms per frame, feature escluse)")

    results = [evaluate_video(v, verbose=False) for v in video_ids]
    summarize_batch(results, tag=tag or f"lite_{source}_seed{seed}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"lite_timing_{source}_seed{seed}.json").write_text(json.dumps({
        "label_source": source, "seed": seed, "device": str(device), "frames": timing["frames"],
        "stream_seconds": round(timing["seconds"], 3), "fps_model": round(fps, 1),
        "feature_model": feature_model}, indent=2), encoding="utf-8")
