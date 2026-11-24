"""
CoTT-Lite - estrazione delle feature visive.

Per ogni video legge i frame gia' estratti a 2 fps (outputs/frames/<video>/frame_XXXXX.jpg)
e calcola un embedding per frame con l'encoder visivo di CLIP.

NOTA: il paper usa EgoVideo (un modello video egocentrico). Qui usiamo CLIP, un modello
per immagini singole, piu' leggero e adatto al Mac: e' una differenza da dichiarare.

Output: outputs/features/<nome_modello>/<video>.pt con
    {"features": Tensor[T, D], "fps": 2, "model": ..., "frames": [nomi file]}
dove T = numero di frame del video e D = dimensione dell'embedding (512 per ViT-B/16).
"""

import re
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FRAMES_DIR = PROJECT_ROOT / "outputs" / "frames"
FEATURES_DIR = PROJECT_ROOT / "outputs" / "features"

DEFAULT_MODEL = "openai/clip-vit-base-patch16"
FPS = 2
BATCH_SIZE = 32


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def model_short_name(model_name):
    """openai/clip-vit-base-patch16 -> clip-vit-base-patch16"""
    return model_name.split("/")[-1]


def features_path(video_id, model_name=DEFAULT_MODEL):
    return FEATURES_DIR / model_short_name(model_name) / f"{video_id}.pt"


def list_frames(video_id):
    frames_dir = FRAMES_DIR / video_id
    files = sorted(frames_dir.glob("frame_*.jpg"), key=lambda p: int(re.search(r"(\d+)", p.stem).group(1)))
    if not files:
        raise FileNotFoundError(f"Nessun frame in {frames_dir}: estrai prima i frame con main.py.")
    return files


class FeatureExtractor:
    """Carica CLIP una sola volta e lo riusa per tutti i video."""

    def __init__(self, model_name=DEFAULT_MODEL, batch_size=BATCH_SIZE):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = pick_device()
        print(f"Carico {model_name} su {self.device} ...")
        self.processor = CLIPImageProcessor.from_pretrained(model_name)
        self.model = CLIPVisionModelWithProjection.from_pretrained(model_name).to(self.device).eval()

    @torch.no_grad()
    def extract(self, frame_files):
        chunks = []
        for i in range(0, len(frame_files), self.batch_size):
            images = [Image.open(p).convert("RGB") for p in frame_files[i:i + self.batch_size]]
            pixels = self.processor(images=images, return_tensors="pt")["pixel_values"].to(self.device)
            embeds = self.model(pixel_values=pixels).image_embeds        # [B, D]
            chunks.append(embeds.float().cpu())
        return torch.cat(chunks, dim=0)                                  # [T, D]


def run(video_ids, model_name=DEFAULT_MODEL, batch_size=BATCH_SIZE, force=False):
    todo = [v for v in video_ids if force or not features_path(v, model_name).exists()]
    print(f"Feature {model_short_name(model_name)}: {len(video_ids)} video, da elaborare {len(todo)}")
    if not todo:
        return

    extractor = FeatureExtractor(model_name, batch_size)
    total_frames, t_start = 0, time.time()

    for idx, vid in enumerate(todo, 1):
        frames = list_frames(vid)
        t0 = time.time()
        feats = extractor.extract(frames)
        elapsed = time.time() - t0

        out = features_path(vid, model_name)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".pt.tmp")
        torch.save({"features": feats, "fps": FPS, "model": model_name,
                    "frames": [p.name for p in frames]}, tmp)
        tmp.replace(out)

        total_frames += len(frames)
        print(f"  [{idx}/{len(todo)}] {vid}: {len(frames)} frame -> {tuple(feats.shape)} "
              f"in {elapsed:.1f}s ({len(frames) / max(elapsed, 1e-6):.0f} frame/s)")

    total = time.time() - t_start
    print(f"Completato: {total_frames} frame in {total:.1f}s. Feature in {FEATURES_DIR / model_short_name(model_name)}")
