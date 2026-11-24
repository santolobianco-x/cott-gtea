"""
Configurazione centrale del progetto CoTT su GTEA.

Tutti i parametri modificabili stanno qui, divisi per fase della pipeline.
Gli altri file li importano (es. `from src.config import FPS`), quindi per
cambiare modello, server o un iperparametro basta modificare questo file.

ATTENZIONE: cambiare un parametro dello Stage 1 invalida i checkpoint gia'
salvati (lo Stage 1 se ne accorge e chiede di rilanciare con --force).
"""

from pathlib import Path

# =============================================================================
# PERCORSI
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]   # cartella principale del progetto
DATA_DIR = PROJECT_ROOT / "data"
VIDEOS_DIR = DATA_DIR / "Videos"                      # video .mp4 di GTEA
GT_DIR = DATA_DIR / "gtea_labels_71" / "labels"       # label (ground truth) di GTEA
OUTPUTS_DIR = PROJECT_ROOT / "outputs"                # tutto cio' che la pipeline produce
FRAMES_DIR = OUTPUTS_DIR / "frames"                   # frame estratti a 2 fps
RESULTS_DIR = PROJECT_ROOT / "results"                # CSV/JSON con le metriche

# =============================================================================
# VIDEO
# =============================================================================
FPS = 2              # frame al secondo analizzati (come nel paper): 1 frame ogni 0.5 s
GTEA_FPS = 15.0      # fps dei video originali: servono a convertire le label in secondi

# =============================================================================
# MODELLO (server compatibile con le API OpenAI, es. llama.cpp)
# =============================================================================
API_URL = "http://127.0.0.1:9931/v1/chat/completions"
MODEL_NAME = "Qwen2.5-VL-3B-Instruct"
IMAGE_SIDE = 336     # lato massimo (px) delle immagini inviate al modello
JPEG_QUALITY = 90    # qualita' della compressione JPEG delle immagini

# =============================================================================
# STAGE 1 - Transition captioning (descrive cosa cambia tra due frame)
# =============================================================================
S1_WINDOW_SIZE = 12      # frame per finestra (12 frame = 6 s a 2 fps, come nel paper)
S1_STRIDE = 6            # passo tra due finestre: si sovrappongono per meta'
S1_MAX_TOKENS = 384
S1_TEMPERATURE = 0.1     # bassa = risposte piu' deterministiche
S1_RETRIES = 2           # tentativi extra se la risposta non e' nel formato richiesto
S1_TIMEOUT = 600         # secondi massimi di attesa per una risposta
S1_PAUSE = 0.2           # pausa tra due chiamate, per non sovraccaricare il server

# =============================================================================
# STAGE 2 - Action grouping (raggruppa le transizioni in azioni, solo testo)
# =============================================================================
S2_BATCH_SIZE = 20                     # transizioni per blocco inviato al modello
S2_BATCH_STRIDE = 15                   # passo tra blocchi: 5 transizioni di sovrapposizione
S2_MAX_TOKENS = 384
S2_RETRY_TEMPERATURES = [0.1, 0.4, 0.7]  # se il parsing fallisce, si riprova piu' "creativi"
S2_TIMEOUT = 600

# =============================================================================
# STAGE 3 - Refinement (rifinisce inizio e fine di ogni azione)
# =============================================================================
S3_PADDING = 1.0        # secondi aggiunti prima e dopo l'azione proposta dallo Stage 2
S3_MAX_FRAMES = 12      # frame massimi inviati al modello per ogni azione
S3_MAX_TOKENS = 256
S3_TEMPERATURE = 0.1
S3_TIMEOUT = 600
S3_PAUSE = 0.3

# =============================================================================
# POST-PROCESSING (aggiunta nostra, NON nel paper)
# =============================================================================
DEDUP_TIOU = 0.5          # due azioni con tIoU >= 0.5 sono considerate duplicate
DEDUP_STRATEGY = "keep"   # keep (come NMS) | mean | union
MIN_CONF = 0.0            # filtro sulla confidenza (0 = disattivato: col 3B non e' informativa)

# =============================================================================
# BASELINE (Figura 1 del paper) e controllo cieco
# =============================================================================
BASE_WINDOW_SIZE = 12                           # clip della sliding-window (6 s, come lo Stage 1)
BASE_MAX_TOKENS = {"slide": 512, "whole": 2048}
BASE_TIMEOUT = {"slide": 600, "whole": 3600}
BASE_RETRY_TEMPERATURES = [0.1, 0.4, 0.7]
GRID_LEN = 2.0            # griglia cieca: segmenti da 2 s...
GRID_PERIOD = 3.0         # ...uno ogni 3 s

# =============================================================================
# VALUTAZIONE
# =============================================================================
TIOU_THRESHOLDS = (0.3, 0.5, 0.7)   # soglie di sovrapposizione per contare un'azione come trovata
N_BOOTSTRAP = 10000                 # ricampionamenti per gli intervalli di confidenza
SEED = 0                            # seme casuale: rende i risultati riproducibili

# =============================================================================
# CoTT-LITE (modello online leggero, App. C del paper)
# =============================================================================
LITE_FEATURE_MODEL = "openai/clip-vit-base-patch16"  # paper: EgoVideo; qui CLIP (piu' leggero)
LITE_FEATURE_BATCH = 32
LITE_HIDDEN = 768          # dimensione dello stato interno della GRU
LITE_LAYERS = 3            # layer della GRU
LITE_BINS = 20             # fasce della progression (histogram loss)
LITE_HISTORY = 64          # frame di storia visti dal modello (32 s)
LITE_EPOCHS = 15
LITE_LR = 3e-4
LITE_WEIGHT_DECAY = 0.01
LITE_BATCH_SIZE = 16
LITE_WINDOW_STRIDE = 8     # passo tra le finestre di training
LITE_ACT_THR = 0.5         # streaming: un'azione inizia se P(azione) supera 0.5...
LITE_PROG_HIGH = 0.5       # ...e finisce se la progression era > 0.5...
LITE_PROG_DROP = 0.3       # ...e crolla di almeno 0.3 (inizia una nuova azione)


# =============================================================================
# FUNZIONI DI UTILITA'
# =============================================================================
def all_video_ids():
    """Nomi dei video in data/Videos, ordinati (es. ['S1_Cheese_C1', ...])."""
    return sorted(p.stem for p in VIDEOS_DIR.glob("*.mp4"))


def subject_of(video_id):
    """Soggetto che ha registrato il video: 'S1_Coffee_C1' -> 'S1'."""
    return video_id.split("_")[0]
