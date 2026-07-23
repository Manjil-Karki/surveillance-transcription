"""Central configuration for the surveillance anomaly detection project."""

from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parent.parent
DATA_DIR     = ROOT / "data" / "dataset"
OUTPUTS_DIR  = ROOT / "outputs"
MODELS_DIR   = OUTPUTS_DIR / "models"
PLOTS_DIR    = OUTPUTS_DIR / "plots"
RESULTS_DIR  = OUTPUTS_DIR / "results"

# Kaggle path (used when DATA_DIR does not exist)
KAGGLE_DATA  = Path("/kaggle/input/ucf-crime-dataset")

def get_data_root():
    if DATA_DIR.exists():
        return DATA_DIR
    if KAGGLE_DATA.exists():
        return KAGGLE_DATA
    raise FileNotFoundError(
        f"Dataset not found at {DATA_DIR} or {KAGGLE_DATA}.\n"
        "Copy the dataset with:\n"
        '  cp -r "/mnt/c/Users/acer/Downloads/archive (4)" '
        f"{DATA_DIR}"
    )

# ── Classes ───────────────────────────────────────────────────────────────────
# Folder name in dataset  →  clean label
CLASS_MAP = {
    "NormalVideos": "Normal",
    "Fighting":     "Fighting",
    "Robbery":      "Robbery",
    "Arson":        "Arson",
    "Burglary":     "Burglary",
}
CLASSES      = list(CLASS_MAP.values())          # ["Normal", "Fighting", ...]
NUM_CLASSES  = len(CLASSES)                      # 5
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for c, i in CLASS_TO_IDX.items()}
ANOMALY_CLASSES = [c for c in CLASSES if c != "Normal"]

# ── Data ──────────────────────────────────────────────────────────────────────
FRAME_H      = 64
FRAME_W      = 64
CHANNELS     = 3        # output channels (reconstruction)
IN_CHANNELS  = 6        # input channels: 3 RGB + 3 frame-difference (motion)
CLIP_LEN     = 16       # frames per clip (16 captures full anomaly events)
TRAIN_STRIDE = 16       # non-overlapping clips for training
TEST_STRIDE  = 8        # 50% overlap for test sliding window
MAX_FRAMES   = None     # no cap — use full dataset; class weights handle imbalance

# ── Augmentation ──────────────────────────────────────────────────────────────
AUG_BRIGHTNESS = 0.2    # brightness jitter range ±
AUG_CONTRAST   = 0.2    # contrast jitter range ±
AUG_NOISE_STD  = 0.02   # Gaussian noise std

# ── Model ─────────────────────────────────────────────────────────────────────
LATENT_DIM   = 256      # CNN encoder output dimension per frame
LSTM_UNITS   = 128
FC_UNITS     = 64
DROPOUT      = 0.5

# ── Training ──────────────────────────────────────────────────────────────────
PRETRAIN_EPOCHS  = 30
JOINT_EPOCHS     = 30
BATCH_SIZE       = 64       # 64 clips × 16 frames = 1024 frames/batch; safe for 8 GB VRAM
LEARNING_RATE    = 1e-4     # reduced from 1e-3 to limit memorisation speed
WEIGHT_DECAY     = 1e-4     # L2 regularisation
LAMBDA           = 0.5      # weight of classification loss in joint loss
                            # tried: 0.1, 0.5, 1.0 (ablation)

# ── Anomaly threshold ─────────────────────────────────────────────────────────
# Determined on validation Normal clips after pre-training
# Placeholder — computed and saved in 02_autoencoder_pretrain notebook
THRESHOLD_2STD   = None
THRESHOLD_3STD   = None
THRESHOLD_OPTIMAL = None    # Youden's J from ROC on validation set

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
