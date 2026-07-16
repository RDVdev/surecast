"""
Central configuration for the forecasting pipeline.
All hyperparameters, paths, and constants live here — no magic numbers elsewhere.
"""
import os
import json
import torch

# ─── Reproducibility ─────────────────────────────────────────────────
RANDOM_SEED = 42

# ─── Data Paths ──────────────────────────────────────────────────────
RAW_DATA_PATH = "data/DataCoSupplyChainDataset.csv"
CLEANED_DATA_PATH = "data/cleaned_dataset.csv"
ENGINEERED_DATA_PATH = "data/engineered_dataset.csv"
PREDICTIONS_PATH = "data/model_predictions.csv"
EVALUATION_METRICS_PATH = "data/evaluation_metrics.csv"

# ─── Model Paths ─────────────────────────────────────────────────────
MODEL_DIR = "models"
TARGET_SCALER_PATH = os.path.join(MODEL_DIR, "target_scaler.pkl")
DL_MODEL_PATH = os.path.join(MODEL_DIR, "surecast_dl.pth")
BEST_SEQ_LEN_PATH = os.path.join(MODEL_DIR, "best_seq_len.json")
META_LEARNER_PATH = os.path.join(MODEL_DIR, "meta_learner.pkl")

# ─── Report Paths ────────────────────────────────────────────────────
REPORTS_DIR = "reports"
PHASE1_REPORT_PATH = os.path.join(REPORTS_DIR, "phase1_summary.txt")
FINAL_REPORT_PATH = os.path.join(REPORTS_DIR, "final_evaluation_report.md")

# ─── Data Splits (temporal) ──────────────────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ─── Sequence Construction ───────────────────────────────────────────
DEFAULT_SEQ_LEN = 8          # fallback if Phase 2 hasn't run yet
CANDIDATE_WINDOW_LENGTHS = [4, 8, 12, 26]
PADDING_STRATEGY = "pad"     # 'pad' or 'exclude'

# ─── Deep Learning Hyperparameters ───────────────────────────────────
DL_BATCH_SIZE = 512
DL_LEARNING_RATE = 0.001
DL_WEIGHT_DECAY = 0.01
DL_EPOCHS = 100
DL_PATIENCE = 15
DL_CHANNELS = ['cnn', 'lstm', 'gru', 'bilstm', 'transformer']

# ─── ML Ensemble ─────────────────────────────────────────────────────
ML_N_ESTIMATORS = 50
ML_MAX_DEPTH = None  # use default

# ─── Fusion ──────────────────────────────────────────────────────────
FUSION_WEIGHT_CANDIDATES = [0.3, 0.4, 0.5, 0.6, 0.7]
STACKING_CV_FOLDS = 5

# ─── Feature Engineering ─────────────────────────────────────────────
ROLLING_WINDOWS = [4, 12, 50]
IMPORTANCE_THRESHOLD = 0.001
MAX_CATEGORICAL_CARDINALITY = 50

# ─── Cross-Validation ────────────────────────────────────────────────
CV_FOLDS = 3
CV_EPOCHS = 15  # reduced for speed during CV

# ─── Phase 6 RLHF ────────────────────────────────────────────────────
DPO_BETA = 0.1
DPO_LEARNING_RATE = 1e-4
DPO_EPOCHS = 3
SYNTHETIC_PAIR_NOISE_SCALE = 0.3  # fraction of std for rejection perturbation

# ─── Device ──────────────────────────────────────────────────────────
def get_device():
    """Auto-detect best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

DEVICE = get_device()

# ─── Utility ─────────────────────────────────────────────────────────
def load_best_seq_len():
    """Load the best sequence length from Phase 2, or fall back to default."""
    if os.path.exists(BEST_SEQ_LEN_PATH):
        with open(BEST_SEQ_LEN_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("best_T", DEFAULT_SEQ_LEN)
    return DEFAULT_SEQ_LEN

def save_best_seq_len(best_T):
    """Persist the best sequence length from Phase 2."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(BEST_SEQ_LEN_PATH, "w", encoding="utf-8") as f:
        json.dump({"best_T": best_T}, f)
