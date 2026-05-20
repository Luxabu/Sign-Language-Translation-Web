import os
from typing import List

# Deployment profile identifier
DEPLOYMENT_MODE = "wlasl2000"
CHECKPOINT_SUBDIR = "wlasl2000"
STRICT_CLASS_DIM = True

# Paths (local to backend/app)
DL_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(DL_DIR, "checkpoints")
RUNS_DIR = os.path.join(DL_DIR, "runs")
DEMO_OUTPUT_DIR = os.path.join(DL_DIR, "demo_output")


def _load_wlasl2000_classes() -> List[str]:
    class_file = os.getenv(
        "WLASL2000_CLASS_LIST_FILE",
        os.path.join(DL_DIR, "class_lists", "wlasl2000_class_list.txt"),
    )
    if not os.path.exists(class_file):
        raise FileNotFoundError(
            f"Could not find WLASL-2000 class list at: {class_file}. "
            "Set WLASL2000_CLASS_LIST_FILE or bundle app/class_lists/wlasl2000_class_list.txt"
        )

    classes: List[str] = []
    with open(class_file, "r", encoding="utf-8") as f:
        for line in f:
            row = line.strip()
            if not row:
                continue
            parts = row.split(maxsplit=1)
            if len(parts) == 2:
                classes.append(parts[1])
            else:
                classes.append(parts[0])

    if len(classes) != 2000:
        raise ValueError(f"Expected 2000 classes, got {len(classes)} from {class_file}")

    return classes


# Classes / labels
CLASSES = _load_wlasl2000_classes()
NUM_CLASSES = len(CLASSES)
CLASS2IDX = {c: i for i, c in enumerate(CLASSES)}
IDX2CLASS = {i: c for c, i in CLASS2IDX.items()}
WLASL_LABEL_ALIASES = {"lose": "losear"}

# Input / model dims
N_FRAMES = 50
N_LANDMARKS = 75
N_COORDS = 2
INPUT_SIZE = N_LANDMARKS * N_COORDS

# Root-relative normalization
ROOTREL_CLIP_MIN = -3.0
ROOTREL_CLIP_MAX = 8.0
ROOTREL_IDX_LEFT_SHOULDER = 11
ROOTREL_IDX_RIGHT_SHOULDER = 12
ROOTREL_FALLBACK_SHOULDER_W = 0.3068
ROOTREL_MIN_SHOULDER_W = 0.02

# Model (scaled for 2000 classes)
CNN_CHANNELS = 256
LSTM_HIDDEN = 256
LSTM_LAYERS = 2
ATTN_HEADS = 4
DROP_CNN = 0.2
DROP_LSTM = 0.5
DROP_CLS = 0.5
FC_HIDDEN = 512

# Camera / realtime
CAM_BUFFER_SIZE = N_FRAMES
CAM_CONFIDENCE = 0.5
PREDICT_EVERY_N = 5
TRAIN_FPS = 25

# Sign spotting
SPOT_VELOCITY_THRESHOLD = 0.012
SPOT_MIN_SIGN_FRAMES = 15
SPOT_MAX_SIGN_FRAMES = 80
SPOT_COOLDOWN_FRAMES = 10
SPOT_PRE_BUFFER = 5
SPOT_POST_BUFFER = 5
SPOT_SMOOTH_WINDOW = 5
SPOT_IDLE_THRESHOLD = 0.005

# Ensemble
ENSEMBLE_MODELS = ["hybrid", "tcn", "bilstm", "transformer"]
ENSEMBLE_WEIGHTS = [0.150000, 0.350000, 0.150000, 0.350000]
ENSEMBLE_MODE = "soft"
ENSEMBLE_USE_KD = False
ENSEMBLE_USE_HARDMINE = False
