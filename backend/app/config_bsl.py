import os

# Deployment profile identifier
DEPLOYMENT_MODE = "bsl"
CHECKPOINT_SUBDIR = "bsl"
# Keep legacy compatibility: existing deployed BSL checkpoints may not exactly match
# the 36-class frontend label set.
STRICT_CLASS_DIM = False

# Paths (local to backend/app)
DL_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(DL_DIR, "checkpoints")
RUNS_DIR = os.path.join(DL_DIR, "runs")
DEMO_OUTPUT_DIR = os.path.join(DL_DIR, "demo_output")

# Classes / labels
CLASSES = sorted(
    [
        "afternoon",
        "always",
        "and",
        "animal",
        "anything",
        "area",
        "ask",
        "because",
        "best",
        "better",
        "big",
        "bird",
        "building",
        "but",
        "call",
        "camera",
        "chicken",
        "die",
        "different",
        "education",
        "good",
        "learn",
        "make",
        "me",
        "morning",
        "my",
        "name",
        "one",
        "our",
        "show",
        "sign",
        "technology",
        "three",
        "two",
        "understand",
        "work",
    ]
)
NUM_CLASSES = len(CLASSES)
CLASS2IDX = {c: i for i, c in enumerate(CLASSES)}
IDX2CLASS = {i: c for c, i in CLASS2IDX.items()}

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

# Model
CNN_CHANNELS = 256
LSTM_HIDDEN = 256
LSTM_LAYERS = 2
ATTN_HEADS = 4
DROP_CNN = 0.1
DROP_LSTM = 0.3
DROP_CLS = 0.4
FC_HIDDEN = 256

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
ENSEMBLE_WEIGHTS = [0.15, 0.40, 0.10, 0.35]
ENSEMBLE_MODE = "soft"
ENSEMBLE_USE_KD = False
ENSEMBLE_USE_HARDMINE = False
