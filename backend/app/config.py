import os

SUPPORTED_DEPLOYMENT_MODES = ("bsl", "wlasl2000")
_raw_mode = os.getenv("SL_DEPLOYMENT_MODE", "bsl").strip().lower()

if _raw_mode in ("wlasl", "wlasl2000", "2000"):
    from app.config_wlasl2000 import *  # noqa: F401,F403

    ACTIVE_DEPLOYMENT_MODE = "wlasl2000"
elif _raw_mode in ("bsl", "bobsl", "36"):
    from app.config_bsl import *  # noqa: F401,F403

    ACTIVE_DEPLOYMENT_MODE = "bsl"
else:
    raise ValueError(
        f"Unsupported SL_DEPLOYMENT_MODE='{_raw_mode}'. "
        f"Supported: {', '.join(SUPPORTED_DEPLOYMENT_MODES)}"
    )

# Backward-compatible alias for existing integrations.
DEPLOYMENT_MODE = ACTIVE_DEPLOYMENT_MODE
