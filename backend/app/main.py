import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes import router as api_router
from app.api.websocket import router as ws_router
from app.config import (
    ACTIVE_DEPLOYMENT_MODE,
    CHECKPOINT_DIR,
    CHECKPOINT_SUBDIR,
    ENSEMBLE_MODELS,
    STRICT_CLASS_DIM,
)


def _validate_startup_assets() -> None:
    if ACTIVE_DEPLOYMENT_MODE != "wlasl2000" or not STRICT_CLASS_DIM:
        return

    missing: list[str] = []
    for model_name in ENSEMBLE_MODELS:
        ckpt = os.path.join(CHECKPOINT_DIR, CHECKPOINT_SUBDIR, f"{model_name}_best.pt")
        if not os.path.exists(ckpt):
            missing.append(ckpt)

    if missing:
        msg = "\n".join(missing)
        raise RuntimeError(
            "WLASL2000 strict deployment requires all ensemble checkpoints. Missing:\n" + msg
        )


def create_app() -> FastAPI:
    """
    FastAPI application factory.

    This is the single entry-point that will be used both by:
      - `uvicorn app.main:create_app` in development
      - any future ASGI deployment.
    """
    app = FastAPI(title="Sign Language Web API", version="0.1.0")

    # CORS: allow frontend on all origins (local dev + production)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["*"],
    )

    _validate_startup_assets()

    app.include_router(api_router)
    app.include_router(ws_router)

    # Serve frontend static files
    frontend_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "frontend", "assets")
    if os.path.exists(frontend_dir):
        app.mount("/assets", StaticFiles(directory=frontend_dir), name="assets")

    return app


app = create_app()
