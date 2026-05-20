from __future__ import annotations

import base64
import io
import os
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import mediapipe as mp
import numpy as np
import torch
from PIL import Image

from app.config import (
    NUM_CLASSES,
    IDX2CLASS,
    CHECKPOINT_DIR,
    CHECKPOINT_SUBDIR,
    STRICT_CLASS_DIM,
    ACTIVE_DEPLOYMENT_MODE,
)
from app.inference.preprocess import (
    filter_transition_frames,
    preprocess_frames,
    extract_keypoints,
)
from app.inference.ensemble import Ensemble
from app.models import get_model


@dataclass
class BackendPredictorConfig:
    model: str = "hybrid"
    mode: str = "spotting"
    confidence: float = 0.55
    ensemble: bool = True
    use_antibias: bool = True
    show_skeleton: bool = False


class _SingleModelPredictor:
    def __init__(self, model_name: str, checkpoint: str | None = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ckpt_num_classes = NUM_CLASSES
        if checkpoint is None:
            candidates = [os.path.join(CHECKPOINT_DIR, CHECKPOINT_SUBDIR, f"{model_name}_best.pt")]
            if not STRICT_CLASS_DIM:
                candidates.append(os.path.join(CHECKPOINT_DIR, f"{model_name}_best.pt"))
            ckpt_path = next((p for p in candidates if os.path.exists(p)), candidates[-1])
        else:
            ckpt_path = checkpoint
        ckpt = torch.load(ckpt_path, map_location=self.device)
        state_dict = ckpt.get("model_state_dict", ckpt.get("model", {}))
        mkw = ckpt.get("model_kwargs", {})
        ckpt_nc = ckpt.get("num_classes", None)
        if ckpt_nc:
            self.ckpt_num_classes = ckpt_nc
        else:
            for key, val in state_dict.items():
                if (
                    ("classifier" in key or "fc" in key)
                    and key.endswith(".weight")
                    and val.ndim == 2
                ):
                    self.ckpt_num_classes = val.shape[0]

        if STRICT_CLASS_DIM and self.ckpt_num_classes != NUM_CLASSES:
            raise RuntimeError(
                f"[SingleModel] Class-dimension mismatch in mode '{ACTIVE_DEPLOYMENT_MODE}': "
                f"checkpoint '{ckpt_path}' outputs {self.ckpt_num_classes}, config expects {NUM_CLASSES}."
            )

        self.model = get_model(model_name, num_classes=self.ckpt_num_classes, **mkw).to(
            self.device
        )
        self.model.load_state_dict(state_dict)
        self.model.eval()

    @torch.no_grad()
    def predict_probs(self, x: torch.Tensor) -> np.ndarray:
        logits = self.model(x.to(self.device))
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        if not STRICT_CLASS_DIM and self.ckpt_num_classes < NUM_CLASSES:
            padded = np.zeros(NUM_CLASSES, dtype=probs.dtype)
            padded[: self.ckpt_num_classes] = probs
            probs = padded
        elif not STRICT_CLASS_DIM and self.ckpt_num_classes > NUM_CLASSES:
            probs = probs[:NUM_CLASSES]
        return probs


class BackendPredictor:
    """
    Inference backend độc lập, không phụ thuộc BOBSL_Lightweight.
    """

    def __init__(self, cfg: BackendPredictorConfig) -> None:
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_antibias = cfg.use_antibias

        # Anti-bias filtering is currently not implemented in this backend.
        # The flag is preserved for API compatibility with the original demo.
        if cfg.ensemble:
            self.ensemble = Ensemble.from_config(device=self.device)
            self.single = None
        else:
            self.ensemble = None
            self.single = _SingleModelPredictor(cfg.model)

        self._holistic = mp.solutions.holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1,
        )

    def close(self) -> None:
        self._holistic.close()

    @staticmethod
    def _decode_base64_image(data: str) -> np.ndarray:
        if "," in data:
            _, data = data.split(",", 1)
        raw = base64.b64decode(data)
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        arr = np.array(img)[:, :, ::-1]
        return arr

    def _predict_from_frames(self, frames: List[np.ndarray]) -> Tuple[str, float]:
        frames = filter_transition_frames(frames)
        x = preprocess_frames(frames)
        if self.ensemble is not None:
            probs = self.ensemble.predict_probs(x)
        else:
            probs = self.single.predict_probs(x)  # type: ignore[union-attr]
        top_idx = int(np.argmax(probs))
        return IDX2CLASS[top_idx], float(probs[top_idx])

    def extract_keypoints_from_bgr(self, frame: np.ndarray) -> dict:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._holistic.process(rgb)
        rgb.flags.writeable = True

        # 75 landmarks (pose + hands) used for model input
        kps = extract_keypoints(results)

        return {
            "pose_hands": kps.tolist()
        }

    def predict_from_keypoints_sequence(self, frames: List[np.ndarray]) -> Tuple[str, float]:
        return self._predict_from_frames(frames)

    def predict_from_base64(self, frame_b64: str) -> Tuple[dict, str, float]:
        frame = self._decode_base64_image(frame_b64)
        keypoints = self.extract_keypoints_from_bgr(frame)
        kps_arr = np.asarray(keypoints["pose_hands"], dtype=np.float32)
        label, conf = self._predict_from_frames([kps_arr])
        return keypoints, label, conf
