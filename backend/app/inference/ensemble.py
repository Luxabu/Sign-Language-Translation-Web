from __future__ import annotations

import os
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Tuple

from app.config import (
    NUM_CLASSES,
    IDX2CLASS,
    ENSEMBLE_MODELS,
    ENSEMBLE_WEIGHTS,
    ENSEMBLE_MODE,
    CHECKPOINT_DIR,
    CHECKPOINT_SUBDIR,
    STRICT_CLASS_DIM,
    ACTIVE_DEPLOYMENT_MODE,
)
from app.models import get_model


class Ensemble:
    """
    Multi-model ensemble cho sign classification.
    """

    def __init__(
        self,
        model_names: List[str] | None = None,
        weights: List[float] | None = None,
        mode: str = "soft",
        device: torch.device | None = None,
    ):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.mode = mode
        self.models: List[Tuple[str, torch.nn.Module, int]] = []
        self.num_classes = NUM_CLASSES
        self.idx2class = IDX2CLASS
        self.model_names = model_names or ENSEMBLE_MODELS
        self.weights = weights or ENSEMBLE_WEIGHTS

        w_sum = sum(self.weights) or 1.0
        self.weights = [w / w_sum for w in self.weights]
        while len(self.weights) < len(self.model_names):
            self.weights.append(1.0 / len(self.model_names))

        ckpt_dir = CHECKPOINT_DIR
        for name in self.model_names:
            candidates = [os.path.join(ckpt_dir, CHECKPOINT_SUBDIR, f"{name}_best.pt")]
            if not STRICT_CLASS_DIM:
                candidates.append(os.path.join(ckpt_dir, f"{name}_best.pt"))
            ckpt_path = next((p for p in candidates if os.path.exists(p)), None)
            if ckpt_path is None:
                continue
            try:
                ckpt = torch.load(ckpt_path, map_location=self.device)
            except FileNotFoundError:
                continue

            sd = ckpt.get("model_state_dict", ckpt.get("model", {}))
            ckpt_num_classes = self.num_classes
            for key, val in sd.items():
                if (
                    ("classifier" in key or "fc" in key)
                    and key.endswith(".weight")
                    and val.ndim == 2
                ):
                    ckpt_num_classes = val.shape[0]

            if STRICT_CLASS_DIM and ckpt_num_classes != self.num_classes:
                raise RuntimeError(
                    f"[Ensemble] Class-dimension mismatch in mode '{ACTIVE_DEPLOYMENT_MODE}': "
                    f"checkpoint '{ckpt_path}' outputs {ckpt_num_classes}, config expects {self.num_classes}."
                )

            mkw = ckpt.get("model_kwargs", {})
            model = get_model(name, num_classes=ckpt_num_classes, **mkw).to(self.device)
            model.load_state_dict(sd)
            model.eval()
            self.models.append((name, model, ckpt_num_classes))

        if not self.models:
            raise RuntimeError(
                "[Ensemble] No models loaded. Copy checkpoints into backend/app/checkpoints."
            )

        loaded_names = [n for n, _, _ in self.models]
        loaded_weights: List[float] = []
        for i, name in enumerate(self.model_names):
            if name in loaded_names:
                loaded_weights.append(self.weights[i] if i < len(self.weights) else 1.0)
        w_sum = sum(loaded_weights) or 1.0
        self.weights = [w / w_sum for w in loaded_weights]

    @classmethod
    def from_config(cls, device: torch.device | None = None) -> "Ensemble":
        return cls(
            model_names=ENSEMBLE_MODELS,
            weights=ENSEMBLE_WEIGHTS,
            mode=ENSEMBLE_MODE,
            device=device,
        )

    @torch.no_grad()
    def predict_probs(self, x: torch.Tensor) -> np.ndarray:
        x = x.to(self.device)
        all_probs = []
        for name, model, ckpt_nc in self.models:
            logits = model(x)
            probs = F.softmax(logits, dim=-1).cpu().numpy()
            if not STRICT_CLASS_DIM and ckpt_nc < self.num_classes:
                if probs.ndim == 1:
                    padded = np.zeros(self.num_classes, dtype=probs.dtype)
                    padded[:ckpt_nc] = probs
                    probs = padded
                else:
                    padded = np.zeros(
                        (probs.shape[0], self.num_classes), dtype=probs.dtype
                    )
                    padded[:, :ckpt_nc] = probs
                    probs = padded
            elif not STRICT_CLASS_DIM and ckpt_nc > self.num_classes:
                probs = probs[..., : self.num_classes]
            all_probs.append(probs)

        if self.mode == "soft":
            combined = np.zeros_like(all_probs[0])
            for prob, w in zip(all_probs, self.weights):
                combined += w * prob
            result = combined
        else:
            votes = np.zeros_like(all_probs[0])
            for prob, w in zip(all_probs, self.weights):
                idx = prob.argmax(axis=-1)
                for b in range(prob.shape[0]):
                    votes[b, idx[b]] += w
            result = votes / votes.sum(axis=-1, keepdims=True)

        if result.shape[0] == 1:
            result = result[0]
        return result

    def predict_label(self, x: torch.Tensor) -> tuple[str, float]:
        probs = self.predict_probs(x)
        top_idx = probs.argmax()
        return self.idx2class[int(top_idx)], float(probs[top_idx])

    @property
    def num_models(self) -> int:
        return len(self.models)
