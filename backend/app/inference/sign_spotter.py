from __future__ import annotations

import numpy as np
from collections import deque
from enum import Enum, auto

from app.config import (
    N_FRAMES,
    N_LANDMARKS,
    SPOT_VELOCITY_THRESHOLD,
    SPOT_MIN_SIGN_FRAMES,
    SPOT_MAX_SIGN_FRAMES,
    SPOT_COOLDOWN_FRAMES,
    SPOT_PRE_BUFFER,
    SPOT_POST_BUFFER,
    SPOT_SMOOTH_WINDOW,
    SPOT_IDLE_THRESHOLD,
)


class SpotterState(Enum):
    IDLE = auto()
    SIGNING = auto()
    COOLDOWN = auto()


class SignSpotter:
    """
    Heuristic sign spotter using hand landmark velocity.
    Logic copy từ sign_spotter.py trong BOBSL.
    """

    def __init__(
        self,
        velocity_threshold: float = SPOT_VELOCITY_THRESHOLD,
        min_sign_frames: int = SPOT_MIN_SIGN_FRAMES,
        max_sign_frames: int = SPOT_MAX_SIGN_FRAMES,
        cooldown_frames: int = SPOT_COOLDOWN_FRAMES,
        pre_buffer: int = SPOT_PRE_BUFFER,
        post_buffer: int = SPOT_POST_BUFFER,
        smooth_window: int = SPOT_SMOOTH_WINDOW,
        idle_threshold: float = SPOT_IDLE_THRESHOLD,
        fps_ratio: float = 1.0,
    ):
        self.velocity_threshold = velocity_threshold
        self.min_sign_frames = min_sign_frames
        self.max_sign_frames = max_sign_frames
        self.cooldown_frames = cooldown_frames
        self.pre_buffer = pre_buffer
        self.post_buffer = post_buffer
        self.smooth_window = smooth_window
        self.idle_threshold = idle_threshold
        self.fps_ratio = fps_ratio

        self.state = SpotterState.IDLE
        self.prev_kps = None
        self.pre_ring = deque(maxlen=pre_buffer)
        self.sign_frames = []
        self.idle_counter = 0
        self.cooldown_counter = 0
        self.vel_history = deque(maxlen=smooth_window)
        self.frame_idx = 0
        self.sign_count = 0
        self._channel_vels = {"body": 0.0, "face": 0.0, "lh": 0.0, "rh": 0.0}

    def reset(self) -> None:
        self.state = SpotterState.IDLE
        self.prev_kps = None
        self.pre_ring.clear()
        self.sign_frames = []
        self.idle_counter = 0
        self.cooldown_counter = 0
        self.vel_history.clear()
        self.frame_idx = 0

    def _compute_channel_velocities(self, kps: np.ndarray) -> dict:
        if self.prev_kps is None:
            return {"body": 0.0, "face": 0.0, "lh": 0.0, "rh": 0.0}

        def _center_vel(indices):
            curr = kps[indices]
            prev = self.prev_kps[indices]
            valid = np.any(curr != 0, axis=1) & np.any(prev != 0, axis=1)
            if not np.any(valid):
                return 0.0
            return float(
                np.linalg.norm(curr[valid].mean(axis=0) - prev[valid].mean(axis=0))
            )

        face_vel = _center_vel(np.arange(0, 11))
        body_vel = _center_vel(np.array([11, 12, 13, 14, 23, 24]))

        lh_curr = kps[33:54]
        lh_prev = self.prev_kps[33:54]
        lh_vel = (
            float(np.linalg.norm(lh_curr.mean(axis=0) - lh_prev.mean(axis=0)))
            if (np.any(lh_curr) and np.any(lh_prev))
            else 0.0
        )

        rh_curr = kps[54:75]
        rh_prev = self.prev_kps[54:75]
        rh_vel = (
            float(np.linalg.norm(rh_curr.mean(axis=0) - rh_prev.mean(axis=0)))
            if (np.any(rh_curr) and np.any(rh_prev))
            else 0.0
        )

        return {"body": body_vel, "face": face_vel, "lh": lh_vel, "rh": rh_vel}

    def _compute_hand_velocity(self, kps: np.ndarray) -> float:
        if self.prev_kps is None:
            return 0.0
        vel = 0.0

        lh_curr = kps[33:54]
        lh_prev = self.prev_kps[33:54]
        if np.any(lh_curr) and np.any(lh_prev):
            lc = lh_curr.mean(axis=0)
            lp = lh_prev.mean(axis=0)
            vel = max(vel, np.linalg.norm(lc - lp))

        rh_curr = kps[54:75]
        rh_prev = self.prev_kps[54:75]
        if np.any(rh_curr) and np.any(rh_prev):
            rc = rh_curr.mean(axis=0)
            rp = rh_prev.mean(axis=0)
            vel = max(vel, np.linalg.norm(rc - rp))

        for idx in (15, 16):
            if np.any(kps[idx]) and np.any(self.prev_kps[idx]):
                wvel = np.linalg.norm(kps[idx] - self.prev_kps[idx])
                vel = max(vel, wvel)

        return float(vel)

    def _smoothed_velocity(self, raw_vel: float) -> float:
        self.vel_history.append(raw_vel)
        if not self.vel_history:
            return raw_vel
        return float(np.mean(self.vel_history))

    def _finalize_sign(self, fps_ratio: float = 1.0):
        frames = list(self.pre_ring) + self.sign_frames
        n = len(frames)
        if n < self.min_sign_frames:
            self.sign_frames = []
            return None

        self.sign_count += 1
        raw_length = n

        if fps_ratio != 1.0 and n > 1:
            target_n = max(1, int(round(n * fps_ratio)))
            indices = np.linspace(0, n - 1, target_n).astype(int)
            frames = [frames[i] for i in indices]
            n = len(frames)

        if n >= N_FRAMES:
            start = (n - N_FRAMES) // 2
            frames = frames[start : start + N_FRAMES]
        else:
            pad_count = N_FRAMES - n
            frames = frames + [frames[-1]] * pad_count

        meta = {
            "raw_length": raw_length,
            "sign_index": self.sign_count,
            "frame_idx": self.frame_idx,
        }
        self.sign_frames = []
        return frames, meta

    def feed(self, kps: np.ndarray):
        self.frame_idx += 1
        result = None

        raw_vel = self._compute_hand_velocity(kps)
        velocity = self._smoothed_velocity(raw_vel)
        self._channel_vels = self._compute_channel_velocities(kps)
        self.prev_kps = kps.copy()

        if self.state == SpotterState.IDLE:
            self.pre_ring.append(kps.copy())
            if velocity > self.velocity_threshold:
                self.state = SpotterState.SIGNING
                self.sign_frames = [kps.copy()]
                self.idle_counter = 0

        elif self.state == SpotterState.SIGNING:
            self.sign_frames.append(kps.copy())
            if velocity < self.idle_threshold:
                self.idle_counter += 1
            else:
                self.idle_counter = 0

            total_len = len(self.pre_ring) + len(self.sign_frames)

            if self.idle_counter >= self.post_buffer:
                result = self._finalize_sign(self.fps_ratio)
                self.state = SpotterState.COOLDOWN
                self.cooldown_counter = 0
            elif total_len >= self.max_sign_frames:
                result = self._finalize_sign(self.fps_ratio)
                self.state = SpotterState.COOLDOWN
                self.cooldown_counter = 0

        elif self.state == SpotterState.COOLDOWN:
            self.cooldown_counter += 1
            if self.cooldown_counter >= self.cooldown_frames:
                self.state = SpotterState.IDLE
                self.pre_ring.clear()
                self.idle_counter = 0

        return result

    @property
    def is_signing(self) -> bool:
        return self.state == SpotterState.SIGNING

    @property
    def current_sign_length(self) -> int:
        if self.state == SpotterState.SIGNING:
            return len(self.pre_ring) + len(self.sign_frames)
        return 0

    def get_debug_info(self) -> dict:
        return {
            "state": self.state.name,
            "velocity": float(self.vel_history[-1]) if self.vel_history else 0.0,
            "smoothed_vel": (
                float(np.mean(self.vel_history)) if self.vel_history else 0.0
            ),
            "sign_len": self.current_sign_length,
            "sign_count": self.sign_count,
            "idle_counter": self.idle_counter,
            "channel_vels": dict(self._channel_vels),
        }
