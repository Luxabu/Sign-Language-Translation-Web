from __future__ import annotations

import numpy as np
import torch

from app.config import (
    N_FRAMES,
    N_LANDMARKS,
    INPUT_SIZE,
    TRAIN_FPS,
    ROOTREL_CLIP_MIN,
    ROOTREL_CLIP_MAX,
    ROOTREL_IDX_LEFT_SHOULDER,
    ROOTREL_IDX_RIGHT_SHOULDER,
    ROOTREL_FALLBACK_SHOULDER_W,
    ROOTREL_MIN_SHOULDER_W,
)


def extract_keypoints(results) -> np.ndarray:
    """Trích xuất 75 landmark (pose + 2 tay) từ kết quả MediaPipe Holistic."""
    kps = np.zeros((N_LANDMARKS, 2), dtype=np.float32)

    if results.pose_landmarks:
        for i, lm in enumerate(results.pose_landmarks.landmark[:33]):
            kps[i] = [lm.x, lm.y]

    if results.left_hand_landmarks:
        for i, lm in enumerate(results.left_hand_landmarks.landmark):
            kps[33 + i] = [lm.x, lm.y]

    if results.right_hand_landmarks:
        for i, lm in enumerate(results.right_hand_landmarks.landmark):
            kps[54 + i] = [lm.x, lm.y]

    return kps





def detection_quality(kps: np.ndarray) -> dict:
    """Đánh giá nhanh chất lượng 1 frame keypoint (75,2)."""
    pose_nonzero = np.any(kps[:33] != 0, axis=1).sum()
    left_nonzero = np.any(kps[33:54] != 0, axis=1).sum()
    right_nonzero = np.any(kps[54:75] != 0, axis=1).sum()
    total_nonzero = np.any(kps != 0, axis=1).sum()

    return {
        "has_pose": pose_nonzero > 5,
        "has_left": left_nonzero > 10,
        "has_right": right_nonzero > 10,
        "hand_count": int(left_nonzero > 10) + int(right_nonzero > 10),
        "nonzero_ratio": total_nonzero / 75.0,
    }


def buffer_has_hands(buffer, min_hand_frames: int = 10) -> bool:
    """Kiểm tra buffer có đủ số frame có tay để suy luận hay không."""
    hand_frames = 0
    for kps in buffer:
        left = np.any(kps[33:54] != 0, axis=1).sum()
        right = np.any(kps[54:75] != 0, axis=1).sum()
        if left > 10 or right > 10:
            hand_frames += 1
    return hand_frames >= min_hand_frames


def filter_transition_frames(frames: list, var_threshold: float = 0.0001) -> list:
    """Lọc các frame flicker / frozen từ MediaPipe (chuyển tiếp)."""
    if len(frames) < 3:
        return frames
    filtered = list(frames)
    for i in range(len(filtered)):
        kps = filtered[i]
        nz = kps[np.any(kps != 0, axis=1)]
        if len(nz) < 5 or nz.var() < var_threshold:
            replaced = False
            for j in range(1, len(filtered)):
                for k in (i + j, i - j):
                    if 0 <= k < len(filtered):
                        nz2 = filtered[k][np.any(filtered[k] != 0, axis=1)]
                        if len(nz2) >= 5 and nz2.var() >= var_threshold:
                            filtered[i] = filtered[k].copy()
                            replaced = True
                            break
                if replaced:
                    break
    return filtered


def resample_fps(
    frames: list, source_fps: float, target_fps: float = TRAIN_FPS
) -> list:
    """Resample danh sách frame keypoint từ source_fps về target_fps."""
    n = len(frames)
    if n == 0 or abs(source_fps - target_fps) < 0.5:
        return frames
    duration = n / source_fps
    target_n = max(1, int(round(duration * target_fps)))
    indices = np.linspace(0, n - 1, target_n).astype(int)
    return [frames[i] for i in indices]


def preprocess_frames(frames: list) -> torch.Tensor:
    """
    frames (list[(75,2)]) → tensor (1, N_FRAMES, INPUT_SIZE)
    với root-relative normalization giống training.
    """
    n = len(frames)
    if n == 0:
        return torch.zeros(1, N_FRAMES, INPUT_SIZE)

    arr = np.stack(frames, axis=0).astype(np.float32)

    if n >= N_FRAMES:
        start = (n - N_FRAMES) // 2
        arr = arr[start : start + N_FRAMES]
    else:
        pad = np.tile(arr[-1:], (N_FRAMES - n, 1, 1))
        arr = np.concatenate([arr, pad], axis=0)

    T = arr.shape[0]
    l_sh = arr[:, ROOTREL_IDX_LEFT_SHOULDER, :]
    r_sh = arr[:, ROOTREL_IDX_RIGHT_SHOULDER, :]

    l_present = np.any(l_sh != 0, axis=1)
    r_present = np.any(r_sh != 0, axis=1)
    both = l_present & r_present

    roots = np.zeros((T, 2), dtype=np.float32)
    scales = np.full(T, ROOTREL_FALLBACK_SHOULDER_W, dtype=np.float32)

    if both.any():
        vi = np.where(both)[0]
        roots[vi] = (l_sh[vi] + r_sh[vi]) / 2.0
        scales[vi] = np.linalg.norm(l_sh[vi] - r_sh[vi], axis=1)

        if not both.all():
            last = -1
            for t in range(T):
                if both[t]:
                    last = t
                elif last >= 0:
                    roots[t] = roots[last]
                    scales[t] = scales[last]
            first = vi[0]
            if first > 0:
                for t in range(first):
                    roots[t] = roots[first]
                    scales[t] = scales[first]

    scales = np.maximum(scales, ROOTREL_MIN_SHOULDER_W)

    out = np.zeros_like(arr)
    for t in range(T):
        nonzero = np.any(arr[t] != 0, axis=1)
        if nonzero.any():
            out[t, nonzero] = (arr[t, nonzero] - roots[t]) / scales[t]

    out = np.clip(out, ROOTREL_CLIP_MIN, ROOTREL_CLIP_MAX)
    out = out.reshape(N_FRAMES, INPUT_SIZE)
    return torch.from_numpy(out).unsqueeze(0).float()
