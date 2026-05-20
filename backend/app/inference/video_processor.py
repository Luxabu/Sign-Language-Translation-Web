from __future__ import annotations

import time
from typing import Any

import cv2
import numpy as np

from app.config import N_FRAMES, TRAIN_FPS


class VideoProcessingError(RuntimeError):
    """Raised when an uploaded video cannot be decoded or processed."""


def process_uploaded_video(
    session,
    video_path: str,
    target_fps: int = TRAIN_FPS,
    window_size: int = N_FRAMES,
    stride: int = 5,
) -> dict[str, Any]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise VideoProcessingError("Cannot open uploaded video file")

    try:
        started_at = time.perf_counter()
        original_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if original_fps <= 0:
            original_fps = float(target_fps)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration_sec = (total_frames / original_fps) if total_frames > 0 else 0.0

        frame_sampling_step = 1
        if target_fps > 0 and original_fps > target_fps:
            frame_sampling_step = max(1, int(round(original_fps / target_fps)))

        keypoint_buffer: list[np.ndarray] = []
        sampled_indices: list[int] = []
        timeline: list[dict[str, Any]] = []

        frame_index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_sampling_step > 1 and (frame_index % frame_sampling_step) != 0:
                frame_index += 1
                continue

            keypoints = session.predictor.extract_keypoints_from_bgr(frame)
            kps_arr = np.asarray(keypoints["pose_hands"], dtype=np.float32)

            keypoint_buffer.append(kps_arr)
            sampled_indices.append(frame_index)

            if len(keypoint_buffer) >= window_size and (
                (len(keypoint_buffer) - window_size) % max(1, stride)
            ) == 0:
                window = keypoint_buffer[-window_size:]
                label, confidence = session.predictor.predict_from_keypoints_sequence(window)
                emitted = session.sentence_builder.feed(label, confidence)

                window_end_frame = sampled_indices[-1]
                timestamp_sec = window_end_frame / original_fps
                timeline.append(
                    {
                        "timestamp_sec": round(float(timestamp_sec), 3),
                        "frame_index": int(window_end_frame),
                        "prediction": label,
                        "confidence": float(confidence),
                        "emitted": emitted is not None,
                    }
                )

            frame_index += 1

        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        return {
            "final_sentence": session.sentence_builder.sentence,
            "timeline": timeline,
            "metadata": {
                "duration_sec": round(float(duration_sec), 3),
                "source_fps": round(float(original_fps), 3),
                "target_fps": int(target_fps),
                "source_frame_count": int(total_frames),
                "processed_frame_count": len(keypoint_buffer),
                "window_size": int(window_size),
                "stride": int(stride),
                "processing_ms": round(float(elapsed_ms), 2),
            },
        }
    except VideoProcessingError:
        raise
    except Exception as exc:
        raise VideoProcessingError(f"Failed while processing video: {exc}") from exc
    finally:
        cap.release()
