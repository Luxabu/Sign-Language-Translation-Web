import asyncio
import json
import threading
import time
from collections import deque
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Dict, List, Optional
import cv2
import numpy as np

from app.inference.predictor import BackendPredictor, BackendPredictorConfig
from app.inference.sentence_builder import SentenceBuilder
from app.config import ACTIVE_DEPLOYMENT_MODE, NUM_CLASSES


router = APIRouter()


class WebcamSession:
    def __init__(self, cfg: BackendPredictorConfig):
        self.cfg = cfg
        self.predictor = BackendPredictor(cfg)
        self.sentence_builder = SentenceBuilder(
            stability_count=2,
            cooldown_sec=1.0,
            confidence_threshold=cfg.confidence,
        )

        self.cap: Optional[cv2.VideoCapture] = None
        self.running = False
        self.frame_buffer: deque = deque(maxlen=32)
        self.buffer_lock = threading.Lock()

        self.window_size = 16  # sliding window for inference
        self.capture_fps = 30
        self.inference_interval = 0.1  # 10 FPS consumer loop

        self.producer_thread: Optional[threading.Thread] = None
        self.consumer_thread: Optional[threading.Thread] = None
        self.websocket: Optional[WebSocket] = None
        self.loop: Optional[asyncio.AbstractEventLoop] = None

    def start_capture(self, websocket: WebSocket) -> None:
        self.websocket = websocket
        self.loop = asyncio.get_event_loop()

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open webcam")

        self.running = True

        self.producer_thread = threading.Thread(target=self._producer_loop, name="webcam-producer", daemon=True)
        self.consumer_thread = threading.Thread(target=self._consumer_loop, name="webcam-consumer", daemon=True)

        self.producer_thread.start()
        self.consumer_thread.start()

    def stop_capture(self) -> None:
        self.running = False

        if self.producer_thread and self.producer_thread.is_alive():
            self.producer_thread.join(timeout=1.0)

        if self.consumer_thread and self.consumer_thread.is_alive():
            self.consumer_thread.join(timeout=1.0)

        if self.cap:
            self.cap.release()
            self.cap = None

        with self.buffer_lock:
            self.frame_buffer.clear()

    def _producer_loop(self) -> None:
        while self.running and self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                with self.buffer_lock:
                    self.frame_buffer.append(frame)
            time.sleep(1.0 / self.capture_fps)

    def _consumer_loop(self) -> None:
        while self.running:
            frames = []
            with self.buffer_lock:
                if len(self.frame_buffer) >= self.window_size:
                    frames = list(self.frame_buffer)[-self.window_size:]

            if frames:
                keypoint_sequence = []
                for frame in frames:
                    keypoints = self.predictor.extract_keypoints_from_bgr(frame)
                    kps_arr = np.asarray(keypoints["pose_hands"], dtype=np.float32)
                    keypoint_sequence.append(kps_arr)

                label, confidence = self.predictor.predict_from_keypoints_sequence(keypoint_sequence)
                self.sentence_builder.feed(label, confidence)

                payload = {
                    "prediction": label,
                    "confidence": float(confidence),
                    "sentence": self.sentence_builder.sentence,
                    "keypoints": keypoints if self.cfg.show_skeleton else None,
                }

                if self.websocket and self.loop and self.loop.is_running():
                    try:
                        future = asyncio.run_coroutine_threadsafe(self.websocket.send_text(json.dumps(payload)), self.loop)
                        future.result(timeout=1.0)
                    except Exception:
                        pass

            time.sleep(self.inference_interval)


class ConnectionManager:
    def __init__(self) -> None:
        self.active_connections: List[WebSocket] = []
        self.sessions: Dict[str, WebcamSession] = {}

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        # Clean up any session for this websocket if needed

    async def send_text(self, message: str, websocket: WebSocket) -> None:
        await websocket.send_text(message)


manager = ConnectionManager()


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """
    Realtime prediction channel with webcam support.

    Commands:
    - {"action": "start_capture", "model": "hybrid", "confidence": 0.55, "fast": true}
    - {"action": "stop_capture"}
    """
    await manager.connect(websocket)
    session: Optional[WebcamSession] = None
    try:
        while True:
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
                action = msg.get("action")

                if action == "start_capture":
                    if session:
                        session.stop_capture()
                    cfg = BackendPredictorConfig(
                        model=msg.get("model", "hybrid"),
                        mode="spotting",
                        confidence=msg.get("confidence", 0.55),
                        ensemble=False,  # Disable ensemble for speed on Raspberry Pi
                        show_skeleton=msg.get("show_skeleton", False),  # TEMP: default True for testing
                    )
                    session = WebcamSession(cfg)
                    session.start_capture(websocket)
                    await websocket.send_text(json.dumps({"status": "capture_started"}))

                elif action == "stop_capture":
                    if session:
                        session.stop_capture()
                        session = None
                        await websocket.send_text(json.dumps({"status": "capture_stopped"}))
                    else:
                        await websocket.send_text(json.dumps({"error": "No active session"}))

                else:
                    await websocket.send_text(json.dumps({"error": "Unknown action"}))

            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({"error": "Invalid JSON"}))

    except WebSocketDisconnect:
        if session:
            session.stop_capture()
        manager.disconnect(websocket)


