import base64
import os
import tempfile
from importlib import import_module
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field
from typing import Dict, List
from uuid import uuid4

from app.inference.predictor import BackendPredictor, BackendPredictorConfig
from app.inference.sentence_builder import SentenceBuilder
from app.inference.video_processor import VideoProcessingError, process_uploaded_video
from app.config import ACTIVE_DEPLOYMENT_MODE, NUM_CLASSES


router = APIRouter()


class StartSessionRequest(BaseModel):
    model_config = {"populate_by_name": True}
    
    model: str = "hybrid"
    mode: str = "spotting"
    confidence: float = 0.55
    ensemble: bool = True
    fast: bool = False
    manual: bool = False
    wlasl: bool = False
    dataset: str | None = None
    show_skeleton: bool = Field(default=False, alias="showSkeleton")


class StartSessionResponse(BaseModel):
    session_id: str
    model: str
    mode: str
    confidence: float
    ensemble: bool
    deployment_mode: str
    num_classes: int
    dataset: str


class PredictRequest(BaseModel):
    session_id: str
    # Phase 1: gửi base64-encoded JPEG frame.
    # Sau có thể chuyển sang gửi keypoints như doc.txt §12.
    frame: str


class PredictResponse(BaseModel):
    prediction: str
    confidence: float
    sentence: List[str]
    keypoints: dict | None = None


class UploadTimelineItem(BaseModel):
    timestamp_sec: float
    frame_index: int
    prediction: str
    confidence: float
    emitted: bool


class UploadVideoMetadata(BaseModel):
    duration_sec: float
    source_fps: float
    target_fps: int
    source_frame_count: int
    processed_frame_count: int
    window_size: int
    stride: int
    processing_ms: float


class UploadVideoResponse(BaseModel):
    sentence: List[str]
    timeline: List[UploadTimelineItem]
    metadata: UploadVideoMetadata


class SpeechTranscribeRequest(BaseModel):
    audio_base64: str
    language: str = "auto"


class SpeechTranscribeResponse(BaseModel):
    text: str
    provider: str


class _SessionState:
    def __init__(self, cfg: BackendPredictorConfig) -> None:
        self.cfg = cfg
        self.predictor = BackendPredictor(cfg)
        self.sentence_builder = SentenceBuilder(
            stability_count=2,
            cooldown_sec=1.0,
            confidence_threshold=cfg.confidence,
        )


_SESSIONS: Dict[str, _SessionState] = {}


_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
_MAX_UPLOAD_BYTES = 10 * 1024 * 1024 * 1024


def _normalize_dataset(dataset: str | None, wlasl: bool) -> str:
    if dataset:
        raw = dataset.strip().lower()
        if raw in {"wlasl", "wlasl2000", "asl", "america", "us"}:
            return "wlasl2000"
        if raw in {"bsl", "bobsl", "uk", "united_kingdom", "united-kingdom"}:
            return "bsl"
        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported dataset value. Use one of: "
                "bsl, uk, bobsl, wlasl, wlasl2000, asl, america, us"
            ),
        )
    return "wlasl2000" if wlasl else "bsl"


def _save_uploaded_video(file: UploadFile) -> str:
    suffix = Path(file.filename or "upload.mp4").suffix.lower()
    if suffix not in _VIDEO_EXTS:
        raise HTTPException(status_code=400, detail="Unsupported video format")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    total = 0
    try:
        while True:
            chunk = file.file.read(8 * 1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > _MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=413, detail="Video exceeds 10GB limit")
            tmp.write(chunk)
        tmp.flush()
        return tmp.name
    except Exception:
        if os.path.exists(tmp.name):
            os.remove(tmp.name)
        raise
    finally:
        tmp.close()


def _decode_base64_audio(data: str) -> bytes:
    payload = data
    if "," in payload:
        payload = payload.split(",", 1)[1]

    try:
        return base64.b64decode(payload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid base64 audio payload") from exc


def _get_speech_recognition_module():
    try:
        return import_module("speech_recognition")
    except Exception:
        return None


@router.get("/health", tags=["system"])
async def health() -> dict:
    return {
        "status": "ok",
        "sessions": len(_SESSIONS),
        "deployment_mode": ACTIVE_DEPLOYMENT_MODE,
        "num_classes": NUM_CLASSES,
    }


@router.get("/deployment-info", tags=["system"])
async def deployment_info() -> dict:
    return {
        "deployment_mode": ACTIVE_DEPLOYMENT_MODE,
        "num_classes": NUM_CLASSES,
        "datasets": [
            {"id": "bsl", "available": ACTIVE_DEPLOYMENT_MODE == "bsl"},
            {"id": "wlasl2000", "available": ACTIVE_DEPLOYMENT_MODE == "wlasl2000"},
        ],
    }


@router.post("/start-session", response_model=StartSessionResponse, tags=["inference"])
@router.post("/start-session", response_model=StartSessionResponse, tags=["inference"])
async def start_session(body: StartSessionRequest) -> StartSessionResponse:
    """
    Khởi tạo một phiên nhận diện mới, load model từ BOBSL_Lightweight.
    """
    cfg = BackendPredictorConfig(
        model=body.model,
        mode=body.mode,
        confidence=body.confidence,
        ensemble=body.ensemble,
        use_antibias=True,
        show_skeleton=body.show_skeleton,
    )

    requested_dataset = _normalize_dataset(body.dataset, body.wlasl)

    if requested_dataset != ACTIVE_DEPLOYMENT_MODE:
        expected = "SL_DEPLOYMENT_MODE=wlasl2000" if requested_dataset == "wlasl2000" else "SL_DEPLOYMENT_MODE=bsl"
        raise HTTPException(
            status_code=409,
            detail=(
                f"Session requested dataset '{requested_dataset}' but backend is running in "
                f"'{ACTIVE_DEPLOYMENT_MODE}'. Restart backend with {expected}."
            ),
        )

    session_id = uuid4().hex
    _SESSIONS[session_id] = _SessionState(cfg)

    return StartSessionResponse(
        session_id=session_id,
        model=body.model,
        mode=body.mode,
        confidence=body.confidence,
        ensemble=body.ensemble,
        deployment_mode=ACTIVE_DEPLOYMENT_MODE,
        num_classes=NUM_CLASSES,
        dataset=requested_dataset,
    )


@router.post("/predict", response_model=PredictResponse, tags=["inference"])
async def predict(body: PredictRequest) -> PredictResponse:
    """
    Nhận một frame base64, chạy MediaPipe + Predictor (demo_camera)
    và cập nhật câu thông qua SentenceBuilder.
    """
    session = _SESSIONS.get(body.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    keypoints, label, conf = session.predictor.predict_from_base64(body.frame)
    if label and conf >= session.cfg.confidence:
        session.sentence_builder.feed(label, conf)

    return PredictResponse(
        prediction=label or "",
        confidence=float(conf),
        sentence=session.sentence_builder.sentence,
        keypoints=keypoints if session.cfg.show_skeleton else None,
    )


@router.post("/upload-video", response_model=UploadVideoResponse, tags=["inference"])
async def upload_video(
    session_id: str = Form(...),
    file: UploadFile = File(...),
    target_fps: int = Form(25),
    stride: int = Form(5),
) -> UploadVideoResponse:
    session = _SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    if target_fps < 1 or target_fps > 60:
        raise HTTPException(status_code=400, detail="target_fps must be in range [1, 60]")
    if stride < 1 or stride > 50:
        raise HTTPException(status_code=400, detail="stride must be in range [1, 50]")

    video_path = _save_uploaded_video(file)

    session.sentence_builder.clear()
    old_cooldown = session.sentence_builder.cooldown_sec
    session.sentence_builder.cooldown_sec = 0.0

    try:
        result = process_uploaded_video(
            session=session,
            video_path=video_path,
            target_fps=target_fps,
            stride=stride,
        )
    except VideoProcessingError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        session.sentence_builder.cooldown_sec = old_cooldown
        try:
            file.file.close()
        except Exception:
            pass
        if os.path.exists(video_path):
            os.remove(video_path)

    return UploadVideoResponse(
        sentence=result["final_sentence"],
        timeline=result["timeline"],
        metadata=result["metadata"],
    )


@router.post("/speech/transcribe", response_model=SpeechTranscribeResponse, tags=["speech"])
async def speech_transcribe(body: SpeechTranscribeRequest) -> SpeechTranscribeResponse:
    sr_module = _get_speech_recognition_module()
    if sr_module is None:
        raise HTTPException(status_code=503, detail="SpeechRecognition package is not installed")

    audio_bytes = _decode_base64_audio(body.audio_base64)
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio payload")

    temp_path: str | None = None
    recognizer = sr_module.Recognizer()

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_bytes)
            temp_path = tmp.name

        with sr_module.AudioFile(temp_path) as source:
            audio_data = recognizer.record(source)

        language = None if body.language in {"", "auto"} else body.language
        if language:
            text = recognizer.recognize_google(audio_data, language=language)
        else:
            text = recognizer.recognize_google(audio_data)

        return SpeechTranscribeResponse(
            text=text.strip(),
            provider="speech_recognition_google",
        )
    except sr_module.UnknownValueError:
        return SpeechTranscribeResponse(text="", provider="speech_recognition_google")
    except sr_module.RequestError as exc:
        raise HTTPException(status_code=503, detail=f"Speech service unavailable: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Audio transcription failed: {exc}") from exc
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
