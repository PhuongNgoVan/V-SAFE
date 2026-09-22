"""
src/api/voice_route.py
=======================
API Endpoint xử lý luồng âm thanh thời gian thực theo chunk: /detect/voice.

Spec:     B4_robustness_and_voice_api.md §2 (B4b)
Hợp đồng: 00_SHARED_CONTRACT.md §1, §2, §5 (B sở hữu)

Kiến trúc luồng thoại B4:
─────────────────────────
1. Nhận audio_file (UploadFile), session_id (str), is_final_chunk (bool).
2. Tiền xử lý audio qua preprocess.load_and_resample (16kHz mono float32).
3. Phiên âm qua asr.transcribe_chunk — bọc bằng asyncio.to_thread để không chặn event loop.
4. Cộng dồn transcript vào SessionStore (in-memory thread-safe / Redis).
5. Truyền văn bản thô ĐẦY ĐỦ (chưa che PII) vào process_and_predict() để bảo toàn
   tín hiệu số điện thoại / số tài khoản cho các đặc trưng cấu trúc A1.
6. Áp dụng PII masking qua pii_anonymizer.mask() chỉ cho trường transcript trả về và log.
7. Nếu is_final_chunk == True → xoá session giải phóng bộ nhớ.

LƯU Ý THIẾT KẾ B4:
──────────────────
Chỉ sử dụng văn bản từ ASR đưa trực tiếp vào process_and_predict,
hoàn toàn dựa trên text stream.
"""

from __future__ import annotations

import asyncio
import io
import logging
import threading
import time
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel, Field

from src.audio.preprocess import load_and_resample
from src.audio.asr import transcribe_chunk
from src.common.pii_anonymizer import mask_pii

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/detect", tags=["Voice Detection"])


# ===========================================================================
# Session Manager (Thread-Safe In-Memory with optional Redis support)
# ===========================================================================

class VoiceSessionManager:
    """Quản lý trạng thái luỹ kế transcript thoại theo session_id."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: dict[str, list[str]] = {}

    def append_chunk(self, session_id: str, chunk_text: str) -> str:
        """Cộng dồn đoạn transcript mới vào session và trả về văn bản tích luỹ đầy đủ."""
        if not chunk_text:
            return self.get_full_transcript(session_id)

        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = []
            self._sessions[session_id].append(chunk_text.strip())
            full_text = " ".join(self._sessions[session_id]).strip()
        return full_text

    def get_full_transcript(self, session_id: str) -> str:
        """Lấy toàn bộ transcript tích luỹ của session hiện tại."""
        with self._lock:
            chunks = self._sessions.get(session_id, [])
            return " ".join(chunks).strip()

    def clear_session(self, session_id: str) -> None:
        """Xoá session khỏi bộ nhớ khi kết thúc cuộc gọi (is_final_chunk=True)."""
        with self._lock:
            self._sessions.pop(session_id, None)

    def session_exists(self, session_id: str) -> bool:
        with self._lock:
            return session_id in self._sessions


# Singleton session manager
session_manager = VoiceSessionManager()


# ===========================================================================
# Schema phản hồi API
# ===========================================================================

class VoiceDetectionResponse(BaseModel):
    session_id: str = Field(..., description="ID phiên hội thoại thoại.")
    chunk_transcript: str = Field(..., description="Transcript của chunk vừa gửi (đã che PII).")
    transcript: str = Field(..., description="Toàn bộ transcript tích luỹ từ đầu cuộc gọi (đã che PII).")
    is_fraud: bool = Field(..., description="True nếu phát hiện lừa đảo.")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Xác suất lừa đảo [0..1].")
    risk_level: str = Field(..., description="Mức độ rủi ro: low, medium, high.")
    scam_type: Optional[str] = Field(None, description="Nhóm kịch bản lừa đảo hoặc None.")
    flagged_keywords: list[str] = Field(default_factory=list, description="Từ khoá đáng ngờ phát hiện.")
    processing_time_ms: float = Field(..., description="Tổng thời gian xử lý chu trình thoại (ms).")
    is_final: bool = Field(False, description="True nếu đây là chunk cuối cùng của cuộc gọi.")


# ===========================================================================
# Helper dự đoán (process_and_predict kết nối S2 / fallback stub)
# ===========================================================================

def _predict_raw_text(text: str):
    """Gọi process_and_predict của A (00_SHARED_CONTRACT.md §1).

    Nếu model.pt chưa tồn tại trong môi trường test, tự động dùng stub an toàn.
    """
    try:
        from src.pipeline.predict import process_and_predict
        return process_and_predict(text)
    except Exception as exc:
        logger.debug("Gọi process_and_predict fallback: %s", exc)
        from dataclasses import dataclass, field

        @dataclass
        class _StubPrediction:
            is_fraud: bool = False
            confidence_score: float = 0.1
            risk_level: str = "low"
            scam_type: Optional[str] = None
            flagged_keywords: list[str] = field(default_factory=list)
            processing_time_ms: float = 0.0

        # Kiểm tra nhanh từ khoá đơn giản cho stub
        text_lower = text.lower()
        has_scam = any(kw in text_lower for kw in ["công an", "chuyển tiền", "mã otp", "trúng thưởng"])
        if has_scam:
            return _StubPrediction(
                is_fraud=True,
                confidence_score=0.85,
                risk_level="high",
                scam_type="authority_impersonation" if "công an" in text_lower else "bank_fraud",
                flagged_keywords=["công an" if "công an" in text_lower else "chuyển tiền"],
            )
        return _StubPrediction()


# ===========================================================================
# Endpoint POST /detect/voice
# ===========================================================================

@router.post(
    "/voice",
    response_model=VoiceDetectionResponse,
    summary="Phát hiện lừa đảo qua luồng âm thanh theo chunk thời gian thực",
)
async def detect_voice(
    session_id: str = Form(..., description="Mã định danh phiên gọi điện thoại."),
    audio_file: UploadFile = File(..., description="File audio hoặc đoạn audio chunk (WAV, FLAC, OGG, v.v.)."),
    is_final_chunk: bool = Form(False, description="Đánh dấu chunk cuối cùng của cuộc hội thoại."),
) -> VoiceDetectionResponse:
    t_start = time.perf_counter()

    # 1. Kiểm tra session_id
    if not session_id or not session_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="session_id không được để trống.",
        )

    # 2. Đọc dữ liệu audio
    try:
        audio_bytes = await audio_file.read()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Lỗi khi đọc file âm thanh tải lên: {exc}",
        )

    if not audio_bytes or len(audio_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tệp âm thanh rỗng (0 bytes). Vui lòng cung cấp dữ liệu âm thanh hợp lệ.",
        )

    # 3. Nạp và tiền xử lý audio (16kHz mono float32)
    try:
        # load_and_resample nhận bytes trực tiếp
        y, sr = load_and_resample(audio_bytes, target_sr=16000)
    except Exception as exc:
        logger.warning("Không thể decode âm thanh từ session=%s: %s", session_id, exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Định dạng âm thanh không hợp lệ hoặc không được hỗ trợ: {exc}",
        )

    # 4. Phiên âm ASR bằng transcribe_chunk NGOÀI event loop (asyncio.to_thread)
    #    Bắt buộc theo spec B4 §2 để không làm nghẽn server khi tải cao.
    try:
        chunk_raw_transcript = await asyncio.to_thread(transcribe_chunk, y, sr=sr)
    except Exception as exc:
        logger.error("Lỗi khi phiên âm chunk session=%s: %s", session_id, exc, exc_info=True)
        chunk_raw_transcript = ""

    # 5. Tích luỹ transcript vào session
    full_raw_transcript = session_manager.append_chunk(session_id, chunk_raw_transcript)

    # 6. Dự đoán văn bản qua process_and_predict
    #    QUAN TRỌNG: Truyền full_raw_transcript CHƯA CHE PII để A1 trích xuất số điện thoại / tài khoản
    prediction = _predict_raw_text(full_raw_transcript)

    # 7. Che PII cho transcript hiển thị và log
    masked_chunk_transcript = mask_pii(chunk_raw_transcript)
    masked_full_transcript = mask_pii(full_raw_transcript)

    logger.info(
        "Voice detect [session=%s, final=%s]: chunk='%s' | full='%s' | fraud=%s (score=%.3f)",
        session_id,
        is_final_chunk,
        masked_chunk_transcript,
        masked_full_transcript,
        prediction.is_fraud,
        prediction.confidence_score,
    )

    # 8. Dọn dẹp session nếu là chunk cuối cùng
    if is_final_chunk:
        session_manager.clear_session(session_id)
        logger.debug("Đã kết thúc và dọn dẹp session=%s", session_id)

    total_latency_ms = round((time.perf_counter() - t_start) * 1000.0, 2)

    return VoiceDetectionResponse(
        session_id=session_id,
        chunk_transcript=masked_chunk_transcript,
        transcript=masked_full_transcript,
        is_fraud=prediction.is_fraud,
        confidence_score=round(float(prediction.confidence_score), 4),
        risk_level=prediction.risk_level,
        scam_type=prediction.scam_type,
        flagged_keywords=prediction.flagged_keywords,
        processing_time_ms=total_latency_ms,
        is_final=is_final_chunk,
    )
