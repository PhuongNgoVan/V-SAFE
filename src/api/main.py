"""
src/api/main.py
===============
Điểm khởi động dịch vụ FastAPI V-SAFE cho API phát hiện lừa đảo tin nhắn và cuộc gọi thoại.

Spec:     B4_robustness_and_voice_api.md §2 (B4b)
Hợp đồng: 00_SHARED_CONTRACT.md §5 (B sở hữu thư mục src/api/)

Kiến trúc:
----------
- Lifespan khởi động nóng: nạp trước mô hình ASR (PhoWhisper) và process_and_predict
  để tránh độ trễ cao (cold-start) ở request đầu tiên.
- Xử lý trực tiếp văn bản từ ASR truyền sang text pipeline.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
import logging
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.voice_route import router as voice_router
import src.audio.asr as asr_module

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Khởi động nóng mô hình ASR và Text Classifier khi start server."""
    logger.info("Khởi động V-SAFE API Server...")

    # 1. Khởi động nóng mô hình ASR PhoWhisper
    try:
        asr_module._STATE.load()
        logger.info("✅ ASR Model đã được nạp sẵn sàng (warm-up).")
    except Exception as exc:
        logger.warning("Chưa thể nạp trước ASR model tại lifespan: %s (sẽ nạp lazy khi gọi).", exc)

    # 2. Khởi động nóng pipeline phân loại văn bản (A3)
    try:
        from src.pipeline.predict import process_and_predict
        # Chạy thử 1 câu ngắn để warm-up cache PyTorch và Tokenizer
        _ = process_and_predict("khởi động hệ thống")
        logger.info("✅ Pipeline Text Classifier (A3) đã được warm-up.")
    except Exception as exc:
        logger.warning("Pipeline text chưa sẵn sàng tại lifespan: %s (sẽ dùng fallback khi cần).", exc)

    yield

    logger.info("Đang tắt V-SAFE API Server...")


def create_app() -> FastAPI:
    """Tạo ứng dụng FastAPI V-SAFE."""
    app = FastAPI(
        title="V-SAFE Anti-Fraud Detection API",
        description="Hệ thống phát hiện cuộc gọi và tin nhắn lừa đảo bằng PhoWhisper và PhoBERT.",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Đăng ký routes
    app.include_router(voice_router)

    @app.get("/health", tags=["Health"])
    async def health_check():
        return {
            "status": "healthy",
            "service": "v-safe-api",
            "version": "1.0.0",
        }

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
