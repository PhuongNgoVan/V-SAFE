"""
src/audio/__init__.py
=====================
Export công khai của package src.audio.

B sở hữu toàn bộ package này (00_SHARED_CONTRACT.md §5).
"""

from src.audio.asr import (
    ASRModelNotReadyError,
    transcribe_chunk,
    transcribe_file,
)

__all__ = [
    "ASRModelNotReadyError",
    "transcribe_chunk",
    "transcribe_file",
]
