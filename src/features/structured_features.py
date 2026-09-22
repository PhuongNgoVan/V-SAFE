"""
src/features/structured_features.py
Trích vector đặc trưng cấu trúc (11 cột) từ văn bản đã tiền xử lý.

Sở hữu: A — xem 00_SHARED_CONTRACT.md §5
Spec:    A1_structured_features.md

Hàm công khai
─────────────
- extract_structured_features(clean_text) -> dict
- features_to_vector(d) -> np.ndarray  (float32, shape=(11,))
- matched_keywords(clean_text) -> list[str]

Hằng số công khai
─────────────────
- FEATURE_ORDER: list[str]  — 11 cột theo thứ tự cố định
- KEYWORDS: dict[str, list[str]]  — từ điển đọc từ YAML
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.common.patterns import (
    BANK_ACCOUNT_PATTERN,
    ID_NUMBER_PATTERN,
    PHONE_PATTERN,
    URL_PATTERN,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thứ tự cột CỐ ĐỊNH — thay đổi ở đây là breaking change với toàn pipeline
# ---------------------------------------------------------------------------
FEATURE_ORDER: list[str] = [
    "n_urgency_kw",
    "n_authority_kw",
    "n_financial_action_kw",
    "n_reward_kw",
    "has_phone_number",
    "has_bank_account_like_number",
    "has_url",
    "has_id_number_request",  # tên giữ nguyên theo kế hoạch; xem ghi chú dưới
    "message_length",
    "uppercase_ratio",
    "exclamation_count",
]
assert len(FEATURE_ORDER) == 11, "FEATURE_ORDER phải có đúng 11 phần tử"

# ---------------------------------------------------------------------------
# Đọc từ điển từ khoá từ YAML (lazy singleton — chỉ đọc một lần)
# ---------------------------------------------------------------------------
_KEYWORDS_CACHE: dict[str, list[str]] | None = None
_YAML_PATH = Path(__file__).resolve().parents[2] / "configs" / "scam_keywords.yaml"


def _load_keywords() -> dict[str, list[str]]:
    """Đọc configs/scam_keywords.yaml và trả về dict nhóm -> [từ khoá]."""
    global _KEYWORDS_CACHE
    if _KEYWORDS_CACHE is None:
        with open(_YAML_PATH, encoding="utf-8") as fh:
            raw: dict[str, Any] = yaml.safe_load(fh)
        _KEYWORDS_CACHE = {k: [str(w).lower() for w in v] for k, v in raw.items()}
        logger.info(
            "Đã nạp từ điển từ khoá: %s nhóm, %d từ",
            len(_KEYWORDS_CACHE),
            sum(len(v) for v in _KEYWORDS_CACHE.values()),
        )
    return _KEYWORDS_CACHE


# Alias công khai (đọc ngay khi import để phát hiện lỗi YAML sớm)
KEYWORDS: dict[str, list[str]] = {}  # sẽ được điền khi module được import lần đầu


def _ensure_keywords() -> dict[str, list[str]]:
    """Đảm bảo KEYWORDS đã được nạp; cập nhật biến module-level."""
    global KEYWORDS
    kw = _load_keywords()
    if not KEYWORDS:
        KEYWORDS.update(kw)
    return kw


# ---------------------------------------------------------------------------
# Hàm nội bộ: đếm số từ khoá khớp trong văn bản (không tính trùng lặp)
# ---------------------------------------------------------------------------
def _count_keywords(text_lower: str, keyword_list: list[str]) -> int:
    """Trả về số từ khoá trong keyword_list xuất hiện trong text_lower.

    Không đếm trùng: mỗi từ khoá chỉ tính một lần dù xuất hiện nhiều lần.
    """
    return sum(1 for kw in keyword_list if kw in text_lower)


# ---------------------------------------------------------------------------
# API công khai
# ---------------------------------------------------------------------------

def extract_structured_features(clean_text: str) -> dict[str, float]:
    """Trích 11 đặc trưng cấu trúc từ *clean_text* (đã tiền xử lý).

    Tham số
    -------
    clean_text : str
        Văn bản đã qua nlp_preprocess.run() — teencode đã giải, bypass-filter
        đã lọc.  Hàm này KHÔNG tự gọi tiền xử lý.

    Trả về
    ------
    dict với đúng 11 khoá tương ứng FEATURE_ORDER.

    Ghi chú về các đặc trưng
    ─────────────────────────
    ``has_id_number_request``
        Tên gây hiểu nhầm (giữ nguyên để khớp kế hoạch).  Thực chất CHỈ
        phát hiện sự hiện diện của chuỗi số 9 hoặc 12 chữ số (dạng CMND/CCCD),
        KHÔNG phát hiện yêu cầu cung cấp.

    ``uppercase_ratio``
        Tỷ lệ ký tự hoa trên tổng ký tự alpha.  Với transcript ASR sau khi
        hạ chữ thường bởi nlp_preprocess, đặc trưng này sẽ = 0 (xem A1.md).

    ``exclamation_count``
        Số dấu '!' — thường = 0 với transcript ASR.  Xem A1.md.
    """
    kw = _ensure_keywords()

    # --- Chuẩn bị ---
    text_lower = clean_text.lower()
    alpha_chars = [c for c in clean_text if c.isalpha()]
    n_alpha = len(alpha_chars)
    upper_chars = [c for c in clean_text if c.isupper()]

    # --- 4 nhóm từ khoá ---
    n_urgency_kw = _count_keywords(text_lower, kw.get("urgency", []))
    n_authority_kw = _count_keywords(text_lower, kw.get("authority", []))
    n_financial_action_kw = _count_keywords(text_lower, kw.get("financial_action", []))
    n_reward_kw = _count_keywords(text_lower, kw.get("reward", []))

    # --- Regex ---
    has_phone_number = int(bool(PHONE_PATTERN.search(clean_text)))
    has_bank_account_like_number = int(bool(BANK_ACCOUNT_PATTERN.search(clean_text)))
    has_url = int(bool(URL_PATTERN.search(clean_text)))
    # Xem ghi chú docstring: chỉ phát hiện *có* số dạng CMND/CCCD
    has_id_number_request = int(bool(ID_NUMBER_PATTERN.search(clean_text)))

    # --- Thống kê văn bản ---
    message_length = len(clean_text)
    uppercase_ratio = len(upper_chars) / n_alpha if n_alpha > 0 else 0.0
    exclamation_count = clean_text.count("!")

    return {
        "n_urgency_kw": float(n_urgency_kw),
        "n_authority_kw": float(n_authority_kw),
        "n_financial_action_kw": float(n_financial_action_kw),
        "n_reward_kw": float(n_reward_kw),
        "has_phone_number": float(has_phone_number),
        "has_bank_account_like_number": float(has_bank_account_like_number),
        "has_url": float(has_url),
        "has_id_number_request": float(has_id_number_request),
        "message_length": float(message_length),
        "uppercase_ratio": float(uppercase_ratio),
        "exclamation_count": float(exclamation_count),
    }


def features_to_vector(d: dict[str, float]) -> np.ndarray:
    """Chuyển dict đặc trưng sang numpy array float32 theo FEATURE_ORDER.

    Tham số
    -------
    d : dict
        Kết quả từ extract_structured_features().

    Trả về
    ------
    np.ndarray, dtype=float32, shape=(11,)
    """
    return np.array([d[col] for col in FEATURE_ORDER], dtype=np.float32)


def matched_keywords(clean_text: str) -> list[str]:
    """Trả danh sách từ khoá khớp trong clean_text (dùng cho flagged_keywords API).

    Trả về
    ------
    list[str] — các từ khoá xuất hiện, không trùng, giữ nguyên thứ tự nhóm.
    """
    kw = _ensure_keywords()
    text_lower = clean_text.lower()
    found: list[str] = []
    for group_keywords in kw.values():
        for keyword in group_keywords:
            if keyword in text_lower and keyword not in found:
                found.append(keyword)
    return found


# ---------------------------------------------------------------------------
# Tiện ích: đo tốc độ trên tập dữ liệu lớn (dùng trong notebook / script)
# ---------------------------------------------------------------------------

def benchmark(texts: list[str]) -> float:
    """Trích đặc trưng cho toàn bộ *texts*, in thời gian, trả về giây/mẫu.

    Cần < 5 giây cho 10 000 mẫu theo nghiệm thu A1.
    """
    t0 = time.perf_counter()
    for t in texts:
        extract_structured_features(t)
    elapsed = time.perf_counter() - t0
    per_sample = elapsed / max(len(texts), 1)
    logger.info(
        "benchmark: %d mẫu trong %.3f s (%.4f ms/mẫu)",
        len(texts),
        elapsed,
        per_sample * 1000,
    )
    print(f"[benchmark] {len(texts)} mẫu | tổng: {elapsed:.3f}s | {per_sample*1000:.4f} ms/mẫu")
    return elapsed
