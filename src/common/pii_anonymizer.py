"""
src/common/pii_anonymizer.py
=============================
Module che giấu thông tin định danh cá nhân (PII) cho log và API response.

Sở hữu: B dùng chung với A (00_SHARED_CONTRACT.md §5, B4_robustness_and_voice_api.md §3)
Các pattern regex được import trực tiếp từ src/common/patterns.py.

Lưu ý:
------
Không che PII trước khi đưa vào pipeline dự đoán (process_and_predict),
vì các đặc trưng số điện thoại, số tài khoản là tín hiệu quan trọng của A1.
Chỉ gọi mask() cho văn bản hiển thị cho người dùng và ghi log.
"""

from __future__ import annotations

import re
from src.common.patterns import (
    PHONE_PATTERN,
    BANK_ACCOUNT_PATTERN,
    ID_NUMBER_PATTERN,
    URL_PATTERN,
)


def mask_pii(text: str) -> str:
    """Che giấu thông tin cá nhân (SĐT, số tài khoản, CMND/CCCD, URL) trong chuỗi.

    Tham số
    -------
    text : str
        Văn bản gốc.

    Trả về
    ------
    str
        Văn bản đã che PII.
    """
    if not text:
        return ""

    # Che URL trước để tránh pattern số làm hỏng URL
    masked = URL_PATTERN.sub("[URL]", text)
    # Che số điện thoại
    masked = PHONE_PATTERN.sub("[PHONE]", masked)
    # Che CMND / CCCD
    masked = ID_NUMBER_PATTERN.sub("[ID_NUMBER]", masked)
    # Che số tài khoản ngân hàng
    masked = BANK_ACCOUNT_PATTERN.sub("[BANK_ACCOUNT]", masked)

    return masked


class PIIAnonymizer:
    """Lớp tiện ích che PII theo thiết kế B4."""

    @staticmethod
    def mask(text: str) -> str:
        return mask_pii(text)
