"""
src/audio/text_norm.py
=======================
Chuẩn hóa văn bản để tính WER theo hợp đồng 00_SHARED_CONTRACT.md §4.

Spec:   B2_audio_preprocess_wer.md §3
Sở hữu: B  (A không dùng — xem 00_SHARED_CONTRACT.md §5)

Quy tắc chuẩn hóa (chốt theo hợp đồng §4, KHÔNG thay đổi)
──────────────────────────────────────────────────────────
1. Chuyển về chữ **thường** (lowercase).
2. Loại bỏ toàn bộ **dấu câu** (punctuation theo Unicode category P*).
3. Gộp **khoảng trắng** thừa (tabs, newlines, multiple spaces → single space).
4. **BẮT BUỘC GIỮ NGUYÊN** dấu thanh tiếng Việt
   (sắc, huyền, hỏi, ngã, nặng — nằm trong tổ hợp ký tự Unicode).
   Không được dùng ``unidecode`` hoặc bất kỳ hàm strip accent nào.

Hàm công khai
──────────────
- ``normalize_for_wer(text)`` → str
    Chuẩn hóa một chuỗi văn bản, dùng cho cả reference và hypothesis.

Ví dụ
─────
>>> normalize_for_wer("Công an, yêu cầu chuyển khoản!")
'công an yêu cầu chuyển khoản'

>>> normalize_for_wer("  XIN CHÀO   BẠN  ")
'xin chào bạn'
"""

from __future__ import annotations

import re
import unicodedata

# ---------------------------------------------------------------------------
# Pattern dấu câu Unicode — lấy tất cả ký tự có category bắt đầu bằng "P"
# (Pc, Pd, Pe, Pf, Pi, Po, Ps) cộng thêm một số ký tự S (Symbol) thường gặp
# trong văn bản tiếng Việt như !?…
# ---------------------------------------------------------------------------
# Regex này khớp bất kỳ ký tự Unicode nào có category P* hoặc S*
# NHƯNG không khớp chữ cái (L*), chữ số (N*), khoảng trắng (Z*),
# hay dấu hợp âm (M* — bao gồm dấu thanh tiếng Việt).
_PUNCT_RE = re.compile(
    r"[\u0021-\u002F"   # !"#$%&'()*+,-./
    r"\u003A-\u0040"    # :;<=>?@
    r"\u005B-\u0060"    # [\]^_`
    r"\u007B-\u007E"    # {|}~
    r"\u00A1-\u00BF"    # ¡¿ và các ký hiệu Latin-1 supplement
    r"\u2000-\u206F"    # General punctuation (…–—""''‹›«»)
    r"\u2E00-\u2E7F"    # Supplemental punctuation
    r"\uFE50-\uFE6F"    # Small form variants
    r"\uFF01-\uFF0F"    # Fullwidth forms
    r"\uFF1A-\uFF20"
    r"\uFF3B-\uFF40"
    r"\uFF5B-\uFF65"
    r"]+",
    re.UNICODE,
)

# Whitespace gộp (tab, newline, multiple spaces → single space)
_WHITESPACE_RE = re.compile(r"\s+", re.UNICODE)


def normalize_for_wer(text: str) -> str:
    """Chuẩn hóa văn bản để tính WER — giữ nguyên dấu thanh tiếng Việt.

    Quy tắc (chốt trong 00_SHARED_CONTRACT.md §4):
    - Chữ thường.
    - Bỏ dấu câu (Unicode P* và ký hiệu phổ biến).
    - Gộp khoảng trắng thừa.
    - **GIỮ NGUYÊN dấu thanh** (sắc/huyền/hỏi/ngã/nặng).

    Tham số
    -------
    text : str
        Chuỗi văn bản gốc (reference hoặc hypothesis).

    Trả về
    ------
    str
        Chuỗi đã chuẩn hóa, strip() hai đầu.

    Ví dụ
    -----
    >>> normalize_for_wer("Công an, yêu cầu chuyển khoản!")
    'công an yêu cầu chuyển khoản'

    >>> normalize_for_wer("Xin chào! Bạn có khỏe không?")
    'xin chào bạn có khỏe không'

    >>> normalize_for_wer("  HELLO   world  ")
    'hello world'

    >>> normalize_for_wer("")
    ''

    Ghi chú
    -------
    **Không** dùng ``unidecode``, ``unicodedata.normalize('NFKD', ...)+encode``
    hay bất kỳ hàm nào strip diacritics — những hàm đó xóa dấu thanh tiếng Việt.

    Hàm sử dụng regex dựa trên codepoint Unicode để xóa dấu câu,
    hoàn toàn an toàn với NFC-encoded Vietnamese text.
    """
    if not text:
        return ""

    # Bước 1: Đảm bảo NFC (tổ hợp ký tự Vietnamese thường ở dạng NFC)
    text = unicodedata.normalize("NFC", text)

    # Bước 2: Chữ thường
    text = text.lower()

    # Bước 3: Xóa dấu câu (giữ nguyên chữ cái + dấu thanh + khoảng trắng + chữ số)
    text = _PUNCT_RE.sub(" ", text)

    # Bước 4: Gộp khoảng trắng thừa
    text = _WHITESPACE_RE.sub(" ", text).strip()

    return text


def normalize_pair(
    reference:  str,
    hypothesis: str,
) -> tuple[str, str]:
    """Chuẩn hóa cặp (reference, hypothesis) cùng lúc.

    Tiện ích dùng trong ``eval_wer.py``.

    Trả về
    ------
    tuple[str, str]
        ``(normalized_ref, normalized_hyp)``
    """
    return normalize_for_wer(reference), normalize_for_wer(hypothesis)
