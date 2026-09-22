"""
src/common/patterns.py
Regex dùng chung cho phát hiện PII và số nhạy cảm.

Sở hữu: A (xem 00_SHARED_CONTRACT.md §5)
B được import trực tiếp từ module này, KHÔNG định nghĩa lại.

LƯU Ý chồng lấn đã biết (xem docs/agent_notes/A1.md):
- BANK_ACCOUNT_PATTERN (\\d{8,16}) cũng khớp số CCCD 12 chữ số và
  nhiều số điện thoại viết liền.  Hành vi này giữ nguyên theo kế hoạch.
"""

import re

# ---------------------------------------------------------------------------
# Số điện thoại Việt Nam
#   - Đầu số: 03x / 05x / 07x / 08x / 09x (di động)
#   - Cũng bắt định dạng quốc tế +84 và 0084
#   - Cho phép dấu cách / gạch ngang giữa các nhóm chữ số
# ---------------------------------------------------------------------------
PHONE_PATTERN = re.compile(
    r"""
    (?:
        (?:\+84|0084)           # mã quốc gia
        [\s\-\.]?
        [3-9]\d{8}              # 9 chữ số còn lại
    |
        0[3-9]\d{8}             # bắt đầu bằng 0, tổng 10 chữ số
    |
        0[3-9](?:[\s\-\.]?\d){8}  # có dấu phân cách bên trong
    |
        \+84[\s\-\.]?[3-9](?:[\s\-\.]?\d){8}  # +84 có phân cách
    )
    """,
    re.VERBOSE,
)

# ---------------------------------------------------------------------------
# Số tài khoản ngân hàng dạng ngắn
#   8–16 chữ số liên tiếp (không phải ngày tháng, không nằm sau dấu #)
#   Chú ý: pattern này cũng khớp CCCD 12 số và SĐT viết liền.
# ---------------------------------------------------------------------------
BANK_ACCOUNT_PATTERN = re.compile(r"(?<!\d)\d{8,16}(?!\d)")

# ---------------------------------------------------------------------------
# URL / đường dẫn web
#   Bắt http/https/ftp và cả dạng rút gọn (bit.ly, tinyurl, ...)
# ---------------------------------------------------------------------------
URL_PATTERN = re.compile(
    r"""
    (?:
        https?://               # http:// hoặc https://
        |
        ftp://
        |
        www\.                   # www. không có scheme
    )
    [\w\-\./?=&%#+:@!,~\[\]]+  # phần còn lại của URL
    |
    # Rút gọn phổ biến (không có scheme)
    (?:bit\.ly|tinyurl\.com|goo\.gl|ow\.ly|t\.co|rb\.gy|shorturl\.at)
    /[\w\-\./?=&%#+:@!,~\[\]]*
    """,
    re.VERBOSE | re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Số CMND (9 chữ số) hoặc CCCD (12 chữ số)
#   Tên trường `has_id_number_request` giữ theo kế hoạch; thực chất
#   chỉ phát hiện *sự hiện diện* của chuỗi số dạng ID, KHÔNG phát
#   hiện *yêu cầu* cung cấp — ghi chú này có trong docstring feature.
# ---------------------------------------------------------------------------
ID_NUMBER_PATTERN = re.compile(r"(?<!\d)(?:\d{9}|\d{12})(?!\d)")
