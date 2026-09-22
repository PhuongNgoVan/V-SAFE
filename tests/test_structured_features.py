"""
tests/test_structured_features.py
Kiểm thử cho src/features/structured_features.py

Chạy: pytest tests/test_structured_features.py -v
"""

import numpy as np
import pytest

from src.features.structured_features import (
    FEATURE_ORDER,
    extract_structured_features,
    features_to_vector,
    matched_keywords,
)


# ---------------------------------------------------------------------------
# Bất biến cơ bản (nghiệm thu A1)
# ---------------------------------------------------------------------------

def test_feature_order_length():
    """FEATURE_ORDER phải có đúng 11 phần tử."""
    assert len(FEATURE_ORDER) == 11


def test_feature_order_names():
    """Tên 11 cột phải khớp spec."""
    expected = [
        "n_urgency_kw",
        "n_authority_kw",
        "n_financial_action_kw",
        "n_reward_kw",
        "has_phone_number",
        "has_bank_account_like_number",
        "has_url",
        "has_id_number_request",
        "message_length",
        "uppercase_ratio",
        "exclamation_count",
    ]
    assert FEATURE_ORDER == expected


def test_vector_shape_empty_string():
    """Chuỗi rỗng: vector shape=(11,), không chia cho 0."""
    d = extract_structured_features("")
    vec = features_to_vector(d)
    assert vec.shape == (11,)
    assert vec.dtype == np.float32
    # uppercase_ratio không chia cho 0
    assert d["uppercase_ratio"] == 0.0


def test_vector_dtype():
    """features_to_vector phải trả float32."""
    d = extract_structured_features("test")
    vec = features_to_vector(d)
    assert vec.dtype == np.float32


# ---------------------------------------------------------------------------
# Trường hợp: tin giả mạo công an (authority + financial)
# ---------------------------------------------------------------------------

def test_fake_police_message():
    """Tin giả mạo công an phải khớp authority và financial_action."""
    text = (
        "Đây là thông báo khẩn từ cơ quan công an. "
        "Yêu cầu anh chuyển khoản 50 triệu vào tài khoản ngân hàng "
        "để phục vụ điều tra. Gấp!"
    )
    d = extract_structured_features(text)
    assert d["n_authority_kw"] >= 1, "Phải phát hiện từ khoá authority"
    assert d["n_financial_action_kw"] >= 1, "Phải phát hiện từ khoá financial_action"
    assert d["n_urgency_kw"] >= 1, "Phải phát hiện từ khoá urgency"
    assert d["exclamation_count"] >= 1


# ---------------------------------------------------------------------------
# Trường hợp: tin trúng thưởng
# ---------------------------------------------------------------------------

def test_lottery_scam_message():
    """Tin trúng thưởng phải khớp nhóm reward."""
    text = (
        "Chúc mừng! Bạn đã trúng thưởng iPhone 15 Pro Max. "
        "Liên hệ ngay để nhận quà tặng miễn phí."
    )
    d = extract_structured_features(text)
    assert d["n_reward_kw"] >= 1, "Phải phát hiện từ khoá reward"
    kws = matched_keywords(text)
    assert len(kws) > 0


# ---------------------------------------------------------------------------
# Trường hợp: tin bình thường (tất cả đặc trưng từ khoá = 0)
# ---------------------------------------------------------------------------

def test_normal_message_all_zero():
    """Tin nhắn bình thường: n_*_kw đều = 0, không có URL / số nhạy cảm."""
    text = "Hôm nay trời đẹp quá, mình đi cà phê nhé?"
    d = extract_structured_features(text)
    assert d["n_urgency_kw"] == 0.0
    assert d["n_authority_kw"] == 0.0
    assert d["n_financial_action_kw"] == 0.0
    assert d["n_reward_kw"] == 0.0
    assert d["has_phone_number"] == 0.0
    assert d["has_url"] == 0.0
    kws = matched_keywords(text)
    assert kws == []


# ---------------------------------------------------------------------------
# Trường hợp: số CCCD 12 chữ số
# ---------------------------------------------------------------------------

def test_cccd_12_digits():
    """Số CCCD 12 chữ số phải được phát hiện bởi has_id_number_request."""
    text = "Cung cấp số CCCD của bạn: 034512345678 để xác minh."
    d = extract_structured_features(text)
    assert d["has_id_number_request"] == 1.0, "Phải phát hiện số CCCD 12 chữ số"


# ---------------------------------------------------------------------------
# Trường hợp: số điện thoại +84 912 345 678
# ---------------------------------------------------------------------------

def test_phone_number_international_format():
    """Số điện thoại định dạng +84 912 345 678 phải được phát hiện."""
    text = "Liên hệ số +84 912 345 678 để được tư vấn."
    d = extract_structured_features(text)
    assert d["has_phone_number"] == 1.0, "Phải phát hiện số điện thoại +84"


def test_phone_number_local_format():
    """Số điện thoại định dạng nội địa 0912345678."""
    text = "Gọi về 0912345678 ngay nhé."
    d = extract_structured_features(text)
    assert d["has_phone_number"] == 1.0


# ---------------------------------------------------------------------------
# Trường hợp: có URL
# ---------------------------------------------------------------------------

def test_has_url():
    """Phát hiện URL trong văn bản."""
    text = "Click vào http://bit.ly/abc123 để nhận thưởng."
    d = extract_structured_features(text)
    assert d["has_url"] == 1.0


# ---------------------------------------------------------------------------
# Trường hợp: uppercase_ratio
# ---------------------------------------------------------------------------

def test_uppercase_ratio():
    """uppercase_ratio tính đúng tỷ lệ chữ hoa / tổng alpha."""
    text = "ABC abc"  # 3 hoa + 3 thường = 6 alpha; ratio = 0.5
    d = extract_structured_features(text)
    assert abs(d["uppercase_ratio"] - 0.5) < 1e-6


# ---------------------------------------------------------------------------
# Trường hợp: message_length
# ---------------------------------------------------------------------------

def test_message_length():
    """message_length là số ký tự thô của văn bản đầu vào."""
    text = "Xin chào!"
    d = extract_structured_features(text)
    assert d["message_length"] == float(len(text))


# ---------------------------------------------------------------------------
# Thứ tự vector phải đúng theo FEATURE_ORDER
# ---------------------------------------------------------------------------

def test_vector_order_matches_feature_order():
    """Các phần tử trong vector phải khớp thứ tự FEATURE_ORDER."""
    text = "Công an yêu cầu chuyển khoản gấp! 0912345678"
    d = extract_structured_features(text)
    vec = features_to_vector(d)
    for i, col in enumerate(FEATURE_ORDER):
        assert vec[i] == pytest.approx(d[col], abs=1e-6), (
            f"Phần tử {i} ({col}) không khớp"
        )
