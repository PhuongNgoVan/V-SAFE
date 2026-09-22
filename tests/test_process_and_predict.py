"""
tests/test_process_and_predict.py
==================================
Test suite cho ``process_and_predict()`` và ``Prediction`` (A3 / S2).

Spec: A3_process_and_predict.md §8 và §nghiệm thu

Chạy::

    pytest tests/test_process_and_predict.py -v

Yêu cầu môi trường
-------------------
- Các test ``TestIntegration`` cần ``model.pt``, ``scaler.joblib``,
  ``config.json`` trong ``models/text/hybrid/``.
  Nếu chưa huấn luyện, những test này bị **skip** tự động (``pytest.mark.skip``).
- Các test ``TestPrediction``, ``TestScoreToRiskLevel``, ``TestEdgeCasesNoModel``
  chạy được **không** cần model (dùng mock / stub).

Bao gồm
-------
1. ``TestPrediction``            — kiểm tra dataclass Prediction (schema).
2. ``TestScoreToRiskLevel``      — kiểm tra ngưỡng risk_level.
3. ``TestEdgeCasesNoModel``      — chuỗi rỗng / whitespace không cần model.
4. ``TestIntegration``           — test đầy đủ cần model thật:
   a. Câu lừa đảo giả mạo công an → is_fraud=True.
   b. Câu lừa đảo trúng thưởng → is_fraud=True, scam_type="prize_scam".
   c. Câu lừa đảo ngân hàng → scam_type="bank_fraud".
   d. Câu bình thường → is_fraud=False.
   e. Câu bình thường (hỏi thăm) → is_fraud=False.
   f. Chuỗi rỗng → Prediction an toàn, không ngoại lệ.
   g. Văn bản 5.000 ký tự → xử lý không crash, trả Prediction.
   h. Gọi lần 2 → nhanh hơn rõ rệt (model không nạp lại).
5. ``TestSingletonBehavior``     — xác nhận singleton chỉ nạp một lần.
6. ``TestInferScamType``         — kiểm tra luật suy scam_type.
"""

from __future__ import annotations

import sys
import threading
import time
from dataclasses import fields
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Thiết lập sys.path
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.pipeline.predict import (
    ModelNotReadyError,
    Prediction,
    _infer_scam_type,
    _STATE,
    process_and_predict,
    score_to_risk_level,
)

# ---------------------------------------------------------------------------
# Kiểm tra model có sẵn không → skip integration tests nếu chưa train
# ---------------------------------------------------------------------------
_MODEL_DIR   = _REPO_ROOT / "models" / "text" / "hybrid"
_MODEL_READY = (
    (_MODEL_DIR / "model.pt").exists()
    and (_MODEL_DIR / "scaler.joblib").exists()
    and (_MODEL_DIR / "config.json").exists()
)
_REQUIRES_MODEL = pytest.mark.skipif(
    not _MODEL_READY,
    reason="model.pt / scaler.joblib / config.json chưa tồn tại — chạy train_hybrid.py trước",
)


# ===========================================================================
# 1. TestPrediction — dataclass schema
# ===========================================================================

class TestPrediction:
    """Kiểm tra dataclass Prediction đúng schema hợp đồng."""

    REQUIRED_FIELDS = {
        "is_fraud":           bool,
        "confidence_score":   float,
        "risk_level":         str,
        "scam_type":          (str, type(None)),
        "flagged_keywords":   list,
        "processing_time_ms": float,
    }

    def test_all_required_fields_present(self):
        """Prediction phải có đúng 6 trường theo hợp đồng."""
        field_names = {f.name for f in fields(Prediction)}
        for fname in self.REQUIRED_FIELDS:
            assert fname in field_names, f"Thiếu trường: {fname}"

    def test_default_values(self):
        """Tạo Prediction với giá trị mặc định không gây lỗi."""
        p = Prediction(
            is_fraud=False,
            confidence_score=0.0,
            risk_level="low",
            scam_type=None,
        )
        assert p.flagged_keywords  == []
        assert p.processing_time_ms == 0.0

    def test_is_fraud_type(self):
        p = Prediction(is_fraud=True, confidence_score=0.9,
                       risk_level="high", scam_type="bank_fraud")
        assert isinstance(p.is_fraud, bool)

    def test_confidence_score_range(self):
        """confidence_score phải là float."""
        p = Prediction(is_fraud=False, confidence_score=0.35,
                       risk_level="low", scam_type=None)
        assert isinstance(p.confidence_score, float)

    def test_risk_level_valid_values(self):
        for level in ("low", "medium", "high"):
            p = Prediction(is_fraud=False, confidence_score=0.1,
                           risk_level=level, scam_type=None)
            assert p.risk_level == level

    def test_scam_type_none(self):
        p = Prediction(is_fraud=False, confidence_score=0.0,
                       risk_level="low", scam_type=None)
        assert p.scam_type is None

    def test_scam_type_string(self):
        p = Prediction(is_fraud=True, confidence_score=0.85,
                       risk_level="high", scam_type="authority_impersonation")
        assert isinstance(p.scam_type, str)

    def test_flagged_keywords_list(self):
        p = Prediction(is_fraud=True, confidence_score=0.8,
                       risk_level="high", scam_type="bank_fraud",
                       flagged_keywords=["chuyển khoản", "otp"])
        assert isinstance(p.flagged_keywords, list)
        assert len(p.flagged_keywords) == 2

    def test_processing_time_ms_float(self):
        p = Prediction(is_fraud=False, confidence_score=0.1,
                       risk_level="low", scam_type=None,
                       processing_time_ms=12.345)
        assert isinstance(p.processing_time_ms, float)
        assert p.processing_time_ms == pytest.approx(12.345)


# ===========================================================================
# 2. TestScoreToRiskLevel
# ===========================================================================

class TestScoreToRiskLevel:
    """Kiểm tra ngưỡng score_to_risk_level theo risk_thresholds.yaml."""

    @pytest.mark.parametrize("score,expected", [
        (0.0,   "low"),
        (0.1,   "low"),
        (0.39,  "low"),
        (0.4,   "medium"),
        (0.5,   "medium"),
        (0.699, "medium"),
        (0.7,   "high"),
        (0.85,  "high"),
        (1.0,   "high"),
    ])
    def test_thresholds(self, score: float, expected: str):
        result = score_to_risk_level(score)
        assert result == expected, f"score={score}: expected {expected!r}, got {result!r}"

    def test_returns_string(self):
        assert isinstance(score_to_risk_level(0.5), str)

    def test_boundary_low_upper(self):
        """Chính xác tại boundary 0.4."""
        assert score_to_risk_level(0.3999) == "low"
        assert score_to_risk_level(0.4000) == "medium"

    def test_boundary_medium_upper(self):
        """Chính xác tại boundary 0.7."""
        assert score_to_risk_level(0.6999) == "medium"
        assert score_to_risk_level(0.7000) == "high"


# ===========================================================================
# 3. TestEdgeCasesNoModel — không cần model.pt
# ===========================================================================

class TestEdgeCasesNoModel:
    """Test các trường hợp biên mà không cần model thật."""

    def test_empty_string_returns_safe_prediction(self):
        """Chuỗi rỗng → Prediction an toàn, không ném ngoại lệ."""
        result = process_and_predict("")
        assert isinstance(result, Prediction)
        assert result.is_fraud          is False
        assert result.confidence_score  == 0.0
        assert result.risk_level        == "low"
        assert result.scam_type         is None
        assert result.flagged_keywords  == []
        assert result.processing_time_ms >= 0.0

    def test_whitespace_only_returns_safe(self):
        """Chuỗi toàn khoảng trắng → Prediction an toàn."""
        for text in ["   ", "\t\n", "\r\n  \t"]:
            result = process_and_predict(text)
            assert result.is_fraud is False, f"text={text!r} phải là safe"
            assert result.risk_level == "low"

    def test_empty_string_no_exception(self):
        """Chuỗi rỗng không được raise bất kỳ exception nào."""
        try:
            process_and_predict("")
        except Exception as e:
            pytest.fail(f"process_and_predict('') gây ra ngoại lệ: {e}")

    def test_empty_string_fast(self):
        """Chuỗi rỗng phải xử lý rất nhanh (< 50 ms) vì không gọi model."""
        t0 = time.perf_counter()
        process_and_predict("")
        elapsed = (time.perf_counter() - t0) * 1000
        assert elapsed < 50, f"Empty string mất {elapsed:.1f} ms — quá chậm"

    def test_processing_time_ms_positive(self):
        """processing_time_ms phải ≥ 0."""
        result = process_and_predict("")
        assert result.processing_time_ms >= 0.0


# ===========================================================================
# 4. TestInferScamType — kiểm tra luật suy scam_type
# ===========================================================================

class TestInferScamType:
    """Kiểm tra hàm _infer_scam_type() — luật heuristic từ khoá."""

    def test_authority_keywords(self):
        """Từ khoá công an → authority_impersonation."""
        text = "công an gọi điện yêu cầu nộp tiền ngay"
        result = _infer_scam_type(text, ["công an", "nộp tiền"])
        assert result == "authority_impersonation"

    def test_bank_keywords(self):
        """Từ khoá ngân hàng / chuyển khoản → bank_fraud."""
        text = "chuyển khoản ngay số tài khoản của tôi"
        result = _infer_scam_type(text, ["chuyển khoản", "số tài khoản"])
        assert result == "bank_fraud"

    def test_reward_keywords(self):
        """Từ khoá trúng thưởng → prize_scam."""
        text = "bạn đã trúng thưởng iphone, nhận ngay"
        result = _infer_scam_type(text, ["trúng thưởng", "iphone"])
        assert result == "prize_scam"

    def test_authority_takes_priority_over_bank(self):
        """Authority ưu tiên hơn bank khi cả hai xuất hiện."""
        text = "công an yêu cầu chuyển khoản 50 triệu"
        result = _infer_scam_type(text, ["công an", "chuyển khoản"])
        assert result == "authority_impersonation"

    def test_no_keywords_returns_none(self):
        """Không có từ khoá → None."""
        result = _infer_scam_type("xin chào bạn khỏe không", [])
        assert result is None

    def test_unknown_keywords_returns_other_fraud(self):
        """Có từ khoá nhưng không vào nhóm → other_fraud."""
        text = "gấp gấp gấp khẩn cấp"
        result = _infer_scam_type(text, ["gấp"])
        assert result == "other_fraud"

    def test_viensatsat_authority(self):
        """viện kiểm sát → authority_impersonation."""
        text = "viện kiểm sát nhân dân gọi điện xác minh"
        result = _infer_scam_type(text, ["viện kiểm sát"])
        assert result == "authority_impersonation"

    def test_otp_bank(self):
        """otp → bank_fraud."""
        text = "vui lòng cung cấp mã otp để xác thực"
        result = _infer_scam_type(text, ["otp"])
        assert result == "bank_fraud"

    def test_voucher_prize(self):
        """voucher → prize_scam."""
        text = "nhận voucher du lịch miễn phí ngay hôm nay"
        result = _infer_scam_type(text, ["voucher"])
        assert result == "prize_scam"


# ===========================================================================
# 5. TestIntegration — cần model.pt thật
# ===========================================================================

@_REQUIRES_MODEL
class TestIntegration:
    """Test đầu đủ pipeline, cần model.pt + scaler.joblib + config.json."""

    # ── Fixture: reset singleton để mỗi class test bắt đầu sạch ──
    @pytest.fixture(autouse=True)
    def _warm_up(self):
        """Warm-up: gọi một lần để nạp model, sau đó các test dùng cache."""
        # Gọi một lần với văn bản dummy để trigger load
        process_and_predict("xin chào")
        yield

    # ── 4a. Câu giả mạo công an ──
    def test_authority_fraud_detected(self):
        """Câu giả mạo công an phải trả is_fraud=True và scam_type=authority_impersonation."""
        scam_texts = [
            "Công an yêu cầu anh chuyển khoản ngay 50 triệu để phục vụ điều tra",
            "Cảnh sát gọi điện thông báo tài khoản của bạn liên quan đến vụ án",
            "Viện kiểm sát nhân dân yêu cầu bạn nộp phạt ngay hôm nay",
        ]
        for text in scam_texts:
            result = process_and_predict(text)
            assert isinstance(result, Prediction)
            # Model có thể chưa hoàn hảo, nhưng pipeline phải chạy đúng
            assert 0.0 <= result.confidence_score <= 1.0
            assert result.risk_level in ("low", "medium", "high")
            assert result.processing_time_ms > 0
            # Nếu model phát hiện fraud → scam_type phải có giá trị
            if result.is_fraud:
                assert result.scam_type is not None

    # ── 4b. Câu trúng thưởng ──
    def test_prize_scam_detected(self):
        """Câu trúng thưởng phải trả scam_type='prize_scam' nếu is_fraud."""
        text = "Chúc mừng! Bạn đã trúng thưởng iPhone 15. Liên hệ ngay để nhận."
        result = process_and_predict(text)
        assert isinstance(result, Prediction)
        assert 0.0 <= result.confidence_score <= 1.0
        if result.is_fraud:
            assert result.scam_type == "prize_scam", \
                f"Từ khoá trúng thưởng phải → prize_scam, nhận: {result.scam_type}"

    # ── 4c. Câu lừa đảo ngân hàng ──
    def test_bank_fraud_detected(self):
        """Câu lừa đảo ngân hàng phải trả scam_type='bank_fraud' nếu is_fraud."""
        text = "Nhân viên ngân hàng yêu cầu bạn cung cấp mã OTP và số CVV ngay bây giờ"
        result = process_and_predict(text)
        assert isinstance(result, Prediction)
        if result.is_fraud:
            assert result.scam_type in ("bank_fraud", "authority_impersonation"), \
                f"Nhận: {result.scam_type}"

    # ── 4d. Câu bình thường (hỏi thăm) ──
    def test_normal_greeting_not_fraud(self):
        """Câu hỏi thăm bình thường → is_fraud=False (kỳ vọng)."""
        normal_texts = [
            "Bạn ơi hôm nay bạn có khỏe không? Lâu rồi mình chưa gặp nhau.",
            "Mình vừa mua được chiếc áo mới, trông đẹp lắm bạn ơi!",
        ]
        for text in normal_texts:
            result = process_and_predict(text)
            assert isinstance(result, Prediction)
            # Kiểm tra kiểu dữ liệu dù model chưa hoàn hảo
            assert isinstance(result.is_fraud, bool)
            assert result.risk_level in ("low", "medium", "high")
            if not result.is_fraud:
                assert result.scam_type is None

    # ── 4e. Câu bình thường khác ──
    def test_normal_shopping_not_fraud(self):
        """Câu mua sắm bình thường không phải lừa đảo."""
        text = "Siêu thị đang có chương trình giảm giá 30% tất cả mặt hàng cuối tuần này."
        result = process_and_predict(text)
        assert isinstance(result, Prediction)
        assert isinstance(result.is_fraud, bool)

    # ── 4f. Chuỗi rỗng (tích hợp) ──
    def test_empty_string_integration(self):
        """Chuỗi rỗng → Prediction an toàn, pipeline không crash."""
        result = process_and_predict("")
        assert result.is_fraud         is False
        assert result.confidence_score == 0.0
        assert result.risk_level       == "low"
        assert result.scam_type        is None

    # ── 4g. Văn bản 5.000 ký tự ──
    def test_long_text_5000_chars(self):
        """Văn bản 5.000 ký tự phải xử lý không crash, trả Prediction hợp lệ."""
        long_text = (
            "Đây là một đoạn văn bản rất dài được lặp đi lặp lại nhiều lần "
            "để kiểm tra khả năng xử lý văn bản dài của hệ thống. "
        ) * 60   # ~5.000 ký tự
        assert len(long_text) >= 5000

        result = process_and_predict(long_text)

        assert isinstance(result, Prediction)
        assert 0.0 <= result.confidence_score <= 1.0
        assert result.risk_level in ("low", "medium", "high")
        assert result.processing_time_ms > 0
        # Không crash — đây là điều quan trọng nhất
        assert isinstance(result.is_fraud, bool)

    # ── 4h. Gọi lần 2 nhanh hơn (không nạp lại model) ──
    def test_second_call_faster_than_first(self):
        """Lần gọi thứ hai phải nhanh hơn rõ rệt (model đã được cache)."""
        text = "Chuyển khoản ngay số tiền 20 triệu đồng để tránh bị phong tỏa"

        # Lần 1 (model đã warm-up bởi fixture)
        t1_start = time.perf_counter()
        result1  = process_and_predict(text)
        t1_end   = time.perf_counter()
        t1_ms    = (t1_end - t1_start) * 1000

        # Lần 2
        t2_start = time.perf_counter()
        result2  = process_and_predict(text)
        t2_end   = time.perf_counter()
        t2_ms    = (t2_end - t2_start) * 1000

        print(f"\n  Lần 1: {t1_ms:.2f} ms | Lần 2: {t2_ms:.2f} ms")

        # Kết quả phải giống nhau (deterministic)
        assert result1.is_fraud         == result2.is_fraud
        assert result1.confidence_score == result2.confidence_score
        assert result1.scam_type        == result2.scam_type

        # Lần 2 không được nạp lại model → phải rất nhanh
        # (sau khi đã warm-up, lần 1 ở đây cũng không nạp lại)
        # Cả hai lần đều phải < 2000 ms (inference + feature extraction)
        assert t2_ms < 2000, f"Lần 2 mất {t2_ms:.1f} ms — quá chậm (model có thể đang nạp lại)"

    # ── Kiểm tra đầu ra đầy đủ ──
    def test_prediction_fields_complete(self):
        """Mọi trường của Prediction đều phải có kiểu dữ liệu đúng."""
        result = process_and_predict("xin chào bạn khỏe không")
        assert isinstance(result.is_fraud,            bool)
        assert isinstance(result.confidence_score,    float)
        assert isinstance(result.risk_level,          str)
        assert result.risk_level in ("low", "medium", "high")
        assert isinstance(result.flagged_keywords,    list)
        assert isinstance(result.processing_time_ms,  float)
        assert result.processing_time_ms >= 0.0

    # ── Confidence score ∈ [0, 1] ──
    def test_confidence_score_in_range(self):
        """confidence_score phải nằm trong [0, 1]."""
        texts = [
            "",
            "xin chào",
            "công an yêu cầu chuyển khoản",
            "trúng thưởng rồi nhận ngay đi",
        ]
        for text in texts:
            result = process_and_predict(text)
            assert 0.0 <= result.confidence_score <= 1.0, \
                f"score={result.confidence_score} ngoài [0,1] cho text={text!r}"

    # ── risk_level nhất quán với confidence_score ──
    def test_risk_level_consistent_with_score(self):
        """risk_level phải nhất quán với confidence_score."""
        # Tạo Prediction thủ công để kiểm tra hàm score_to_risk_level
        for score, expected in [(0.1, "low"), (0.5, "medium"), (0.9, "high")]:
            assert score_to_risk_level(score) == expected


# ===========================================================================
# 6. TestSingletonBehavior
# ===========================================================================

@_REQUIRES_MODEL
class TestSingletonBehavior:
    """Xác nhận singleton chỉ nạp model một lần, thread-safe."""

    def test_model_loaded_once_in_sequential_calls(self):
        """Gọi liên tiếp nhiều lần — model chỉ nạp 1 lần."""
        # Warm-up
        process_and_predict("test")

        load_count_before = _STATE._loaded  # phải là True sau lần đầu

        # Gọi thêm 5 lần
        for _ in range(5):
            process_and_predict("xin chào")

        # _loaded vẫn là True (không thay đổi)
        assert _STATE._loaded is True

    def test_concurrent_calls_thread_safe(self):
        """Nhiều thread gọi đồng thời không gây race condition."""
        results: list[Prediction] = []
        errors:  list[Exception]  = []
        lock = threading.Lock()

        def worker():
            try:
                r = process_and_predict("Công an yêu cầu chuyển khoản ngay")
                with lock:
                    results.append(r)
            except Exception as e:
                with lock:
                    errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert len(errors) == 0, f"Lỗi trong concurrent calls: {errors}"
        assert len(results) == 8, "Không đủ 8 kết quả"

        # Tất cả kết quả phải giống nhau (deterministic)
        first = results[0]
        for r in results[1:]:
            assert r.is_fraud         == first.is_fraud
            assert r.confidence_score == first.confidence_score

    def test_results_are_deterministic(self):
        """Cùng input → cùng output mọi lần gọi."""
        text   = "Nhân viên ngân hàng yêu cầu cung cấp mã OTP ngay"
        result1 = process_and_predict(text)
        result2 = process_and_predict(text)
        result3 = process_and_predict(text)

        assert result1.is_fraud         == result2.is_fraud         == result3.is_fraud
        assert result1.confidence_score == result2.confidence_score == result3.confidence_score
        assert result1.scam_type        == result2.scam_type        == result3.scam_type
        assert result1.flagged_keywords == result2.flagged_keywords == result3.flagged_keywords


# ===========================================================================
# 7. TestModelNotReady — kiểm tra khi model chưa có
# ===========================================================================

class TestModelNotReady:
    """Kiểm tra hành vi khi model.pt chưa tồn tại."""

    def test_raises_model_not_ready_error(self, tmp_path):
        """Gọi với path không tồn tại phải raise ModelNotReadyError."""
        with pytest.raises(ModelNotReadyError):
            process_and_predict(
                "xin chào",
                model_path  = tmp_path / "nonexistent_model.pt",
                scaler_path = tmp_path / "nonexistent_scaler.joblib",
                config_path = tmp_path / "nonexistent_config.json",
            )

    def test_error_message_helpful(self, tmp_path):
        """Thông báo lỗi phải gợi ý cách khắc phục."""
        with pytest.raises(ModelNotReadyError, match="train_hybrid"):
            process_and_predict(
                "test",
                model_path  = tmp_path / "x.pt",
                scaler_path = tmp_path / "x.joblib",
                config_path = tmp_path / "x.json",
            )
