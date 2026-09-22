"""
tests/test_voice_api_robustness.py
===================================
Test suite cho Task B4:
- scripts/eval_robustness.py (B4a)
- src/api/voice_route.py & src/api/main.py (B4b)
- scripts/bench_latency.py (B4b)

Chạy::

    pytest tests/test_voice_api_robustness.py -v
"""

from __future__ import annotations

import io
from pathlib import Path
import sys
from unittest.mock import MagicMock, patch
import wave

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# sys.path
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.eval_robustness import (
    compute_classification_metrics,
    evaluate_robustness_for_dataframe,
    simulate_chunked_robustness,
)
from src.api.main import app
from src.api.voice_route import session_manager
from scripts.bench_latency import run_latency_benchmark, generate_4s_audio_chunk


# ===========================================================================
# Helper sinh WAV bytes hợp lệ cho test
# ===========================================================================

def make_valid_wav_bytes(duration_sec: float = 1.0, sr: int = 16000) -> bytes:
    """Tạo byte stream file WAV chuẩn mono 16kHz."""
    n_samples = int(duration_sec * sr)
    t = np.linspace(0, duration_sec, n_samples, endpoint=False, dtype=np.float32)
    y = (0.2 * np.sin(2 * np.pi * 440 * t) * 32767.0).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(y.tobytes())
    return buf.getvalue()


# ===========================================================================
# 1. Test chỉ số phân loại & độ bền B4a
# ===========================================================================

class TestRobustnessMetrics:
    """Kiểm tra tính toán Accuracy, Macro-F1, FNR, FPR và độ suy giảm Δ."""

    def test_compute_metrics_perfect(self):
        """Phân loại hoàn hảo → Accuracy=1.0, F1=1.0, FNR=0.0, FPR=0.0."""
        y_true = [1, 1, 0, 0]
        y_pred = [1, 1, 0, 0]
        m = compute_classification_metrics(y_true, y_pred)
        assert m["accuracy"] == 1.0
        assert m["macro_f1"] == 1.0
        assert m["fnr"] == 0.0
        assert m["fpr"] == 0.0

    def test_compute_metrics_all_fn(self):
        """Tất cả mẫu lừa đảo bị dự đoán là an toàn → FNR = 1.0 (100% bỏ sót)."""
        y_true = [1, 1, 1, 0]
        y_pred = [0, 0, 0, 0]
        m = compute_classification_metrics(y_true, y_pred)
        assert m["fnr"] == 1.0
        assert m["tp"] == 0
        assert m["fn"] == 3

    def test_delta_and_threshold_evaluation(self):
        """Kiểm tra tính toán độ suy giảm Δ và cảnh báo vượt ngưỡng 5%."""
        # Giả lập dữ liệu: Clean bắt được 100%, ASR bị bỏ sót 50%
        df = pd.DataFrame({
            "audio_path": ["a1.wav", "a2.wav", "a3.wav", "a4.wav"],
            "transcript_reference": [
                "công an gọi bạn",
                "yêu cầu chuyển tiền",
                "hôm nay trời đẹp",
                "xin chào bạn nhé",
            ],
            # ASR làm hỏng câu 2
            "transcript_asr": [
                "công an gọi bạn",
                "yêu cầu mua đồ chơi",  # Bị sai từ khóa lừa đảo
                "hôm nay trời đẹp",
                "xin chào bạn nhé",
            ],
            "label": [1, 1, 0, 0],
            "region": ["bac", "bac", "nam", "nam"],
            "tone": ["ap_luc", "ap_luc", "trung_tinh", "trung_tinh"],
            "speaker_id": ["SPK1", "SPK1", "SPK2", "SPK2"],
        })

        def mock_predict(t: str):
            is_fraud = any(w in t.lower() for w in ["công an", "chuyển tiền"])
            return is_fraud, 0.9 if is_fraud else 0.1, "high" if is_fraud else "low", None

        summary, reg_tone, new_fns = evaluate_robustness_for_dataframe(
            df=df,
            condition="clean",
            model="phowhisper-base",
            predictor=mock_predict,
        )

        assert summary["acc_clean"] == 1.0
        assert summary["acc_asr"] == 0.75
        assert summary["delta_acc"] == -0.25  # Giảm 25% (> 5% ngưỡng)
        assert summary["fnr_clean"] == 0.0
        assert summary["fnr_asr"] == 0.50
        assert summary["delta_fnr"] == 0.50   # FNR tăng 50%
        assert "Đề xuất ASR error augmentation" in summary["threshold_conclusion"]

        # Kiểm tra bảng theo region × tone
        assert not reg_tone.empty
        assert "delta_acc" in reg_tone.columns
        assert "delta_fnr" in reg_tone.columns

        # Kiểm tra New FN được phát hiện
        assert len(new_fns) == 1
        assert new_fns[0]["audio_path"] == "a2.wav"
        assert "chuyển tiền" in new_fns[0]["lost_or_corrupted_keywords"]


# ===========================================================================
# 2. Test mô phỏng độ bền theo Chunk 4s
# ===========================================================================

class TestChunkedRobustness:
    """Kiểm tra logic mô phỏng độ bền luỹ kế theo chunk thời gian thực."""

    def test_simulate_chunked_robustness(self):
        df = pd.DataFrame({
            "audio_path": ["test_call.wav"],
            # Câu dài khoảng 20 từ → chia ~2 chunk
            "transcript_asr": [
                "xin chào bạn tôi là cán bộ công an yêu cầu bạn xác minh tài khoản chuyển tiền ngay"
            ],
            "label": [1],
        })

        def mock_predict(t: str):
            is_fraud = "chuyển tiền" in t.lower() or "công an" in t.lower()
            return is_fraud, 0.85 if is_fraud else 0.1, "high" if is_fraud else "low", None

        chunk_df = simulate_chunked_robustness(
            df=df,
            condition="clean",
            words_per_chunk=8,
            predictor=mock_predict,
        )

        assert not chunk_df.empty
        assert "cumulative_seconds" in chunk_df.columns
        assert "is_stable" in chunk_df.columns
        assert chunk_df["cumulative_seconds"].iloc[0] == 4.0
        if len(chunk_df) > 1:
            assert chunk_df["cumulative_seconds"].iloc[1] == 8.0


# ===========================================================================
# 3. Test Route API /detect/voice (B4b)
# ===========================================================================

class TestVoiceRouteAPI:
    """Kiểm tra endpoint /detect/voice, luồng tích luỹ session, và PII masking."""

    @pytest.fixture(autouse=True)
    def clean_session_manager(self):
        """Dọn dẹp session_manager trước và sau mỗi test."""
        session_manager._sessions.clear()
        yield
        session_manager._sessions.clear()

    @patch("src.api.voice_route.transcribe_chunk")
    @patch("src.api.voice_route._predict_raw_text")
    def test_voice_route_multiple_chunks_accumulation(self, mock_pred, mock_asr):
        """Gửi nhiều chunk cùng session_id → transcript tích luỹ, chunk cuối xoá session."""
        client = TestClient(app)
        session_id = "test_sess_001"

        wav_bytes = make_valid_wav_bytes(duration_sec=1.0)

        from dataclasses import dataclass, field
        @dataclass
        class MockP:
            is_fraud: bool = True
            confidence_score: float = 0.88
            risk_level: str = "high"
            scam_type: str = "authority_impersonation"
            flagged_keywords: list[str] = field(default_factory=lambda: ["công an"])

        mock_pred.return_value = MockP()

        # --- Chunk 1: Chưa kết thúc ---
        mock_asr.return_value = "cán bộ công an gọi"
        res1 = client.post(
            "/detect/voice",
            data={"session_id": session_id, "is_final_chunk": False},
            files={"audio_file": ("chunk1.wav", wav_bytes, "audio/wav")},
        )
        assert res1.status_code == 200
        data1 = res1.json()
        assert data1["session_id"] == session_id
        assert data1["chunk_transcript"] == "cán bộ công an gọi"
        assert data1["transcript"] == "cán bộ công an gọi"
        assert data1["is_final"] is False
        assert session_manager.session_exists(session_id) is True

        # --- Chunk 2: Kết thúc cuộc gọi ---
        mock_asr.return_value = "yêu cầu chuyển tiền vào tài khoản"
        res2 = client.post(
            "/detect/voice",
            data={"session_id": session_id, "is_final_chunk": True},
            files={"audio_file": ("chunk2.wav", wav_bytes, "audio/wav")},
        )
        assert res2.status_code == 200
        data2 = res2.json()
        assert data2["chunk_transcript"] == "yêu cầu chuyển tiền vào tài khoản"
        # Kiểm tra cộng dồn cả 2 chunk
        assert data2["transcript"] == "cán bộ công an gọi yêu cầu chuyển tiền vào tài khoản"
        assert data2["is_final"] is True
        # Đã dọn dẹp session sau chunk cuối
        assert session_manager.session_exists(session_id) is False

    @patch("src.api.voice_route.transcribe_chunk")
    @patch("src.api.voice_route._predict_raw_text")
    def test_voice_route_pii_masking_behavior(self, mock_pred, mock_asr):
        """Transcript trả về được che PII (SĐT, STK), nhưng predictor nhận raw text."""
        client = TestClient(app)
        session_id = "test_pii_sess"
        wav_bytes = make_valid_wav_bytes(duration_sec=1.0)

        # ASR trả về chứa SĐT và số tài khoản
        raw_asr = "liên hệ số điện thoại 0987654321 và chuyển tiền số 123456789012"
        mock_asr.return_value = raw_asr

        from dataclasses import dataclass, field
        @dataclass
        class MockP:
            is_fraud: bool = True
            confidence_score: float = 0.95
            risk_level: str = "high"
            scam_type: str = "bank_fraud"
            flagged_keywords: list[str] = field(default_factory=lambda: ["chuyển tiền"])

        mock_pred.return_value = MockP()

        res = client.post(
            "/detect/voice",
            data={"session_id": session_id, "is_final_chunk": True},
            files={"audio_file": ("pii.wav", wav_bytes, "audio/wav")},
        )
        assert res.status_code == 200
        data = res.json()

        # Transcript trả về phải được che PII
        assert "[PHONE]" in data["transcript"] or "[PHONE]" in data["chunk_transcript"]
        assert ("[BANK_ACCOUNT]" in data["transcript"] or "[ID_NUMBER]" in data["transcript"])
        assert "0987654321" not in data["transcript"]
        assert "123456789012" not in data["transcript"]

        # Predictor PHẢI nhận raw_text (không che) để giữ số điện thoại/tài khoản cho A1
        pred_arg = mock_pred.call_args[0][0]
        assert "0987654321" in pred_arg
        assert "123456789012" in pred_arg

    def test_voice_route_empty_audio_file(self):
        """Tải lên file rỗng (0 bytes) → 400 Bad Request."""
        client = TestClient(app)
        res = client.post(
            "/detect/voice",
            data={"session_id": "sess_empty", "is_final_chunk": False},
            files={"audio_file": ("empty.wav", b"", "audio/wav")},
        )
        assert res.status_code == 400
        assert "rỗng" in res.json()["detail"].lower()

    def test_voice_route_corrupted_audio(self):
        """Tải lên dữ liệu không phải định dạng âm thanh → 400 Bad Request."""
        client = TestClient(app)
        res = client.post(
            "/detect/voice",
            data={"session_id": "sess_bad", "is_final_chunk": False},
            files={"audio_file": ("corrupt.wav", b"not a wav file binary content", "audio/wav")},
        )
        assert res.status_code == 400

    def test_voice_route_empty_session_id(self):
        """session_id rỗng → 400 Bad Request."""
        client = TestClient(app)
        wav_bytes = make_valid_wav_bytes()
        res = client.post(
            "/detect/voice",
            data={"session_id": "   ", "is_final_chunk": False},
            files={"audio_file": ("test.wav", wav_bytes, "audio/wav")},
        )
        assert res.status_code == 400


# ===========================================================================
# 4. Kiểm tra loại bỏ Fusion & Acoustic trong src/api/
# ===========================================================================

class TestNoAcousticFusionLeak:
    """Nghiệm thu spec B4: không còn bất kỳ dấu vết nào của Fusion hay Librosa features trong src/api/."""

    def test_src_api_contains_no_fusion_or_acoustic(self):
        api_dir = _REPO_ROOT / "src" / "api"
        forbidden_keywords = ["fusion", "acoustic", "asyncio.gather"]

        for py_file in api_dir.glob("*.py"):
            content = py_file.read_text(encoding="utf-8").lower()
            for kw in forbidden_keywords:
                assert kw not in content, (
                    f"Phát hiện từ khoá cấm '{kw}' trong file {py_file.name}"
                )


# ===========================================================================
# 5. Test Benchmark Latency (B4b)
# ===========================================================================

class TestLatencyBenchmark:
    """Kiểm tra runner benchmark độ trễ xuất đủ 4 thành phần và cấu hình máy."""

    @patch("scripts.bench_latency.transcribe_chunk", return_value="cán bộ công an gọi")
    def test_run_latency_benchmark_structure(self, mock_asr, tmp_path):
        out_csv = tmp_path / "latency.csv"
        df_bench = run_latency_benchmark(n_chunks=3, n_warmup=1, output_path=out_csv)

        assert out_csv.exists()
        assert len(df_bench) == 4

        expected_components = {
            "ASR theo chunk 4s",
            "NLP preprocess",
            "PhoBERT + Hybrid prediction",
            "Tổng pipeline chu trình thoại",
        }
        assert set(df_bench["component"]) == expected_components

        # Đảm bảo KHÔNG CÓ dòng "Fusion"
        assert "Fusion" not in df_bench["component"].values

        # Kiểm tra các cột bắt buộc
        req_cols = ["component", "p50_ms", "p95_ms", "max_ms", "mean_ms", "n_runs", "cpu", "cores", "ram", "os"]
        for c in req_cols:
            assert c in df_bench.columns

        for p50 in df_bench["p50_ms"]:
            assert p50 >= 0.0
