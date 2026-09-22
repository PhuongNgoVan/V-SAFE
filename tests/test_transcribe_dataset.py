"""
tests/test_transcribe_dataset.py
==================================
Test suite cho Task B3:
- scripts/transcribe_dataset.py
- scripts/keyword_asr_errors.py

Chạy::

    pytest tests/test_transcribe_dataset.py -v

Yêu cầu kiểm thử
────────────────
- Dữ liệu hoàn toàn mock — không phụ thuộc vào file audio thật trên đĩa
  và không cần tải trước trọng số PhoWhisper.
- Kiểm tra tính đúng đắn của logic tính WER, chuẩn hoá tiếng Việt có dấu.
- Kiểm tra schema các cột output của tất cả các file CSV theo hợp đồng §3 và spec B3.
- Kiểm tra logic phân tích keyword recall và trích xuất lỗi phiên âm điển hình.
"""

from __future__ import annotations

from pathlib import Path
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# sys.path
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.transcribe_dataset import (
    compute_row_wer,
    aggregate_wer_by_region_tone,
    transcribe_audio_dataset,
    _REQUIRED_METADATA_COLS,
)
from scripts.keyword_asr_errors import (
    load_scam_keywords,
    count_keyword_occurrences,
    find_misrecognition,
    calculate_keyword_recall_for_df,
    select_typical_errors,
    parse_model_and_condition_from_filename,
)


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def sample_metadata_df() -> pd.DataFrame:
    """Tạo DataFrame metadata mẫu chuẩn hợp đồng 00_SHARED_CONTRACT.md §3."""
    return pd.DataFrame({
        "audio_path": [
            "audio_01.wav",
            "audio_02.wav",
            "audio_03.wav",
            "audio_04.wav",
            "audio_05.wav",
            "audio_06.wav",
        ],
        "transcript_reference": [
            "công an yêu cầu chuyển tiền gấp trong hôm nay",
            "xin chào đây là ngân hàng nhà nước gọi xác nhận giao dịch",
            "anh đã trúng thưởng một chiếc xe máy và quà tặng",
            "cục thuế thông báo bạn còn nợ thuế cần nộp tiền ngay",
            "chúc mừng bạn nhận thưởng chương trình khuyến mãi",
            "cán bộ điều tra yêu cầu cung cấp mã otp và số tài khoản",
        ],
        "label": [1, 1, 1, 1, 0, 1],
        "scam_type": [
            "gia_mao_co_quan",
            "gia_mao_ngan_hang",
            "trung_thuong",
            "gia_mao_co_quan",
            "khac",
            "gia_mao_ngan_hang",
        ],
        "region": ["bac", "bac", "trung", "trung", "nam", "nam"],
        "tone": ["ap_luc", "trung_tinh", "ap_luc", "trung_tinh", "trung_tinh", "ap_luc"],
        "speaker_id": ["SPK_01", "SPK_02", "SPK_03", "SPK_04", "SPK_05", "SPK_06"],
    })


@pytest.fixture
def mock_scam_keywords() -> dict[str, list[str]]:
    """Từ điển từ khóa mẫu đủ 4 nhóm."""
    return {
        "urgency": ["gấp", "trong hôm nay", "ngay"],
        "authority": ["công an", "ngân hàng nhà nước", "cục thuế", "cán bộ"],
        "financial_action": ["chuyển tiền", "xác nhận giao dịch", "nộp tiền", "mã otp", "số tài khoản"],
        "reward": ["trúng thưởng", "quà tặng", "nhận thưởng", "khuyến mãi"],
    }


# ===========================================================================
# 1. Test tính toán WER
# ===========================================================================

class TestWERCalculation:
    """Kiểm tra logic tính WER và tuân thủ quy tắc chuẩn hoá theo hợp đồng §4."""

    def test_wer_exact_match(self):
        """Khớp chính xác từng từ → WER = 0.0."""
        ref = "công an yêu cầu chuyển tiền gấp"
        hyp = "công an yêu cầu chuyển tiền gấp"
        assert compute_row_wer(ref, hyp) == 0.0

    def test_wer_case_and_punctuation_invariance(self):
        """Bỏ qua hoa thường và dấu câu, giữ nguyên dấu thanh → WER = 0.0."""
        ref = "Công an, yêu cầu chuyển tiền gấp!"
        hyp = "công an yêu cầu chuyển tiền gấp."
        assert compute_row_wer(ref, hyp) == 0.0

    def test_wer_vietnamese_accent_sensitivity(self):
        """Dấu thanh khác biệt ('công an' vs 'công ăn') phải tính là lỗi thay thế (WER > 0)."""
        ref = "công an yêu cầu"
        hyp = "công ăn yêu cầu"
        wer_val = compute_row_wer(ref, hyp)
        assert wer_val > 0.0
        # 1 từ sai ('ăn' thay vì 'an') trên 4 từ → WER = 1/4 = 0.25
        assert round(wer_val, 2) == 0.25

    def test_wer_completely_different(self):
        """Khác biệt hoàn toàn → WER = 1.0 (hoặc >= 1.0 nếu có thêm từ)."""
        ref = "chuyển tiền ngay"
        hyp = "xin chào bạn"
        assert compute_row_wer(ref, hyp) == 1.0

    def test_wer_empty_edges(self):
        """Xử lý an toàn các trường hợp chuỗi rỗng và NaN."""
        # Cả hai rỗng → 0.0
        assert compute_row_wer("", "") == 0.0
        assert compute_row_wer(None, None) == 0.0
        assert compute_row_wer(float("nan"), float("nan")) == 0.0

        # Ref rỗng, hyp có chữ → 1.0
        assert compute_row_wer("", "alo xin chào") == 1.0

        # Ref có chữ, hyp rỗng (im lặng hoặc lỗi ASR) → 1.0
        assert compute_row_wer("công an gọi", "") == 1.0
        assert compute_row_wer("công an gọi", None) == 1.0


# ===========================================================================
# 2. Test cấu trúc cột Output CSV
# ===========================================================================

class TestOutputColumnsStructure:
    """Kiểm tra cấu trúc cột của tất cả các file CSV theo hợp đồng §3 và spec B3."""

    def test_transcripts_asr_columns_schema(self, sample_metadata_df):
        """transcripts_asr_<model>_<condition>.csv phải chứa metadata + 4 cột bắt buộc."""
        df = sample_metadata_df.copy()
        df["transcript_asr"] = ["công an yêu cầu chuyển tiền gấp"] * len(df)
        df["asr_latency_ms"] = [150.25] * len(df)
        df["audio_seconds"] = [3.5] * len(df)
        df["wer"] = [0.0] * len(df)

        # Kiểm tra metadata gốc
        for col in _REQUIRED_METADATA_COLS:
            assert col in df.columns, f"Thiếu cột metadata bắt buộc: {col}"

        # Kiểm tra 4 cột kết quả phiên âm
        expected_output_cols = ["transcript_asr", "asr_latency_ms", "audio_seconds", "wer"]
        for col in expected_output_cols:
            assert col in df.columns, f"Thiếu cột output ASR bắt buộc: {col}"

    def test_wer_region_tone_columns_schema(self, sample_metadata_df):
        """wer_region_tone_<condition>.csv phải chứa thông tin phân nhóm và metadata tái lập."""
        df = sample_metadata_df.copy()
        df["wer"] = [0.1, 0.2, 0.15, 0.05, 0.3, 0.25]

        agg_df = aggregate_wer_by_region_tone(
            df=df,
            model="phowhisper-base",
            condition="clean",
            seed=42,
            git_commit="mock_commit",
            timestamp="2026-09-23T00:00:00Z",
        )

        required_schema = [
            "region",
            "tone",
            "wer_mean",
            "wer_median",
            "n",
            "note",
            "model",
            "condition",
            "compute_type",
            "seed",
            "git_commit",
            "timestamp",
        ]
        for col in required_schema:
            assert col in agg_df.columns, f"Thiếu cột trong bảng tổng hợp WER: {col}"

    def test_keyword_recall_columns_schema(self, sample_metadata_df, mock_scam_keywords):
        """keyword_recall_asr.csv phải có đầy đủ cột thống kê tần suất và tỷ lệ recall."""
        df = sample_metadata_df.copy()
        df["transcript_asr"] = df["transcript_reference"]  # Giữ nguyên 100%

        recall_df, _ = calculate_keyword_recall_for_df(
            df=df,
            keywords_dict=mock_scam_keywords,
            condition="clean",
            model="phowhisper-base",
            git_commit="mock_commit",
            seed=42,
        )

        expected_cols = [
            "condition",
            "model",
            "level",
            "category",
            "keyword",
            "ref_count",
            "retained_count",
            "keyword_recall",
            "ref_samples",
            "recalled_samples",
            "sample_recall",
            "git_commit",
            "timestamp",
            "seed",
        ]
        for col in expected_cols:
            assert col in recall_df.columns, f"Thiếu cột trong bảng keyword recall: {col}"

    def test_asr_error_examples_columns_schema(self, sample_metadata_df, mock_scam_keywords):
        """asr_error_examples.csv phải có đầy đủ thông tin vị trí lỗi, ngữ cảnh và mã ví dụ."""
        df = sample_metadata_df.copy()
        # Giả lập lỗi: thay 'công an' bằng 'công ăn'
        df["transcript_asr"] = [
            "công ăn yêu cầu chuyển tiền gấp trong hôm nay",
            "xin chào đây là ngân hàng nhà nước gọi xác nhận giao dịch",
            "anh đã trúng thưởng một chiếc xe máy và quà tặng",
            "cục thuế thông báo bạn còn nợ thuế cần nộp tiền ngay",
            "chúc mừng bạn nhận thưởng chương trình khuyến mãi",
            "cán bộ điều tra yêu cầu cung cấp mã otp và số tài khoản",
        ]

        _, errors = calculate_keyword_recall_for_df(
            df=df,
            keywords_dict=mock_scam_keywords,
            condition="clean",
            model="phowhisper-base",
            git_commit="mock_commit",
            seed=42,
        )
        selected_errors = select_typical_errors(errors, max_examples=30)

        expected_cols = [
            "example_id",
            "condition",
            "model",
            "category",
            "target_keyword",
            "detected_misrecognition",
            "ref_snippet",
            "asr_snippet",
            "region",
            "tone",
            "speaker_id",
            "transcript_reference",
            "transcript_asr",
            "git_commit",
            "timestamp",
            "seed",
        ]
        for col in expected_cols:
            assert col in selected_errors.columns, f"Thiếu cột trong bảng lỗi ASR: {col}"


# ===========================================================================
# 3. Test tổng hợp theo Region × Tone
# ===========================================================================

class TestRegionToneAggregation:
    """Kiểm tra logic nhóm dữ liệu theo 3 miền × 2 sắc thái."""

    def test_aggregation_values(self):
        """Mean, median và số mẫu n phải được tính chính xác."""
        df = pd.DataFrame({
            "region": ["bac", "bac", "nam", "nam"],
            "tone": ["ap_luc", "ap_luc", "trung_tinh", "trung_tinh"],
            "wer": [0.10, 0.20, 0.05, 0.15],
        })

        agg = aggregate_wer_by_region_tone(
            df=df,
            model="phowhisper-base",
            condition="clean",
            seed=42,
            git_commit="test_hash",
        )

        assert len(agg) == 2

        bac_apluc = agg[(agg["region"] == "bac") & (agg["tone"] == "ap_luc")].iloc[0]
        assert bac_apluc["n"] == 2
        assert bac_apluc["wer_mean"] == 0.15
        assert bac_apluc["wer_median"] == 0.15
        assert bac_apluc["note"] == "n nhỏ"  # n < 30
        assert bac_apluc["seed"] == 42
        assert bac_apluc["git_commit"] == "test_hash"

    def test_empty_dataframe_aggregation(self):
        """Xử lý an toàn khi DataFrame rỗng."""
        agg = aggregate_wer_by_region_tone(pd.DataFrame(), "phowhisper-base", "clean")
        assert agg.empty
        assert "region" in agg.columns


# ===========================================================================
# 4. Test logic Keyword Recall
# ===========================================================================

class TestKeywordRecallLogic:
    """Kiểm tra logic so sánh từ khóa và tính tỷ lệ giữ lại."""

    def test_count_keyword_occurrences(self):
        """Đếm đúng từ khóa đơn và từ khóa ghép với ranh giới từ."""
        text = "công an huyện thông báo công an xã phải khẩn cấp"
        assert count_keyword_occurrences("công an", text) == 2
        assert count_keyword_occurrences("khẩn cấp", text) == 1
        assert count_keyword_occurrences("thuế", text) == 0
        # "an" không được khớp nếu chỉ là chuỗi con bên trong từ khác ("hoang", "mang")
        assert count_keyword_occurrences("an", "hoang mang lo lang") == 0

    def test_perfect_keyword_recall(self, mock_scam_keywords):
        """Khi ASR giữ nguyên toàn bộ từ khóa, recall = 1.0 (100%)."""
        df = pd.DataFrame({
            "transcript_reference": ["công an yêu cầu chuyển tiền gấp"],
            "transcript_asr": ["công an yêu cầu chuyển tiền gấp"],
            "region": ["bac"],
            "tone": ["ap_luc"],
            "speaker_id": ["SPK_1"],
        })

        recall_df, errors = calculate_keyword_recall_for_df(
            df=df,
            keywords_dict=mock_scam_keywords,
            condition="clean",
        )

        # Kiểm tra từ khóa cụ thể 'công an'
        ca_row = recall_df[recall_df["keyword"] == "công an"].iloc[0]
        assert ca_row["ref_count"] == 1
        assert ca_row["retained_count"] == 1
        assert ca_row["keyword_recall"] == 1.0
        assert len(errors) == 0

    def test_partial_and_zero_keyword_recall(self, mock_scam_keywords):
        """Khi từ khóa bị mất hoặc sai dấu thanh, recall phản ánh chính xác."""
        df = pd.DataFrame({
            "transcript_reference": [
                "công an gọi cho bạn",
                "công an tiếp tục yêu cầu",
                "gặp cán bộ công an",
            ],
            # 'công an' chỉ xuất hiện ở câu 1, câu 2 bị mất, câu 3 biến thành 'công ăn'
            "transcript_asr": [
                "công an gọi cho bạn",
                "người ta tiếp tục yêu cầu",
                "gặp cán bộ công ăn",
            ],
            "region": ["bac", "bac", "bac"],
            "tone": ["ap_luc", "ap_luc", "ap_luc"],
            "speaker_id": ["SPK_1", "SPK_2", "SPK_3"],
        })

        recall_df, errors = calculate_keyword_recall_for_df(
            df=df,
            keywords_dict=mock_scam_keywords,
            condition="snr10",
        )

        ca_row = recall_df[recall_df["keyword"] == "công an"].iloc[0]
        assert ca_row["ref_count"] == 3
        assert ca_row["retained_count"] == 1
        assert round(ca_row["keyword_recall"], 4) == 0.3333
        assert ca_row["ref_samples"] == 3
        assert ca_row["recalled_samples"] == 1
        assert round(ca_row["sample_recall"], 4) == 0.3333

        # Đã phát hiện 2 ca lỗi của 'công an'
        ca_errors = [e for e in errors if e["target_keyword"] == "công an"]
        assert len(ca_errors) == 2


# ===========================================================================
# 5. Test trích xuất lỗi phiên âm (ASR Error Examples)
# ===========================================================================

class TestASRErrorExtraction:
    """Kiểm tra khả năng tìm lỗi thay thế (công an → công ăn) hoặc bỏ sót từ khóa."""

    def test_find_misrecognition_substitution(self):
        """Tìm thấy từ thay thế cụ thể: 'công an' → 'công ăn'."""
        ref = "gặp cán bộ công an quận"
        hyp = "gặp cán bộ công ăn quận"

        ref_snip, asr_snip, detected = find_misrecognition("công an", ref, hyp)

        assert "[công an]" in ref_snip
        assert detected == "công ăn"
        assert "[công ăn]" in asr_snip

    def test_find_misrecognition_deletion(self):
        """Phát hiện từ khóa bị bỏ sót / omitted."""
        ref = "yêu cầu chuyển tiền ngay lập tức"
        hyp = "yêu cầu ngay lập tức"

        ref_snip, asr_snip, detected = find_misrecognition("chuyển tiền", ref, hyp)

        assert "[chuyển tiền]" in ref_snip
        assert "bỏ sót" in detected.lower() or "omitted" in detected.lower()

    def test_select_typical_errors_cap_at_max(self):
        """Giới hạn đúng 30 lỗi điển hình và đánh số thứ tự từ 1."""
        fake_errors = []
        for i in range(50):
            fake_errors.append({
                "condition": "clean",
                "model": "phowhisper-base",
                "category": "authority",
                "target_keyword": f"keyword_{i}",
                "detected_misrecognition": f"misrecognized_{i}",
                "ref_snippet": f"ref snippet {i}",
                "asr_snippet": f"asr snippet {i}",
                "region": "bac",
                "tone": "ap_luc",
                "speaker_id": f"SPK_{i}",
                "transcript_reference": f"full ref {i}",
                "transcript_asr": f"full asr {i}",
                "git_commit": "abc",
                "timestamp": "now",
                "seed": 42,
            })

        selected = select_typical_errors(fake_errors, max_examples=30)
        assert len(selected) == 30
        assert selected["example_id"].tolist() == list(range(1, 31))


# ===========================================================================
# 6. Test Mock Pipeline tích hợp (End-to-End không cần audio thật)
# ===========================================================================

class TestPipelineMockIntegration:
    """Kiểm tra luồng thực thi transcribe_audio_dataset bằng mock."""

    @patch("scripts.transcribe_dataset.load_and_resample")
    @patch("scripts.transcribe_dataset.make_noisy")
    @patch("scripts.transcribe_dataset.asr_module.transcribe_chunk")
    def test_transcribe_dataset_clean_pipeline(
        self,
        mock_transcribe,
        mock_noisy,
        mock_load,
        sample_metadata_df,
        tmp_path,
    ):
        """Pipeline điều kiện 'clean': load_and_resample được gọi, KHÔNG gọi make_noisy."""
        # Mock audio: 1 giây audio 16kHz
        dummy_audio = np.zeros(16000, dtype=np.float32)
        mock_load.return_value = (dummy_audio, 16000)
        mock_transcribe.return_value = "công an yêu cầu chuyển tiền"

        # Tạo file audio giả trong tmp_path để path.exists() trả về True
        for row in sample_metadata_df["audio_path"]:
            (tmp_path / row).touch()

        transcripts_df, wer_df = transcribe_audio_dataset(
            metadata_df=sample_metadata_df,
            data_dir=tmp_path,
            model_name="phowhisper-base",
            condition="clean",
            batch_limit=3,
        )

        assert len(transcripts_df) == 3
        assert mock_load.call_count == 3
        assert mock_transcribe.call_count == 3
        # Không được gọi make_noisy ở điều kiện clean
        mock_noisy.assert_not_called()

        # Kiểm tra các giá trị kết quả
        for val in transcripts_df["audio_seconds"]:
            assert val == 1.0
        for val in transcripts_df["asr_latency_ms"]:
            assert val >= 0.0
        for val in transcripts_df["wer"]:
            assert 0.0 <= val <= 1.0

        assert not wer_df.empty
        assert "wer_mean" in wer_df.columns

    @patch("scripts.transcribe_dataset.load_and_resample")
    @patch("scripts.transcribe_dataset.make_noisy")
    @patch("scripts.transcribe_dataset.asr_module.transcribe_chunk")
    def test_transcribe_dataset_noisy_pipeline(
        self,
        mock_transcribe,
        mock_noisy,
        mock_load,
        sample_metadata_df,
        tmp_path,
    ):
        """Pipeline điều kiện 'snr10': phải gọi make_noisy với snr_db=10."""
        dummy_audio = np.zeros(16000, dtype=np.float32)
        mock_load.return_value = (dummy_audio, 16000)
        mock_noisy.return_value = dummy_audio
        mock_transcribe.return_value = "phiên âm có nhiễu"

        for row in sample_metadata_df["audio_path"]:
            (tmp_path / row).touch()

        transcripts_df, wer_df = transcribe_audio_dataset(
            metadata_df=sample_metadata_df,
            data_dir=tmp_path,
            model_name="phowhisper-base",
            condition="snr10",
            batch_limit=2,
            seed=42,
        )

        assert len(transcripts_df) == 2
        assert mock_noisy.call_count == 2
        # Kiểm tra tham số snr_db trong make_noisy
        _, kwargs = mock_noisy.call_args
        assert kwargs["snr_db"] == 10
        assert kwargs["sr"] == 16000

    def test_parse_model_and_condition_from_filename(self):
        """Tự động tách model và condition từ tên file."""
        p1 = Path("data/audio/transcripts_asr_phowhisper-base_clean.csv")
        m1, c1 = parse_model_and_condition_from_filename(p1)
        assert m1 == "phowhisper-base"
        assert c1 == "clean"

        p2 = Path("data/audio/transcripts_asr_whisper-base_snr10.csv")
        m2, c2 = parse_model_and_condition_from_filename(p2)
        assert m2 == "whisper-base"
        assert c2 == "snr10"

        p3 = Path("data/audio/transcripts_asr_phowhisper-small_snr5.csv")
        m3, c3 = parse_model_and_condition_from_filename(p3)
        assert m3 == "phowhisper-small"
        assert c3 == "snr5"
