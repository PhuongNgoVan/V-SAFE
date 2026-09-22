"""
tests/test_chapter3_report_b.py
================================
Test suite cho Task B5:
- scripts/make_tables_b.py
- reports/chapter3_asr_robustness.md
- docs/agent_notes/HANDOFF_B.md

Chạy::

    pytest tests/test_chapter3_report_b.py -v
"""

from __future__ import annotations

from pathlib import Path
import re
import sys

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# sys.path
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.make_tables_b import (
    generate_all_tables,
    generate_wer_model_table,
    generate_wer_region_tone_table,
    generate_keyword_recall_table,
    generate_asr_error_examples_table,
    generate_robustness_summary_table,
    generate_latency_table,
    generate_chunked_robustness_table,
)


# ===========================================================================
# 1. Test sinh bảng biểu tự động (scripts/make_tables_b.py)
# ===========================================================================

class TestMakeTablesB:
    """Kiểm tra tính đúng đắn và chuẩn định dạng Markdown của các bảng sinh tự động."""

    def test_generate_all_tables_returns_dict(self, tmp_path):
        """Hàm generate_all_tables phải trả về dict chứa đủ 7 bảng thành phần."""
        tables = generate_all_tables(results_dir=tmp_path)
        expected_keys = {
            "wer_model_table",
            "wer_region_tone_table",
            "keyword_recall_table",
            "asr_error_examples_table",
            "robustness_summary_table",
            "latency_table",
            "chunked_robustness_table",
        }
        assert expected_keys.issubset(tables.keys())

    def test_markdown_table_structure(self, tmp_path):
        """Các bảng sinh ra phải đúng định dạng Markdown (| header | và |---|)."""
        tables = generate_all_tables(results_dir=tmp_path)
        for name, table_str in tables.items():
            assert table_str.strip().startswith("|"), f"Bảng {name} không bắt đầu bằng ký tự '|'"
            assert "|:---" in table_str or "|---" in table_str, f"Bảng {name} thiếu dòng phân cách tiêu đề"

    def test_latency_table_has_no_fusion(self, tmp_path):
        """Bảng độ trễ latency tuyệt đối KHÔNG chứa dòng hoặc từ 'Fusion'."""
        latency_str = generate_latency_table(results_dir=tmp_path)
        assert "fusion" not in latency_str.lower(), "Bảng latency có chứa từ cấm 'Fusion'"
        assert "ASR" in latency_str
        assert "p50" in latency_str or "p95" in latency_str

    def test_dynamic_table_generation_from_csv(self, tmp_path):
        """Khi có file CSV thật, make_tables_b phải đọc trực tiếp từ CSV thay vì dùng fallback."""
        csv_file = tmp_path / "wer_by_model.csv"
        df_real = pd.DataFrame([
            {"model": "test-model-x", "dataset": "vivos", "condition": "clean", "n_samples": 42, "WER": 7.5, "note": "ok"}
        ])
        df_real.to_csv(csv_file, index=False)

        table_str = generate_wer_model_table(results_dir=tmp_path)
        assert "test-model-x" in table_str
        assert "7.50%" in table_str


# ===========================================================================
# 2. Test cấu trúc và nội dung báo cáo Chương 3 (reports/chapter3_asr_robustness.md)
# ===========================================================================

class TestChapter3ReportContent:
    """Kiểm tra báo cáo reports/chapter3_asr_robustness.md đầy đủ 6 phần bắt buộc."""

    @pytest.fixture
    def report_text(self) -> str:
        report_path = _REPO_ROOT / "reports" / "chapter3_asr_robustness.md"
        assert report_path.exists(), f"Không tìm thấy file báo cáo: {report_path}"
        return report_path.read_text(encoding="utf-8")

    def test_report_has_all_6_sections(self, report_text):
        """Báo cáo phải chứa đủ 6 phần nội dung chuẩn theo B5 spec §2."""
        patterns = [
            r"##\s*1\.\s*Mô tả Dữ liệu Âm thanh",
            r"##\s*2\.\s*Đánh Giá Hiệu Năng ASR",
            r"##\s*3\.\s*Đánh Giá Độ Bền",
            r"##\s*4\.\s*Phân Tích Độ Trễ",
            r"##\s*5\.\s*Quản Trị Rủi Ro & Giới Hạn",
            r"##\s*6\.\s*Lý Do Đơn Giản Hoá Kiến Trúc",
        ]
        for pat in patterns:
            match = re.search(pat, report_text, re.IGNORECASE)
            assert match is not None, f"Báo cáo thiếu tiêu đề phần khớp với pattern: {pat}"

    def test_report_separates_vinai_published_numbers(self, report_text):
        """Báo cáo phải ghi rõ số liệu của VinAI là số của tác giả, tách riêng số tự đo."""
        assert "VinAI" in report_text or "tác giả" in report_text
        assert "VIVOS" in report_text
        assert "tự đo" in report_text

    def test_report_specifies_test_only_dataset(self, report_text):
        """Báo cáo phải nêu rõ tập thu âm chỉ dùng làm tập test ASR / độ bền, không huấn luyện."""
        text_lower = report_text.lower()
        assert "không" in text_lower and ("huấn luyện" in text_lower or "train" in text_lower)
        assert "test" in text_lower or "kiểm thử" in text_lower

    def test_report_mentions_speaker_disjoint_and_codec(self, report_text):
        """Báo cáo phải đề cập đến speaker_id và mô phỏng telephone codec."""
        assert "speaker_id" in report_text or "người nói" in report_text
        assert "codec" in report_text.lower() or "8 khz" in report_text.lower() or "8khz" in report_text.lower()

    def test_report_covers_all_4_risks(self, report_text):
        """Báo cáo phải đề cập đầy đủ 4 rủi ro: R8, R9, R1, R10."""
        for r_code in ["R8", "R9", "R1", "R10"]:
            assert r_code in report_text, f"Báo cáo chưa phân tích rủi ro {r_code}"

    def test_report_latency_has_no_fusion(self, report_text):
        """Trong mục latency của báo cáo, không được có dòng Fusion."""
        # Tìm phần 4
        sec4_match = re.search(r"##\s*4\..*?(?=##\s*5\.|\Z)", report_text, re.DOTALL | re.IGNORECASE)
        assert sec4_match is not None
        sec4_text = sec4_match.group(0)
        assert "fusion" not in sec4_text.lower()


# ===========================================================================
# 3. Test tài liệu bàn giao kỹ thuật S3 (docs/agent_notes/HANDOFF_B.md)
# ===========================================================================

class TestHandoffBDocument:
    """Kiểm tra biên bản bàn giao mốc S3 trong docs/agent_notes/HANDOFF_B.md."""

    def test_handoff_b_exists_and_contains_contract_conformance(self):
        handoff_path = _REPO_ROOT / "docs" / "agent_notes" / "HANDOFF_B.md"
        assert handoff_path.exists(), f"Không tìm thấy {handoff_path}"
        content = handoff_path.read_text(encoding="utf-8")

        assert "process_and_predict" in content
        assert "FNR" in content
        assert "Macro-F1" in content or "f1_score" in content
        assert "normalize_for_wer" in content
        assert "Fusion" in content or "acoustic" in content
