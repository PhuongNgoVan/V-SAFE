"""
tests/test_ablation.py
======================
Test suite cho Task A4:
- scripts/run_ablation.py
- results/ablation_summary.csv
- results/ablation_metrics.json
- results/hybrid_error_analysis.csv
- reports/chapter3_hybrid_ablation.md

Chạy::

    pytest tests/test_ablation.py -v
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# sys.path
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.run_ablation import (
    compute_bootstrap_ci,
    compute_mcnemar_test,
    compute_metrics,
    generate_ablation_markdown_table,
    run_ablation,
)
from src.features.structured_features import FEATURE_ORDER


# ===========================================================================
# 1. Kiểm tra Pipeline Run Ablation & File Kết Quả
# ===========================================================================
class TestRunAblationPipeline:
    """Kiểm tra scripts/run_ablation.py sinh đầy đủ và đúng định dạng kết quả."""

    @pytest.fixture(scope="class")
    def ablation_run_output(self, tmp_path_factory):
        """Chạy thực nghiệm ablation ở chế độ mock trong thư mục tạm để kiểm thử."""
        tmp_dir = tmp_path_factory.mktemp("ablation_test_run")
        res_dir = tmp_dir / "results"
        out_dir = tmp_dir / "models"

        output = run_ablation(
            results_dir=res_dir,
            output_dir=out_dir,
            seed=42,
            mock=True,
            n_bootstrap=100,  # 100 lần để test chạy nhanh
            epochs=2,
            device_str="cpu",
        )
        return {
            "output": output,
            "res_dir": res_dir,
            "summary_csv": res_dir / "ablation_summary.csv",
            "metrics_json": res_dir / "ablation_metrics.json",
            "error_csv": res_dir / "hybrid_error_analysis.csv",
        }

    def test_creates_all_three_result_files(self, ablation_run_output):
        """Xác nhận script sinh đủ 3 file kết quả bắt buộc."""
        assert ablation_run_output["summary_csv"].exists(), "Thiếu results/ablation_summary.csv"
        assert ablation_run_output["metrics_json"].exists(), "Thiếu results/ablation_metrics.json"
        assert ablation_run_output["error_csv"].exists(), "Thiếu results/hybrid_error_analysis.csv"

    def test_summary_csv_schema_and_models(self, ablation_run_output):
        """Kiểm tra schema ablation_summary.csv chứa đúng 4 mô hình M0-M3 và các chỉ số."""
        df = pd.read_csv(ablation_run_output["summary_csv"])
        assert len(df) == 4, f"Bảng summary phải có đúng 4 dòng (M0-M3), nhận {len(df)}"

        expected_cols = {
            "model_id",
            "model_name",
            "input_features",
            "feature_dim",
            "accuracy",
            "macro_f1",
            "fnr",
            "fpr",
        }
        assert expected_cols.issubset(df.columns), f"Thiếu cột trong summary CSV: {expected_cols - set(df.columns)}"

        model_ids = df["model_id"].tolist()
        assert model_ids == ["M0", "M1", "M2", "M3"], f"Danh sách model_id không đúng: {model_ids}"

        # Kiểm tra khoảng giá trị hợp lệ [0, 1]
        for col in ["accuracy", "macro_f1", "fnr", "fpr"]:
            assert (df[col] >= 0.0).all() and (df[col] <= 1.0).all(), f"Giá trị {col} không nằm trong [0, 1]"

        # Kiểm tra số chiều feature
        dim_map = dict(zip(df["model_id"], df["feature_dim"]))
        assert dim_map["M0"] == 768
        assert dim_map["M1"] == 779
        assert dim_map["M2"] == 777
        assert dim_map["M3"] == 11

    def test_metrics_json_structure(self, ablation_run_output):
        """Kiểm tra cấu trúc và tính đầy đủ của file ablation_metrics.json."""
        with open(ablation_run_output["metrics_json"], "r", encoding="utf-8") as f:
            data = json.load(f)

        assert "metadata" in data
        assert data["metadata"]["seed"] == 42
        assert "timestamp" in data["metadata"]
        assert "git_commit" in data["metadata"]

        assert "models" in data
        assert set(data["models"].keys()) == {"M0", "M1", "M2", "M3"}

        assert "statistical_tests" in data
        assert "m0_vs_m1" in data["statistical_tests"]
        m_tests = data["statistical_tests"]["m0_vs_m1"]
        assert "mcnemar" in m_tests
        assert "bootstrap_ci_95" in m_tests

        # McNemar keys
        assert "chi2" in m_tests["mcnemar"]
        assert "p_value" in m_tests["mcnemar"]
        assert "contingency_table" in m_tests["mcnemar"]

        # Bootstrap keys
        assert "delta_macro_f1" in m_tests["bootstrap_ci_95"]
        assert "delta_fnr" in m_tests["bootstrap_ci_95"]

    def test_error_analysis_csv_content(self, ablation_run_output):
        """Kiểm tra file phân tích lỗi có đủ các cột và ưu tiên FN trước FP."""
        df_err = pd.read_csv(ablation_run_output["error_csv"])
        assert len(df_err) > 0, "File phân tích lỗi không được rỗng"
        assert len(df_err) <= 20, "File phân tích lỗi tối đa 20 mẫu"

        required_cols = {
            "sample_id",
            "text",
            "true_label",
            "predicted_label",
            "confidence_score",
            "error_type",
            "root_cause",
            "flagged_keywords",
        }
        assert required_cols.issubset(df_err.columns)

        # Kiểm tra ưu tiên: nếu có FN thì FN phải đứng trước FP
        error_types = df_err["error_type"].tolist()
        if "FN" in error_types and "FP" in error_types:
            first_fp_idx = error_types.index("FP")
            last_fn_idx = max(i for i, t in enumerate(error_types) if t == "FN")
            assert last_fn_idx < first_fp_idx, "Mẫu FN phải đứng trước mẫu FP trong danh sách phân tích lỗi"


# ===========================================================================
# 2. Kiểm tra Tính Toán Thống Kê (McNemar & Bootstrap 95% CI)
# ===========================================================================
class TestStatisticalCalculations:
    """Kiểm tra độ chính xác và tính đúng đắn của các hàm kiểm định thống kê."""

    def test_mcnemar_identical_predictions(self):
        """Khi hai mô hình dự đoán giống hệt nhau, chi2 = 0 và p-value = 1.0."""
        y_true = np.array([1, 0, 1, 1, 0, 0, 1, 0])
        y_pred = np.array([1, 0, 1, 1, 0, 0, 1, 0])
        res = compute_mcnemar_test(y_true, y_pred, y_pred)
        assert res["chi2"] == 0.0
        assert res["p_value"] == 1.0
        assert res["significant"] is False

    def test_mcnemar_known_discordant(self):
        """Kiểm tra McNemar khi có mẫu bất đồng rõ rệt."""
        y_true = np.array([1] * 20 + [0] * 20)
        # M0 đoán đúng hết
        pred_m0 = y_true.copy()
        # M1 đoán sai 10 mẫu lừa đảo
        pred_m1 = y_true.copy()
        pred_m1[:10] = 0

        res = compute_mcnemar_test(y_true, pred_m0, pred_m1)
        # b = 10 (M0 đúng, M1 sai), c = 0
        assert res["contingency_table"]["b"] == 10
        assert res["contingency_table"]["c"] == 0
        assert 0.0 <= res["p_value"] <= 1.0
        assert res["chi2"] > 0.0

    def test_bootstrap_ci_bounds(self):
        """Kiểm tra Bootstrap CI 95%: lower <= mean <= upper."""
        rng = np.random.RandomState(42)
        y_true = rng.randint(0, 2, size=100)
        p_m0 = rng.randint(0, 2, size=100)
        p_m1 = rng.randint(0, 2, size=100)

        res = compute_bootstrap_ci(y_true, p_m0, p_m1, n_iterations=200, seed=42)

        ci_f1 = res["delta_macro_f1"]
        assert ci_f1["ci_95_lower"] <= ci_f1["mean"] <= ci_f1["ci_95_upper"]

        ci_fnr = res["delta_fnr"]
        assert ci_fnr["ci_95_lower"] <= ci_fnr["mean"] <= ci_fnr["ci_95_upper"]

    def test_compute_metrics_standard_contract(self):
        """Kiểm tra compute_metrics tuân thủ FNR = FN / (FN + TP) theo 00_SHARED_CONTRACT.md §4."""
        y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        y_pred = np.array([1, 1, 0, 0, 0, 0, 1, 1])
        # tp=2, fn=2 -> fnr = 2 / 4 = 0.5
        # tn=2, fp=2 -> fpr = 2 / 4 = 0.5
        m = compute_metrics(y_true, y_pred)
        assert m["tp"] == 2
        assert m["fn"] == 2
        assert m["fp"] == 2
        assert m["tn"] == 2
        assert m["fnr"] == 0.5
        assert m["fpr"] == 0.5
        assert m["accuracy"] == 0.5


# ===========================================================================
# 3. Kiểm tra Nội Dung Báo Cáo reports/chapter3_hybrid_ablation.md
# ===========================================================================
class TestChapter3ReportContent:
    """Kiểm tra báo cáo reports/chapter3_hybrid_ablation.md đầy đủ các mục bắt buộc."""

    @pytest.fixture(scope="class")
    def report_text(self) -> str:
        report_path = _REPO_ROOT / "reports" / "chapter3_hybrid_ablation.md"
        assert report_path.exists(), f"Không tìm thấy file báo cáo: {report_path}"
        return report_path.read_text(encoding="utf-8")

    def test_contains_all_11_features(self, report_text):
        """Báo cáo phải chứa đủ tên của cả 11 đặc trưng trong FEATURE_ORDER."""
        for feat in FEATURE_ORDER:
            assert feat in report_text, f"Báo cáo thiếu mô tả đặc trưng: {feat}"

    def test_contains_architecture_and_training(self, report_text):
        """Báo cáo phải mô tả kiến trúc HybridTextClassifier và quy trình huấn luyện."""
        assert "HybridTextClassifier" in report_text
        assert "PhoBERT" in report_text
        assert "StandardScaler" in report_text
        assert "Early Stopping" in report_text or "early stopping" in report_text.lower()
        assert "ngưỡng" in report_text.lower()

    def test_contains_ablation_table_m0_to_m3(self, report_text):
        """Báo cáo phải có bảng Ablation với đủ 4 mô hình M0, M1, M2, M3."""
        for mid in ["M0", "M1", "M2", "M3"]:
            assert f"**{mid}**" in report_text or f"| {mid} " in report_text or f"| **{mid}**" in report_text

        # Bảng phải có các chỉ số chính
        assert "Accuracy" in report_text
        assert "Macro-F1" in report_text
        assert "FNR" in report_text
        assert "FPR" in report_text

    def test_contains_statistical_tests(self, report_text):
        """Báo cáo phải có kết quả kiểm định McNemar và Bootstrap 95% CI."""
        assert "McNemar" in report_text or "mcnemar" in report_text.lower()
        assert "Bootstrap" in report_text or "bootstrap" in report_text.lower()
        assert "95%" in report_text
        assert "Khoảng tin cậy" in report_text or "khoảng tin cậy" in report_text.lower()

    def test_contains_error_analysis(self, report_text):
        """Báo cáo phải có phần phân tích lỗi False Negatives và False Positives."""
        assert "Phân Tích Lỗi" in report_text or "Phân tích lỗi" in report_text
        assert "False Negative" in report_text or "FN" in report_text
        assert "False Positive" in report_text or "FP" in report_text

    def test_contains_mandatory_limitations_section(self, report_text):
        """Báo cáo BẮT BUỘC có mục Giới hạn hệ thống (prosody giọng điệu & lệch phân phối SMS/ASR)."""
        assert "Giới Hạn Hệ Thống" in report_text or "Giới hạn hệ thống" in report_text
        # Mất tín hiệu giọng điệu / prosody
        assert "giọng điệu" in report_text.lower() or "prosody" in report_text.lower()
        # Lệch phân phối giữa SMS và transcript ASR
        assert "lệch phân phối" in report_text.lower() or "domain shift" in report_text.lower()
        assert "sms" in report_text.lower() and "asr" in report_text.lower()

    def test_contains_mandatory_future_work_section(self, report_text):
        """Báo cáo BẮT BUỘC có mục Hướng phát triển (bổ sung tốc độ nói speech-rate từ timestamps)."""
        assert "Hướng Phát Triển" in report_text or "Hướng phát triển" in report_text
        assert "tốc độ nói" in report_text.lower() or "speech-rate" in report_text.lower() or "speech rate" in report_text.lower()
        assert "faster-whisper" in report_text.lower() or "phowhisper" in report_text.lower() or "timestamp" in report_text.lower()

    def test_data_consistency_with_saved_results(self, report_text):
        """Kiểm tra tính nhất quán: các giá trị trong báo cáo khớp với results/ablation_summary.csv."""
        summary_csv = _REPO_ROOT / "results" / "ablation_summary.csv"
        if summary_csv.exists():
            df = pd.read_csv(summary_csv)
            for _, r in df.iterrows():
                mid = r["model_id"]
                acc_pct = f"{r['accuracy'] * 100:.2f}%"
                f1_pct = f"{r['macro_f1'] * 100:.2f}%"
                assert mid in report_text, f"Báo cáo thiếu mã mô hình {mid}"
                assert acc_pct in report_text, f"Báo cáo không khớp Accuracy của {mid}: {acc_pct}"
                assert f1_pct in report_text, f"Báo cáo không khớp Macro-F1 của {mid}: {f1_pct}"
