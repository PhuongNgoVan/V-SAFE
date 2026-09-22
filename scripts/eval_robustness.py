"""
scripts/eval_robustness.py
===========================
Đo độ bền của hệ thống phát hiện lừa đảo trước lỗi phiên âm ASR (Rủi ro R10).

Spec:     B4_robustness_and_voice_api.md §1–4
Hợp đồng: 00_SHARED_CONTRACT.md §1, §3, §4 (B sở hữu, gọi A qua process_and_predict)

Chức năng:
──────────
1. Đọc transcript_reference (đường chuẩn sạch) và transcript_asr từ các file:
   data/audio/transcripts_asr_<model>_<condition>.csv (clean, snr20, snr10, snr5).
2. Nạp pipeline phân loại qua process_and_predict() từ src.pipeline.predict
   (tự động fallback sang keyword/rule-based predictor nếu model.pt chưa sẵn sàng).
3. Tính Accuracy, Macro-F1, FNR, FPR cho transcript_reference và transcript_asr;
   tính độ suy giảm Δ (ASR − sạch).
4. Phân rã theo region × tone (3 miền × 2 sắc thái).
5. Phân tích các ca False Negative mới phát sinh (New FN: Sạch đúng nhưng ASR sai).
6. Mô phỏng đánh giá độ bền theo luỹ kế chunk 4s.

Đầu ra:
───────
- results/robustness_summary.csv
- results/robustness_by_region_tone.csv
- results/robustness_new_fn.csv
- results/robustness_chunked.csv
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import logging
from pathlib import Path
import subprocess
import sys
from typing import Callable, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# ---------------------------------------------------------------------------
# sys.path
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.audio.text_norm import normalize_for_wer
from scripts.keyword_asr_errors import load_scam_keywords, count_keyword_occurrences

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hằng số mặc định
# ---------------------------------------------------------------------------
_DEFAULT_DATA_DIR = _REPO_ROOT / "data" / "audio"
_DEFAULT_RESULTS_DIR = _REPO_ROOT / "results"
_DEFAULT_CONDITIONS = ["clean", "snr20", "snr10", "snr5"]
_DEFAULT_MODEL = "phowhisper-base"
_DEFAULT_SEED = 42
_SMALL_N_THRESHOLD = 30
_ROBUSTNESS_DELTA_THRESHOLD = 0.05  # Ngưỡng 5 điểm phần trăm theo spec B4 §4


# ===========================================================================
# Tiện ích hệ thống
# ===========================================================================

def get_git_commit() -> str:
    """Lấy git commit hash ngắn."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=str(_REPO_ROOT),
        )
        return res.stdout.strip()
    except Exception:
        return "unknown"


def get_timestamp() -> str:
    """Lấy timestamp ISO-8601 UTC."""
    return datetime.now(timezone.utc).isoformat()


# ===========================================================================
# Dự đoán văn bản qua process_and_predict hoặc Fallback
# ===========================================================================

def _fallback_keyword_predict(text: str, keywords_dict: dict[str, list[str]]) -> tuple[bool, float, str, Optional[str]]:
    """Dự đoán dựa trên từ khóa lừa đảo khi mô hình PyTorch chưa sẵn sàng."""
    norm_text = normalize_for_wer(text)
    if not norm_text:
        return False, 0.05, "low", None

    hit_categories: dict[str, int] = {}
    total_hits = 0
    for cat, kws in keywords_dict.items():
        cat_count = sum(count_keyword_occurrences(kw, norm_text) for kw in kws)
        if cat_count > 0:
            hit_categories[cat] = cat_count
            total_hits += cat_count

    if total_hits == 0:
        return False, 0.15, "low", None

    # Xác định scam_type từ category nổi trội
    top_cat = max(hit_categories.items(), key=lambda x: x[1])[0]
    scam_type_map = {
        "authority": "authority_impersonation",
        "financial_action": "bank_fraud",
        "reward": "prize_scam",
        "urgency": "other_fraud",
    }
    scam_type = scam_type_map.get(top_cat, "other_fraud")

    # Điểm confidence ước tính
    score = min(0.98, 0.50 + total_hits * 0.15)
    risk_level = "high" if score >= 0.7 else "medium"
    return True, score, risk_level, scam_type


def get_predictor() -> Callable[[str], tuple[bool, float, str, Optional[str]]]:
    """Trả về hàm dự đoán text → (is_fraud, score, risk_level, scam_type).

    Ưu tiên gọi process_and_predict từ src.pipeline.predict (A sở hữu).
    Nếu mô hình chưa được huấn luyện (chưa có model.pt), fallback sang rule-based.
    """
    try:
        from src.pipeline.predict import process_and_predict
        # Thử nghiệm với chuỗi rỗng để kiểm tra model đã sẵn sàng chưa
        test_pred = process_and_predict("")
        logger.info("✅ Đã kết nối thành công với src.pipeline.predict.process_and_predict (A3)")

        def _predict_wrapper(text: str) -> tuple[bool, float, str, Optional[str]]:
            p = process_and_predict(text)
            return p.is_fraud, p.confidence_score, p.risk_level, p.scam_type

        return _predict_wrapper
    except Exception as exc:
        logger.warning(
            "Chưa thể nạp model PyTorch từ src.pipeline.predict (%s). "
            "Kích hoạt bộ dự đoán Fallback Keyword-Based để chạy đánh giá độ bền.",
            exc,
        )
        kw_dict = load_scam_keywords()

        def _predict_fallback_wrapper(text: str) -> tuple[bool, float, str, Optional[str]]:
            return _fallback_keyword_predict(text, kw_dict)

        return _predict_fallback_wrapper


# ===========================================================================
# Tính toán chỉ số phân loại theo Hợp đồng §4
# ===========================================================================

def compute_classification_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    """Tính toán Accuracy, Macro-F1, FNR, FPR theo chuẩn Hợp đồng §4.

    Quy tắc:
    - Nhãn dương = 1 (lừa đảo).
    - FNR = FN / (FN + TP)
    - FPR = FP / (FP + TN)
    - Macro-F1 = sklearn.metrics.f1_score(y_true, y_pred, average="macro")
    """
    if not y_true or len(y_true) != len(y_pred):
        return {"accuracy": 0.0, "macro_f1": 0.0, "fnr": 0.0, "fpr": 0.0}

    y_t = np.array(y_true, dtype=int)
    y_p = np.array(y_pred, dtype=int)

    acc = float(accuracy_score(y_t, y_p))
    f1 = float(f1_score(y_t, y_p, average="macro", zero_division=0))

    # Tính ma trận nhầm lẫn
    tp = int(np.sum((y_t == 1) & (y_p == 1)))
    fn = int(np.sum((y_t == 1) & (y_p == 0)))
    fp = int(np.sum((y_t == 0) & (y_p == 1)))
    tn = int(np.sum((y_t == 0) & (y_p == 0)))

    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(f1, 4),
        "fnr": round(fnr, 4),
        "fpr": round(fpr, 4),
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
    }


# ===========================================================================
# B4a — Phân tích độ bền & trích xuất New FN
# ===========================================================================

def evaluate_robustness_for_dataframe(
    df: pd.DataFrame,
    condition: str,
    model: str = _DEFAULT_MODEL,
    predictor: Optional[Callable] = None,
    keywords_dict: Optional[dict[str, list[str]]] = None,
    git_commit: Optional[str] = None,
    timestamp: Optional[str] = None,
    seed: int = _DEFAULT_SEED,
) -> tuple[dict, pd.DataFrame, list[dict]]:
    """Đánh giá độ bền cho một DataFrame transcript (chứa transcript_reference và transcript_asr).

    Trả về:
    -------
    tuple[summary_dict, region_tone_df, new_fn_list]
    """
    pred_fn = predictor or get_predictor()
    kw_dict = keywords_dict or load_scam_keywords()
    commit_str = git_commit or get_git_commit()
    time_str = timestamp or get_timestamp()

    y_true: list[int] = [int(v) for v in df["label"]]
    preds_clean: list[int] = []
    preds_asr: list[int] = []
    scores_clean: list[float] = []
    scores_asr: list[float] = []

    for _, row in df.iterrows():
        ref_text = str(row.get("transcript_reference", ""))
        asr_text = str(row.get("transcript_asr", ""))

        is_fraud_c, score_c, _, _ = pred_fn(ref_text)
        is_fraud_a, score_a, _, _ = pred_fn(asr_text)

        preds_clean.append(int(is_fraud_c))
        preds_asr.append(int(is_fraud_a))
        scores_clean.append(score_c)
        scores_asr.append(score_a)

    df_eval = df.copy()
    df_eval["pred_clean"] = preds_clean
    df_eval["pred_asr"] = preds_asr
    df_eval["score_clean"] = scores_clean
    df_eval["score_asr"] = scores_asr

    # 1. Metrics tổng thể
    m_clean = compute_classification_metrics(y_true, preds_clean)
    m_asr = compute_classification_metrics(y_true, preds_asr)

    delta_acc = round(m_asr["accuracy"] - m_clean["accuracy"], 4)
    delta_f1 = round(m_asr["macro_f1"] - m_clean["macro_f1"], 4)
    delta_fnr = round(m_asr["fnr"] - m_clean["fnr"], 4)
    delta_fpr = round(m_asr["fpr"] - m_clean["fpr"], 4)

    # Quyết định ngưỡng theo spec B4 §4
    exceeds_threshold = (delta_acc < -_ROBUSTNESS_DELTA_THRESHOLD) or (delta_fnr > _ROBUSTNESS_DELTA_THRESHOLD)
    threshold_conclusion = "Đề xuất ASR error augmentation" if exceeds_threshold else "Không cần augmentation"

    summary_record = {
        "condition": condition,
        "model": model,
        "n_samples": len(df),
        "acc_clean": m_clean["accuracy"],
        "acc_asr": m_asr["accuracy"],
        "delta_acc": delta_acc,
        "f1_clean": m_clean["macro_f1"],
        "f1_asr": m_asr["macro_f1"],
        "delta_f1": delta_f1,
        "fnr_clean": m_clean["fnr"],
        "fnr_asr": m_asr["fnr"],
        "delta_fnr": delta_fnr,
        "fpr_clean": m_clean["fpr"],
        "fpr_asr": m_asr["fpr"],
        "delta_fpr": delta_fpr,
        "threshold_conclusion": threshold_conclusion,
        "git_commit": commit_str,
        "timestamp": time_str,
        "seed": seed,
    }

    # 2. Phân rã theo region × tone
    region_tone_records: list[dict] = []
    if "region" in df_eval.columns and "tone" in df_eval.columns:
        for (region, tone), grp in df_eval.groupby(["region", "tone"], observed=False):
            n_sub = len(grp)
            if n_sub == 0:
                continue
            sub_y = grp["label"].astype(int).tolist()
            sub_c = grp["pred_clean"].tolist()
            sub_a = grp["pred_asr"].tolist()

            sm_clean = compute_classification_metrics(sub_y, sub_c)
            sm_asr = compute_classification_metrics(sub_y, sub_a)

            note = "n nhỏ" if n_sub < _SMALL_N_THRESHOLD else ""
            region_tone_records.append({
                "condition": condition,
                "model": model,
                "region": str(region),
                "tone": str(tone),
                "n_samples": n_sub,
                "acc_clean": sm_clean["accuracy"],
                "acc_asr": sm_asr["accuracy"],
                "delta_acc": round(sm_asr["accuracy"] - sm_clean["accuracy"], 4),
                "f1_clean": sm_clean["macro_f1"],
                "f1_asr": sm_asr["macro_f1"],
                "delta_f1": round(sm_asr["macro_f1"] - sm_clean["macro_f1"], 4),
                "fnr_clean": sm_clean["fnr"],
                "fnr_asr": sm_asr["fnr"],
                "delta_fnr": round(sm_asr["fnr"] - sm_clean["fnr"], 4),
                "fpr_clean": sm_clean["fpr"],
                "fpr_asr": sm_asr["fpr"],
                "delta_fpr": round(sm_asr["fpr"] - sm_clean["fpr"], 4),
                "note": note,
                "git_commit": commit_str,
                "timestamp": time_str,
                "seed": seed,
            })

    # 3. Phân tích FN mới phát sinh do ASR (Sạch đúng: pred_clean == 1; ASR sai: pred_asr == 0 với nhãn thật == 1)
    new_fn_records: list[dict] = []
    for idx, row in df_eval.iterrows():
        is_true_fraud = (int(row["label"]) == 1)
        clean_correct = (row["pred_clean"] == 1)
        asr_missed = (row["pred_asr"] == 0)

        if is_true_fraud and clean_correct and asr_missed:
            ref_norm = normalize_for_wer(str(row.get("transcript_reference", "")))
            asr_norm = normalize_for_wer(str(row.get("transcript_asr", "")))

            # Tìm từ khoá bị mất hoặc bị biến dạng
            lost_kws: list[str] = []
            for cat, kws in kw_dict.items():
                for kw in kws:
                    c_ref = count_keyword_occurrences(kw, ref_norm)
                    c_asr = count_keyword_occurrences(kw, asr_norm)
                    if c_ref > c_asr:
                        lost_kws.append(f"{kw} ({cat}: {c_ref}→{c_asr})")

            lost_str = "; ".join(lost_kws) if lost_kws else "[ngữ nghĩa câu bị biến dạng]"
            new_fn_records.append({
                "audio_path": str(row.get("audio_path", f"sample_{idx}")),
                "condition": condition,
                "model": model,
                "region": str(row.get("region", "")),
                "tone": str(row.get("tone", "")),
                "speaker_id": str(row.get("speaker_id", "")),
                "label": 1,
                "pred_clean": 1,
                "score_clean": round(float(row["score_clean"]), 4),
                "pred_asr": 0,
                "score_asr": round(float(row["score_asr"]), 4),
                "lost_or_corrupted_keywords": lost_str,
                "transcript_reference": str(row.get("transcript_reference", "")),
                "transcript_asr": str(row.get("transcript_asr", "")),
                "git_commit": commit_str,
                "timestamp": time_str,
                "seed": seed,
            })

    return summary_record, pd.DataFrame(region_tone_records), new_fn_records


# ===========================================================================
# Mô phỏng độ bền theo Chunk 4s (Spec B4 §3)
# ===========================================================================

def simulate_chunked_robustness(
    df: pd.DataFrame,
    condition: str = "clean",
    model: str = _DEFAULT_MODEL,
    words_per_chunk: int = 10,  # ~10 từ mỗi chunk 4s theo tốc độ nói tiếng Việt
    predictor: Optional[Callable] = None,
    git_commit: Optional[str] = None,
    timestamp: Optional[str] = None,
    seed: int = _DEFAULT_SEED,
) -> pd.DataFrame:
    """Mô phỏng thời gian thực: phân rã phiên âm theo từng chunk 4s luỹ kế và đo độ ổn định."""
    if df.empty or "transcript_asr" not in df.columns:
        return pd.DataFrame()

    pred_fn = predictor or get_predictor()
    commit_str = git_commit or get_git_commit()
    time_str = timestamp or get_timestamp()

    chunk_rows: list[dict] = []

    for idx, row in df.iterrows():
        raw_text = str(row.get("transcript_asr", ""))
        audio_path = str(row.get("audio_path", f"sample_{idx}"))
        true_label = int(row.get("label", 0))

        words = raw_text.split()
        if not words:
            continue

        # Chia transcript thành các đoạn luỹ kế
        total_chunks = max(1, (len(words) + words_per_chunk - 1) // words_per_chunk)
        final_is_fraud, final_score, final_risk, final_type = pred_fn(raw_text)

        cum_words: list[str] = []
        for c_idx in range(1, total_chunks + 1):
            chunk_slice = words[:c_idx * words_per_chunk]
            cum_transcript = " ".join(chunk_slice)
            cum_seconds = round(c_idx * 4.0, 1)

            c_fraud, c_score, c_risk, c_type = pred_fn(cum_transcript)
            is_stable = (c_fraud == final_is_fraud)

            chunk_rows.append({
                "audio_path": audio_path,
                "condition": condition,
                "model": model,
                "true_label": true_label,
                "chunk_index": c_idx,
                "cumulative_seconds": cum_seconds,
                "is_fraud": c_fraud,
                "confidence_score": round(float(c_score), 4),
                "risk_level": c_risk,
                "scam_type": c_type,
                "is_stable": is_stable,
                "cumulative_transcript": cum_transcript,
                "git_commit": commit_str,
                "timestamp": time_str,
                "seed": seed,
            })

    return pd.DataFrame(chunk_rows)


# ===========================================================================
# CLI
# ===========================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Task B4a: Đánh giá độ bền trước lỗi ASR, phân rã theo region × tone và phát hiện New FN."
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=_DEFAULT_CONDITIONS,
        help="Danh sách các điều kiện âm thanh cần đánh giá (mặc định: clean, snr20, snr10, snr5).",
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=_DEFAULT_DATA_DIR,
        help="Thư mục chứa các file transcripts_asr_*.csv (mặc định: data/audio).",
    )
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=_DEFAULT_RESULTS_DIR,
        help="Thư mục lưu kết quả CSV (mặc định: results).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=_DEFAULT_MODEL,
        help="Tên mô hình ASR (mặc định: phowhisper-base).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=_DEFAULT_SEED,
        help="Seed tái lập (mặc định: 42).",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    args.results_dir.mkdir(parents=True, exist_ok=True)
    predictor = get_predictor()
    kw_dict = load_scam_keywords()
    commit_str = get_git_commit()
    time_str = get_timestamp()

    all_summaries: list[dict] = []
    all_region_tones: list[pd.DataFrame] = []
    all_new_fns: list[dict] = []
    all_chunks: list[pd.DataFrame] = []

    for cond in args.conditions:
        csv_file = args.data_dir / f"transcripts_asr_{args.model}_{cond}.csv"
        if not csv_file.exists():
            logger.warning(
                "Không tìm thấy file %s. Vui lòng chạy scripts/transcribe_dataset.py --condition %s trước.",
                csv_file.name, cond
            )
            continue

        logger.info("Đang đánh giá độ bền điều kiện: %s (file: %s)", cond, csv_file.name)
        try:
            df = pd.read_csv(csv_file)
        except Exception as exc:
            logger.error("Không đọc được %s: %s", csv_file, exc)
            continue

        summary_rec, reg_tone_df, new_fns = evaluate_robustness_for_dataframe(
            df=df,
            condition=cond,
            model=args.model,
            predictor=predictor,
            keywords_dict=kw_dict,
            git_commit=commit_str,
            timestamp=time_str,
            seed=args.seed,
        )
        all_summaries.append(summary_rec)
        all_region_tones.append(reg_tone_df)
        all_new_fns.extend(new_fns)

        # Mô phỏng chunk
        chunk_df = simulate_chunked_robustness(
            df=df,
            condition=cond,
            model=args.model,
            predictor=predictor,
            git_commit=commit_str,
            timestamp=time_str,
            seed=args.seed,
        )
        if not chunk_df.empty:
            all_chunks.append(chunk_df)

    # 1. Lưu robustness_summary.csv
    summary_path = args.results_dir / "robustness_summary.csv"
    if all_summaries:
        df_summary = pd.DataFrame(all_summaries)
        df_summary.to_csv(summary_path, index=False, encoding="utf-8")
        logger.info("✅ Đã lưu kết quả tổng quan vào: %s", summary_path)

        print("\n═════════════ TỔNG HỢP ĐỘ BỀN TRƯỚC LỖI ASR (R10) ═════════════")
        print(df_summary[["condition", "model", "acc_clean", "acc_asr", "delta_acc", "fnr_clean", "fnr_asr", "delta_fnr", "threshold_conclusion"]].to_string(index=False))

    # 2. Lưu robustness_by_region_tone.csv
    reg_path = args.results_dir / "robustness_by_region_tone.csv"
    if all_region_tones:
        df_reg = pd.concat(all_region_tones, ignore_index=True)
        df_reg.to_csv(reg_path, index=False, encoding="utf-8")
        logger.info("✅ Đã lưu kết quả theo region × tone vào: %s", reg_path)

    # 3. Lưu robustness_new_fn.csv
    fn_path = args.results_dir / "robustness_new_fn.csv"
    df_new_fn = pd.DataFrame(all_new_fns)
    df_new_fn.to_csv(fn_path, index=False, encoding="utf-8")
    logger.info("✅ Đã lưu %d ca New False Negative vào: %s", len(df_new_fn), fn_path)

    # 4. Lưu robustness_chunked.csv
    chunk_path = args.results_dir / "robustness_chunked.csv"
    if all_chunks:
        df_chunks = pd.concat(all_chunks, ignore_index=True)
        df_chunks.to_csv(chunk_path, index=False, encoding="utf-8")
        logger.info("✅ Đã lưu kết quả đánh giá theo chunk vào: %s", chunk_path)


if __name__ == "__main__":
    main()
