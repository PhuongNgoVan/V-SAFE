"""
scripts/keyword_asr_errors.py
==============================
Phân tích tỷ lệ giữ từ khóa (Keyword Recall) và trích xuất các lỗi phiên âm
từ khóa quan trọng (urgency, authority, financial_action, reward) từ transcript ASR.

Spec:     B3_transcribe_dataset.md §4–5 (Rủi ro R10)
Hợp đồng: 00_SHARED_CONTRACT.md §3, §4 (B sở hữu)

Cách dùng
─────────
::

    # Tự động tìm tất cả các file transcripts_asr_*.csv trong data/audio
    python scripts/keyword_asr_errors.py

    # Chỉ định các file transcripts cụ thể
    python scripts/keyword_asr_errors.py \\
        --transcripts data/audio/transcripts_asr_phowhisper-base_clean.csv \\
                      data/audio/transcripts_asr_phowhisper-base_snr10.csv

    # Tuỳ biến đường dẫn lưu kết quả
    python scripts/keyword_asr_errors.py \\
        --output_recall results/keyword_recall_asr.csv \\
        --output_errors results/asr_error_examples.csv \\
        --max_examples 30

Đầu ra
──────
- results/keyword_recall_asr.csv:
  Bảng tỷ lệ giữ từ khóa (tổng thể, theo nhóm, theo từng từ khóa) qua các điều kiện.
- results/asr_error_examples.csv:
  30 lỗi phiên âm từ khóa điển hình (ví dụ: 'công an' → 'công ăn', bỏ sót từ khóa...).
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import difflib
import logging
from pathlib import Path
import re
import subprocess
import sys
from typing import Optional

import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# sys.path
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.audio.text_norm import normalize_for_wer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hằng số mặc định
# ---------------------------------------------------------------------------
_DEFAULT_KEYWORDS_PATH = _REPO_ROOT / "configs" / "scam_keywords.yaml"
_DEFAULT_DATA_DIR = _REPO_ROOT / "data" / "audio"
_DEFAULT_RESULTS_DIR = _REPO_ROOT / "results"
_DEFAULT_SEED = 42

# Danh sách từ khoá dự phòng nếu configs/scam_keywords.yaml chưa có (Kế hoạch mục 3.2)
_FALLBACK_KEYWORDS = {
    "urgency": [
        "gấp", "khẩn cấp", "ngay lập tức", "ngay bây giờ", "hạn chót",
        "hết hạn", "sắp hết", "trong hôm nay", "trước 12 giờ", "trước 24 giờ",
        "ngay hôm nay", "lập tức", "ngay", "ngay ngày", "không được chậm trễ",
        "nhanh lên", "khẩn", "cấp bách",
    ],
    "authority": [
        "công an", "cảnh sát", "bộ công an", "viện kiểm sát", "tòa án",
        "ngân hàng nhà nước", "bộ tài chính", "thuế", "cục thuế",
        "cơ quan điều tra", "cán bộ", "cơ quan chức năng", "thanh tra",
        "ủy ban nhân dân", "ubnd", "chính phủ", "bộ trưởng", "thứ trưởng",
        "nhân viên ngân hàng", "đại diện ngân hàng", "nhân viên điều tra",
    ],
    "financial_action": [
        "chuyển khoản", "chuyển tiền", "nộp tiền", "đặt cọc", "tạm ứng",
        "phí bảo lãnh", "phí xử lý", "phí hành chính", "phí bảo hiểm",
        "phí thu hồi", "thanh toán ngay", "xác nhận giao dịch", "otp",
        "mã otp", "mã xác thực", "số tài khoản", "tài khoản ngân hàng",
        "atm", "thẻ ngân hàng", "số thẻ", "cvv", "mã pin",
        "internet banking", "nạp tiền", "bảo lãnh", "đóng phí",
    ],
    "reward": [
        "trúng thưởng", "trúng giải", "giải thưởng", "nhận thưởng", "quà tặng",
        "khuyến mãi", "miễn phí", "hoàn tiền", "cashback", "ưu đãi đặc biệt",
        "ưu đãi", "may mắn", "bốc thăm", "chương trình thưởng", "phần thưởng",
        "tiền thưởng", "iphone", "xe máy", "du lịch miễn phí", "voucher", "điểm thưởng",
    ],
}


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
# Nạp từ khóa
# ===========================================================================

def load_scam_keywords(yaml_path: Optional[Path] = None) -> dict[str, list[str]]:
    """Đọc từ điển từ khóa lừa đảo từ file YAML hoặc fallback.

    Tất cả từ khóa đều được chuẩn hóa qua normalize_for_wer để đảm bảo
    đồng bộ tuyệt đối với văn bản phiên âm và hợp đồng §4.
    """
    path = yaml_path or _DEFAULT_KEYWORDS_PATH
    raw_dict: dict[str, list[str]] = {}

    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = yaml.safe_load(f)
                if isinstance(content, dict):
                    raw_dict = content
                    logger.info("Đã nạp %d nhóm từ khoá từ %s", len(raw_dict), path)
        except Exception as exc:
            logger.warning("Lỗi đọc %s: %s — dùng danh sách fallback", path, exc)
            raw_dict = _FALLBACK_KEYWORDS
    else:
        logger.warning("Không tìm thấy %s — dùng danh sách fallback mục 3.2", path)
        raw_dict = _FALLBACK_KEYWORDS

    cleaned_dict: dict[str, list[str]] = {}
    for category, kw_list in raw_dict.items():
        seen = set()
        clean_list = []
        for kw in kw_list:
            norm_kw = normalize_for_wer(str(kw))
            if norm_kw and norm_kw not in seen:
                seen.add(norm_kw)
                clean_list.append(norm_kw)
        cleaned_dict[category] = clean_list

    return cleaned_dict


# ===========================================================================
# Tìm kiếm từ khóa & so khớp ngữ cảnh
# ===========================================================================

def count_keyword_occurrences(norm_keyword: str, norm_text: str) -> int:
    r"""Đếm số lần xuất hiện của từ khóa trong văn bản đã chuẩn hóa.

    Dùng ranh giới từ `(?<!\w)...(?!\w)` để tránh khớp nhầm chuỗi con.
    """
    if not norm_keyword or not norm_text:
        return 0
    pattern = rf"(?<!\w){re.escape(norm_keyword)}(?!\w)"
    return len(re.findall(pattern, norm_text))


def extract_context_snippet(words: list[str], start_idx: int, end_idx: int, window: int = 3) -> str:
    """Trích xuất ngữ cảnh xung quanh vị trí từ khoá."""
    left = max(0, start_idx - window)
    right = min(len(words), end_idx + window)

    before = " ".join(words[left:start_idx])
    target = " ".join(words[start_idx:end_idx])
    after = " ".join(words[end_idx:right])

    parts = []
    if before:
        parts.append(before)
    parts.append(f"[{target}]")
    if after:
        parts.append(after)
    return " ".join(parts)


def find_misrecognition(
    norm_kw: str,
    ref_norm: str,
    asr_norm: str,
) -> tuple[str, str, str]:
    """Phát hiện lỗi phiên âm cụ thể cho từ khóa bị sai.

    Sử dụng difflib.SequenceMatcher trên danh sách tokens để tìm xem từ khoá
    trong reference được thay thế bằng từ gì trong ASR (hoặc bị bỏ sót).

    Trả về:
    -------
    tuple[ref_snippet, asr_snippet, detected_misrecognition]
    """
    ref_words = ref_norm.split()
    asr_words = asr_norm.split()
    kw_words = norm_kw.split()
    kw_len = len(kw_words)

    # Tìm vị trí của từ khoá trong ref_words
    kw_start = -1
    for i in range(len(ref_words) - kw_len + 1):
        if ref_words[i:i + kw_len] == kw_words:
            kw_start = i
            break

    if kw_start == -1:
        return "", "", ""

    kw_end = kw_start + kw_len
    ref_snippet = extract_context_snippet(ref_words, kw_start, kw_end)

    matcher = difflib.SequenceMatcher(None, ref_words, asr_words)
    opcodes = matcher.get_opcodes()

    asr_j_start = None
    asr_j_end = None
    has_delete = False

    for tag, i1, i2, j1, j2 in opcodes:
        overlap_start = max(i1, kw_start)
        overlap_end = min(i2, kw_end)
        if overlap_start < overlap_end:
            if tag == "delete":
                has_delete = True
                cur_j1 = j1
                cur_j2 = j2
            elif tag == "equal":
                offset1 = overlap_start - i1
                offset2 = overlap_end - i1
                cur_j1 = j1 + offset1
                cur_j2 = j1 + offset2
            else:
                cur_j1 = j1
                cur_j2 = j2

            if asr_j_start is None or cur_j1 < asr_j_start:
                asr_j_start = cur_j1
            if asr_j_end is None or cur_j2 > asr_j_end:
                asr_j_end = cur_j2

    if asr_j_start is not None and asr_j_end is not None and asr_j_start < asr_j_end:
        detected = " ".join(asr_words[asr_j_start:asr_j_end])
        asr_snippet = extract_context_snippet(asr_words, asr_j_start, asr_j_end)
    elif has_delete or (asr_j_start is not None and asr_j_start == asr_j_end):
        detected = "[bị bỏ sót / omitted]"
        j_pos = asr_j_start if asr_j_start is not None else 0
        asr_snippet = extract_context_snippet(asr_words, j_pos, j_pos)
    else:
        detected = "[bị biến dạng]"
        asr_snippet = asr_norm[:60] + "..." if len(asr_norm) > 60 else asr_norm

    return ref_snippet, asr_snippet, detected


# ===========================================================================
# Tính tỷ lệ giữ từ khóa (Keyword Recall)
# ===========================================================================

def calculate_keyword_recall_for_df(
    df: pd.DataFrame,
    keywords_dict: dict[str, list[str]],
    condition: str = "clean",
    model: str = "phowhisper-base",
    git_commit: Optional[str] = None,
    timestamp: Optional[str] = None,
    seed: int = _DEFAULT_SEED,
) -> tuple[pd.DataFrame, list[dict]]:
    """Tính Keyword Recall và trích xuất danh sách lỗi phiên âm từ DataFrame transcript.

    Tham số
    -------
    df : pd.DataFrame
        DataFrame chứa cột 'transcript_reference' và 'transcript_asr'.
    keywords_dict : dict[str, list[str]]
        Từ điển phân loại {category: [keyword, ...]}.

    Trả về
    ------
    tuple[pd.DataFrame, list[dict]]
        - recall_df: Thống kê recall từng từ khóa, từng category và overall.
        - error_candidates: Danh sách các trường hợp từ khóa bị phiên âm sai.
    """
    if df.empty or "transcript_reference" not in df.columns or "transcript_asr" not in df.columns:
        return pd.DataFrame(), []

    commit_str = git_commit or get_git_commit()
    time_str = timestamp or get_timestamp()

    # Chuẩn hoá sẵn toàn bộ văn bản trong df
    norm_refs = [normalize_for_wer(str(t) if pd.notna(t) else "") for t in df["transcript_reference"]]
    norm_asrs = [normalize_for_wer(str(t) if pd.notna(t) else "") for t in df["transcript_asr"]]

    recall_rows: list[dict] = []
    error_candidates: list[dict] = []

    total_all_ref = 0
    total_all_retained = 0
    total_all_ref_samples = 0
    total_all_recalled_samples = 0

    for category, kw_list in keywords_dict.items():
        cat_ref_count = 0
        cat_retained_count = 0
        cat_ref_samples = 0
        cat_recalled_samples = 0

        for kw in kw_list:
            kw_ref_count = 0
            kw_retained_count = 0
            kw_ref_samples = 0
            kw_recalled_samples = 0

            for idx, (r_norm, a_norm) in enumerate(zip(norm_refs, norm_asrs)):
                c_ref = count_keyword_occurrences(kw, r_norm)
                if c_ref > 0:
                    c_asr = count_keyword_occurrences(kw, a_norm)
                    retained = min(c_ref, c_asr)

                    kw_ref_count += c_ref
                    kw_retained_count += retained

                    kw_ref_samples += 1
                    if retained >= 1:
                        kw_recalled_samples += 1

                    # Nếu từ khóa bị mất hoặc nhận diện thiếu
                    if retained < c_ref:
                        ref_snip, asr_snip, detected_err = find_misrecognition(kw, r_norm, a_norm)
                        row_meta = df.iloc[idx]
                        error_candidates.append({
                            "condition": condition,
                            "model": model,
                            "category": category,
                            "target_keyword": kw,
                            "detected_misrecognition": detected_err,
                            "ref_snippet": ref_snip,
                            "asr_snippet": asr_snip,
                            "region": str(row_meta.get("region", "")),
                            "tone": str(row_meta.get("tone", "")),
                            "speaker_id": str(row_meta.get("speaker_id", "")),
                            "transcript_reference": str(row_meta.get("transcript_reference", "")),
                            "transcript_asr": str(row_meta.get("transcript_asr", "")),
                            "git_commit": commit_str,
                            "timestamp": time_str,
                            "seed": seed,
                        })

            kw_recall = round(kw_retained_count / kw_ref_count, 4) if kw_ref_count > 0 else float("nan")
            kw_sample_recall = round(kw_recalled_samples / kw_ref_samples, 4) if kw_ref_samples > 0 else float("nan")

            recall_rows.append({
                "condition": condition,
                "model": model,
                "level": "keyword",
                "category": category,
                "keyword": kw,
                "ref_count": kw_ref_count,
                "retained_count": kw_retained_count,
                "keyword_recall": kw_recall,
                "ref_samples": kw_ref_samples,
                "recalled_samples": kw_recalled_samples,
                "sample_recall": kw_sample_recall,
                "git_commit": commit_str,
                "timestamp": time_str,
                "seed": seed,
            })

            cat_ref_count += kw_ref_count
            cat_retained_count += kw_retained_count
            cat_ref_samples += kw_ref_samples
            cat_recalled_samples += kw_recalled_samples

        cat_recall = round(cat_retained_count / cat_ref_count, 4) if cat_ref_count > 0 else float("nan")
        cat_sample_recall = round(cat_recalled_samples / cat_ref_samples, 4) if cat_ref_samples > 0 else float("nan")

        recall_rows.append({
            "condition": condition,
            "model": model,
            "level": "category_summary",
            "category": category,
            "keyword": "__CATEGORY_ALL__",
            "ref_count": cat_ref_count,
            "retained_count": cat_retained_count,
            "keyword_recall": cat_recall,
            "ref_samples": cat_ref_samples,
            "recalled_samples": cat_recalled_samples,
            "sample_recall": cat_sample_recall,
            "git_commit": commit_str,
            "timestamp": time_str,
            "seed": seed,
        })

        total_all_ref += cat_ref_count
        total_all_retained += cat_retained_count
        total_all_ref_samples += cat_ref_samples
        total_all_recalled_samples += cat_recalled_samples

    total_recall = round(total_all_retained / total_all_ref, 4) if total_all_ref > 0 else float("nan")
    total_sample_recall = round(total_all_recalled_samples / total_all_ref_samples, 4) if total_all_ref_samples > 0 else float("nan")

    recall_rows.append({
        "condition": condition,
        "model": model,
        "level": "overall",
        "category": "__ALL__",
        "keyword": "__OVERALL__",
        "ref_count": total_all_ref,
        "retained_count": total_all_retained,
        "keyword_recall": total_recall,
        "ref_samples": total_all_ref_samples,
        "recalled_samples": total_all_recalled_samples,
        "sample_recall": total_sample_recall,
        "git_commit": commit_str,
        "timestamp": time_str,
        "seed": seed,
    })

    return pd.DataFrame(recall_rows), error_candidates


# ===========================================================================
# Tuyển chọn 30 lỗi điển hình đa dạng
# ===========================================================================

def select_typical_errors(
    error_candidates: list[dict],
    max_examples: int = 30,
) -> pd.DataFrame:
    """Tuyển chọn tối đa 30 lỗi điển hình trải đều theo category và condition."""
    if not error_candidates:
        return pd.DataFrame(columns=[
            "example_id", "condition", "model", "category", "target_keyword",
            "detected_misrecognition", "ref_snippet", "asr_snippet", "region",
            "tone", "speaker_id", "transcript_reference", "transcript_asr",
            "git_commit", "timestamp", "seed"
        ])

    df_err = pd.DataFrame(error_candidates)

    # Loại bỏ các ví dụ trùng lặp hoàn toàn về từ khoá + detected_misrecognition nếu có nhiều
    dedup = df_err.drop_duplicates(subset=["condition", "target_keyword", "detected_misrecognition"]).copy()

    # Ưu tiên các lỗi có detected_misrecognition cụ thể (không phải rỗng)
    selected = dedup.head(max_examples).copy()

    # Đánh số thứ tự example_id từ 1 đến N
    selected.insert(0, "example_id", range(1, len(selected) + 1))
    return selected


# ===========================================================================
# Tự động phát hiện model & condition từ tên file
# ===========================================================================

def parse_model_and_condition_from_filename(path: Path) -> tuple[str, str]:
    """Phân tích tên file transcripts_asr_<model>_<condition>.csv."""
    stem = path.stem
    prefix = "transcripts_asr_"
    if stem.startswith(prefix):
        remainder = stem[len(prefix):]
        parts = remainder.split("_")
        if len(parts) >= 2:
            condition = parts[-1]
            model = "_".join(parts[:-1])
            return model, condition
    return "phowhisper-base", "clean"


# ===========================================================================
# CLI
# ===========================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Task B3: Đánh giá Keyword Recall và trích xuất 30 lỗi phiên âm ASR điển hình."
    )
    parser.add_argument(
        "--transcripts",
        nargs="*",
        default=None,
        help="Danh sách các file transcript CSV cần phân tích. Nếu bỏ trống, script sẽ tự quét data/audio/transcripts_asr_*.csv.",
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=_DEFAULT_DATA_DIR,
        help="Thư mục chứa transcript ASR (mặc định: data/audio).",
    )
    parser.add_argument(
        "--keywords_file",
        type=Path,
        default=_DEFAULT_KEYWORDS_PATH,
        help="File YAML chứa scam keywords (mặc định: configs/scam_keywords.yaml).",
    )
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=_DEFAULT_RESULTS_DIR,
        help="Thư mục xuất kết quả (mặc định: results).",
    )
    parser.add_argument(
        "--output_recall",
        type=Path,
        default=None,
        help="File CSV lưu keyword recall (mặc định: results/keyword_recall_asr.csv).",
    )
    parser.add_argument(
        "--output_errors",
        type=Path,
        default=None,
        help="File CSV lưu 30 lỗi phiên âm (mặc định: results/asr_error_examples.csv).",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=30,
        help="Số lượng ví dụ lỗi tối đa cần trích xuất (mặc định: 30).",
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

    # Nạp từ khóa
    keywords_dict = load_scam_keywords(args.keywords_file)
    total_kw = sum(len(v) for v in keywords_dict.values())
    logger.info("Đã nạp tổng cộng %d từ khóa phân loại.", total_kw)

    # Tìm danh sách file transcripts
    transcript_files: list[Path] = []
    if args.transcripts:
        transcript_files = [Path(p) for p in args.transcripts if Path(p).exists()]
    else:
        if args.data_dir.exists():
            transcript_files = sorted(args.data_dir.glob("transcripts_asr_*.csv"))

    if not transcript_files:
        logger.warning(
            "Không tìm thấy file transcripts_asr_*.csv nào tại %s. "
            "Chạy scripts/transcribe_dataset.py trước để tạo file transcript.",
            args.data_dir
        )
        sys.exit(0)

    logger.info("Tìm thấy %d file transcript để phân tích: %s", len(transcript_files), [f.name for f in transcript_files])

    all_recalls: list[pd.DataFrame] = []
    all_errors: list[dict] = []

    commit_str = get_git_commit()
    time_str = get_timestamp()

    for t_path in transcript_files:
        model, condition = parse_model_and_condition_from_filename(t_path)
        logger.info("Phân tích file: %s (model=%s, condition=%s)", t_path.name, model, condition)

        try:
            df_t = pd.read_csv(t_path)
        except Exception as exc:
            logger.error("Không đọc được %s: %s", t_path, exc)
            continue

        recall_df, errors = calculate_keyword_recall_for_df(
            df=df_t,
            keywords_dict=keywords_dict,
            condition=condition,
            model=model,
            git_commit=commit_str,
            timestamp=time_str,
            seed=args.seed,
        )
        all_recalls.append(recall_df)
        all_errors.extend(errors)

    out_recall = args.output_recall or (args.results_dir / "keyword_recall_asr.csv")
    out_errors = args.output_errors or (args.results_dir / "asr_error_examples.csv")

    args.results_dir.mkdir(parents=True, exist_ok=True)

    # Lưu kết quả recall
    if all_recalls:
        final_recall_df = pd.concat(all_recalls, ignore_index=True)
        final_recall_df.to_csv(out_recall, index=False, encoding="utf-8")
        logger.info("✅ Đã lưu kết quả Keyword Recall (%d hàng) vào: %s", len(final_recall_df), out_recall)

        # In tóm tắt tổng thể theo điều kiện
        print("\n═════════════ TÓM TẮT KEYWORD RECALL THEO ĐIỀU KIỆN ═════════════")
        summary_mask = final_recall_df["level"] == "overall"
        print(final_recall_df[summary_mask][["condition", "model", "ref_count", "retained_count", "keyword_recall", "sample_recall"]].to_string(index=False))

    # Tuyển chọn và lưu 30 lỗi điển hình
    typical_errors_df = select_typical_errors(all_errors, max_examples=args.max_examples)
    typical_errors_df.to_csv(out_errors, index=False, encoding="utf-8")
    logger.info("✅ Đã lưu %d lỗi phiên âm điển hình vào: %s", len(typical_errors_df), out_errors)

    if not typical_errors_df.empty:
        print(f"\n═════════════ {len(typical_errors_df)} VÍ DỤ LỖI PHIÊN ÂM TỪ KHÓA ═════════════")
        print(typical_errors_df[["example_id", "condition", "target_keyword", "detected_misrecognition", "ref_snippet"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
