"""
scripts/transcribe_dataset.py
==============================
Sinh transcript ASR cho toàn bộ tập thu âm đã gán nhãn, đo độ trễ,
tính WER từng file và tổng hợp theo phân nhóm region × tone.

Spec:     B3_transcribe_dataset.md §1–3
Hợp đồng: 00_SHARED_CONTRACT.md §2, §3, §4 (B sở hữu)

Cách dùng
─────────
::

    # Phiên âm tập sạch với PhoWhisper (mặc định)
    python scripts/transcribe_dataset.py --model phowhisper-base --condition clean

    # Phiên âm với nhiễu viễn thông (SNR 20 dB, 10 dB, 5 dB)
    python scripts/transcribe_dataset.py --model phowhisper-base --condition snr20
    python scripts/transcribe_dataset.py --model phowhisper-base --condition snr10
    python scripts/transcribe_dataset.py --model phowhisper-base --condition snr5

    # Chạy mô hình whisper-base gốc
    python scripts/transcribe_dataset.py --model whisper-base --condition clean

Đầu ra
──────
- data/audio/transcripts_asr_<model>_<condition>.csv:
  gồm metadata + transcript_asr + asr_latency_ms + audio_seconds + wer
- results/wer_region_tone_<condition>.csv:
  tổng hợp theo region × tone (mean, median, n, git_commit, seed, timestamp...)
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import logging
from pathlib import Path
import subprocess
import sys
import time
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# sys.path
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.audio.preprocess import load_and_resample, make_noisy
from src.audio.text_norm import normalize_for_wer
import src.audio.asr as asr_module

try:
    import jiwer
    _JIWER_AVAILABLE = True
except ImportError:
    _JIWER_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hằng số mặc định theo hợp đồng và spec
# ---------------------------------------------------------------------------
_DEFAULT_SEED = 42
_DEFAULT_DATA_DIR = _REPO_ROOT / "data" / "audio"
_DEFAULT_RESULTS_DIR = _REPO_ROOT / "results"
_SMALL_N_THRESHOLD = 30

_CONDITION_SNR_MAP = {
    "clean": None,
    "snr20": 20,
    "snr10": 10,
    "snr5": 5,
}

_REQUIRED_METADATA_COLS = {
    "audio_path",
    "transcript_reference",
    "label",
    "scam_type",
    "region",
    "tone",
    "speaker_id",
}


# ===========================================================================
# Tiện ích hệ thống & metadata
# ===========================================================================

def get_git_commit() -> str:
    """Lấy git commit hash ngắn của commit hiện tại."""
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
    """Lấy timestamp ISO-8601 theo chuẩn UTC."""
    return datetime.now(timezone.utc).isoformat()


def resolve_audio_path(path_str: str, data_dir: Path) -> Path:
    """Tìm đường dẫn file âm thanh thực tế trên đĩa."""
    p = Path(path_str)
    if p.is_absolute() and p.exists():
        return p
    if (data_dir / p).exists():
        return data_dir / p
    if (_REPO_ROOT / p).exists():
        return _REPO_ROOT / p
    return data_dir / p


# ===========================================================================
# Tính WER
# ===========================================================================

def compute_row_wer(ref: str | float | None, hyp: str | float | None) -> float:
    """Tính WER cho một hàng bằng jiwer sau khi chuẩn hoá bằng normalize_for_wer.

    Quy tắc theo hợp đồng 00_SHARED_CONTRACT.md §4:
    - Cả hai vế được chuẩn hoá qua normalize_for_wer: chữ thường, bỏ dấu câu,
      gộp khoảng trắng thừa, GIỮ NGUYÊN dấu thanh tiếng Việt.
    - Xử lý biên:
      * ref rỗng & hyp rỗng → 0.0
      * ref rỗng & hyp có chữ → 1.0
      * ref có chữ & hyp rỗng → 1.0 (toàn bộ deletions)
    """
    if not _JIWER_AVAILABLE:
        raise ImportError("jiwer chưa cài đặt. Cài bằng: pip install jiwer")

    ref_str = "" if pd.isna(ref) else str(ref)
    hyp_str = "" if pd.isna(hyp) else str(hyp)

    ref_clean = normalize_for_wer(ref_str)
    hyp_clean = normalize_for_wer(hyp_str)

    if not ref_clean:
        return 0.0 if not hyp_clean else 1.0
    if not hyp_clean:
        return 1.0

    wer_val = float(jiwer.wer(ref_clean, hyp_clean))
    return round(wer_val, 4)


# ===========================================================================
# Tổng hợp WER theo region × tone
# ===========================================================================

def aggregate_wer_by_region_tone(
    df: pd.DataFrame,
    model: str,
    condition: str,
    compute_type: str = "int8",
    seed: int = _DEFAULT_SEED,
    git_commit: Optional[str] = None,
    timestamp: Optional[str] = None,
) -> pd.DataFrame:
    """Tổng hợp WER theo region × tone (mean, median, n) kèm thông tin nghiệm thu.

    Trả về DataFrame gồm:
    - region, tone
    - wer_mean, wer_median, n, note ("n nhỏ" nếu n < 30)
    - model, condition, compute_type, seed, git_commit, timestamp
    """
    if df.empty or "wer" not in df.columns or "region" not in df.columns or "tone" not in df.columns:
        return pd.DataFrame(columns=[
            "region", "tone", "wer_mean", "wer_median", "n", "note",
            "model", "condition", "compute_type", "seed", "git_commit", "timestamp"
        ])

    commit_str = git_commit if git_commit is not None else get_git_commit()
    time_str = timestamp if timestamp is not None else get_timestamp()

    rows: list[dict] = []
    # Nhóm theo region và tone
    for (region, tone), group in df.groupby(["region", "tone"], observed=False):
        wers = group["wer"].dropna().tolist()
        n = len(wers)
        mean_val = float(np.mean(wers)) if n > 0 else float("nan")
        median_val = float(np.median(wers)) if n > 0 else float("nan")
        note = "n nhỏ" if n < _SMALL_N_THRESHOLD else ""

        rows.append({
            "region": str(region),
            "tone": str(tone),
            "wer_mean": round(mean_val, 4),
            "wer_median": round(median_val, 4),
            "n": n,
            "note": note,
            "model": model,
            "condition": condition,
            "compute_type": compute_type,
            "seed": seed,
            "git_commit": commit_str,
            "timestamp": time_str,
        })

    return pd.DataFrame(rows)


# ===========================================================================
# Cấu hình Model ASR
# ===========================================================================

def configure_asr_model(
    model_name: str,
    device: str = "cpu",
    compute_type: str = "int8",
) -> None:
    """Cấu hình model trong src.audio.asr singleton trước khi phiên âm.

    Hỗ trợ cả phowhisper-base (CTranslate2) và whisper-base gốc.
    """
    if model_name == "phowhisper-base":
        # Dùng model_path từ configs/asr.yaml hoặc models/asr/phowhisper-base-ct2
        ct2_dir = _REPO_ROOT / "models" / "asr" / "phowhisper-base-ct2"
        if ct2_dir.exists() and not asr_module._STATE._loaded:
            asr_module._STATE.load()
    elif model_name == "whisper-base":
        from faster_whisper import WhisperModel
        logger.info("Nạp whisper-base gốc (faster-whisper) device=%s", device)
        whisper_model = WhisperModel("base", device=device, compute_type="float32")
        asr_module._STATE._loaded = True
        asr_module._STATE.model = whisper_model
        asr_module._STATE.cfg = {
            "language": "vi",
            "beam_size": 5,
            "best_of": 5,
            "temperature": 0.0,
            "vad_filter": True,
            "min_silence_duration_ms": 500,
            "condition_on_previous_text": False,
            "device": device,
            "compute_type": "float32",
        }
    else:
        logger.info("Model tùy chọn: %s", model_name)


# ===========================================================================
# Pipeline phiên âm dataset chính
# ===========================================================================

def transcribe_audio_dataset(
    metadata_df: pd.DataFrame,
    data_dir: Path,
    model_name: str = "phowhisper-base",
    condition: str = "clean",
    seed: int = _DEFAULT_SEED,
    compute_type: str = "int8",
    batch_limit: Optional[int] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Thực thi phiên âm cho toàn bộ tập dữ liệu trong metadata_df.

    Tham số
    -------
    metadata_df : pd.DataFrame
        Dữ liệu đọc từ metadata.csv.
    data_dir : Path
        Thư mục chứa file audio.
    model_name : str
        Tên mô hình ('phowhisper-base', 'whisper-base'...).
    condition : str
        'clean', 'snr20', 'snr10', 'snr5'.
    seed : int
        Seed cho make_noisy để đảm bảo tính tái lập.
    compute_type : str
        Định dạng tính toán (int8, float32...).
    batch_limit : int, optional
        Giới hạn số file xử lý (dùng khi test).

    Trả về
    ------
    tuple[pd.DataFrame, pd.DataFrame]
        (transcripts_df, wer_summary_df)
    """
    if metadata_df.empty:
        logger.warning("metadata_df rỗng, trả về kết quả rỗng.")
        return pd.DataFrame(), pd.DataFrame()

    df = metadata_df.copy()
    if batch_limit is not None and batch_limit > 0:
        df = df.iloc[:batch_limit].copy()

    # Kiểm tra cột bắt buộc
    missing_cols = _REQUIRED_METADATA_COLS - set(df.columns)
    if missing_cols:
        raise ValueError(f"metadata thiếu cột bắt buộc theo hợp đồng: {missing_cols}")

    transcripts_asr: list[str] = []
    asr_latencies_ms: list[float] = []
    audio_durations_sec: list[float] = []
    wer_list: list[float] = []

    snr_db = _CONDITION_SNR_MAP.get(condition, None)
    if snr_db is None and condition != "clean":
        try:
            snr_db = int(condition.replace("snr", "").replace("db", "").replace("noisy_", ""))
        except ValueError:
            snr_db = 10

    total_files = len(df)
    logger.info(
        "Bắt đầu phiên âm dataset: n=%d, model=%s, condition=%s, snr_db=%s",
        total_files, model_name, condition, snr_db
    )

    for idx, row in df.iterrows():
        raw_path = str(row["audio_path"])
        audio_file = resolve_audio_path(raw_path, data_dir)
        ref_text = str(row.get("transcript_reference", ""))

        if not audio_file.exists():
            logger.warning("Không tìm thấy file audio: %s (row=%d)", audio_file, idx)
            transcripts_asr.append("")
            asr_latencies_ms.append(0.0)
            audio_durations_sec.append(0.0)
            wer_list.append(1.0 if ref_text.strip() else 0.0)
            continue

        try:
            # 1. Nạp audio qua load_and_resample (16kHz mono float32)
            y, sr = load_and_resample(str(audio_file), target_sr=16000)

            # 2. Áp dụng make_noisy nếu condition != 'clean'
            if condition != "clean" and snr_db is not None:
                # Seed cố định + idx để mỗi file có sample noise khác nhau nhưng tái lập
                file_seed = seed + int(idx)
                y = make_noisy(y, sr=sr, snr_db=snr_db, seed=file_seed)

            audio_sec = round(len(y) / float(sr), 3)

            # 3. Phiên âm cả file qua asr.transcribe_chunk (đo asr_latency_ms)
            t_start = time.perf_counter()
            asr_text = asr_module.transcribe_chunk(y, sr=sr)
            latency_ms = round((time.perf_counter() - t_start) * 1000.0, 2)

            # 4. Tính WER từng file
            wer_val = compute_row_wer(ref_text, asr_text)

            transcripts_asr.append(asr_text)
            asr_latencies_ms.append(latency_ms)
            audio_durations_sec.append(audio_sec)
            wer_list.append(wer_val)

        except Exception as exc:
            logger.error("Lỗi khi xử lý file %s: %s", audio_file.name, exc, exc_info=True)
            transcripts_asr.append("")
            asr_latencies_ms.append(0.0)
            audio_durations_sec.append(0.0)
            wer_list.append(1.0 if ref_text.strip() else 0.0)

        if (len(transcripts_asr)) % 20 == 0 or len(transcripts_asr) == total_files:
            logger.info("  Đã phiên âm %d/%d file", len(transcripts_asr), total_files)

    df["transcript_asr"] = transcripts_asr
    df["asr_latency_ms"] = asr_latencies_ms
    df["audio_seconds"] = audio_durations_sec
    df["wer"] = wer_list

    # Tổng hợp theo region × tone
    wer_summary_df = aggregate_wer_by_region_tone(
        df=df,
        model=model_name,
        condition=condition,
        compute_type=compute_type,
        seed=seed,
    )

    return df, wer_summary_df


# ===========================================================================
# CLI
# ===========================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Task B3: Phiên âm tập dữ liệu audio, đo độ trễ và tính WER theo region × tone."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="phowhisper-base",
        help="Tên mô hình ASR (mặc định: phowhisper-base, hỗ trợ whisper-base).",
    )
    parser.add_argument(
        "--condition",
        type=str,
        default="clean",
        choices=["clean", "snr20", "snr10", "snr5"],
        help="Điều kiện âm thanh: clean, snr20, snr10, snr5 (mặc định: clean).",
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=_DEFAULT_DATA_DIR,
        help="Thư mục chứa metadata.csv và file audio (mặc định: data/audio).",
    )
    parser.add_argument(
        "--metadata_file",
        type=Path,
        default=None,
        help="Đường dẫn trực tiếp file metadata.csv (nếu khác {data_dir}/metadata.csv).",
    )
    parser.add_argument(
        "--output_file",
        type=Path,
        default=None,
        help="Đường dẫn file lưu transcripts (mặc định: data/audio/transcripts_asr_<model>_<condition>.csv).",
    )
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=_DEFAULT_RESULTS_DIR,
        help="Thư mục lưu kết quả CSV (mặc định: results).",
    )
    parser.add_argument(
        "--wer_output_file",
        type=Path,
        default=None,
        help="Đường dẫn lưu bảng WER theo region x tone (mặc định: results/wer_region_tone_<condition>.csv).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=_DEFAULT_SEED,
        help="Seed ngẫu nhiên cho make_noisy và tái lập kết quả (mặc định: 42).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device chạy mô hình: 'cpu' hoặc 'cuda' (mặc định: cpu).",
    )
    parser.add_argument(
        "--compute_type",
        type=str,
        default="int8",
        help="Kiểu số thực thi CTranslate2: int8, float32, float16 (mặc định: int8).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Giới hạn số file xử lý (tiện ích cho debug/kiểm tra nhanh).",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    meta_path = args.metadata_file if args.metadata_file else (args.data_dir / "metadata.csv")
    if not meta_path.exists():
        logger.error("Không tìm thấy file metadata tại: %s", meta_path)
        logger.info("Vui lòng chuẩn bị file metadata.csv theo hợp đồng 00_SHARED_CONTRACT.md §3.")
        sys.exit(1)

    try:
        metadata_df = pd.read_csv(meta_path)
    except Exception as exc:
        logger.error("Không đọc được metadata từ %s: %s", meta_path, exc)
        sys.exit(1)

    logger.info("Đọc được %d dòng từ %s", len(metadata_df), meta_path)

    # Cấu hình model
    try:
        configure_asr_model(args.model, device=args.device, compute_type=args.compute_type)
    except Exception as exc:
        logger.warning("Không thể nạp trước model %s: %s (sẽ thử khi transcribe)", args.model, exc)

    # Chạy pipeline phiên âm
    transcripts_df, wer_summary_df = transcribe_audio_dataset(
        metadata_df=metadata_df,
        data_dir=args.data_dir,
        model_name=args.model,
        condition=args.condition,
        seed=args.seed,
        compute_type=args.compute_type,
        batch_limit=args.limit,
    )

    # Đường dẫn output
    output_csv = args.output_file or (args.data_dir / f"transcripts_asr_{args.model}_{args.condition}.csv")
    wer_csv = args.wer_output_file or (args.results_dir / f"wer_region_tone_{args.condition}.csv")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    wer_csv.parent.mkdir(parents=True, exist_ok=True)

    transcripts_df.to_csv(output_csv, index=False, encoding="utf-8")
    logger.info("✅ Đã lưu file transcript ASR (%d dòng) vào: %s", len(transcripts_df), output_csv)

    wer_summary_df.to_csv(wer_csv, index=False, encoding="utf-8")
    logger.info("✅ Đã lưu kết quả WER region × tone (%d dòng) vào: %s", len(wer_summary_df), wer_csv)

    print("\n═════════════ BẢNG TỔNG HỢP WER THEO REGION × TONE ═════════════")
    if not wer_summary_df.empty:
        print(wer_summary_df.to_string(index=False))
    else:
        print("Không có kết quả tổng hợp.")


if __name__ == "__main__":
    main()
