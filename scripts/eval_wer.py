"""
scripts/eval_wer.py
====================
Đo WER (Word Error Rate) cho PhoWhisper và Whisper gốc.

Spec:   B2_audio_preprocess_wer.md §5–6
Hợp đồng: 00_SHARED_CONTRACT.md §4

Cách dùng
─────────
::

    # Đo trên tập tự thu (clean)
    python scripts/eval_wer.py \\
        --models phowhisper-base whisper-base \\
        --data_dir data/audio \\
        --condition clean

    # Đo trên tập tự thu sau make_noisy (SNR = 10 dB)
    python scripts/eval_wer.py \\
        --models phowhisper-base \\
        --data_dir data/audio \\
        --condition noisy --snr_db 10

    # Đo tất cả 3 điều kiện (clean + 3 mức SNR)
    python scripts/eval_wer.py \\
        --models phowhisper-base whisper-base \\
        --data_dir data/audio \\
        --condition all

Đầu ra
──────
- ``results/wer_by_model.csv``       : (model, dataset, condition, n_samples, WER, note)
- ``results/wer_by_region_tone.csv`` : (model, condition, region, tone, n_samples, WER, note)

Ghi chú
───────
- Ô nào n < 30 → cột ``note`` = "n nhỏ".
- WER tính bằng ``jiwer`` sau khi chuẩn hóa hai vế qua ``normalize_for_wer``.
- Số WER công bố của VinAI (8.46% base, 6.33% small, 10.41% tiny) chỉ là mốc
  đối chiếu trong báo cáo — không ghi vào kết quả thực nghiệm của nhóm.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# sys.path
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.audio.text_norm import normalize_for_wer          # noqa: E402
from src.audio.preprocess import load_and_resample, make_noisy  # noqa: E402

try:
    import jiwer                                            # type: ignore
    _JIWER_OK = True
except ImportError:
    _JIWER_OK = False
    logger.error("jiwer chưa cài — pip install jiwer")

# ---------------------------------------------------------------------------
# Hằng số
# ---------------------------------------------------------------------------
_SMALL_N_THRESHOLD = 30          # n < 30 → đánh dấu "n nhỏ"
_SNR_LEVELS        = [20, 10, 5]  # Ba mức SNR mô phỏng nhiễu viễn thông
_RESULTS_DIR       = _REPO_ROOT / "results"

# Ánh xạ tên model → HuggingFace model ID hoặc đường dẫn local
_MODEL_MAP: dict[str, str] = {
    "phowhisper-base":  str(_REPO_ROOT / "models" / "asr" / "phowhisper-base-ct2"),
    "phowhisper-tiny":  str(_REPO_ROOT / "models" / "asr" / "phowhisper-tiny-ct2"),
    "phowhisper-small": str(_REPO_ROOT / "models" / "asr" / "phowhisper-small-ct2"),
    "whisper-base":     "base",   # faster-whisper model name
    "whisper-small":    "small",
    "whisper-tiny":     "tiny",
}


# ===========================================================================
# Dataclass kết quả
# ===========================================================================

@dataclass
class WERRow:
    """Một hàng trong kết quả WER."""
    model:      str
    dataset:    str
    condition:  str
    n_samples:  int
    wer:        float
    note:       str = ""

    def to_dict(self) -> dict:
        return {
            "model":     self.model,
            "dataset":   self.dataset,
            "condition": self.condition,
            "n_samples": self.n_samples,
            "WER":       round(self.wer * 100, 4),   # phần trăm
            "note":      self.note,
        }


@dataclass
class WERRegionRow:
    """Một hàng trong kết quả WER theo region × tone."""
    model:      str
    condition:  str
    region:     str
    tone:       str
    n_samples:  int
    wer:        float
    note:       str = ""

    def to_dict(self) -> dict:
        return {
            "model":     self.model,
            "condition": self.condition,
            "region":    self.region,
            "tone":      self.tone,
            "n_samples": self.n_samples,
            "WER":       round(self.wer * 100, 4),
            "note":      self.note,
        }


# ===========================================================================
# Tiện ích
# ===========================================================================

def _note(n: int) -> str:
    """Trả 'n nhỏ' nếu n < threshold, ngược lại trả ''."""
    return "n nhỏ" if n < _SMALL_N_THRESHOLD else ""


def _compute_wer(
    references:  list[str],
    hypotheses:  list[str],
) -> float:
    """Tính WER bằng jiwer sau khi chuẩn hóa hai vế.

    Cả hai vế đều đi qua ``normalize_for_wer`` trước khi tính.
    """
    if not _JIWER_OK:
        raise ImportError("jiwer chưa cài — pip install jiwer")

    norm_refs  = [normalize_for_wer(r) for r in references]
    norm_hyps  = [normalize_for_wer(h) for h in hypotheses]

    # Bỏ qua các cặp reference rỗng (tránh chia cho 0)
    pairs = [(r, h) for r, h in zip(norm_refs, norm_hyps) if r.strip()]
    if not pairs:
        logger.warning("Không có cặp reference hợp lệ để tính WER")
        return float("nan")

    refs_clean, hyps_clean = zip(*pairs)
    wer_val = jiwer.wer(list(refs_clean), list(hyps_clean))
    return float(wer_val)


def _transcribe_batch(
    model_name:  str,
    audio_paths: list[Path],
    condition:   str,
    snr_db:      int = 10,
    noise_seed:  int = 42,
) -> list[str]:
    """Phiên âm danh sách file audio bằng model ASR.

    Tham số
    -------
    model_name : str
        Tên model (key trong _MODEL_MAP).
    audio_paths : list[Path]
        Danh sách đường dẫn file WAV/FLAC.
    condition : str
        ``"clean"`` hoặc ``"noisy_<snr_db>dB"``.
    snr_db : int
        Dùng khi ``condition`` chứa "noisy".
    noise_seed : int
        Seed cho ``make_noisy``.

    Trả về
    ------
    list[str]
        Danh sách transcript, cùng thứ tự với ``audio_paths``.
    """
    try:
        from faster_whisper import WhisperModel  # type: ignore
    except ImportError:
        raise ImportError(
            "faster-whisper chưa cài — pip install faster-whisper"
        )

    model_path = _MODEL_MAP.get(model_name, model_name)
    is_phowhisper = "phowhisper" in model_name.lower()
    device = "cpu"

    logger.info("Nạp model: %s (path=%s)", model_name, model_path)
    model = WhisperModel(
        model_path,
        device       = device,
        compute_type = "int8" if is_phowhisper else "float32",
    )

    transcripts: list[str] = []
    is_noisy = "noisy" in condition

    for i, audio_path in enumerate(audio_paths):
        try:
            y, sr = load_and_resample(str(audio_path))
            if is_noisy:
                y = make_noisy(y, sr=sr, snr_db=snr_db, seed=noise_seed)

            segments_iter, _ = model.transcribe(
                y,
                language               = "vi",
                vad_filter             = True,
                min_silence_duration_ms = 500,
                beam_size              = 5,
                temperature            = 0.0,
            )
            parts = [seg.text for seg in segments_iter if seg.text.strip()]
            transcript = " ".join(parts).strip()
        except Exception as exc:
            logger.warning("Lỗi phiên âm file %s: %s", audio_path.name, exc)
            transcript = ""

        transcripts.append(transcript)

        if (i + 1) % 10 == 0:
            logger.info("  Đã xử lý %d/%d file", i + 1, len(audio_paths))

    return transcripts


# ===========================================================================
# Nạp metadata
# ===========================================================================

def _load_metadata(data_dir: Path) -> pd.DataFrame:
    """Đọc metadata.csv; kiểm tra cột bắt buộc theo hợp đồng §3."""
    meta_path = data_dir / "metadata.csv"
    if not meta_path.exists():
        logger.error(
            "Không tìm thấy %s — xem docs/agent_notes/B2.md", meta_path
        )
        return pd.DataFrame()

    df = pd.read_csv(meta_path)
    required_cols = {
        "audio_path", "transcript_reference", "label",
        "scam_type", "region", "tone", "speaker_id",
    }
    missing = required_cols - set(df.columns)
    if missing:
        logger.error(
            "metadata.csv thiếu cột: %s — dừng phần phụ thuộc dữ liệu", missing
        )
        return pd.DataFrame()

    logger.info(
        "Metadata: %d hàng | region=%s | tone=%s | speakers=%d",
        len(df),
        df["region"].unique().tolist(),
        df["tone"].unique().tolist(),
        df["speaker_id"].nunique(),
    )
    return df


# ===========================================================================
# Đánh giá WER
# ===========================================================================

def evaluate_model_wer(
    model_name: str,
    df:         pd.DataFrame,
    data_dir:   Path,
    condition:  str,
    snr_db:     int = 10,
    dataset:    str = "thu-am-tu-thu",
) -> list[WERRow]:
    """Đánh giá WER của một model trên một điều kiện.

    Trả về danh sách ``WERRow`` (thường 1 row, trừ khi có nhiều subset).
    """
    if df.empty:
        return []

    audio_paths = [data_dir / p for p in df["audio_path"]]
    existing    = [p for p in audio_paths if p.exists()]

    if not existing:
        logger.warning("Không tìm thấy file audio nào trong %s", data_dir)
        return []

    # Giữ chỉ các hàng có file tồn tại
    mask = [p.exists() for p in audio_paths]
    df_ok = df[mask].reset_index(drop=True)

    references  = df_ok["transcript_reference"].tolist()
    audio_ok    = [data_dir / p for p in df_ok["audio_path"]]

    logger.info(
        "Đánh giá model=%s condition=%s n=%d", model_name, condition, len(audio_ok)
    )
    hypotheses = _transcribe_batch(model_name, audio_ok, condition, snr_db)

    wer_val = _compute_wer(references, hypotheses)
    n       = len(references)

    return [WERRow(
        model     = model_name,
        dataset   = dataset,
        condition = condition,
        n_samples = n,
        wer       = wer_val,
        note      = _note(n),
    )]


def evaluate_region_tone_wer(
    model_name: str,
    df:         pd.DataFrame,
    data_dir:   Path,
    condition:  str,
    snr_db:     int = 10,
) -> list[WERRegionRow]:
    """Đánh giá WER theo phân nhóm region × tone."""
    if df.empty:
        return []

    rows: list[WERRegionRow] = []
    for (region, tone), grp in df.groupby(["region", "tone"]):
        audio_paths = [data_dir / p for p in grp["audio_path"]]
        existing_mask = [p.exists() for p in audio_paths]
        grp_ok = grp[existing_mask].reset_index(drop=True)

        if grp_ok.empty:
            continue

        refs  = grp_ok["transcript_reference"].tolist()
        paths = [data_dir / p for p in grp_ok["audio_path"]]
        hyps  = _transcribe_batch(model_name, paths, condition, snr_db)

        wer_val = _compute_wer(refs, hyps)
        n       = len(refs)
        rows.append(WERRegionRow(
            model     = model_name,
            condition = condition,
            region    = str(region),
            tone      = str(tone),
            n_samples = n,
            wer       = wer_val,
            note      = _note(n),
        ))

    return rows


# ===========================================================================
# Ghi kết quả
# ===========================================================================

def _write_csv(rows: list[dict], out_path: Path) -> None:
    """Ghi danh sách dict ra CSV."""
    if not rows:
        logger.warning("Không có kết quả để ghi vào %s", out_path)
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("✅ Đã ghi %d hàng → %s", len(rows), out_path)


# ===========================================================================
# CLI
# ===========================================================================

def _build_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Đo WER cho PhoWhisper / Whisper trên dữ liệu thu âm.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--models", nargs="+",
        default=["phowhisper-base", "whisper-base"],
        help="Danh sách model cần đánh giá. "
             "Hỗ trợ: phowhisper-base, phowhisper-small, phowhisper-tiny, "
             "whisper-base, whisper-small, whisper-tiny.",
    )
    ap.add_argument(
        "--data_dir", type=Path, default=_REPO_ROOT / "data" / "audio",
        help="Thư mục chứa metadata.csv và các file audio.",
    )
    ap.add_argument(
        "--condition", choices=["clean", "noisy", "all"], default="clean",
        help="Điều kiện đánh giá: "
             "'clean' (không nhiễu), "
             "'noisy' (SNR từ --snr_db), "
             "'all' (clean + 3 mức SNR: 20/10/5 dB).",
    )
    ap.add_argument(
        "--snr_db", type=int, default=10,
        help="Mức SNR (dB) khi condition='noisy'. Mặc định 10.",
    )
    ap.add_argument(
        "--noise_seed", type=int, default=42,
        help="Seed RNG cho make_noisy. Cố định để tái lập kết quả.",
    )
    ap.add_argument(
        "--dataset", default="thu-am-tu-thu",
        help="Tên tập dữ liệu ghi vào CSV.",
    )
    ap.add_argument(
        "--results_dir", type=Path, default=_RESULTS_DIR,
        help="Thư mục lưu kết quả CSV.",
    )
    return ap.parse_args()


def main() -> None:
    args = _build_args()

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = _load_metadata(args.data_dir)
    if df.empty:
        logger.error(
            "Không có dữ liệu hợp lệ. Xem docs/agent_notes/B2.md để kiểm tra metadata."
        )
        sys.exit(1)

    # Xác định danh sách (condition, snr_db) cần chạy
    if args.condition == "all":
        conditions: list[tuple[str, int]] = [("clean", 0)] + [
            (f"noisy_{snr}dB", snr) for snr in _SNR_LEVELS
        ]
    elif args.condition == "clean":
        conditions = [("clean", 0)]
    else:
        conditions = [(f"noisy_{args.snr_db}dB", args.snr_db)]

    model_rows:  list[dict] = []
    region_rows: list[dict] = []

    for model_name in args.models:
        logger.info("═══ Model: %s ═══", model_name)
        for cond_str, snr in conditions:
            try:
                m_rows = evaluate_model_wer(
                    model_name = model_name,
                    df         = df,
                    data_dir   = args.data_dir,
                    condition  = cond_str,
                    snr_db     = snr,
                    dataset    = args.dataset,
                )
                model_rows.extend(r.to_dict() for r in m_rows)

                r_rows = evaluate_region_tone_wer(
                    model_name = model_name,
                    df         = df,
                    data_dir   = args.data_dir,
                    condition  = cond_str,
                    snr_db     = snr,
                )
                region_rows.extend(r.to_dict() for r in r_rows)

            except Exception as exc:
                logger.error(
                    "Lỗi model=%s condition=%s: %s", model_name, cond_str, exc,
                    exc_info=True,
                )

    # Ghi kết quả
    _write_csv(model_rows,  args.results_dir / "wer_by_model.csv")
    _write_csv(region_rows, args.results_dir / "wer_by_region_tone.csv")

    # In tóm tắt
    if model_rows:
        print("\n── WER Summary ──")
        for r in model_rows:
            flag = f"  [{r['note']}]" if r["note"] else ""
            print(
                f"  {r['model']:20s} | {r['condition']:15s} | "
                f"n={r['n_samples']:4d} | WER={r['WER']:6.2f}%{flag}"
            )


if __name__ == "__main__":
    main()
