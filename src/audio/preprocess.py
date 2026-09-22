"""
src/audio/preprocess.py
========================
Tiền xử lý audio cho pipeline ASR và đánh giá WER.

Spec:   B2_audio_preprocess_wer.md §1–2
Sở hữu: B (00_SHARED_CONTRACT.md §5)

Hàm công khai
──────────────
- ``load_and_resample(source, target_sr)`` → (np.ndarray, int)
    Nạp file/bytes âm thanh, chuyển mono float32, chuẩn hóa biên độ, resample.
- ``split_chunks(y, sr, seconds, overlap)`` → list[np.ndarray]
    Cắt mảng thành các đoạn ngắn có chồng lấn.
- ``make_noisy(y, sr, snr_db, seed)`` → np.ndarray
    Mô phỏng nhiễu viễn thông để đánh giá độ bền (KHÔNG dùng để train).

Lưu ý thiết kế
──────────────
- ``load_and_resample`` chấp nhận ``str | Path | bytes | BinaryIO``
  → linh hoạt với cả file local, bytes từ API, và BinaryIO từ stream.
- ``make_noisy`` chỉ dùng cho **evaluation** — không augment training data.
- Không có code trích đặc trưng âm học (spec B2 §1).
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Union

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional imports — graceful degradation với thông báo rõ ràng
# ---------------------------------------------------------------------------
try:
    import soundfile as sf          # type: ignore
    _SF_OK = True
except ImportError:
    _SF_OK = False
    sf = None   # type: ignore[assignment]
    logger.warning("soundfile chưa cài — pip install soundfile")

try:
    import librosa                  # type: ignore
    _LIBROSA_OK = True
except ImportError:
    _LIBROSA_OK = False
    librosa = None  # type: ignore[assignment]
    logger.warning("librosa chưa cài — pip install librosa")

# Kiểu đầu vào được chấp nhận bởi load_and_resample
AudioSource = Union[str, Path, bytes, io.IOBase]


# ===========================================================================
# load_and_resample
# ===========================================================================

def load_and_resample(
    source:    AudioSource,
    target_sr: int = 16000,
) -> tuple[np.ndarray, int]:
    """Nạp âm thanh từ file/bytes, chuẩn hóa và resample về ``target_sr``.

    Pipeline bên trong
    ------------------
    1. Đọc audio (soundfile) — hỗ trợ WAV, FLAC, OGG; MP3 cần ``ffmpeg``.
    2. Chuyển stereo → mono (trung bình kênh).
    3. Ép dtype về float32.
    4. Chuẩn hóa biên độ nhẹ (*soft peak normalization*): chia cho
       ``max(|y|)`` nếu ``max > 1.0``, giữ nguyên nếu đã trong [−1, 1].
       Không cắt xén (clip), không thay đổi tỷ lệ động nếu không cần.
    5. Resample về ``target_sr`` Hz nếu sample rate gốc khác.

    Tham số
    -------
    source : str | Path | bytes | BinaryIO
        Đường dẫn file, bytes thô của file audio, hoặc file-like object.
    target_sr : int, optional
        Sample rate đầu ra. Mặc định 16 000 Hz (yêu cầu của PhoWhisper).

    Trả về
    ------
    tuple[np.ndarray, int]
        ``(y, target_sr)`` — mảng float32 mono 1-D và sample rate đích.

    Raises
    ------
    ImportError
        Nếu ``soundfile`` hoặc ``librosa`` chưa được cài đặt.
    FileNotFoundError
        Nếu ``source`` là đường dẫn không tồn tại.
    RuntimeError
        Nếu file audio không đọc được (định dạng không hỗ trợ…).
    """
    if not _SF_OK:
        raise ImportError("soundfile chưa cài — pip install soundfile")

    # ── Đọc audio ──────────────────────────────────────────────────────────
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy file audio: {path}")
        y_raw, sr_orig = sf.read(str(path), dtype="float32", always_2d=False)
    elif isinstance(source, bytes):
        buf = io.BytesIO(source)
        y_raw, sr_orig = sf.read(buf, dtype="float32", always_2d=False)
    else:
        # file-like object (BinaryIO)
        y_raw, sr_orig = sf.read(source, dtype="float32", always_2d=False)

    # ── Stereo → mono ───────────────────────────────────────────────────────
    if y_raw.ndim == 2:
        logger.debug("load_and_resample: stereo (%s) → mono", y_raw.shape)
        # shape (N, C) → mean over channels axis
        y_raw = y_raw.mean(axis=1)

    # ── Đảm bảo float32 ────────────────────────────────────────────────────
    y: np.ndarray = y_raw.astype(np.float32)

    # ── Chuẩn hóa biên độ nhẹ (soft peak norm) ─────────────────────────────
    peak = float(np.abs(y).max())
    if peak > 1.0:
        logger.debug("load_and_resample: peak=%.4f > 1.0 → normalize", peak)
        y = y / peak
    elif peak == 0.0:
        logger.debug("load_and_resample: mảng im lặng hoàn toàn (peak=0)")
        # Giữ nguyên — không chia cho 0

    # ── Resample nếu cần ───────────────────────────────────────────────────
    if sr_orig != target_sr:
        if not _LIBROSA_OK:
            raise ImportError(
                f"librosa chưa cài — cần để resample {sr_orig}→{target_sr} Hz. "
                "pip install librosa"
            )
        logger.debug("load_and_resample: resample %d→%d Hz", sr_orig, target_sr)
        y = librosa.resample(y, orig_sr=sr_orig, target_sr=target_sr)

    return y.astype(np.float32), target_sr


# ===========================================================================
# split_chunks
# ===========================================================================

def split_chunks(
    y:       np.ndarray,
    sr:      int   = 16000,
    seconds: float = 4.0,
    overlap: float = 0.5,
) -> list[np.ndarray]:
    """Cắt mảng âm thanh thành các đoạn ngắn có chồng lấn.

    Tham số
    -------
    y : np.ndarray
        Mảng float32 mono 1-D.
    sr : int, optional
        Sample rate của ``y`` (Hz). Mặc định 16 000.
    seconds : float, optional
        Độ dài mỗi đoạn (giây). Spec B2: 3–5 s; mặc định 4 s.
    overlap : float, optional
        Tỷ lệ chồng lấn [0, 1). ``0.5`` → 50% chồng lấn (step = seconds/2).
        Mặc định 0.5.

    Trả về
    ------
    list[np.ndarray]
        Danh sách các đoạn; đoạn cuối có thể ngắn hơn (phần dư).
        Trả ``[y]`` nếu tổng độ dài ≤ chunk_len (không cần cắt).

    Raises
    ------
    ValueError
        Nếu ``overlap`` ∉ [0, 1) hoặc ``seconds`` ≤ 0.
    """
    if seconds <= 0:
        raise ValueError(f"seconds phải > 0, nhận {seconds}")
    if not (0.0 <= overlap < 1.0):
        raise ValueError(f"overlap phải ∈ [0, 1), nhận {overlap}")

    chunk_len = int(seconds * sr)
    step_len  = max(1, int(chunk_len * (1.0 - overlap)))

    if len(y) <= chunk_len:
        return [y.copy()]

    chunks: list[np.ndarray] = []
    start = 0
    while start < len(y):
        end   = start + chunk_len
        chunk = y[start:end]
        chunks.append(chunk.copy())
        if end >= len(y):
            break
        start += step_len

    return chunks


# ===========================================================================
# make_noisy
# ===========================================================================

def make_noisy(
    y:      np.ndarray,
    sr:     int = 16000,
    snr_db: int = 10,
    seed:   int = 42,
) -> np.ndarray:
    """Mô phỏng nhiễu viễn thông — **chỉ dùng để đánh giá độ bền, không train**.

    Hai bước mô phỏng:
    1. **Telephone codec simulation**: hạ mẫu về 8 kHz → nội suy lại 16 kHz
       (mất thông tin tần số cao > 4 kHz, như điện thoại PSTN/GSM cũ).
    2. **AWGN** (Additive White Gaussian Noise): cộng nhiễu Gaussian trắng
       theo tỷ lệ SNR cho trước.

    Tham số
    -------
    y : np.ndarray
        Audio float32 mono 16 kHz đầu vào.
    sr : int, optional
        Sample rate của ``y``. Mặc định 16 000 Hz.
    snr_db : int, optional
        Tỷ lệ tín-hiệu/nhiễu (dB). Spec B2 §2: 20 / 10 / 5 dB.
        Mặc định 10 dB.
    seed : int, optional
        Seed RNG để đảm bảo kết quả tái lập. Mặc định 42.

    Trả về
    ------
    np.ndarray
        Audio có nhiễu, cùng độ dài với ``y``, dtype float32.

    Ghi chú
    -------
    Hàm này **không** thay đổi độ dài mảng — ``len(output) == len(y)``.
    Kết quả hoàn toàn tái lập với cùng ``seed``.
    """
    if not _LIBROSA_OK:
        raise ImportError(
            "librosa chưa cài — cần để resample trong make_noisy. "
            "pip install librosa"
        )

    y = np.array(y, dtype=np.float32)
    n = len(y)

    # ── Bước 1: Telephone codec (hạ 16kHz → 8kHz → 16kHz) ─────────────────
    sr_phone = 8000
    y_down = librosa.resample(y, orig_sr=sr, target_sr=sr_phone)
    y_up   = librosa.resample(y_down, orig_sr=sr_phone, target_sr=sr)

    # Cắt / pad về đúng độ dài gốc (librosa resample có thể off-by-one)
    if len(y_up) > n:
        y_up = y_up[:n]
    elif len(y_up) < n:
        y_up = np.pad(y_up, (0, n - len(y_up)))

    # ── Bước 2: AWGN theo SNR ───────────────────────────────────────────────
    rng = np.random.default_rng(seed)

    signal_power = float(np.mean(y_up ** 2))
    if signal_power == 0.0:
        # Im lặng → cộng nhiễu rất nhỏ để vẫn trả về mảng cùng kích thước
        signal_power = 1e-10

    # SNR (dB) = 10 * log10(P_signal / P_noise)  →  P_noise = P_signal / 10^(SNR/10)
    noise_power  = signal_power / (10 ** (snr_db / 10.0))
    noise        = rng.normal(0.0, float(np.sqrt(noise_power)), size=n).astype(np.float32)

    y_noisy = y_up + noise

    # Soft peak normalization để tránh clipping
    peak = float(np.abs(y_noisy).max())
    if peak > 1.0:
        y_noisy = y_noisy / peak

    return y_noisy.astype(np.float32)
