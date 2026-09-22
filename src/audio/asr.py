"""
src/audio/asr.py
=================
Module ASR — phiên âm âm thanh tiếng Việt bằng PhoWhisper (CTranslate2).

Spec:   B1_asr_setup_phowhisper.md §4
Hợp đồng: 00_SHARED_CONTRACT.md §2  (B sở hữu)

Hàm công khai (theo hợp đồng — KHÔNG thay đổi chữ ký sau S2)
──────────────────────────────────────────────────────────────
- ``transcribe_chunk(y, sr=16000)`` → str
    Nhận mảng numpy float32 mono, trả chuỗi phiên âm tiếng Việt.
- ``transcribe_file(path)`` → str
    Đọc file âm thanh rồi gọi ``transcribe_chunk``.

Singleton
─────────
``WhisperModel`` được khởi tạo **một lần duy nhất** (lazy, thread-safe)
nhờ ``threading.Lock``.  Cấu hình đọc từ ``configs/asr.yaml``.

Xử lý biên
──────────
- Mảng im lặng / không có segment → trả ``""`` (không ném lỗi).
- ``sr != 16000`` → resample về 16 kHz bằng librosa.
- Input không phải float32 → tự ép kiểu và cảnh báo.
- Input stereo → lấy trung bình kênh → mono float32.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Đường dẫn mặc định
# ---------------------------------------------------------------------------
_REPO_ROOT  = Path(__file__).resolve().parents[2]
_ASR_CFG    = _REPO_ROOT / "configs" / "asr.yaml"

# ---------------------------------------------------------------------------
# Import faster-whisper với thông báo rõ ràng khi chưa cài
# ---------------------------------------------------------------------------
try:
    from faster_whisper import WhisperModel  # type: ignore
    _FASTER_WHISPER_AVAILABLE = True
except ImportError:
    _FASTER_WHISPER_AVAILABLE = False
    WhisperModel = None  # type: ignore[assignment,misc]
    logger.warning(
        "faster-whisper chưa được cài đặt. "
        "Cài bằng: pip install faster-whisper"
    )

# ---------------------------------------------------------------------------
# Import librosa (chỉ dùng để resample — không dùng cho đặc trưng âm học)
# ---------------------------------------------------------------------------
try:
    import librosa  # type: ignore
    _LIBROSA_AVAILABLE = True
except ImportError:
    _LIBROSA_AVAILABLE = False
    logger.warning(
        "librosa chưa được cài đặt — resample sẽ không hoạt động. "
        "Cài bằng: pip install librosa"
    )

# ---------------------------------------------------------------------------
# Import soundfile để đọc file âm thanh
# ---------------------------------------------------------------------------
try:
    import soundfile as sf  # type: ignore
    _SOUNDFILE_AVAILABLE = True
except ImportError:
    _SOUNDFILE_AVAILABLE = False
    logger.warning(
        "soundfile chưa được cài đặt. "
        "Cài bằng: pip install soundfile"
    )


# ===========================================================================
# Ngoại lệ
# ===========================================================================

class ASRModelNotReadyError(RuntimeError):
    """Raise khi model CTranslate2 chưa được convert hoặc không tìm thấy."""


# ===========================================================================
# Đọc cấu hình
# ===========================================================================

def _load_asr_config(cfg_path: Path = _ASR_CFG) -> dict:
    """Đọc configs/asr.yaml, trả về dict với giá trị mặc định an toàn."""
    defaults: dict = {
        "model_path":             "models/asr/phowhisper-base-ct2",
        "device":                 "cpu",
        "compute_type":           "int8",
        "language":               "vi",
        "vad_filter":             True,
        "min_silence_duration_ms": 500,
        "beam_size":              5,
        "best_of":                5,
        "temperature":            0.0,
        "condition_on_previous_text": False,
    }
    if not cfg_path.exists():
        logger.warning("Không tìm thấy %s — dùng cấu hình mặc định.", cfg_path)
        return defaults

    with open(cfg_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    merged = {**defaults, **raw}
    logger.debug("Đọc ASR config từ %s: %s", cfg_path, merged)
    return merged


# ===========================================================================
# Lazy singleton — WhisperModel
# ===========================================================================

class _ASRState:
    """Container thread-safe cho singleton WhisperModel."""

    def __init__(self) -> None:
        self._lock    = threading.Lock()
        self._loaded  = False
        self.model:   Optional[object] = None  # WhisperModel
        self.cfg:     dict = {}

    def load(self, cfg_path: Path = _ASR_CFG) -> None:
        """Nạp WhisperModel từ cấu hình — gọi tối đa một lần (thread-safe)."""
        if self._loaded:
            return
        with self._lock:
            if self._loaded:   # double-checked locking
                return
            self._do_load(cfg_path)
            self._loaded = True

    def _do_load(self, cfg_path: Path) -> None:
        if not _FASTER_WHISPER_AVAILABLE:
            raise ASRModelNotReadyError(
                "faster-whisper chưa được cài đặt. "
                "Chạy: pip install faster-whisper"
            )

        cfg = _load_asr_config(cfg_path)
        self.cfg = cfg

        # Giải quyết đường dẫn model (tương đối → tuyệt đối)
        model_path_raw = cfg["model_path"]
        model_path = Path(model_path_raw)
        if not model_path.is_absolute():
            model_path = _REPO_ROOT / model_path

        # Kiểm tra thư mục model tồn tại
        if not model_path.exists():
            raise ASRModelNotReadyError(
                f"Không tìm thấy model CTranslate2 tại: {model_path}\n"
                "Chạy script convert trước: bash scripts/convert_phowhisper.sh"
            )

        logger.info(
            "Đang nạp WhisperModel: %s (device=%s, compute_type=%s)…",
            model_path, cfg["device"], cfg["compute_type"],
        )
        self.model = WhisperModel(
            str(model_path),
            device=cfg["device"],
            compute_type=cfg["compute_type"],
        )
        logger.info("✅ WhisperModel nạp xong từ %s", model_path)


# Module-level singleton
_STATE = _ASRState()


def _ensure_loaded(cfg_path: Path = _ASR_CFG) -> _ASRState:
    """Đảm bảo WhisperModel đã được nạp; nếu chưa thì nạp ngay."""
    _STATE.load(cfg_path)
    return _STATE


# ===========================================================================
# Tiền xử lý audio
# ===========================================================================

def _to_mono_float32(y: np.ndarray) -> np.ndarray:
    """Chuyển mảng audio về mono float32.

    - Stereo (2-D) → lấy trung bình các kênh.
    - Ép dtype về float32 nếu cần.
    """
    if y.ndim == 2:
        logger.debug("Audio stereo → chuyển sang mono (mean)")
        y = y.mean(axis=0) if y.shape[0] < y.shape[1] else y.mean(axis=1)
    if y.dtype != np.float32:
        logger.debug("Audio dtype=%s → ép về float32", y.dtype)
        y = y.astype(np.float32)
    return y


def _resample(y: np.ndarray, sr_orig: int, sr_target: int = 16000) -> np.ndarray:
    """Resample về sr_target Hz bằng librosa.

    Parameters
    ----------
    y : np.ndarray
        Mảng float32 mono.
    sr_orig : int
        Sample rate gốc.
    sr_target : int
        Sample rate đích (mặc định 16000 Hz).
    """
    if not _LIBROSA_AVAILABLE:
        raise ASRModelNotReadyError(
            f"librosa chưa cài đặt — không thể resample {sr_orig}→{sr_target} Hz. "
            "Cài bằng: pip install librosa"
        )
    logger.debug("Resample %d → %d Hz", sr_orig, sr_target)
    return librosa.resample(y, orig_sr=sr_orig, target_sr=sr_target)


def _is_silence(y: np.ndarray, threshold: float = 1e-4) -> bool:
    """Kiểm tra mảng có phải im lặng không (RMS < threshold)."""
    rms = float(np.sqrt(np.mean(y ** 2)))
    return rms < threshold


# ===========================================================================
# API công khai — hợp đồng 00_SHARED_CONTRACT.md §2
# ===========================================================================

def transcribe_chunk(
    y:  np.ndarray,
    sr: int = 16000,
    *,
    cfg_path: Path = _ASR_CFG,
) -> str:
    """Phiên âm đoạn audio numpy float32 mono sang text tiếng Việt.

    Tham số
    -------
    y : np.ndarray
        Mảng audio, dtype float32, mono (hoặc stereo — sẽ tự convert).
        Nếu dtype không phải float32 → tự ép kiểu và cảnh báo.
    sr : int, optional
        Sample rate của ``y``. Mặc định: 16000 Hz.
        Nếu ``sr != 16000`` → resample về 16 kHz trước khi phiên âm.
    cfg_path : Path, optional
        Đường dẫn file cấu hình ASR (chủ yếu dùng trong tests).

    Trả về
    ------
    str
        Chuỗi phiên âm tiếng Việt đã strip.  Trả ``""`` nếu:
        - Mảng im lặng / không có năng lượng âm thanh.
        - VAD không phát hiện segment nào.
        - Model không tạo ra token nào.

    Raises
    ------
    ASRModelNotReadyError
        Nếu model chưa được convert hoặc faster-whisper chưa cài.
    ValueError
        Nếu ``y`` rỗng hoặc không phải ndarray.

    Ghi chú
    -------
    Hàm này **không** gọi lại NLP preprocess — transcript thô được trả
    nguyên bản cho route của B để B tự gán PII mask trước khi lưu.
    """
    # ── Validate đầu vào ──
    if not isinstance(y, np.ndarray):
        raise ValueError(f"y phải là numpy ndarray, nhận {type(y)}")
    if y.size == 0:
        logger.debug("transcribe_chunk: mảng rỗng → trả ''")
        return ""

    # ── Tiền xử lý ──
    y = _to_mono_float32(y)

    # Resample về 16 kHz TRƯỚC khi kiểm tra im lặng.
    # Lý do: (1) silence check phải chạy trên audio đã ở đúng sample-rate;
    #         (2) đảm bảo resample luôn được gọi khi sr != 16000 (testable).
    if sr != 16000:
        y = _resample(y, sr_orig=sr, sr_target=16000)

    # Kiểm tra im lặng nhanh (sau resample, trước khi gọi model)
    if _is_silence(y):
        logger.debug("transcribe_chunk: phát hiện im lặng (RMS < 1e-4) → trả ''")
        return ""

    # ── Đảm bảo model nạp ──
    state = _ensure_loaded(cfg_path)
    cfg   = state.cfg

    # ── Phiên âm ──
    # Chú ý: ASRModelNotReadyError KHÔNG được nuốt — phải văng ra ngoài
    # để caller biết model chưa sẵn sàng (phân biệt với lỗi runtime thuần tuý).
    try:
        segments_iter, _info = state.model.transcribe(
            y,
            language               = cfg.get("language", "vi"),
            beam_size              = int(cfg.get("beam_size", 5)),
            best_of                = int(cfg.get("best_of", 5)),
            temperature            = float(cfg.get("temperature", 0.0)),
            vad_filter             = bool(cfg.get("vad_filter", True)),
            min_silence_duration_ms = int(cfg.get("min_silence_duration_ms", 500)),
            condition_on_previous_text = bool(
                cfg.get("condition_on_previous_text", False)
            ),
        )
        # Ghép text các segment
        parts = [seg.text for seg in segments_iter if seg.text.strip()]
        transcript = " ".join(parts).strip()
    except ASRModelNotReadyError:
        # Không nuốt — để propagate lên caller
        raise
    except Exception as exc:
        # Chỉ nuốt các lỗi runtime từ model.transcribe() (OOM, CUDA error…)
        logger.error("Lỗi runtime khi phiên âm: %s", exc, exc_info=True)
        return ""

    return transcript


def transcribe_file(
    path: str,
    *,
    cfg_path: Path = _ASR_CFG,
) -> str:
    """Đọc file âm thanh và phiên âm sang text tiếng Việt.

    Tiện ích dành cho script độc lập và đánh giá — B dùng trong pipeline
    batch hoặc khi nhận đường dẫn file thay vì mảng numpy.

    Tham số
    -------
    path : str
        Đường dẫn file âm thanh (WAV, FLAC, MP3, OGG…).
        Định dạng đọc được bởi ``soundfile``; MP3 cần ``ffmpeg``.
    cfg_path : Path, optional
        Đường dẫn config ASR (dùng trong tests).

    Trả về
    ------
    str
        Chuỗi phiên âm, hoặc ``""`` nếu im lặng / lỗi.

    Raises
    ------
    FileNotFoundError
        Nếu ``path`` không tồn tại.
    ASRModelNotReadyError
        Nếu model chưa được convert.
    """
    audio_path = Path(path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file âm thanh: {audio_path}")

    if not _SOUNDFILE_AVAILABLE:
        raise ASRModelNotReadyError(
            "soundfile chưa cài đặt — không thể đọc file âm thanh. "
            "Cài bằng: pip install soundfile"
        )

    try:
        y, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
    except Exception as exc:
        logger.error("Không đọc được file %s: %s", path, exc)
        raise

    logger.debug("Đọc file: %s (sr=%d, shape=%s)", audio_path.name, sr, y.shape)
    return transcribe_chunk(y, sr, cfg_path=cfg_path)


# ===========================================================================
# CLI nhanh (nghiệm thu B1 §38)
# ===========================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Dùng: python src/audio/asr.py <đường_dẫn_file_âm_thanh>")
        sys.exit(1)

    file_path = sys.argv[1]
    print(f"Phiên âm: {file_path}")
    result = transcribe_file(file_path)
    print(f"Kết quả : {result!r}")
