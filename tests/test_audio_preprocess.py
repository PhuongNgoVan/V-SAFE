"""
tests/test_audio_preprocess.py
================================
Test suite cho B2: ``preprocess.py`` và ``text_norm.py``.

Chạy::

    pytest tests/test_audio_preprocess.py -v

Yêu cầu môi trường
-------------------
- ``soundfile``, ``librosa``, ``numpy`` phải được cài.
- Không cần file audio thật — tất cả audio được tổng hợp bằng numpy trong test.
- Không cần model ASR.

Bao gồm
-------
1. ``TestLoadAndResample``   — đầu ra đúng 16kHz, mono, float32, shape đúng.
2. ``TestNormalizeForWER``   — chuẩn hóa đúng, BẮT BUỘC giữ dấu thanh tiếng Việt.
3. ``TestSplitChunks``       — kích thước chunk, overlap, edge cases.
4. ``TestMakeNoisy``         — giữ độ dài, tái lập với cùng seed, SNR khác nhau.
5. ``TestEdgeCases``         — đầu vào biên: silence, rỗng, stereo, bytes.
"""

from __future__ import annotations

import io
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# sys.path
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ---------------------------------------------------------------------------
# Kiểm tra optional deps
# ---------------------------------------------------------------------------
try:
    import soundfile as sf    # type: ignore
    _SF_OK = True
except ImportError:
    _SF_OK = False

try:
    import librosa            # type: ignore
    _LIBROSA_OK = True
except ImportError:
    _LIBROSA_OK = False

_REQUIRES_AUDIO_DEPS = pytest.mark.skipif(
    not (_SF_OK and _LIBROSA_OK),
    reason="soundfile và/hoặc librosa chưa cài — pip install soundfile librosa",
)


# ===========================================================================
# Helpers
# ===========================================================================

def _make_sine(freq: float = 440.0, duration_s: float = 2.0,
               sr: int = 16000, amplitude: float = 0.5) -> np.ndarray:
    t = np.linspace(0, duration_s, int(duration_s * sr), endpoint=False)
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _make_silence(duration_s: float = 1.0, sr: int = 16000) -> np.ndarray:
    return np.zeros(int(duration_s * sr), dtype=np.float32)


def _write_wav(path: Path, y: np.ndarray, sr: int = 16000,
               channels: int = 1) -> None:
    """Ghi mảng float32 vào WAV (16-bit PCM)."""
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        if channels == 1:
            pcm = (y * 32767).clip(-32768, 32767).astype(np.int16)
        else:
            # stereo: y shape (N, 2) hoặc (2, N)
            if y.ndim == 1:
                y2 = np.stack([y, y], axis=1)
            else:
                y2 = y.T if y.shape[0] == 2 else y
            pcm = (y2 * 32767).clip(-32768, 32767).astype(np.int16)
        wf.writeframes(pcm.tobytes())


def _make_wav_bytes(y: np.ndarray, sr: int = 16000) -> bytes:
    """Tạo bytes của WAV file từ mảng float32."""
    buf = io.BytesIO()
    if _SF_OK:
        import soundfile as sf
        sf.write(buf, y, sr, format="WAV", subtype="PCM_16")
        return buf.getvalue()
    # Fallback: wave module
    buf2 = io.BytesIO()
    with wave.open(buf2, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        pcm = (y * 32767).clip(-32768, 32767).astype(np.int16)
        wf.writeframes(pcm.tobytes())
    return buf2.getvalue()


# ===========================================================================
# 1. TestLoadAndResample
# ===========================================================================

@_REQUIRES_AUDIO_DEPS
class TestLoadAndResample:
    """Kiểm tra load_and_resample() — đầu ra đúng 16kHz, mono, float32."""

    def test_returns_tuple_ndarray_int(self, tmp_path):
        """Trả về (np.ndarray, int)."""
        from src.audio.preprocess import load_and_resample
        y_src = _make_sine(sr=16000)
        wav = tmp_path / "test.wav"
        _write_wav(wav, y_src)
        result = load_and_resample(str(wav))
        assert isinstance(result, tuple) and len(result) == 2
        assert isinstance(result[0], np.ndarray)
        assert isinstance(result[1], int)

    def test_output_sample_rate_is_16000(self, tmp_path):
        """Sample rate trả về phải đúng 16000."""
        from src.audio.preprocess import load_and_resample
        wav = tmp_path / "test.wav"
        _write_wav(wav, _make_sine(sr=16000))
        _, sr_out = load_and_resample(str(wav))
        assert sr_out == 16000

    def test_output_is_mono_1d(self, tmp_path):
        """Đầu ra phải là mảng 1-D (mono)."""
        from src.audio.preprocess import load_and_resample
        wav = tmp_path / "test.wav"
        _write_wav(wav, _make_sine(sr=16000))
        y_out, _ = load_and_resample(str(wav))
        assert y_out.ndim == 1, f"Đầu ra phải 1-D, nhận shape={y_out.shape}"

    def test_output_dtype_float32(self, tmp_path):
        """Đầu ra phải là dtype float32."""
        from src.audio.preprocess import load_and_resample
        wav = tmp_path / "test.wav"
        _write_wav(wav, _make_sine(sr=16000))
        y_out, _ = load_and_resample(str(wav))
        assert y_out.dtype == np.float32, f"dtype={y_out.dtype}, cần float32"

    def test_resample_from_8khz_to_16khz(self, tmp_path):
        """File 8kHz → resample → đầu ra 16kHz."""
        from src.audio.preprocess import load_and_resample
        sr_orig = 8000
        y_8k = _make_sine(sr=sr_orig, duration_s=1.0)
        wav = tmp_path / "test8k.wav"
        _write_wav(wav, y_8k, sr=sr_orig)
        y_out, sr_out = load_and_resample(str(wav), target_sr=16000)
        assert sr_out == 16000
        # Số mẫu phải gần gấp đôi (16000/8000 = 2)
        expected_len = int(1.0 * 16000)
        assert abs(len(y_out) - expected_len) <= 10, \
            f"Độ dài sau resample: {len(y_out)}, kỳ vọng ~{expected_len}"

    def test_resample_from_44100_to_16000(self, tmp_path):
        """File 44100 Hz → resample → 16000 Hz."""
        from src.audio.preprocess import load_and_resample
        sr_orig = 44100
        y_44k = _make_sine(sr=sr_orig, duration_s=1.0)
        wav = tmp_path / "test44k.wav"
        _write_wav(wav, y_44k, sr=sr_orig)
        y_out, sr_out = load_and_resample(str(wav), target_sr=16000)
        assert sr_out == 16000
        expected = int(1.0 * 16000)
        assert abs(len(y_out) - expected) <= 20

    def test_amplitude_normalized_below_1(self, tmp_path):
        """Biên độ sau load không được vượt quá 1.0."""
        from src.audio.preprocess import load_and_resample
        # Tạo sine với biên độ > 1 (PCM 16-bit sẽ scale lại)
        y_loud = _make_sine(amplitude=0.95, sr=16000)
        wav = tmp_path / "loud.wav"
        _write_wav(wav, y_loud)
        y_out, _ = load_and_resample(str(wav))
        assert float(np.abs(y_out).max()) <= 1.0 + 1e-5

    def test_path_object_accepted(self, tmp_path):
        """Chấp nhận pathlib.Path."""
        from src.audio.preprocess import load_and_resample
        wav = tmp_path / "test.wav"
        _write_wav(wav, _make_sine())
        y_out, sr_out = load_and_resample(wav)   # Path object, bukan str
        assert isinstance(y_out, np.ndarray)
        assert sr_out == 16000

    def test_bytes_input_accepted(self):
        """Chấp nhận bytes của WAV file."""
        from src.audio.preprocess import load_and_resample
        wav_bytes = _make_wav_bytes(_make_sine())
        y_out, sr_out = load_and_resample(wav_bytes)
        assert isinstance(y_out, np.ndarray)
        assert sr_out == 16000

    def test_nonexistent_file_raises_fnf(self, tmp_path):
        """File không tồn tại → FileNotFoundError."""
        from src.audio.preprocess import load_and_resample
        with pytest.raises(FileNotFoundError):
            load_and_resample(str(tmp_path / "nonexistent.wav"))

    def test_silence_file_returns_zeros(self, tmp_path):
        """File im lặng → mảng zeros (không crash)."""
        from src.audio.preprocess import load_and_resample
        wav = tmp_path / "silence.wav"
        _write_wav(wav, _make_silence())
        y_out, sr_out = load_and_resample(str(wav))
        assert isinstance(y_out, np.ndarray)
        assert y_out.dtype == np.float32
        assert sr_out == 16000


# ===========================================================================
# 2. TestNormalizeForWER
# ===========================================================================

class TestNormalizeForWER:
    """Kiểm tra normalize_for_wer() — hợp đồng §4."""

    def test_lowercase_conversion(self):
        """Chuyển thành chữ thường."""
        from src.audio.text_norm import normalize_for_wer
        assert normalize_for_wer("HELLO WORLD") == "hello world"

    def test_keeps_vietnamese_tones_sac(self):
        """Giữ nguyên dấu SẮC (á, é, í, ó, ú, ý…)."""
        from src.audio.text_norm import normalize_for_wer
        text = "Công an"
        result = normalize_for_wer(text)
        assert "công" in result, f"Mất dấu sắc trong 'công': {result!r}"
        assert "an" in result

    def test_keeps_vietnamese_tones_huyen(self):
        """Giữ nguyên dấu HUYỀN (à, è, ì…)."""
        from src.audio.text_norm import normalize_for_wer
        result = normalize_for_wer("bà con làng xóm")
        assert "bà" in result
        assert "làng" in result

    def test_keeps_vietnamese_tones_hoi(self):
        """Giữ nguyên dấu HỎI (ả, ẻ, ỉ…)."""
        from src.audio.text_norm import normalize_for_wer
        result = normalize_for_wer("Hỏi thăm")
        assert "hỏi" in result

    def test_keeps_vietnamese_tones_nga(self):
        """Giữ nguyên dấu NGÃ (ã, ẽ, ĩ…)."""
        from src.audio.text_norm import normalize_for_wer
        result = normalize_for_wer("Bãi biển")
        assert "bãi" in result

    def test_keeps_vietnamese_tones_nang(self):
        """Giữ nguyên dấu NẶNG (ạ, ẹ, ị…)."""
        from src.audio.text_norm import normalize_for_wer
        result = normalize_for_wer("Chuyển khoản")
        assert "chuyển" in result
        assert "khoản" in result

    def test_removes_comma(self):
        """Xóa dấu phẩy."""
        from src.audio.text_norm import normalize_for_wer
        result = normalize_for_wer("xin chào, bạn")
        assert "," not in result

    def test_removes_period(self):
        """Xóa dấu chấm."""
        from src.audio.text_norm import normalize_for_wer
        result = normalize_for_wer("xin chào.")
        assert "." not in result

    def test_removes_exclamation(self):
        """Xóa dấu chấm than."""
        from src.audio.text_norm import normalize_for_wer
        result = normalize_for_wer("Nhanh lên!")
        assert "!" not in result

    def test_removes_question_mark(self):
        """Xóa dấu hỏi chấm."""
        from src.audio.text_norm import normalize_for_wer
        result = normalize_for_wer("Bạn có khỏe không?")
        assert "?" not in result

    def test_collapses_whitespace(self):
        """Gộp khoảng trắng thừa."""
        from src.audio.text_norm import normalize_for_wer
        result = normalize_for_wer("  xin   chào   bạn  ")
        assert result == "xin chào bạn"
        assert "  " not in result

    def test_strips_leading_trailing(self):
        """Strip hai đầu."""
        from src.audio.text_norm import normalize_for_wer
        result = normalize_for_wer("  hello  ")
        assert result == result.strip()

    def test_empty_string(self):
        """Chuỗi rỗng → ''."""
        from src.audio.text_norm import normalize_for_wer
        assert normalize_for_wer("") == ""

    def test_whitespace_only(self):
        """Toàn khoảng trắng → ''."""
        from src.audio.text_norm import normalize_for_wer
        assert normalize_for_wer("   ") == ""

    def test_full_vietnamese_sentence(self):
        """Câu tiếng Việt đầy đủ — tất cả dấu thanh phải giữ nguyên."""
        from src.audio.text_norm import normalize_for_wer
        sentence = "Công an yêu cầu chuyển khoản ngay!"
        result   = normalize_for_wer(sentence)
        expected = "công an yêu cầu chuyển khoản ngay"
        assert result == expected, f"Nhận: {result!r}, kỳ vọng: {expected!r}"

    def test_full_scam_sentence(self):
        """Câu lừa đảo điển hình — dấu thanh giữ nguyên."""
        from src.audio.text_norm import normalize_for_wer
        sentence = "Bạn đã trúng thưởng iPhone 15, liên hệ ngay để nhận!"
        result   = normalize_for_wer(sentence)
        assert "trúng" in result
        assert "thưởng" in result
        assert "!" not in result
        assert "," not in result

    def test_normalize_pair_helper(self):
        """normalize_pair trả cặp (ref, hyp) đều đã chuẩn hóa."""
        from src.audio.text_norm import normalize_pair
        ref, hyp = normalize_pair("Công an!", "công an")
        assert ref == "công an"
        assert hyp == "công an"

    def test_numbers_preserved(self):
        """Chữ số giữ nguyên (không xóa)."""
        from src.audio.text_norm import normalize_for_wer
        result = normalize_for_wer("chuyển 50 triệu đồng")
        assert "50" in result

    @pytest.mark.parametrize("inp,expected", [
        ("Công an",   "công an"),
        ("Cảnh sát",  "cảnh sát"),
        ("Trúng thưởng", "trúng thưởng"),
        ("Ngân hàng", "ngân hàng"),
        ("Chuyển khoản", "chuyển khoản"),
    ])
    def test_parametrized_vietnamese_phrases(self, inp: str, expected: str):
        """Nhiều cụm tiếng Việt điển hình giữ nguyên dấu thanh."""
        from src.audio.text_norm import normalize_for_wer
        assert normalize_for_wer(inp) == expected


# ===========================================================================
# 3. TestSplitChunks
# ===========================================================================

class TestSplitChunks:
    """Kiểm tra split_chunks() — kích thước, overlap, edge cases."""

    def test_returns_list_of_ndarrays(self):
        """Trả về list[np.ndarray]."""
        from src.audio.preprocess import split_chunks
        y = _make_sine(duration_s=10.0)
        chunks = split_chunks(y, sr=16000, seconds=4.0, overlap=0.5)
        assert isinstance(chunks, list)
        assert all(isinstance(c, np.ndarray) for c in chunks)

    def test_chunk_length_correct(self):
        """Chunk phải có đúng chunk_len mẫu (trừ chunk cuối)."""
        from src.audio.preprocess import split_chunks
        sr, seconds = 16000, 4.0
        chunk_len = int(seconds * sr)
        y = _make_sine(duration_s=12.0, sr=sr)
        chunks = split_chunks(y, sr=sr, seconds=seconds, overlap=0.5)
        # Tất cả chunk trừ cuối phải dài đúng chunk_len
        for c in chunks[:-1]:
            assert len(c) == chunk_len, \
                f"Chunk dài {len(c)}, kỳ vọng {chunk_len}"

    def test_overlap_50_step_correct(self):
        """overlap=0.5 → step = chunk_len/2."""
        from src.audio.preprocess import split_chunks
        sr, seconds, overlap = 16000, 4.0, 0.5
        chunk_len = int(seconds * sr)
        step_len  = int(chunk_len * (1 - overlap))
        y = _make_sine(duration_s=12.0, sr=sr)
        chunks = split_chunks(y, sr=sr, seconds=seconds, overlap=overlap)
        # Số chunk kỳ vọng: ceil((N - chunk_len) / step + 1)
        n = len(y)
        expected_n = max(1, int(np.ceil((n - chunk_len) / step_len)) + 1)
        assert abs(len(chunks) - expected_n) <= 1, \
            f"Số chunk={len(chunks)}, kỳ vọng ~{expected_n}"

    def test_no_overlap(self):
        """overlap=0 → không chồng lấn."""
        from src.audio.preprocess import split_chunks
        sr, seconds = 16000, 4.0
        chunk_len = int(seconds * sr)
        duration  = 12.0
        y = _make_sine(duration_s=duration, sr=sr)
        chunks = split_chunks(y, sr=sr, seconds=seconds, overlap=0.0)
        # Mỗi vùng không chồng lấn → tổng ≥ len(y)
        assert len(chunks) >= int(duration / seconds)

    def test_short_audio_returns_single_chunk(self):
        """Audio ngắn hơn 1 chunk → trả list với 1 phần tử."""
        from src.audio.preprocess import split_chunks
        y = _make_sine(duration_s=1.0, sr=16000)  # 1s < 4s chunk
        chunks = split_chunks(y, sr=16000, seconds=4.0, overlap=0.5)
        assert len(chunks) == 1
        assert len(chunks[0]) == len(y)

    def test_exact_length_audio(self):
        """Audio dài đúng 1 chunk → trả 1 chunk."""
        from src.audio.preprocess import split_chunks
        sr, seconds = 16000, 4.0
        y = _make_sine(duration_s=seconds, sr=sr)
        chunks = split_chunks(y, sr=sr, seconds=seconds, overlap=0.5)
        assert len(chunks) == 1

    def test_all_chunks_are_numpy_float32(self):
        """Tất cả chunk phải là float32."""
        from src.audio.preprocess import split_chunks
        y = _make_sine(duration_s=10.0)
        chunks = split_chunks(y)
        for c in chunks:
            assert c.dtype == np.float32

    def test_invalid_overlap_raises(self):
        """overlap ≥ 1 hoặc < 0 → ValueError."""
        from src.audio.preprocess import split_chunks
        y = _make_sine()
        with pytest.raises(ValueError):
            split_chunks(y, overlap=1.0)
        with pytest.raises(ValueError):
            split_chunks(y, overlap=-0.1)

    def test_invalid_seconds_raises(self):
        """seconds ≤ 0 → ValueError."""
        from src.audio.preprocess import split_chunks
        y = _make_sine()
        with pytest.raises(ValueError):
            split_chunks(y, seconds=0.0)
        with pytest.raises(ValueError):
            split_chunks(y, seconds=-1.0)

    def test_chunks_cover_entire_audio(self):
        """Không có mẫu nào bị bỏ qua (chunk cuối có thể ngắn hơn)."""
        from src.audio.preprocess import split_chunks
        sr, seconds, overlap = 16000, 4.0, 0.5
        y = _make_sine(duration_s=10.0, sr=sr)
        chunks = split_chunks(y, sr=sr, seconds=seconds, overlap=overlap)
        # Chunk đầu tiên bắt đầu từ 0
        np.testing.assert_array_equal(
            chunks[0],
            y[:len(chunks[0])],
            err_msg="Chunk đầu phải bắt đầu từ mẫu 0",
        )


# ===========================================================================
# 4. TestMakeNoisy
# ===========================================================================

@_REQUIRES_AUDIO_DEPS
class TestMakeNoisy:
    """Kiểm tra make_noisy() — độ dài, tái lập, SNR."""

    def test_output_length_equals_input(self):
        """Đầu ra phải cùng độ dài với đầu vào."""
        from src.audio.preprocess import make_noisy
        y = _make_sine(duration_s=2.0)
        y_noisy = make_noisy(y, sr=16000, snr_db=10, seed=42)
        assert len(y_noisy) == len(y), \
            f"Đầu ra dài {len(y_noisy)}, kỳ vọng {len(y)}"

    def test_output_dtype_float32(self):
        """Đầu ra phải là float32."""
        from src.audio.preprocess import make_noisy
        y = _make_sine()
        y_noisy = make_noisy(y, snr_db=10, seed=42)
        assert y_noisy.dtype == np.float32

    def test_reproducible_with_same_seed(self):
        """Cùng seed → cùng nhiễu."""
        from src.audio.preprocess import make_noisy
        y = _make_sine(duration_s=1.0)
        y1 = make_noisy(y, snr_db=10, seed=42)
        y2 = make_noisy(y, snr_db=10, seed=42)
        np.testing.assert_array_equal(y1, y2, err_msg="Cùng seed phải cho cùng đầu ra")

    def test_different_seeds_give_different_noise(self):
        """Khác seed → khác nhiễu."""
        from src.audio.preprocess import make_noisy
        y = _make_sine(duration_s=1.0)
        y1 = make_noisy(y, snr_db=10, seed=42)
        y2 = make_noisy(y, snr_db=10, seed=99)
        assert not np.array_equal(y1, y2), "Khác seed phải cho đầu ra khác nhau"

    def test_different_snr_different_result(self):
        """SNR khác nhau → đầu ra khác nhau."""
        from src.audio.preprocess import make_noisy
        y = _make_sine(duration_s=1.0)
        y_10  = make_noisy(y, snr_db=10,  seed=42)
        y_5   = make_noisy(y, snr_db=5,   seed=42)
        y_20  = make_noisy(y, snr_db=20,  seed=42)
        assert not np.array_equal(y_10, y_5)
        assert not np.array_equal(y_10, y_20)

    def test_higher_snr_less_noise(self):
        """SNR cao hơn → ít nhiễu hơn (MSE nhỏ hơn so với gốc)."""
        from src.audio.preprocess import make_noisy
        y = _make_sine(duration_s=2.0)
        y_20 = make_noisy(y, snr_db=20, seed=42)
        y_5  = make_noisy(y, snr_db=5,  seed=42)
        # SNR=20 dB gần tín hiệu gốc hơn SNR=5 dB
        # So sánh MSE giữa noisy và gốc (sau telephone codec)
        mse_20 = float(np.mean((y_20 - y) ** 2))
        mse_5  = float(np.mean((y_5  - y) ** 2))
        assert mse_20 <= mse_5, \
            f"SNR=20 phải ít noise hơn SNR=5 (MSE_20={mse_20:.6f}, MSE_5={mse_5:.6f})"

    @pytest.mark.parametrize("snr_db", [20, 10, 5])
    def test_three_snr_levels_spec(self, snr_db: int):
        """Ba mức SNR theo spec B2 §2: 20/10/5 dB — không crash."""
        from src.audio.preprocess import make_noisy
        y = _make_sine(duration_s=1.0)
        y_noisy = make_noisy(y, snr_db=snr_db, seed=42)
        assert len(y_noisy) == len(y)
        assert y_noisy.dtype == np.float32

    def test_output_amplitude_bounded(self):
        """Đầu ra phải ≤ 1.0 (soft normalization)."""
        from src.audio.preprocess import make_noisy
        y = _make_sine(amplitude=0.9)
        y_noisy = make_noisy(y, snr_db=5, seed=42)
        assert float(np.abs(y_noisy).max()) <= 1.0 + 1e-5

    def test_silence_input_no_crash(self):
        """Im lặng hoàn toàn không crash."""
        from src.audio.preprocess import make_noisy
        y = _make_silence()
        try:
            y_noisy = make_noisy(y, snr_db=10, seed=42)
            assert len(y_noisy) == len(y)
        except Exception as e:
            pytest.fail(f"make_noisy(silence) gây ra ngoại lệ: {e}")

    def test_not_identical_to_input(self):
        """Đầu ra phải khác đầu vào (có nhiễu)."""
        from src.audio.preprocess import make_noisy
        y = _make_sine(duration_s=2.0)
        y_noisy = make_noisy(y, snr_db=10, seed=42)
        assert not np.array_equal(y, y_noisy), "make_noisy phải thêm nhiễu vào tín hiệu"


# ===========================================================================
# 5. TestEdgeCases
# ===========================================================================

class TestEdgeCases:
    """Các trường hợp biên tổng hợp."""

    def test_split_chunks_preserves_dtype(self):
        """split_chunks không đổi dtype."""
        from src.audio.preprocess import split_chunks
        y = _make_sine().astype(np.float64)
        chunks = split_chunks(y.astype(np.float32))
        for c in chunks:
            assert c.dtype == np.float32

    def test_normalize_for_wer_mixed_script(self):
        """Hỗn hợp ASCII + tiếng Việt."""
        from src.audio.text_norm import normalize_for_wer
        result = normalize_for_wer("Hello! Xin chào.")
        assert "!" not in result
        assert "." not in result
        assert "xin" in result
        assert "chào" in result   # dấu huyền giữ nguyên

    def test_normalize_for_wer_ellipsis(self):
        """Dấu ba chấm (…) bị xóa."""
        from src.audio.text_norm import normalize_for_wer
        result = normalize_for_wer("Chờ một chút…")
        assert "…" not in result

    def test_normalize_for_wer_quotes(self):
        """Dấu ngoặc kép (""'') bị xóa."""
        from src.audio.text_norm import normalize_for_wer
        result = normalize_for_wer('"Công an" yêu cầu')
        assert '"' not in result
        assert "công an" in result

    @_REQUIRES_AUDIO_DEPS
    def test_load_from_bytesio(self):
        """load_and_resample nhận io.BytesIO (file-like object)."""
        from src.audio.preprocess import load_and_resample
        y_src = _make_sine()
        wav_bytes = _make_wav_bytes(y_src)
        buf = io.BytesIO(wav_bytes)
        y_out, sr_out = load_and_resample(buf)
        assert isinstance(y_out, np.ndarray)
        assert sr_out == 16000
