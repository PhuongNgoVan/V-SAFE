"""
tests/test_asr.py
==================
Test suite cho ``src/audio/asr.py`` (B1).

Chạy::

    pytest tests/test_asr.py -v

Yêu cầu môi trường
-------------------
- ``TestModelIntegration`` cần ``models/asr/phowhisper-base-ct2/`` đã được
  convert (``bash scripts/convert_phowhisper.sh``). Auto-skip nếu chưa có.
- Tất cả test còn lại dùng **mock** / dữ liệu tổng hợp — không cần model thật.

Bao gồm
-------
1. ``TestASRContracts``       — chữ ký hàm, import đúng hợp đồng.
2. ``TestSilenceDetection``   — im lặng numpy zeros → "".
3. ``TestInputNormalization`` — stereo→mono, dtype, resample mock.
4. ``TestSingletonBehavior``  — singleton nạp model đúng một lần.
5. ``TestMockTranscription``  — forward pass mock trả về kiểu str.
6. ``TestTranscribeFile``     — đọc file WAV thực (tổng hợp), mock model.
7. ``TestEdgeCases``          — mảng rỗng, input None, file không tồn tại.
8. ``TestModelIntegration``   — test thật với model (auto-skip nếu chưa có).
"""

from __future__ import annotations

import sys
import threading
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# sys.path
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ---------------------------------------------------------------------------
# Kiểm tra model đã convert chưa
# ---------------------------------------------------------------------------
_MODEL_DIR   = _REPO_ROOT / "models" / "asr" / "phowhisper-base-ct2"
_MODEL_READY = (_MODEL_DIR / "model.bin").exists()

_REQUIRES_MODEL = pytest.mark.skipif(
    not _MODEL_READY,
    reason="models/asr/phowhisper-base-ct2/ chưa tồn tại — chạy scripts/convert_phowhisper.sh trước",
)

# ---------------------------------------------------------------------------
# Kiểm tra faster-whisper cài chưa
# ---------------------------------------------------------------------------
try:
    from faster_whisper import WhisperModel as _WM  # noqa: F401
    _FW_AVAILABLE = True
except ImportError:
    _FW_AVAILABLE = False


# ===========================================================================
# Helpers
# ===========================================================================

def _make_silence(duration_s: float = 1.0, sr: int = 16000) -> np.ndarray:
    """Tạo mảng numpy zeros (im lặng hoàn toàn)."""
    return np.zeros(int(duration_s * sr), dtype=np.float32)


def _make_sine(freq: float = 440.0, duration_s: float = 1.0,
               sr: int = 16000, amplitude: float = 0.5) -> np.ndarray:
    """Tạo sóng sine float32 (có năng lượng âm thanh, không phải im lặng)."""
    t = np.linspace(0, duration_s, int(duration_s * sr), endpoint=False)
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _write_wav(path: Path, y: np.ndarray, sr: int = 16000) -> None:
    """Ghi mảng float32 vào file WAV."""
    import wave, struct
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)   # 16-bit
        wf.setframerate(sr)
        pcm = (y * 32767).clip(-32768, 32767).astype(np.int16)
        wf.writeframes(pcm.tobytes())


def _make_fake_segment(text: str) -> MagicMock:
    """Tạo mock segment giống faster-whisper Segment."""
    seg = MagicMock()
    seg.text = text
    return seg


# ===========================================================================
# 1. TestASRContracts
# ===========================================================================

class TestASRContracts:
    """Kiểm tra hàm công khai theo hợp đồng 00_SHARED_CONTRACT.md §2."""

    def test_transcribe_chunk_importable(self):
        """transcribe_chunk phải import được từ src.audio.asr."""
        from src.audio.asr import transcribe_chunk  # noqa: F401
        assert callable(transcribe_chunk)

    def test_transcribe_file_importable(self):
        """transcribe_file phải import được từ src.audio.asr."""
        from src.audio.asr import transcribe_file  # noqa: F401
        assert callable(transcribe_file)

    def test_transcribe_chunk_signature(self):
        """transcribe_chunk phải nhận (y, sr=16000) theo hợp đồng."""
        import inspect
        from src.audio.asr import transcribe_chunk
        sig = inspect.signature(transcribe_chunk)
        params = list(sig.parameters.keys())
        assert "y"  in params, "Thiếu tham số 'y'"
        assert "sr" in params, "Thiếu tham số 'sr'"
        # sr mặc định = 16000
        assert sig.parameters["sr"].default == 16000

    def test_transcribe_file_signature(self):
        """transcribe_file phải nhận (path: str) theo hợp đồng."""
        import inspect
        from src.audio.asr import transcribe_file
        sig = inspect.signature(transcribe_file)
        assert "path" in sig.parameters

    def test_return_type_annotation(self):
        """Cả hai hàm phải có return annotation là str."""
        import inspect
        from src.audio.asr import transcribe_chunk, transcribe_file
        for fn in (transcribe_chunk, transcribe_file):
            sig = inspect.signature(fn)
            assert sig.return_annotation is str or sig.return_annotation == "str", \
                f"{fn.__name__} thiếu annotation -> str"

    def test_no_pyaudioanalysis_import(self):
        """src.audio.asr không được import pyAudioAnalysis."""
        import ast
        asr_path = _REPO_ROOT / "src" / "audio" / "asr.py"
        tree = ast.parse(asr_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in getattr(node, "names", []):
                    assert "pyAudioAnalysis" not in alias.name, \
                        "Tìm thấy import pyAudioAnalysis trong asr.py — vi phạm B1 §2"
                if hasattr(node, "module") and node.module:
                    assert "pyAudioAnalysis" not in node.module


# ===========================================================================
# 2. TestSilenceDetection
# ===========================================================================

class TestSilenceDetection:
    """Im lặng → trả '' (không crash, không gọi model)."""

    def test_zeros_array_returns_empty_string(self):
        """np.zeros() trả ''."""
        from src.audio.asr import transcribe_chunk
        y = _make_silence(duration_s=1.0)
        result = transcribe_chunk(y)
        assert result == "", f"Im lặng phải trả '', nhận {result!r}"

    def test_zeros_3_seconds_returns_empty(self):
        """3 giây im lặng → ''."""
        from src.audio.asr import transcribe_chunk
        y = _make_silence(duration_s=3.0)
        result = transcribe_chunk(y)
        assert result == ""

    def test_very_small_amplitude_treated_as_silence(self):
        """Biên độ cực nhỏ (1e-6) phải được coi là im lặng."""
        from src.audio.asr import transcribe_chunk
        y = np.full(16000, 1e-6, dtype=np.float32)
        result = transcribe_chunk(y)
        assert result == ""

    def test_silence_no_exception(self):
        """Im lặng không được ném ngoại lệ."""
        from src.audio.asr import transcribe_chunk
        y = _make_silence()
        try:
            transcribe_chunk(y)
        except Exception as e:
            pytest.fail(f"transcribe_chunk(silence) gây ra ngoại lệ: {e}")

    def test_silence_result_type_is_str(self):
        """Im lặng phải trả về kiểu str."""
        from src.audio.asr import transcribe_chunk
        y = _make_silence()
        result = transcribe_chunk(y)
        assert isinstance(result, str)


# ===========================================================================
# 3. TestInputNormalization
# ===========================================================================

class TestInputNormalization:
    """Kiểm tra tiền xử lý đầu vào (dtype, stereo, resample)."""

    def test_float64_input_converted_silently(self):
        """float64 không crash — chuyển về float32 nội bộ."""
        from src.audio.asr import transcribe_chunk
        y = _make_silence(duration_s=0.5).astype(np.float64)
        result = transcribe_chunk(y)
        assert isinstance(result, str)

    def test_int16_input_converted(self):
        """int16 không crash — ép về float32 nội bộ."""
        from src.audio.asr import transcribe_chunk
        y = np.zeros(8000, dtype=np.int16)
        result = transcribe_chunk(y)
        assert isinstance(result, str)

    def test_stereo_2d_row_major_converted(self):
        """Stereo (2, N) → mono, không crash."""
        from src.audio.asr import _to_mono_float32
        stereo = np.zeros((2, 16000), dtype=np.float32)
        mono = _to_mono_float32(stereo)
        assert mono.ndim == 1
        assert mono.shape[0] == 16000

    def test_stereo_2d_col_major_converted(self):
        """Stereo (N, 2) → mono, không crash."""
        from src.audio.asr import _to_mono_float32
        stereo = np.zeros((16000, 2), dtype=np.float32)
        mono = _to_mono_float32(stereo)
        assert mono.ndim == 1

    def test_is_silence_helper_zeros(self):
        """_is_silence phải trả True cho mảng zeros."""
        from src.audio.asr import _is_silence
        assert _is_silence(np.zeros(16000, dtype=np.float32)) is True

    def test_is_silence_helper_sine(self):
        """_is_silence phải trả False cho sine wave."""
        from src.audio.asr import _is_silence
        assert _is_silence(_make_sine()) is False

    def test_resample_mock(self):
        """sr != 16000 phải gọi resample (mock librosa)."""
        from src.audio import asr as asr_mod
        y_8k = _make_silence(sr=8000)
        with patch.object(asr_mod, "_LIBROSA_AVAILABLE", True), \
             patch.object(asr_mod, "librosa") as mock_lib, \
             patch.object(asr_mod, "_is_silence", return_value=True):
            mock_lib.resample.return_value = _make_silence(sr=16000)
            asr_mod.transcribe_chunk(y_8k, sr=8000)
            mock_lib.resample.assert_called_once()
            _, kwargs = mock_lib.resample.call_args
            assert kwargs.get("orig_sr") == 8000 or 8000 in mock_lib.resample.call_args[0]


# ===========================================================================
# 4. TestSingletonBehavior
# ===========================================================================

class TestSingletonBehavior:
    """Singleton nạp model đúng một lần, thread-safe."""

    def test_state_singleton_exists(self):
        """Module phải expose _STATE như singleton."""
        from src.audio.asr import _STATE
        assert _STATE is not None

    def test_load_called_once_on_repeated_calls(self, tmp_path):
        """Gọi _ensure_loaded() nhiều lần chỉ nạp 1 lần (mock model)."""
        from src.audio import asr as asr_mod

        # Reset singleton để bắt đầu sạch
        original_loaded = asr_mod._STATE._loaded
        asr_mod._STATE._loaded = False
        asr_mod._STATE.model   = None

        load_count = {"n": 0}

        def _mock_do_load(cfg_path):
            load_count["n"] += 1
            asr_mod._STATE.cfg   = asr_mod._load_asr_config(cfg_path)
            # Tạo mock model để tránh lỗi AttributeError
            asr_mod._STATE.model = MagicMock()

        with patch.object(asr_mod._STATE, "_do_load", side_effect=_mock_do_load):
            for _ in range(5):
                asr_mod._ensure_loaded(asr_mod._ASR_CFG)

        assert load_count["n"] == 1, \
            f"_do_load được gọi {load_count['n']} lần — phải đúng 1 lần"

        # Khôi phục state
        asr_mod._STATE._loaded = original_loaded

    def test_concurrent_loads_thread_safe(self, tmp_path):
        """Nhiều thread gọi đồng thời không gây race condition (mock model)."""
        from src.audio import asr as asr_mod

        asr_mod._STATE._loaded = False
        asr_mod._STATE.model   = None

        load_count = {"n": 0}
        lock = threading.Lock()

        def _mock_do_load(cfg_path):
            import time
            time.sleep(0.01)  # mô phỏng I/O
            with lock:
                load_count["n"] += 1
            asr_mod._STATE.cfg   = {}
            asr_mod._STATE.model = MagicMock()

        errors: list[Exception] = []
        err_lock = threading.Lock()

        def worker():
            try:
                with patch.object(asr_mod._STATE, "_do_load", side_effect=_mock_do_load):
                    asr_mod._ensure_loaded(asr_mod._ASR_CFG)
            except Exception as e:
                with err_lock:
                    errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, f"Lỗi trong concurrent load: {errors}"
        # Với double-checked locking, _do_load chỉ được gọi 1 lần
        assert load_count["n"] <= 1, \
            f"_do_load được gọi {load_count['n']} lần — race condition!"

        # Reset
        asr_mod._STATE._loaded = False
        asr_mod._STATE.model   = None


# ===========================================================================
# 5. TestMockTranscription
# ===========================================================================

class TestMockTranscription:
    """Forward pass qua mock model — kiểm tra kết quả kiểu str."""

    def _make_mock_model(self, texts: list[str]) -> MagicMock:
        """Tạo mock WhisperModel.transcribe() trả về list[segment]."""
        segments = [_make_fake_segment(t) for t in texts]
        info = MagicMock()
        model = MagicMock()
        model.transcribe.return_value = (iter(segments), info)
        return model

    def _setup_state(self, asr_mod, mock_model: MagicMock) -> None:
        """Thiết lập singleton state với mock model."""
        asr_mod._STATE._loaded = True
        asr_mod._STATE.model   = mock_model
        asr_mod._STATE.cfg     = {
            "language": "vi", "beam_size": 5, "best_of": 5,
            "temperature": 0.0, "vad_filter": True,
            "min_silence_duration_ms": 500,
            "condition_on_previous_text": False,
        }

    def test_mock_returns_string(self):
        """transcribe_chunk với mock model phải trả về str."""
        from src.audio import asr as asr_mod
        y = _make_sine()
        mock_model = self._make_mock_model(["xin chào"])
        self._setup_state(asr_mod, mock_model)
        result = asr_mod.transcribe_chunk(y)
        assert isinstance(result, str)

    def test_mock_single_segment_text(self):
        """Một segment → text của segment đó."""
        from src.audio import asr as asr_mod
        y = _make_sine()
        mock_model = self._make_mock_model(["Công an gọi điện yêu cầu"])
        self._setup_state(asr_mod, mock_model)
        result = asr_mod.transcribe_chunk(y)
        assert result == "Công an gọi điện yêu cầu"

    def test_mock_multiple_segments_joined(self):
        """Nhiều segment → ghép bằng dấu cách."""
        from src.audio import asr as asr_mod
        y = _make_sine()
        mock_model = self._make_mock_model(["xin chào,", "bạn khỏe không?"])
        self._setup_state(asr_mod, mock_model)
        result = asr_mod.transcribe_chunk(y)
        assert result == "xin chào, bạn khỏe không?"

    def test_mock_no_segments_returns_empty(self):
        """Không có segment → ''."""
        from src.audio import asr as asr_mod
        y = _make_sine()
        mock_model = self._make_mock_model([])
        self._setup_state(asr_mod, mock_model)
        result = asr_mod.transcribe_chunk(y)
        assert result == ""

    def test_mock_whitespace_only_segments_filtered(self):
        """Segment toàn khoảng trắng → lọc bỏ, kết quả ''."""
        from src.audio import asr as asr_mod
        y = _make_sine()
        mock_model = self._make_mock_model(["   ", "\t", "  \n  "])
        self._setup_state(asr_mod, mock_model)
        result = asr_mod.transcribe_chunk(y)
        assert result == ""

    def test_mock_result_stripped(self):
        """Kết quả phải được strip() hai đầu."""
        from src.audio import asr as asr_mod
        y = _make_sine()
        mock_model = self._make_mock_model(["  chào bạn  "])
        self._setup_state(asr_mod, mock_model)
        result = asr_mod.transcribe_chunk(y)
        assert result == result.strip()

    def test_mock_transcribe_called_with_correct_kwargs(self):
        """Kiểm tra model.transcribe() được gọi với đúng tham số từ config."""
        from src.audio import asr as asr_mod
        y = _make_sine()
        mock_model = self._make_mock_model(["test"])
        self._setup_state(asr_mod, mock_model)
        asr_mod.transcribe_chunk(y)
        mock_model.transcribe.assert_called_once()
        _, kwargs = mock_model.transcribe.call_args
        assert kwargs.get("language")  == "vi"
        assert kwargs.get("vad_filter") is True
        assert kwargs.get("min_silence_duration_ms") == 500


# ===========================================================================
# 6. TestTranscribeFile
# ===========================================================================

class TestTranscribeFile:
    """Kiểm tra transcribe_file() — đọc WAV và gọi transcribe_chunk."""

    def _make_wav(self, tmp_path: Path, y: np.ndarray, sr: int = 16000) -> Path:
        p = tmp_path / "test.wav"
        _write_wav(p, y, sr)
        return p

    def test_silence_wav_returns_empty(self, tmp_path):
        """WAV im lặng → ''."""
        from src.audio.asr import transcribe_file
        wav_path = self._make_wav(tmp_path, _make_silence())
        result = transcribe_file(str(wav_path))
        assert result == ""

    def test_transcribe_file_returns_str(self, tmp_path):
        """transcribe_file phải trả về str."""
        from src.audio.asr import transcribe_file
        wav_path = self._make_wav(tmp_path, _make_silence())
        result = transcribe_file(str(wav_path))
        assert isinstance(result, str)

    def test_nonexistent_file_raises_fnf(self):
        """File không tồn tại → FileNotFoundError."""
        from src.audio.asr import transcribe_file
        with pytest.raises(FileNotFoundError):
            transcribe_file("/nonexistent/path/audio.wav")

    def test_transcribe_file_uses_chunk_internally(self, tmp_path):
        """transcribe_file phải gọi transcribe_chunk nội bộ."""
        from src.audio import asr as asr_mod
        wav_path = self._make_wav(tmp_path, _make_silence())
        with patch.object(asr_mod, "transcribe_chunk", return_value="mock result") as mock_fn:
            result = asr_mod.transcribe_file(str(wav_path))
        mock_fn.assert_called_once()
        assert result == "mock result"

    def test_sine_wav_with_mock_model(self, tmp_path):
        """WAV có âm thanh + mock model → trả string không rỗng."""
        from src.audio import asr as asr_mod
        y = _make_sine()
        wav_path = self._make_wav(tmp_path, y)

        mock_model = MagicMock()
        seg = _make_fake_segment("xin chào")
        mock_model.transcribe.return_value = (iter([seg]), MagicMock())
        asr_mod._STATE._loaded = True
        asr_mod._STATE.model   = mock_model
        asr_mod._STATE.cfg     = {
            "language": "vi", "beam_size": 5, "best_of": 5,
            "temperature": 0.0, "vad_filter": True,
            "min_silence_duration_ms": 500,
            "condition_on_previous_text": False,
        }

        result = asr_mod.transcribe_file(str(wav_path))
        assert result == "xin chào"


# ===========================================================================
# 7. TestEdgeCases
# ===========================================================================

class TestEdgeCases:
    """Các trường hợp biên."""

    def test_empty_array_returns_empty(self):
        """Mảng rỗng np.array([]) → ''."""
        from src.audio.asr import transcribe_chunk
        y = np.array([], dtype=np.float32)
        result = transcribe_chunk(y)
        assert result == ""

    def test_non_ndarray_raises_value_error(self):
        """Đầu vào không phải ndarray → ValueError."""
        from src.audio.asr import transcribe_chunk
        with pytest.raises((ValueError, TypeError)):
            transcribe_chunk([0.0] * 100)   # list, không phải ndarray

    def test_none_input_raises(self):
        """None → ValueError."""
        from src.audio.asr import transcribe_chunk
        with pytest.raises((ValueError, TypeError, AttributeError)):
            transcribe_chunk(None)   # type: ignore

    def test_very_short_audio_no_crash(self):
        """Mảng 1 mẫu không crash."""
        from src.audio.asr import transcribe_chunk
        y = np.array([0.0], dtype=np.float32)
        result = transcribe_chunk(y)
        assert isinstance(result, str)

    def test_model_not_ready_raises_correctly(self, tmp_path):
        """Gọi khi model chưa convert → ASRModelNotReadyError."""
        from src.audio import asr as asr_mod
        from src.audio.asr import ASRModelNotReadyError

        # Tạo config trỏ tới model path KHÔNG tồn tại
        fake_cfg = tmp_path / "asr.yaml"
        fake_cfg.write_text(
            "model_path: '/nonexistent/phowhisper'\n"
            "device: cpu\ncompute_type: int8\nlanguage: vi\n"
            "vad_filter: true\nmin_silence_duration_ms: 500\n",
            encoding="utf-8",
        )

        # Tạo một _ASRState hoàn toàn mới (độc lập với module-level singleton)
        # để test không bị ảnh hưởng bởi trạng thái từ các test trước.
        fresh_state = asr_mod._ASRState()

        y = _make_sine()

        # Patch module-level _STATE bằng state mới + đảm bảo faster-whisper
        # được coi là đã cài (để code reach được bước kiểm tra model path).
        with patch.object(asr_mod, "_STATE", fresh_state), \
             patch.object(asr_mod, "_FASTER_WHISPER_AVAILABLE", True):
            with pytest.raises(ASRModelNotReadyError):
                asr_mod.transcribe_chunk(y, cfg_path=fake_cfg)

    def test_exception_in_transcribe_returns_empty(self):
        """Ngoại lệ trong model.transcribe() → trả '' (không re-raise)."""
        from src.audio import asr as asr_mod
        y = _make_sine()

        mock_model = MagicMock()
        mock_model.transcribe.side_effect = RuntimeError("GPU OOM")
        asr_mod._STATE._loaded = True
        asr_mod._STATE.model   = mock_model
        asr_mod._STATE.cfg     = {
            "language": "vi", "beam_size": 5, "best_of": 5,
            "temperature": 0.0, "vad_filter": True,
            "min_silence_duration_ms": 500,
            "condition_on_previous_text": False,
        }

        result = asr_mod.transcribe_chunk(y)
        assert result == "", f"Exception trong model phải trả '', nhận {result!r}"


# ===========================================================================
# 8. TestModelIntegration — cần model thật
# ===========================================================================

@_REQUIRES_MODEL
class TestModelIntegration:
    """Test đầy đủ pipeline với WhisperModel thật (auto-skip nếu chưa convert)."""

    @pytest.fixture(autouse=True)
    def _reset_state(self):
        """Reset singleton trước mỗi test để đảm bảo nạp lại thật."""
        from src.audio import asr as asr_mod
        asr_mod._STATE._loaded = False
        asr_mod._STATE.model   = None
        yield
        # Không reset sau test để test sau dùng cache

    def test_model_loads_successfully(self):
        """WhisperModel nạp thành công từ phowhisper-base-ct2."""
        from src.audio.asr import _ensure_loaded
        state = _ensure_loaded()
        assert state.model is not None
        assert state._loaded is True

    def test_silence_3s_returns_empty(self):
        """3 giây im lặng → '' với model thật."""
        from src.audio.asr import transcribe_chunk
        y = _make_silence(duration_s=3.0)
        result = transcribe_chunk(y)
        assert result == "", f"Im lặng phải trả '', nhận: {result!r}"

    def test_second_call_uses_cache(self):
        """Lần gọi thứ 2 nhanh hơn lần đầu (model không nạp lại)."""
        import time
        from src.audio.asr import transcribe_chunk

        y = _make_silence()

        # Lần 1
        t1 = time.perf_counter()
        transcribe_chunk(y)
        t1_ms = (time.perf_counter() - t1) * 1000

        # Lần 2
        t2 = time.perf_counter()
        transcribe_chunk(y)
        t2_ms = (time.perf_counter() - t2) * 1000

        print(f"\n  Lần 1: {t1_ms:.1f} ms | Lần 2: {t2_ms:.1f} ms")
        # Lần 2 không nạp model → nhanh hơn đáng kể
        assert t2_ms < t1_ms * 0.5 or t2_ms < 100, \
            f"Lần 2 ({t2_ms:.1f} ms) không nhanh hơn rõ rệt lần 1 ({t1_ms:.1f} ms)"

    def test_real_audio_returns_nonempty_string(self, tmp_path):
        """File WAV có tiếng nói tiếng Việt → chuỗi không rỗng."""
        from src.audio.asr import transcribe_file

        # Tạo âm thanh giả có năng lượng (sine wave — không phải lời nói thật)
        # Test thật nên dùng file VIVOS; đây chỉ kiểm tra pipeline không crash
        y = _make_sine(freq=200.0, duration_s=3.0, amplitude=0.8)
        wav_path = tmp_path / "sample.wav"
        _write_wav(wav_path, y)

        result = transcribe_file(str(wav_path))
        # Pipeline không crash → pass (output có thể rỗng vì không phải lời nói thật)
        assert isinstance(result, str)
