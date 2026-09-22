# B1 — Cài PhoWhisper (CTranslate2) và module ASR

**Người phụ trách:** B · **Tuần:** 1 · **Phụ thuộc:** không · **Nguồn:** kế hoạch mục 2.2–2.3

## Mục tiêu
Thay `openai/whisper-base` bằng `vinai/PhoWhisper-base` chạy qua `faster-whisper` (int8, CPU), giữ nguyên API tầng serving.

## Việc cần làm
1. **Kiểm tra repo:** tìm code ASR hiện có (`faster-whisper`, `WhisperModel`). Ghi lại đường dẫn, chữ ký hàm cũ vào `docs/agent_notes/B1.md`.
2. **Cài phụ thuộc** (cập nhật `requirements.txt` với phiên bản đã cố định): `transformers ctranslate2 faster-whisper jiwer soundfile librosa`. Xoá `pyAudioAnalysis` nếu có; **không** thêm Librosa cho đặc trưng âm học (chỉ dùng để resample nếu cần).
3. **Viết `scripts/convert_phowhisper.sh`:**
   ```bash
   ct2-transformers-converter \
     --model vinai/PhoWhisper-base \
     --output_dir models/asr/phowhisper-base-ct2 \
     --copy_files tokenizer.json preprocessor_config.json \
     --quantization int8
   ```
   Kế hoạch gốc chỉ copy `tokenizer_config.json`; `faster-whisper` thường cần thêm `tokenizer.json` và `preprocessor_config.json`. Chạy thử — nếu thiếu file nào thì bổ sung `--copy_files` cho đến khi nạp được. Script phải **idempotent** (bỏ qua nếu thư mục đã đủ file).
4. **Viết `src/audio/asr.py`** theo `00_SHARED_CONTRACT.md` mục 2:
   - Nạp `WhisperModel` **một lần** (lazy singleton), `device="cpu"`, `compute_type="int8"`; tên/đường dẫn model đọc từ `configs/asr.yaml` (để đổi sang `small`/`tiny` không phải sửa code).
   - `transcribe_chunk(y, sr=16000)`: nhận numpy float32 mono; nếu `sr != 16000` thì resample; `language="vi"`, `vad_filter=True`, `min_silence_duration_ms=500`; nối text các segment, `strip()`.
   - `transcribe_file(path)`: đọc file rồi gọi `transcribe_chunk`.
   - Đoạn im lặng/không có segment → trả `""`, không ném lỗi.
5. **Test** `tests/test_asr.py`: nạp model; phiên âm 1 file mẫu tiếng Việt ngắn (lấy từ VIVOS hoặc tự thu) và kiểm tra kết quả là chuỗi không rỗng; 3 giây im lặng → `""`.
6. **Quyết định lưu checkpoint:** nếu > 100 MB, thêm vào Git LFS hoặc `.gitignore` kèm hướng dẫn chạy script trong `README` (không convert lại khi build Docker).

## Điểm cần lưu ý
- Nếu môi trường không truy cập được HuggingFace, **dừng và báo** — không thay bằng model khác.
- Chưa chọn `small` hay `base`: quyết định ở B4 dựa trên số đo, mặc định `base`.

## Đầu ra
`scripts/convert_phowhisper.sh`, `configs/asr.yaml`, `src/audio/asr.py`, `tests/test_asr.py`, `models/asr/phowhisper-base-ct2/`, `docs/agent_notes/B1.md`.

## Nghiệm thu
- [ ] `bash scripts/convert_phowhisper.sh` chạy hai lần liên tiếp không lỗi.
- [ ] `pytest tests/test_asr.py` pass.
- [ ] `python -c "from src.audio.asr import transcribe_file; print(transcribe_file('<file mẫu>'))"` in ra tiếng Việt có dấu hợp lý.
- [ ] Không còn import `pyAudioAnalysis` hay module acoustic trong `src/`.
