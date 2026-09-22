# B2 — Tiền xử lý audio, chuẩn hoá văn bản và đo WER

**Người phụ trách:** B · **Tuần:** 2 · **Phụ thuộc:** B1 · **Nguồn:** kế hoạch mục 2.4, 5.2 (Tuần 1–2)

## Mục tiêu
Có bộ dữ liệu audio chuẩn hoá và bộ đo WER tái lập, so sánh PhoWhisper với Whisper gốc.

## Việc cần làm
1. **`src/audio/preprocess.py`:** `load_and_resample(source, target_sr=16000) -> (np.ndarray, int)` (nhận bytes hoặc đường dẫn; mono; float32; chuẩn hoá biên độ nhẹ, không cắt xén), và `split_chunks(y, sr, seconds=4, overlap=0.5)` (chunk 3–5 s theo kế hoạch). Không viết trích đặc trưng âm học.
2. **Mô phỏng nhiễu viễn thông** `make_noisy(y, sr)`: hạ xuống 8 kHz rồi nâng lại 16 kHz, cộng nhiễu Gaussian với SNR ngẫu nhiên có seed (đặt 3 mức: 20/10/5 dB). Chỉ tạo **dữ liệu test độ bền**, không dùng để huấn luyện.
3. **`src/audio/text_norm.py`:** `normalize_for_wer(text)` theo hợp đồng mục 4 (chữ thường, bỏ dấu câu, giữ dấu thanh, gộp khoảng trắng). Test riêng với vài câu có dấu.
4. **Kiểm tra `data/audio/metadata.csv`:** đủ cột theo hợp đồng; báo cáo số file theo `region × tone × label`, số `speaker_id`. Nếu thiếu cột hoặc file audio, ghi vào `docs/agent_notes/B2.md` và **dừng phần phụ thuộc dữ liệu** — không tự tạo dữ liệu giả.
5. **`scripts/eval_wer.py`:** đo WER (`jiwer`) cho danh sách model qua tham số `--models`:
   - `phowhisper-base` (bắt buộc), `whisper-base` gốc (bắt buộc để so sánh), `phowhisper-tiny`/`small` (tuỳ chọn),
   - trên (a) VIVOS test (nếu có sẵn, hoặc mẫu con ≥ 200 câu), (b) tập tự thu sạch, (c) tập tự thu sau `make_noisy` ở 3 mức SNR.
   Xuất `results/wer_by_model.csv` (model, tập, điều kiện, n_mẫu, WER) và `results/wer_by_region_tone.csv`.
6. **Nêu rõ độ tin cậy:** với mỗi ô WER có n < 30 mẫu, đánh dấu "n nhỏ" trong CSV.

## Điểm cần lưu ý
- Số WER trong kế hoạch (8,46% cho base, 6,33% small, 10,41% tiny) là số công bố trên VIVOS; **không chép vào kết quả của nhóm**, chỉ dùng làm mốc đối chiếu trong báo cáo.
- Whisper gốc cần chạy đúng tham số `language="vi"` để so sánh công bằng.

## Đầu ra
`src/audio/preprocess.py`, `src/audio/text_norm.py`, `scripts/eval_wer.py`, `results/wer_by_model.csv`, `results/wer_by_region_tone.csv`, `tests/test_audio_preprocess.py`.

## Nghiệm thu
- [ ] `pytest tests/test_audio_preprocess.py` pass (resample ra đúng 16 kHz, mono, chunk đúng độ dài).
- [ ] `results/wer_by_model.csv` có ít nhất PhoWhisper-base và Whisper gốc trên tập tự thu.
- [ ] Chạy lại `eval_wer.py` cho cùng số liệu (seed nhiễu cố định).
- [ ] Ô nào n < 30 đều có cờ "n nhỏ".
