# B3 — Sinh transcript ASR cho toàn bộ tập thu âm

**Người phụ trách:** B · **Tuần:** 3 · **Phụ thuộc:** B1, B2 · **Nguồn:** kế hoạch mục 5.2 (Tuần 3)

## Mục tiêu
Tạo `transcript_asr` cho mọi file thu âm đã gán nhãn — đầu vào cho phân tích WER theo miền/sắc thái và cho thí nghiệm độ bền ở B4.

## Việc cần làm
1. **`scripts/transcribe_dataset.py`** `--model phowhisper-base --condition {clean,snr20,snr10,snr5}`:
   đọc `metadata.csv` → `load_and_resample` → phiên âm **cả file** (không chia chunk; đường chunk kiểm ở B4) → ghi `data/audio/transcripts_asr_<model>_<condition>.csv` gồm metadata + `transcript_asr` + `asr_latency_ms` + `audio_seconds`.
2. **Chạy 4 điều kiện** (sạch + 3 mức nhiễu) cho `phowhisper-base`; chạy thêm `whisper-base` gốc ở điều kiện sạch cho bảng so sánh.
3. **Tính WER từng file** (cột `wer`) bằng `normalize_for_wer` + `jiwer`, rồi tổng hợp theo `region` (Bắc/Trung/Nam) × `tone` (trung tính/áp lực) → `results/wer_region_tone_<condition>.csv` (mean, median, n).
4. **Phân tích lỗi ASR trên từ khoá quan trọng** `scripts/keyword_asr_errors.py`: với từng từ khoá trong `configs/scam_keywords.yaml` (của A) đếm số lần xuất hiện trong `transcript_reference` và số lần còn nguyên trong `transcript_asr` → **tỷ lệ giữ từ khoá (keyword recall)** theo điều kiện. Lưu `results/keyword_recall_asr.csv`. Đây là số liệu trực tiếp cho rủi ro R10.
5. **Liệt kê 30 lỗi phiên âm điển hình** (từ khoá bị sai, ví dụ "công an" → "công ăn") vào `results/asr_error_examples.csv` để dùng cho báo cáo và cho tuỳ chọn augmentation ở B4.

## Điểm cần lưu ý
- Nếu `configs/scam_keywords.yaml` chưa có (A1 chưa xong), tạm dùng danh sách trong kế hoạch mục 3.2 và ghi chú; không tự chế từ khoá khác.
- Ghi `git_commit`, tên model, `compute_type` vào file kết quả.

## Đầu ra
`scripts/transcribe_dataset.py`, `scripts/keyword_asr_errors.py`, các CSV trong `data/audio/` và `results/`.

## Nghiệm thu
- [ ] Mỗi file transcript có số dòng bằng `metadata.csv`, không có `transcript_asr` bị thiếu ngoài trường hợp im lặng thật (liệt kê các ca đó).
- [ ] Có bảng WER theo 3 miền × 2 sắc thái cho ít nhất điều kiện sạch.
- [ ] `keyword_recall_asr.csv` có đủ 4 điều kiện.
- [ ] Không có hàng nào của `speaker_id` bị lẫn giữa các split (nếu split đã gán).
