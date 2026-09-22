# B4 — Đo độ bền trước lỗi ASR, cập nhật `/detect/voice`, benchmark latency

**Người phụ trách:** B · **Tuần:** 4–5 · **Phụ thuộc:** B3; **B4a cần S2** (A3 xong) · **Nguồn:** kế hoạch mục 5.2 (Tuần 4–5), 5.3, 6 (R10)

Chia hai phần: **B4a** (robustness, cần `process_and_predict`) và **B4b** (API + latency). B4b có thể làm song song trong lúc chờ S2, dùng bản giả lập (stub) của hàm.

## B4a — Độ bền trước lỗi ASR (rủi ro R10)

### Việc cần làm
1. **`scripts/eval_robustness.py`:** với mỗi điều kiện (`clean`, `snr20`, `snr10`, `snr5`):
   - Chạy `process_and_predict()` trên `transcript_reference` (**đường chuẩn sạch**) và trên `transcript_asr`.
   - Tính Accuracy, Macro-F1, FNR, FPR cho từng đường; tính Δ (ASR − sạch).
   - Tách theo `region` × `tone`.
   → `results/robustness_summary.csv`, `results/robustness_by_region_tone.csv`.
2. **Phân tích FN mới phát sinh do ASR:** các mẫu sạch đúng nhưng ASR sai; cho mỗi mẫu ghi từ khoá bị mất/sai → `results/robustness_new_fn.csv`. Đối chiếu với `keyword_recall_asr.csv` (B3).
3. **Độ bền ở mức chunk** (mô phỏng thời gian thực): chia file 4 s, phiên âm luỹ kế từng chunk, ghi dự đoán sau chunk 1, 2, 3, … → `results/robustness_chunked.csv`. Trả lời: cần bao nhiêu giây để kết luận ổn định?
4. **Kết luận theo ngưỡng (do nhóm quyết định, không tự quyết):** nếu Δ Accuracy hoặc Δ FNR vượt 5 điểm phần trăm ở điều kiện sạch, ghi rõ trong `reports` và **đề xuất** (không tự làm) augmentation lỗi ASR điển hình từ `asr_error_examples.csv` cho A. Nếu không vượt, ghi "không cần".
5. **Quyết định `base` hay `small`:** chạy lại 1–2 bước trên với `phowhisper-small` (đổi `configs/asr.yaml`), so sánh WER và latency; ghi bảng đánh đổi. Không tự đổi mặc định nếu latency p95 chunk vượt ngưỡng đã chốt cho demo (mặc định gợi ý: 2 s/chunk 4 s; điền ngưỡng thực tế nhóm chọn).

## B4b — Route `/detect/voice` và latency

### Việc cần làm
1. **Cập nhật route** theo kế hoạch mục 5.3: `load_and_resample → asr.transcribe_chunk → append_transcript_chunk (Redis) → pii_anonymizer.mask (chỉ để log/hiển thị) → process_and_predict(full_transcript)`. Bỏ `asyncio.gather` và mọi lời gọi acoustic/fusion.
2. **Chạy ASR ngoài event loop:** `transcribe_chunk` là tác vụ CPU chặn; bọc bằng `await asyncio.to_thread(...)` (hoặc `run_in_executor`) để không đứng các request khác. Kế hoạch gốc gọi đồng bộ trong hàm `async` — chỉnh lại và ghi chú.
3. **Đưa văn bản gốc (chưa che PII) vào `process_and_predict`**, chỉ dùng bản đã che cho log và trường `transcript` trả về; nếu che trước khi phân loại thì mất số điện thoại/số tài khoản là tín hiệu của A1. Ghi rõ quyết định này trong `docs/agent_notes/B4.md`.
4. **`lifespan()`:** bỏ nạp `fusion_model`; nạp ASR và `process_and_predict` (khởi động sẵn để tránh chậm ở request đầu).
5. **Test API** `tests/test_voice_route.py`: 2–3 chunk cùng `session_id` (transcript được cộng dồn), `is_final_chunk=True` xoá session, file rỗng/định dạng sai trả lỗi 4xx rõ ràng (không 500).
6. **Benchmark latency** `scripts/bench_latency.py` — chạy ≥ 50 chunk 4 s, sau khi khởi động nóng: đo riêng **NLP preprocess / PhoBERT + Hybrid / ASR theo chunk / tổng route**; báo p50, p95, max → `results/latency.csv`. Không còn dòng "Fusion". Ghi cấu hình máy (CPU, số nhân, RAM).
7. Cập nhật `docker/` và `requirements.txt` (bỏ phụ thuộc acoustic); build Docker Compose và xác nhận service khởi động, `/detect/voice` chạy trong container.

## Điểm cần lưu ý
- Route `/detect/text` và phần Redis/PII/UX **không viết lại**; chỉ chạm phần voice.
- Nếu `process_and_predict` chưa sẵn sàng, dùng stub trả `Prediction` cố định để làm B4b, và **gỡ stub** trước khi báo hoàn thành.

## Đầu ra
`scripts/eval_robustness.py`, `scripts/bench_latency.py`, `src/api/voice_route.py` (cập nhật), `tests/test_voice_route.py`, `results/robustness_*.csv`, `results/latency.csv`, `docs/agent_notes/B4.md`.

## Nghiệm thu
- [ ] `robustness_summary.csv` có đủ 4 điều kiện × {sạch, ASR} × 4 chỉ số + Δ.
- [ ] `pytest tests/test_voice_route.py` pass; không còn stub.
- [ ] `grep -rn "fusion\|acoustic\|asyncio.gather" src/api/` không còn kết quả liên quan tới luồng thoại.
- [ ] `latency.csv` có p50/p95 cho từng thành phần và cấu hình máy.
- [ ] `docker compose up` chạy được và `/detect/voice` trả phản hồi hợp lệ trong container.
