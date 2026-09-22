# A4 — Ablation PhoBERT-only vs Hybrid & báo cáo Chương 3

**Người phụ trách:** A · **Tuần:** 3 (chạy) → 5–6 (hoàn thiện) · **Phụ thuộc:** A2 (số liệu), A3 + B3 (mục tuỳ chọn) · **Nguồn:** kế hoạch mục 3.3 cuối, mục 8

## Mục tiêu
Thí nghiệm ablation thay thế bảng so sánh Early/Late Fusion đã bỏ, và viết phần báo cáo tương ứng.

## Việc cần làm
1. **Ablation chính** `scripts/run_ablation.py`, cùng split, cùng seed, cùng ngưỡng chọn trên Val:
   | Cấu hình | Mô tả |
   |---|---|
   | M0 | PhoBERT-only (baseline hiện có) |
   | M1 | Hybrid đầy đủ (768 + 11) |
   | M2 | Hybrid bỏ 2 cột đặc thù kênh (`uppercase_ratio`, `exclamation_count`) |
   | M3 | Chỉ 11 đặc trưng cấu trúc (không PhoBERT), dùng Logistic Regression — mốc tham chiếu |
   Với mỗi cấu hình: Accuracy, Macro-F1, FNR, FPR. Nếu dữ liệu nhỏ, chạy 5-fold CV (mean ± std) cho cả 4.
2. **Kiểm định ý nghĩa:** với M0 vs M1, thực hiện McNemar (cùng tập Test) hoặc bootstrap 1.000 lần cho ΔF1 và ΔFNR; báo khoảng tin cậy 95%. Ghi rõ nếu **không** có khác biệt có ý nghĩa.
3. **Phân tích lỗi:** liệt kê 20 mẫu M1 sai (FN trước, FP sau), gắn nhóm nguyên nhân (thiếu từ khoá, từ khoá gây nhiễu, văn phong lạ…), lưu `results/hybrid_error_analysis.csv`.
4. **Mục tuỳ chọn khi có transcript của B (sau B3):** chạy M0/M1/M2 trên `transcript_reference` để so sánh cột đặc thù kênh. Không bắt buộc nếu B chưa xong.
5. **Viết `reports/chapter3_hybrid_ablation.md`** gồm: mô tả đặc trưng (bảng 11 cột), kiến trúc, quy trình huấn luyện, **bảng ablation**, kiểm định, phân tích lỗi, giới hạn. Phần "Giới hạn" bắt buộc nêu: mất tín hiệu giọng điệu (kế hoạch mục 0), và rủi ro lệch phân phối giữa SMS và transcript ASR.
6. **Đưa mục "Hướng phát triển"** (kế hoạch mục 7: thêm tốc độ nói từ timestamp của `faster-whisper` vào `structured_features`) vào cuối báo cáo — chỉ mô tả, không cài đặt.

## Điểm cần lưu ý
- Chỉ trích số liệu từ file trong `results/`; bảng trong báo cáo sinh bằng script, không gõ tay.
- Kết luận phải khớp số liệu: nếu Hybrid ≈ PhoBERT-only thì viết đúng như vậy và nêu lợi ích còn lại (khả năng diễn giải, `flagged_keywords`).

## Đầu ra
`scripts/run_ablation.py`, `results/ablation_*.{json,csv}`, `results/hybrid_error_analysis.csv`, `reports/chapter3_hybrid_ablation.md`.

## Nghiệm thu
- [ ] Bảng ablation có đủ M0–M3 × 4 chỉ số, sinh tự động từ `results/`.
- [ ] Có khoảng tin cậy hoặc p-value cho M0 vs M1.
- [ ] Báo cáo có mục Giới hạn và Hướng phát triển.
- [ ] Chạy lại `run_ablation.py` cho bảng giống hệt (cùng seed).
