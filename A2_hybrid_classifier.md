# A2 — `HybridTextClassifier` (PhoBERT 768-d + 11 đặc trưng cấu trúc)

**Người phụ trách:** A · **Tuần:** 2–3 · **Phụ thuộc:** A1 · **Nguồn:** kế hoạch mục 3.3

## Mục tiêu
Huấn luyện classifier lai và lưu checkpoint đủ để serving nạp lại (kèm scaler).

## Đầu vào
- Baseline PhoBERT-only hiện có và hàm `extract_text_embedding()` (tìm trong repo; nếu chưa có thì viết theo cùng cách baseline đang lấy embedding — ghi rõ pooling nào: `[CLS]` hay mean).
- `src/features/structured_features.py` (A1); split `text_{train,val,test}.csv`.

## Việc cần làm
1. **Tính và cache embedding** cho Train/Val/Test vào `data/cache/phobert_emb_{split}.npy` (kèm hash của văn bản để phát hiện cache cũ). Nếu PhoBERT baseline được fine-tune, dùng **encoder đã fine-tune**; nếu không, dùng PhoBERT gốc đóng băng. Ghi lựa chọn vào `config.json`.
2. **Tính structured features** cho cả ba split bằng cùng `nlp_preprocess.run()`.
3. **Chuẩn hoá:** `StandardScaler` **fit chỉ trên Train**, áp cho các cột số (`n_*_kw, message_length, uppercase_ratio, exclamation_count`); cột boolean giữ 0/1. Lưu `scaler.joblib`.
4. **`src/models/hybrid_classifier.py`:** class `HybridTextClassifier` đúng kế hoạch (`768+11 → 128 → ReLU → Dropout 0.3 → num_classes`).
5. **Huấn luyện** `scripts/train_hybrid.py`: AdamW, class weight theo tần suất nhãn Train, Early Stopping theo **Macro-F1 trên Val** (patience 5), seed 42. Lưu log từng epoch vào `results/hybrid_train_log.csv`.
6. **Chọn ngưỡng** quyết định trên Val (mặc định 0.5; nếu FNR Val > 5% thì thử hạ ngưỡng và ghi lại đánh đổi FPR). Ngưỡng cuối lưu trong `config.json`, **không** chọn trên Test.
7. **Lưu** `models/text/hybrid/{model.pt, scaler.joblib, config.json}`; `config.json` chứa `structured_dim, feature_order, hidden=128, dropout, threshold, embedding_source, seed, git_commit`.
8. Đánh giá một lần trên Test → `results/hybrid_test_metrics.json` (Accuracy, Macro-F1, FNR, FPR, ma trận nhầm lẫn, số mẫu mỗi lớp).

## Điểm cần lưu ý
- Nếu tập dữ liệu nhỏ (vài trăm mẫu), báo cáo thêm kết quả **5-fold stratified CV** (mean ± std) thay vì một lần chạy — số Test đơn lẻ trên tập nhỏ dễ đánh lừa.
- Không fit scaler trên Train+Val+Test, không chọn ngưỡng/epoch bằng Test.
- Nếu Hybrid **không** tốt hơn PhoBERT-only, vẫn báo cáo đúng số liệu; đây là kết quả hợp lệ cho ablation ở A4.

## Đầu ra
`src/models/hybrid_classifier.py`, `scripts/train_hybrid.py`, `models/text/hybrid/*`, `results/hybrid_test_metrics.json`, `results/hybrid_train_log.csv`.

## Nghiệm thu
- [ ] `python scripts/train_hybrid.py --seed 42` chạy hết, chạy lại cho cùng kết quả (±0.5% F1).
- [ ] Nạp lại `model.pt` + `scaler.joblib` ở tiến trình mới cho dự đoán trùng khớp với lúc train (test tự động).
- [ ] Không có dòng nào trong code gọi `scaler.fit` trên dữ liệu ngoài Train.
- [ ] `hybrid_test_metrics.json` có đủ Accuracy, Macro-F1, FNR, FPR.
