# A3 — `process_and_predict()` — điểm bàn giao cho B (S2)

**Người phụ trách:** A · **Tuần:** 4 (đầu tuần) · **Phụ thuộc:** A2 · **Nguồn:** kế hoạch mục 5.1 Tuần 4, mục 4

## Mục tiêu
Đóng gói toàn bộ đường xử lý văn bản thành **một hàm**, dùng chung cho `/detect/text` và `/detect/voice`.

## Việc cần làm
1. **Tạo `src/pipeline/predict.py`** theo `00_SHARED_CONTRACT.md` mục 1:
   `text → nlp_preprocess.run → extract_text_embedding → extract_structured_features → scaler → HybridTextClassifier → Prediction`.
2. **Nạp model một lần** (lazy singleton, an toàn đa luồng). Model, scaler, ngưỡng đọc từ `models/text/hybrid/config.json`.
3. **`scam_type`:** dùng cơ chế phân loại 4 nhóm hiện có trong `text_model`; nếu chưa có, suy ra bằng luật từ nhóm từ khoá nổi trội (authority → giả mạo công an, financial+bank → giả mạo ngân hàng, reward → trúng thưởng, …) và ghi rõ trong docstring rằng đây là luật, không phải model.
4. **`flagged_keywords`** lấy từ `matched_keywords()` (A1). **`risk_level`** dùng `score_to_risk_level()` hiện có; nếu chưa có, đặt `low < 0.4 ≤ medium < 0.7 ≤ high` trong `configs/risk_thresholds.yaml`.
5. **Đo `processing_time_ms`** bằng `time.perf_counter()` trong hàm.
6. **Xử lý biên:** chuỗi rỗng / toàn khoảng trắng → `Prediction(is_fraud=False, confidence_score=0.0, risk_level="low", ...)`, không ném lỗi; văn bản rất dài → cắt theo `max_length` của tokenizer, không crash.
7. **Cập nhật `/detect/text`** (nếu đang dùng đường xử lý cũ) để gọi `process_and_predict`. Chỉ thay lời gọi, không đụng phần Redis/PII.
8. **Test tích hợp** `tests/test_process_and_predict.py`: 5 câu lừa đảo, 5 câu bình thường, chuỗi rỗng, chuỗi 5.000 ký tự, gọi 2 lần liên tiếp (lần 2 không nạp lại model).
9. **Báo B** (ghi vào `docs/agent_notes/HANDOFF_A.md`): đường dẫn checkpoint, cách import, thời gian trung bình mỗi lần gọi (CPU).

## Điểm cần lưu ý
- Hàm nhận văn bản **thô**; đừng tiền xử lý hai lần (một lần ở route, một lần ở hàm).
- Không đổi chữ ký hàm sau S2 mà không báo B.

## Nghiệm thu
- [ ] `pytest tests/test_process_and_predict.py` pass.
- [ ] `python -c "from src.pipeline.predict import process_and_predict as p; print(p('Công an yêu cầu chuyển khoản ngay'))"` chạy được và trả `Prediction`.
- [ ] Lần gọi thứ hai nhanh hơn rõ rệt lần đầu (không nạp lại model) — in cả hai thời gian.
- [ ] `/detect/text` trả kết quả giống `process_and_predict` cho cùng đầu vào.
- [ ] `HANDOFF_A.md` đã viết → **S2 đạt**.
