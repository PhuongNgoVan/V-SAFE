# BỘ TÀI LIỆU GIAO VIỆC CHO AI AGENT — CẢI TIẾN ASR (2 NGƯỜI)

Nguồn: `KE_HOACH_CAI_TIEN_ASR_SIMPLIFIED.md`. Kế hoạch gốc chia 3 thành viên (TV1/TV2/TV3); bản này gộp lại cho **2 người**.

## 1. Phân công

| | **Người A — NLP & Hybrid Classifier** | **Người B — ASR, Robustness & Serving** |
|---|---|---|
| Gộp từ | TV1 (NLP) | TV2 (Speech) + phần serving của TV3 |
| Trách nhiệm chính | Đặc trưng cấu trúc, `HybridTextClassifier`, `process_and_predict()`, ablation PhoBERT-only vs Hybrid | PhoWhisper (CT2), tiền xử lý audio, đo WER, đo độ bền trước lỗi ASR, route `/detect/voice`, benchmark latency |
| Deliverable code | `src/features/structured_features.py`, `src/models/hybrid_classifier.py`, `src/pipeline/predict.py` | `src/audio/asr.py`, `src/audio/preprocess.py`, `src/api/voice_route.py` (cập nhật), `scripts/convert_phowhisper.sh` |
| Deliverable báo cáo | `reports/chapter3_hybrid_ablation.md` | `reports/chapter3_asr_robustness.md`, bảng latency |
| Số bước | 4 (A1–A4) | 5 (B1–B5) |

Lý do chia như vậy: A sở hữu toàn bộ "phần văn bản" (một model, một pipeline); B sở hữu mọi thứ liên quan đến âm thanh và triển khai. Hai bên chỉ giao nhau ở **một hàm duy nhất**: `process_and_predict()` (xem `00_SHARED_CONTRACT.md`).

> Phần TV3 không đổi (Redis session, PII Anonymizer, UX cảnh báo, Docker Compose, route `/detect/text`) **thuộc B nhưng chỉ bảo trì**, không viết lại. Nếu B quá tải, chuyển bước B4 sang A sau khi A4 xong.

## 2. Thứ tự & điểm đồng bộ

```
Tuần 1   A1 ─────────┐         B1 ─────────┐
Tuần 2   A2          │         B2          │
Tuần 3   A2 (ablation) ─▶ A4   B3 (sinh transcript ASR)
              │                        │
Tuần 4        └─▶ A3 (process_and_predict) ══▶ [S2] ══▶ B4a (robustness)
Tuần 5   A4 hoàn thiện                B4b (voice API + latency)
Tuần 6   A4 báo cáo ─────────────▶ [S3] ◀───────── B5 báo cáo
```

| Mã | Điểm đồng bộ | Điều kiện |
|---|---|---|
| **S1** (ngày 1) | Chốt `00_SHARED_CONTRACT.md` | Cả hai xác nhận chữ ký hàm, schema, đường dẫn |
| **S2** (đầu Tuần 4) | A bàn giao `process_and_predict()` + checkpoint model | Test A3 pass; B bắt đầu B4a |
| **S3** (Tuần 6) | Gộp số liệu vào Chương 3/4 | Cả hai file `reports/*.md` đã có số liệu thật |

A và B **độc lập hoàn toàn** từ S1 đến S2: B3 dùng `transcript_reference` sạch làm dữ liệu chuẩn nên không cần chờ model của A.

## 3. Quy ước chung cho mọi AI agent

1. **Đọc repo trước khi viết code.** Kế hoạch gốc nhắc tới các module đã có (`nlp_preprocess`, `pii_anonymizer`, `text_model`, baseline PhoBERT). Chạy `find . -name "*.py" | head -100` và đọc các module liên quan; nếu tên/đường dẫn khác giả định trong file bước, **dùng tên thực tế** và ghi lại chênh lệch vào `docs/agent_notes/<mã_bước>.md`.
2. **Không bịa số liệu.** Mọi con số WER/Accuracy/F1/FNR/latency phải do code chạy ra và lưu file (`results/*.json|csv`). Nếu không chạy được (thiếu dữ liệu, thiếu mạng, thiếu GPU), ghi rõ "chưa đo" — không điền số ước tính.
3. **Tách Train/Val/Test trước khi fit bất cứ thứ gì** (scaler, từ điển, threshold). Không để dữ liệu Test rò vào Train, kể cả khi augment.
4. **Đặt seed cố định** (`42`) và ghi vào file kết quả.
5. **Mỗi bước kết thúc bằng checklist nghiệm thu.** Chỉ báo hoàn thành khi mọi mục được tích và có lệnh kiểm chứng chạy được.
6. **Không sửa file ngoài phạm vi bước.** Nếu phát hiện lỗi ở module người kia, ghi vào `docs/agent_notes/HANDOFF_<A|B>.md` thay vì tự sửa.
7. **Commit nhỏ**, thông điệp dạng `[A1] thêm extract_structured_features + test`.
8. Khi gặp mâu thuẫn trong kế hoạch, làm theo mục "Điểm cần lưu ý" của file bước, và báo lại người phụ trách.

## 4. Cách giao cho agent

Với mỗi bước, đưa cho agent prompt:

```
Đọc README.md, 00_SHARED_CONTRACT.md và <đường dẫn file bước>.
Thực hiện đúng file bước đó, tuân thủ "Quy ước chung" trong README.
Khi xong, in ra checklist nghiệm thu với kết quả từng mục (PASS/FAIL) kèm lệnh đã chạy.
```

## 5. Danh sách file

```
agent_tasks/
├── README.md
├── 00_SHARED_CONTRACT.md
├── A_nlp_hybrid/
│   ├── A1_structured_features.md
│   ├── A2_hybrid_classifier.md
│   ├── A3_process_and_predict.md
│   └── A4_ablation_report.md
└── B_asr_serving/
    ├── B1_asr_setup_phowhisper.md
    ├── B2_audio_preprocess_wer.md
    ├── B3_transcribe_dataset.md
    ├── B4_robustness_and_voice_api.md
    └── B5_chapter3_report.md
```
"# V-SAFE" 
