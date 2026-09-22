# MASTER PROJECT PLAN
## Đề tài: Nghiên cứu và phát triển hệ thống nhận diện và chống lừa đảo qua tin nhắn và cuộc gọi
**Nhóm 6 — PTIT — Học phần Phương pháp luận nghiên cứu khoa học — 2026**

---

## 1. DANH SÁCH DATASET KHẢ DỤNG CHO TIẾNG VIỆT

> Không có bộ dữ liệu "vishing tiếng Việt" công khai quy mô lớn nào tồn tại sẵn (đây chính là Khoảng trống nghiên cứu 2 đã nêu ở Chương 1 báo cáo PPLNCKH) — nhóm phải **kết hợp** dữ liệu công khai + tự thu thập + tự giả lập. Bảng dưới liệt kê mọi nguồn có thể tận dụng ngay, theo từng nhóm.

### 1.1. Dataset văn bản (SMS/OTT) — dùng cho Thành viên 1

| # | Dataset | Quy mô | Ngôn ngữ | Link | Cách dùng trong đề tài |
|---|---|---|---|---|---|
| 1 | **ViSpamDetection** (UIT, Son T. Luu) | Spam review TMĐT tiếng Việt (v1 & v2) | Tiếng Việt | `https://huggingface.co/datasets/sonlam1102/vispamdetection` (gated, liên hệ `sonlt@uit.edu.vn`) | Nguồn văn phong lừa đảo/spam tiếng Việt thật, dùng bổ sung negative/positive samples |
| 2 | **ViHSD** (UIT — Vietnamese Hate Speech Detection) | ~33.400 comment, 3 nhãn (CLEAN/OFFENSIVE/HATE) | Tiếng Việt | `https://huggingface.co/datasets/sonlam1102/vihsd` hoặc `https://github.com/sonlam1102/vihsd` | Không dùng trực tiếp cho nhãn lừa đảo, nhưng tận dụng làm nguồn **negative samples đa dạng văn phong mạng xã hội** (để mô hình không nhầm ngôn ngữ tục tĩu với lừa đảo) |
| 3 | **UCI/Kaggle SMS Spam Collection** | 5.574 tin (747 spam) — tiếng Anh | Tiếng Anh | `https://archive.ics.uci.edu/dataset/228/sms+spam+collection`, `https://huggingface.co/datasets/ucirvine/sms_spam` | Không dùng trực tiếp (khác ngôn ngữ) — chỉ dùng để **tham khảo cấu trúc gán nhãn / pipeline benchmark** khi thiết kế bộ tiếng Việt |
| 4 | **Vietnamese Spam Post in Social Network** (Kaggle) | Bài đăng mạng xã hội tiếng Việt gắn nhãn spam | Tiếng Việt | `https://www.kaggle.com/datasets/victorhoward2/vietnamese-spam-post-in-social-network` | Nguồn bổ sung mẫu spam/lừa đảo tiếng Việt thật |
| 5 | **Dữ liệu tự thu thập — Cổng Không gian mạng quốc gia (khonggianmang.vn)** | Phản ánh lừa đảo thực tế, cập nhật liên tục | Tiếng Việt | `https://khonggianmang.vn/` | Nguồn chính cho 4 kịch bản lừa đảo — cần crawl/thu thập thủ công có chọn lọc, tự gán nhãn |
| 6 | **Dữ liệu tự thu thập — Cổng cảnh báo an toàn thông tin (Cục ATTT, canhbao.khonggianmang.vn)** | Số liệu & case study thực tế 2024–2025 | Tiếng Việt | Tham chiếu báo cáo PPLNCKH mục 1, tài liệu tham khảo [24] | Dùng làm căn cứ thống kê tỷ trọng 4 kịch bản khi thiết kế phân bố lớp |
| 7 | **SMS OTP / tin nhắn biến động số dư ngân hàng (tự tổng hợp)** | Tự thu thập từ tin nhắn cá nhân được ẩn danh hoá | Tiếng Việt | Không có nguồn public — tự thu thập nội bộ nhóm, **ẩn danh hoá trước khi lưu** | Nguồn chính cho lớp "An toàn" (nhãn 0), đảm bảo mô hình không báo nhầm SMS OTP hợp lệ |

### 1.2. Dataset teencode / từ điển chuẩn hoá — dùng cho `teencode_dict.json`

| # | Nguồn | Link | Ghi chú |
|---|---|---|---|
| 1 | **VietnameseTextNormalizer / teencode-dict (cộng đồng GitHub)** | Tìm kiếm từ khoá `teencode vietnamese dictionary github` | Nhiều repo cộng đồng dạng `teen_code.txt`, cần rà soát thủ công độ tin cậy trước khi dùng |
| 2 | **underthesea — bộ từ điển viết tắt tích hợp sẵn** | `https://github.com/undertheseanlp/underthesea` | Có thể tận dụng module chuẩn hoá văn bản sẵn có, rồi mở rộng thủ công theo văn phong lừa đảo (ck, tk, tr, ib, sđt, stk, cmnd...) |
| 3 | **Tự xây dựng thủ công (khuyến nghị chính)** | — | Thành viên 1 tổng hợp từ dữ liệu thực tế thu thập được ở mục 1.1, ưu tiên các từ xuất hiện trong ngữ cảnh tài chính/ngân hàng/pháp luật |

### 1.3. Dataset âm thanh tiếng Việt — dùng cho Thành viên 2

| # | Dataset | Quy mô | Link | Cách dùng trong đề tài |
|---|---|---|---|---|
| 1 | **VIVOS** (AILAB, VNUHCM) | 15 giờ thu âm, 11.660 câu train / 760 câu test, đơn giọng | `https://huggingface.co/datasets/AILAB-VNUHCM/vivos` | Fine-tune/benchmark ASR (Whisper) trên giọng đọc rõ, làm baseline "điều kiện lý tưởng" |
| 2 | **VLSP 2020 ASR Challenge dataset** | Tập thi ASR tiếng Việt chính thức, đa dạng nguồn | Đăng ký qua ban tổ chức VLSP (`https://vlsp.org.vn/`) | Nếu xin được quyền truy cập: benchmark ASR ở điều kiện đa dạng hơn VIVOS |
| 3 | **Mozilla Common Voice (vi)** | Giọng đọc cộng đồng đóng góp, đa giọng | `https://commonvoice.mozilla.org/vi/datasets` | Bổ sung đa dạng giọng nói (nhiều người nói khác nhau) cho việc test độ bền ASR |
| 4 | **Bộ dữ liệu tự thu âm/giả lập của nhóm** | Do nhóm tự ghi âm — kịch bản theo 4 loại lừa đảo, 3 miền Bắc/Trung/Nam | Nội bộ | Nguồn **chính** cho bài toán vishing — bắt buộc vì không có dataset vishing tiếng Việt công khai |
| 5 | **Kho hiệu ứng nhiễu nền (background noise)** | Tiếng còi xe, văn phòng, đường phố | Freesound.org (`https://freesound.org/`, cần lọc theo giấy phép CC0/CC-BY) hoặc tự thu | Trộn (augment) vào bộ dữ liệu giả lập để mô phỏng điều kiện viễn thông thực tế |

### 1.4. Mô hình/thư viện nền tảng cần tải về

| Công cụ | Nguồn tải | Dùng để |
|---|---|---|
| `vinai/phobert-base-v2` | `https://huggingface.co/vinai/phobert-base-v2` (`pip install transformers`, tự động tải khi `from_pretrained`) | Embedding ngữ nghĩa 768-d |
| `underthesea` | `https://github.com/undertheseanlp/underthesea` (`pip install underthesea`) | Tách từ ghép tiếng Việt (bắt buộc trước khi đưa vào PhoBERT). **Đã chốt thay cho `VnCoreNLP`/`py_vncorenlp`** vì không cần JVM — tránh lỗi `JavaNotFound` khi đóng gói Docker trên image `python:3.11-slim` (xem Risk Register R6) |
| `faster-whisper` | `https://github.com/SYSTRAN/faster-whisper` (`pip install faster-whisper`) | ASR tốc độ cao (CTranslate2, nhanh gấp ~4 lần Whisper gốc) |
| `silero-vad` | `https://github.com/snakers4/silero-vad` (tích hợp sẵn trong `faster-whisper` qua `vad_filter=True`) | Cắt khoảng lặng trước khi bóc băng |
| `librosa` | `pip install librosa` | Trích đặc trưng Pitch/Energy/Speech Rate |
| `FastAPI` + `uvicorn` | `pip install fastapi uvicorn[standard]` | RESTful API async |
| `ONNX Runtime` | `pip install onnxruntime` | Tối ưu suy luận PhoBERT sau khi export/quantize |
| `Docker` + `Docker Compose` | `https://docs.docker.com/get-docker/` | Container hoá 2 dịch vụ (`app` gộp API Gateway + Inference Worker, và `redis`) — xem lý do gộp ở Risk Register R6 |

### 1.5. Tài liệu nghiên cứu cần tham khảo (trích Danh mục tài liệu tham khảo, báo cáo PPLNCKH)

| Chủ đề | Tài liệu | Vai trò |
|---|---|---|
| Nền tảng mô hình | Nguyen & Nguyen (2020), *"PhoBERT: Pre-trained language models for Vietnamese"*, EMNLP Findings | Cơ sở lựa chọn PhoBERT |
| Nền tảng ASR | Radford et al. (2023), *"Robust Speech Recognition via Large-Scale Weak Supervision"*, ICML/PMLR | Cơ sở lựa chọn Whisper |
| Nền tảng NLP tiếng Việt | Vu et al. (2018), *"VnCoreNLP: A Vietnamese NLP Toolkit"*, NAACL | Cơ sở tách từ |
| Multimodal fusion | Baltrušaitis, Ahuja, Morency (2018), *"Multimodal Machine Learning: A Survey and Taxonomy"*, IEEE TPAMI | Cơ sở lý luận Early/Late Fusion |
| Hệ thống tham chiếu gần nhất | Lokare et al. (2025), *"VishGuard: AI-Powered Real-Time Defense Against Voice Phishing"*, IEEE ICTBIG | Benchmark kiến trúc gần nhất, cần vượt qua giới hạn đơn kênh của công trình này |
| Smishing baseline | Samad et al. (2023), *"SmishGuard"*, IJACSA | Tham khảo kỹ thuật TF-IDF + LDA + XGBoost cho baseline so sánh |
| Kiến trúc kinh điển | Vaswani et al. (2017) *Attention Is All You Need*; Hochreiter & Schmidhuber (1997) *LSTM* | Nền tảng lý thuyết Transformer/BiLSTM dùng trong Chương 3 |

---

## 2. TIMELINE TỔNG THỂ — 6 TUẦN

```
Tuần 1 ██████░░░░░░░░░░░░░░░░░░░░░░░░  Setup + Thu thập dữ liệu + Từ điển teencode
Tuần 2 ░░░░░░██████░░░░░░░░░░░░░░░░░░  Module tiền xử lý Text (NLP) & Audio (ASR/Librosa)
Tuần 3 ░░░░░░░░░░░░██████░░░░░░░░░░░░  Huấn luyện Baseline: PhoBERT/BiLSTM/SVM + Audio model
Tuần 4 ░░░░░░░░░░░░░░░░░░██████░░░░░░  Thực nghiệm Fusion: Early vs Late + so sánh metric
Tuần 5 ░░░░░░░░░░░░░░░░░░░░░░░░██████  FastAPI + Docker + tối ưu độ trễ (quantization)
Tuần 6 ░░░░░░░░░░░░░░░░░░░░░░░░░░░░██  Demo UI + Load test + Biểu đồ Chương 3/4 + hoàn thiện báo cáo
```

### 2.1. Milestones & Definition of Done (DoD)

| Mốc | Cuối tuần | Deliverable | Definition of Done |
|---|---|---|---|
| **M1 — Data Foundation** | Tuần 1 | `teencode_dict.json`, tập SMS gán nhãn thô (≥ 1.500 mẫu), 10 bản ghi audio giả lập mẫu | Dữ liệu đã qua vòng ẩn danh hoá PII; tỷ lệ An toàn:Lừa đảo trong khoảng 60:40–50:50; repo Git + Docker khởi tạo, README chạy được |
| **M2 — Preprocessing Pipeline** | Tuần 2 | Module `nlp_preprocess.py`, module `voice_preprocess.py` (VAD + resample 16kHz) | Chạy được end-to-end trên ≥ 20 mẫu thật; unit test pass; log không lộ PII |
| **M3 — Baseline Models** | Tuần 3 | Checkpoint PhoBERT/BiLSTM/SVM (text) + model audio baseline | Bảng so sánh Accuracy/F1/FNR trên tập test giữ lại (hold-out ≥ 20%); FNR được ghi rõ, không chỉ Accuracy |
| **M4 — Fusion Experiments** | Tuần 4 | Checkpoint Early Fusion + Late Fusion, đồ thị so sánh | Có bảng so sánh định lượng 2 phương án hợp nhất trên cùng tập test; chọn được phương án khuyến nghị kèm lý do |
| **M5 — Serving Layer** | Tuần 5 | API FastAPI `/detect/text`, `/detect/voice` (hỗ trợ audio chunk 3–5s); `docker-compose.yml` chạy 2 service (`app` + `redis`) | `docker-compose up` khởi động thành công; benchmark latency p50/p95 cho từng endpoint; PII Anonymizer chạy trước log (kể cả với số có khoảng trắng/dấu chấm xen giữa) |
| **M6 — Demo & Báo cáo** | Tuần 6 | Web Demo (Streamlit/React), báo cáo Chương 3 + Chương 4 hoàn chỉnh, video demo | Demo chạy được cảnh báo Đỏ/Vàng/Xanh trực quan; load test ghi nhận số liệu; báo cáo có đủ bảng/biểu đồ số liệu thực nghiệm |

---

## 3. MA TRẬN TRÁCH NHIỆM (RACI MATRIX)

**Quy ước:** R = Responsible (thực hiện) · A = Accountable (chịu trách nhiệm cuối) · C = Consulted (được tham vấn) · I = Informed (được thông báo)

| Hạng mục công việc | TV1 (NLP & Text) | TV2 (Speech & Fusion) | TV3 (Serving/API/UX) |
|---|---|---|---|
| Thu thập & gán nhãn dữ liệu SMS | **R/A** | C | I |
| Thu thập/giả lập dữ liệu âm thanh 3 miền | I | **R/A** | C |
| Xây dựng `teencode_dict.json` | **R/A** | I | I |
| Module tiền xử lý NLP (clean/teencode/segment) | **R/A** | C | I |
| Module ASR (faster-whisper) | I | **R/A** | C |
| Module trích đặc trưng âm học (Librosa) | I | **R/A** | I |
| Huấn luyện mô hình Text (PhoBERT/BiLSTM/SVM) | **R/A** | I | I |
| Huấn luyện mô hình Audio | I | **R/A** | I |
| Module Multimodal Fusion (Early/Late) | C | **R/A** | I |
| Module PII Anonymizer | C | I | **R/A** |
| API FastAPI (`/detect/text`, `/detect/voice`) | C | C | **R/A** |
| Docker & Docker Compose | I | I | **R/A** |
| Web Demo & cơ chế cảnh báo UX | I | C | **R/A** |
| Benchmark độ trễ & load test | C | C | **R/A** |
| Viết Chương 3 (thực nghiệm mô hình) | **R/A** | **R/A** | C |
| Viết Chương 4 (hệ thống & kiểm thử) | I | C | **R/A** |
| Data Drift Monitoring / MLOps | C | C | **R/A** |

---

## 4. QUẢN TRỊ RỦI RO (RISK REGISTER)

| # | Rủi ro | Xác suất | Tác động | Chiến lược giảm thiểu | Chủ sở hữu |
|---|---|---|---|---|---|
| R1 | **Nghẽn độ trễ thời gian thực** — pipeline ASR + PhoBERT + Fusion vượt ngưỡng chấp nhận được trên CPU | Cao | Cao | (a) Dùng `faster-whisper` với `compute_type="int8"`; (b) Quantize PhoBERT sang INT8 qua `optimum`/`onnxruntime`; (c) Xử lý ASR và trích đặc trưng âm học song song bằng `asyncio.gather`; (d) Benchmark từng module riêng biệt trước khi ghép pipeline để cô lập bottleneck | TV2 & TV3 |
| R2 | **Thiếu hụt dữ liệu vishing tiếng Việt** — không có dataset công khai quy mô lớn | Cao | Cao | (a) Tự giả lập kịch bản dựa trên 4 loại lừa đảo đã xác định, có kịch bản thoại chi tiết theo 3 miền, **thu âm thật** là nguồn chính cho cả 3 miền; (b) Dùng `edge-tts` **chỉ để nhân nhanh số lượng mẫu giọng Bắc** — lưu ý `edge-tts` chỉ có 2 giọng tiếng Việt (`vi-VN-HoaiMyNeural`, `vi-VN-NamMinhNeural`, đều giọng chuẩn/Bắc), **không thể thay thế** việc thu âm thật cho giọng Trung/Nam; (c) Data augmentation: trộn nhiễu nền, thay đổi tốc độ/pitch nhẹ để nhân bản mẫu | TV2 |
| R3 | **Lệch phân phối nhãn (data drift) khi kịch bản lừa đảo đổi theo thời sự** | Trung bình | Trung bình | (a) Thiết kế module Data Drift Detector theo dõi từ khoá thời sự mới; (b) Kiến trúc tách rời Classification Head khỏi backbone để retrain nhanh không cần train lại từ đầu; (c) Ghi rõ trong báo cáo đây là giới hạn đã biết (known limitation), không phải lỗi hệ thống | TV1 & TV3 |
| R4 | **Mất cân bằng lớp (imbalanced data)** dẫn tới mô hình thiên lệch về lớp đa số | Trung bình | Cao | (a) Kiểm soát tỷ lệ thu thập theo mục tiêu 60:40; (b) Áp dụng class weighting hoặc oversampling (SMOTE cho đặc trưng số, không áp dụng trực tiếp cho raw text) nếu tỷ lệ thực tế lệch nhiều; (c) Luôn báo cáo Macro F1 và FNR song song với Accuracy | TV1 |
| R5 | **Rò rỉ PII trong log/checkpoint**, kể cả các biến thể số có khoảng trắng/dấu chấm xen giữa (`098 123 4567`) mà regex đơn giản bỏ sót | Thấp | Rất cao (vi phạm đạo đức nghiên cứu) | (a) PII Anonymizer chạy bắt buộc ở tầng Input Layer trước khi dữ liệu chạm bất kỳ module nào khác; (b) Chuẩn hoá chuỗi (xoá khoảng trắng/dấu chấm/gạch ngang giữa các chữ số) **trước khi** áp regex quét PII; (c) Code review chéo giữa 3 thành viên trước khi merge vào nhánh chính; (d) Không commit dữ liệu thô lên Git — dùng `.gitignore` cho thư mục `data/raw/` | TV3 (chủ trì), cả nhóm review |
| R6 | **Xung đột môi trường Docker & phụ thuộc thư viện** — `VnCoreNLP`/`py_vncorenlp` yêu cầu máy ảo Java (JRE/JDK), trong khi image nền `python:3.11-slim` không có sẵn Java, dẫn tới lỗi `JavaNotFound` ngay khi khởi động container; tách `inference-worker` thành container riêng như thiết kế ban đầu cũng không có ý nghĩa nếu không kèm message broker thật (Celery/RQ) | Đã phát sinh trong review kỹ thuật | Cao | **Quyết định đã chốt (xem AGENT_SYSTEM_PROMPT.md mục 3.6):** (a) Thay `VnCoreNLP` bằng `underthesea` (thuần Python, không cần JVM) làm công cụ tách từ chính thức — không còn là phương án dự phòng; (b) Gộp `api-gateway` và `inference-worker` thành một service `app` duy nhất trong `docker-compose.yml`, chỉ giữ `redis` làm service phụ trợ — dựng broker thật không cần thiết cho quy mô tải của một demo prototype 3 người/6 tuần; (c) Pin version cụ thể trong `requirements.txt` để tránh breaking change từ `transformers` | TV1 & TV3 |
| R7 | **Quá tải khối lượng công việc Tuần 6** (Demo + Load test + Viết báo cáo cùng lúc) | Trung bình | Trung bình | (a) Chuẩn bị khung báo cáo Chương 3/4 từ Tuần 3–4, chỉ điền số liệu vào Tuần 6; (b) Demo UI làm khung sườn từ Tuần 5, Tuần 6 chỉ tích hợp API thật | Cả nhóm |
| R8 | **Overfitting nghiêm trọng** — PhoBERT-base-v2 (~135M tham số) full fine-tune trên chỉ 1.500–2.500 mẫu SMS; MLP Early Fusion (773→256→64→2) quá lớn so với vài chục–trăm mẫu audio; vector text 768-d áp đảo vector audio 5-d nếu không chuẩn hoá thang đo | Cao | Cao | **Quyết định đã chốt:** (a) Freeze 8/12 tầng Transformer đầu của PhoBERT, chỉ huấn luyện các tầng trên cùng, kết hợp `EarlyStoppingCallback(patience=2)` — **không dùng LoRA** (kỹ thuật đúng nhưng tốn 2–3 ngày học `peft` mà lợi ích biên nhỏ ở quy mô dữ liệu này; chỉ cân nhắc lại nếu freeze+early-stopping vẫn overfit rõ sau khi có kết quả thật); (b) Bắt buộc `StandardScaler` cho 5 chiều đặc trưng âm học trước khi nối vào vector 768-d; (c) Thu nhỏ MLP về dạng cổ chai hẹp hơn nếu vẫn overfit, tăng Dropout lên 0.5; (d) Đặt Late Fusion làm phương án đối chứng trọng tâm | TV1 & TV2 |
| R9 | **Rò rỉ dữ liệu chéo (data leakage) khi Data Augmentation** — nếu nhân bản mẫu (LLM cho text, nhiễu/tốc độ cho audio) **trước** khi chia Train/Val/Test, các biến thể từ cùng 1 mẫu gốc lọt vào cả 2 tập, gây Accuracy ảo (>99%) trong phòng thí nghiệm nhưng thất bại trên dữ liệu thực tế | Trung bình | Rất cao (làm sai toàn bộ số liệu Chương 3) | **Quy tắc bắt buộc:** (a) Chia `Train (70%) / Val (15%) / Test (15%)` trên **mẫu gốc** trước, lưu 3 file `train_ids.txt/val_ids.txt/test_ids.txt`; (b) Chỉ chạy Data Augmentation **sau đó và chỉ trên tập Train**; (c) Viết hàm `assert_no_leakage(train_ids, test_ids)` kiểm tra giao nhau rỗng, chạy như một bước kiểm tra bắt buộc trước khi train bất kỳ mô hình nào | TV1 & TV2 |

---

## 5. QUY TRÌNH LÀM VIỆC NHÓM (KHUYẾN NGHỊ)

- **Git flow:** nhánh `main` (ổn định) ← `dev` ← nhánh cá nhân `feature/nlp-*`, `feature/audio-*`, `feature/api-*`. Merge vào `dev` qua Pull Request, tối thiểu 1 reviewer.
- **Họp đồng bộ:** 15–20 phút cuối mỗi tuần, đối chiếu với Milestone tương ứng trong mục 2.1; cập nhật rủi ro ở mục 4 nếu phát sinh mới.
- **Cấu trúc thư mục dùng chung** (thống nhất ngay Tuần 1):

```
project-root/
├── data/
│   ├── raw/            # KHÔNG commit lên Git
│   ├── processed/
│   └── teencode_dict.json
├── src/
│   ├── nlp/             # Thành viên 1
│   ├── audio/            # Thành viên 2
│   ├── fusion/            # Thành viên 2
│   ├── api/               # Thành viên 3
│   └── common/
│       └── pii_anonymizer.py
├── notebooks/
├── docker/
│   ├── Dockerfile.app       # gộp API Gateway + Inference Worker (xem Risk Register R6)
│   └── docker-compose.yml   # 2 service: app + redis
├── tests/
├── config.yaml
└── README.md
```
