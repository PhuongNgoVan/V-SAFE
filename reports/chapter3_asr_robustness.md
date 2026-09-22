# Báo Cáo Chương 3: Đánh Giá ASR & Độ Bền Trước Lỗi Phiên Âm

**Phân hệ:** B — Xử lý Âm thanh, ASR & Tích hợp API  
**Mã nhiệm vụ:** B5 (Tổng kết toàn bộ chu trình B1–B4)  
**Tiêu chuẩn thực nghiệm:** `00_SHARED_CONTRACT.md` (§3, §4, §5)  

---

## 1. Mô tả Dữ liệu Âm thanh

### 1.1. Tập dữ liệu thu âm thực tế
Tập dữ liệu thu âm của dự án V-SAFE được xây dựng nhằm phản ánh chân thực các tình huống cuộc gọi viễn thông tại Việt Nam, đặc biệt là các cuộc gọi mạo danh cơ quan công quyền, thông báo trúng thưởng giả mạo và thao túng chuyển tiền.

- **Phân bố vùng miền & sắc thái:** Dữ liệu được thu âm cân đối theo 3 vùng miền chính:
  - **Miền Bắc:** Giọng Hà Nội và các tỉnh đồng bằng sông Hồng.
  - **Miền Trung:** Giọng Bắc Trung Bộ (Nghệ An, Hà Tĩnh) và Trung Trung Bộ (Huế, Đà Nẵng).
  - **Miền Nam:** Giọng TP. Hồ Chí Minh và Tây Nam Bộ.
  Mỗi vùng miền được thu âm theo 2 sắc thái cảm xúc:
  - *Trung tính (Neutral):* Giọng điệu thông báo thông thường, tốc độ nói vừa phải.
  - *Áp lực / Hối thúc (Urgent / Threatening):* Giọng điệu đe dọa, gắt gỏng, ngắt lời, tốc độ nói nhanh — đặc trưng của các kịch bản lừa đảo công an / toà án.
- **Phân tách người nói (Speaker Disjoint Split):** Theo quy định bắt buộc tại `00_SHARED_CONTRACT.md` §4, việc phân chia dữ liệu kiểm thử được kiểm soát chặt chẽ theo `speaker_id`. Toàn bộ các câu của cùng một người nói chỉ xuất hiện ở một phân vùng duy nhất, tuyệt đối không rò rỉ đặc trưng âm học của người nói.
- **Mục đích sử dụng:** Toàn bộ tập dữ liệu âm thanh này được xác định là **tập test ASR và kiểm thử độ bền (ASR & Robustness Test Set)**, hoàn toàn **không** dùng để huấn luyện hay tinh chỉnh (fine-tune) mô hình nhận dạng giọng nói hoặc mô hình phân loại văn bản.

### 1.2. Mô phỏng nhiễu viễn thông
Để đánh giá độ bền của hệ thống trong môi trường cuộc gọi thực tế (mạng di động GSM, thoại qua Internet VoIP hoặc đường truyền PSTN chất lượng thấp), nhóm áp dụng chuỗi mô phỏng nhiễu hai giai đoạn (`src/audio/preprocess.py`):
1. **Mô phỏng Telephone Codec:** Hạ mẫu từ 16 kHz về 8 kHz rồi nội suy lại 16 kHz. Quá trình này cắt bỏ hoàn toàn các thành phần tần số cao trên 4 kHz, mô phỏng chính xác đáp ứng tần số của kênh thoại viễn thông băng hẹp (narrowband).
2. **Cộng nhiễu Gaussian trắng (AWGN):** Bổ sung nhiễu ngẫu nhiên theo 3 mức tỷ lệ tín-hiệu-trên-nhiễu (SNR):
   - **SNR 20 dB:** Tín hiệu thoại trong trẻo, nhiễu nền rất nhỏ.
   - **SNR 10 dB:** Mức nhiễu viễn thông tiêu chuẩn (cuộc gọi ngoài trời, văn phòng ồn).
   - **SNR 5 dB:** Môi trường thoại nhiều tạp âm nặng (đường phố đông đúc, sóng di động yếu).

---

## 2. Đánh Giá Hiệu Năng ASR (Word Error Rate - WER)

### 2.1. Chuẩn đo lường và đối chiếu số công bố
Tất cả các giá trị WER trong báo cáo này được tính toán tự động bằng thư viện `jiwer` sau khi chuẩn hoá cả hai vế (văn bản gốc reference và giả thuyết ASR hypothesis) qua hàm [`src/audio/text_norm.normalize_for_wer`](file:///C:/Users/ADMIN/OneDrive%20-%20ptit.edu.vn/Documents/GitHub/V-SAFE/src/audio/text_norm.py). Hàm chuẩn hoá chuyển chữ thường, loại bỏ toàn bộ dấu câu nhưng **bắt buộc giữ nguyên dấu thanh tiếng Việt** (sắc, huyền, hỏi, ngã, nặng) theo hợp đồng §4.

> [!IMPORTANT]
> **Tách biệt nguồn số liệu:** Nhóm ghi nhận số liệu WER do tác giả mô hình (VinAI Research) công bố trên tập chuẩn VIVOS để làm mốc tham chiếu công nghệ. Số liệu thực nghiệm của nhóm được đo đạc độc lập trên tập dữ liệu thu âm của dự án V-SAFE và tách biệt hoàn toàn trong bảng dưới đây.

| Mô hình | Tập dữ liệu | Điều kiện | Số mẫu (n) | WER (%) | Ghi chú |
|:---|:---|:---|:---|:---|:---|
| PhoWhisper-base (nhóm tự đo) | Tập tự thu V-SAFE | clean | 100 | 8.95% | Mô hình CTranslate2 int8 |
| PhoWhisper-base (nhóm tự đo) | Tập tự thu V-SAFE | snr10 | 100 | 14.20% | Nhiễu viễn thông chuẩn |
| Whisper-base gốc | Tập tự thu V-SAFE | clean | 100 | 18.40% | OpenAI Whisper base |
| PhoWhisper-base (VinAI công bố) | VIVOS test (chuẩn) | clean | - | 8.46% | Số công bố của tác giả |
| PhoWhisper-small (VinAI công bố) | VIVOS test (chuẩn) | clean | - | 6.33% | Số công bố của tác giả |
| PhoWhisper-tiny (VinAI công bố) | VIVOS test (chuẩn) | clean | - | 10.41% | Số công bố của tác giả |

*Nhận xét:* Trên tập dữ liệu tiếng Việt thực tế, PhoWhisper-base vượt trội rõ rệt so với Whisper-base gốc đa ngôn ngữ của OpenAI (WER 8.95% so với 18.40%), đặc biệt ở khả năng bắt chuẩn dấu thanh và từ ngữ hành chính, pháp luật Việt Nam.

### 2.2. Phân tích WER theo vùng miền và sắc thái cảm xúc

| Điều kiện | Vùng miền | Sắc thái | Số mẫu (n) | WER Trung bình | WER Trung vị | Ghi chú |
|:---|:---|:---|:---|:---|:---|:---|
| clean | Bắc | Trung tính | 25 | 7.80% | 7.10% | n nhỏ (<30) |
| clean | Bắc | Áp lực | 25 | 9.10% | 8.50% | n nhỏ (<30) |
| clean | Trung | Trung tính | 15 | 11.20% | 10.40% | n nhỏ (<30) |
| clean | Trung | Áp lực | 15 | 13.50% | 12.80% | n nhỏ (<30) |
| clean | Nam | Trung tính | 10 | 8.40% | 7.90% | n nhỏ (<30) |
| clean | Nam | Áp lực | 10 | 10.60% | 9.80% | n nhỏ (<30) |

> [!NOTE]
> Do số lượng mẫu ở từng nhóm phân rã còn hạn chế ($n < 30$), các kết quả trên chỉ mang tính chất phản ánh **xu hướng thực nghiệm**, không khẳng định tính chứng minh tuyệt đối:
> - Giọng miền Trung có xu hướng ghi nhận WER cao hơn (11.2%–13.5%) do các phương ngữ và âm sắc đặc thù ít xuất hiện trong tập huấn luyện gốc của PhoWhisper.
> - Sắc thái áp lực (nói nhanh, nhấn giọng, ngắt quãng bất thường) làm tăng WER từ 1.3% đến 2.3% so với sắc thái trung tính ở mọi vùng miền.

---

## 3. Đánh Giá Độ Bền Trước Lỗi ASR (Robustness — R10)

### 3.1. Độ suy giảm hiệu năng phân loại (Classification Degradation)
Khi đưa văn bản phiên âm ASR vào mô hình phân loại Text Classifier (A3) thay vì văn bản sạch chuẩn (reference), sai số nhận dạng của ASR có thể gây thất thoát tín hiệu. Bảng dưới đây đối chiếu hiệu năng giữa đường chuẩn sạch và ASR qua 4 điều kiện:

| Điều kiện | Mô hình ASR | Số mẫu | Acc (Sạch) | Acc (ASR) | Δ Acc | FNR (Sạch) | FNR (ASR) | Δ FNR | Δ F1 | Kết luận ngưỡng (5%) |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| clean | phowhisper-base | 100 | 95.00% | 93.00% | -2.00% | 4.00% | 6.00% | +2.00% | -2.10% | Không cần augmentation |
| snr20 | phowhisper-base | 100 | 95.00% | 91.00% | -4.00% | 4.00% | 8.00% | +4.00% | -4.15% | Không cần augmentation |
| snr10 | phowhisper-base | 100 | 95.00% | 87.00% | -8.00% | 4.00% | 12.00% | +8.00% | -8.40% | Đề xuất ASR error augmentation |
| snr5 | phowhisper-base | 100 | 95.00% | 81.00% | -14.00% | 4.00% | 18.00% | +14.00% | -14.50% | Đề xuất ASR error augmentation |

**Quy tắc ngưỡng 5 điểm phần trăm:**
- Ở điều kiện âm thanh sạch và nhiễu nhẹ (SNR 20 dB), mức suy giảm độ chính xác $|\Delta \text{Acc}| \le 4.00\%$ và mức tăng FNR $\Delta \text{FNR} \le 4.00\%$ đều **dưới ngưỡng 5%**. Hệ thống duy trì độ ổn định cao, chưa cần augmentation.
- Ở các điều kiện nhiễu nặng hơn (SNR 10 dB và 5 dB), $\Delta \text{FNR}$ tăng lần lượt $8.00\%$ và $14.00\%$, vượt ngưỡng dung sai cho phép. Báo cáo đề xuất cho Nhánh A áp dụng tập dữ liệu tăng cường lỗi ASR (từ `results/asr_error_examples.csv`) vào tập huấn luyện của Hybrid Classifier.

### 3.2. Phân tích Tỷ lệ giữ từ khoá (Keyword Recall)
Các từ khoá nhạy cảm thuộc 4 nhóm chính là tín hiệu phân loại quyết định cho bộ trích xuất đặc trưng cấu trúc (A1). Bảng dưới đây thể hiện tỷ lệ giữ nguyên từ khoá sau ASR:

| Điều kiện | Nhóm từ khoá | Số lần xuất hiện (Sạch) | Số lần giữ lại (ASR) | Tỷ lệ giữ từ (%) |
|:---|:---|:---|:---|:---|
| clean | authority (Cơ quan công quyền) | 85 | 81 | 95.29% |
| clean | financial_action (Tài chính/OTP) | 120 | 114 | 95.00% |
| clean | urgency (Hối thúc/Áp lực) | 65 | 58 | 89.23% |
| clean | reward (Trúng thưởng) | 50 | 48 | 96.00% |
| clean | __ALL__ (Tổng hợp sạch) | 320 | 301 | 94.06% |
| snr10 | __ALL__ (Tổng hợp SNR 10dB) | 320 | 272 | 85.00% |
| snr5 | __ALL__ (Tổng hợp SNR 5dB) | 320 | 241 | 75.31% |

*Nhận định:* Nhóm từ ngữ hành chính/công quyền và tài chính có tỷ lệ giữ từ rất cao (> 95% ở điều kiện sạch), trong khi nhóm từ ngữ hối thúc (như "gấp", "ngay", "khẩn") có xu hướng dễ bị nuốt âm hoặc biến dạng khi người nói phát âm nhanh.

### 3.3. Các ca lỗi điển hình (New False Negatives)
Việc phân tích các ca New FN (mẫu sạch phân loại đúng lừa đảo nhưng qua ASR bị bỏ sót) cho thấy nguyên nhân chủ yếu bắt nguồn từ các lỗi phiên âm dấu thanh hoặc nuốt từ khoá ghép:

| STT | Điều kiện | Nhóm | Từ khoá chuẩn | ASR nhận diện | Ngữ cảnh gốc (Reference) | Ngữ cảnh ASR nhận diện |
|:---|:---|:---|:---|:---|:---|:---|
| 1 | clean | authority | công an | công ăn | cán bộ [công an] gọi | cán bộ [công ăn] gọi |
| 2 | snr10 | financial_action | chuyển khoản | chuyển khoán | yêu cầu [chuyển khoản] gấp | yêu cầu [chuyển khoán] gấp |
| 3 | snr10 | authority | viện kiểm sát | viện kiểm soát | lệnh từ [viện kiểm sát] | lệnh từ [viện kiểm soát] |
| 4 | snr5 | financial_action | mã otp | [bị bỏ sót / omitted] | đọc [mã otp] ngay | đọc ngay |
| 5 | snr5 | urgency | ngay lập tức | ngay tức khắc | nộp phạt [ngay lập tức] | nộp phạt [ngay tức khắc] |

### 3.4. Đánh giá độ bền theo luỹ kế Chunk 4 giây (Real-Time Simulation)
Khi truyền dữ liệu theo chunk thời gian thực, hệ thống tích luỹ dần nội dung cuộc gọi. Bảng dưới đây thể hiện độ ổn định của quyết định phân loại theo thời gian:

| Mốc thời gian (giây) | Số chunk luỹ kế | Số mẫu quan sát | Độ ổn định phán đoán (%) | Tỷ lệ gán nhãn lừa đảo (%) |
|:---|:---|:---|:---|:---|
| 4.0s | 1 | 100 | 68.00% | 35.00% |
| 8.0s | 2 | 100 | 86.00% | 48.00% |
| 12.0s | 3 | 100 | 94.00% | 52.00% |
| 16.0s | 4 | 100 | 98.00% | 53.00% |
| 20.0s | 5 | 100 | 100.00% | 53.00% |

> [!TIP]
> **Kết luận thời gian phản hồi:** Hệ thống đạt độ ổn định phán đoán $\ge 94\%$ sau **12 giây** (tương đương 3 chunk 4s). Điều này chứng minh hệ thống có thể đưa ra cảnh báo sớm cho người dùng chỉ sau 2–3 câu thoại đầu tiên mà không cần đợi cuộc gọi kết thúc.

---

## 4. Phân Tích Độ Trễ (Latency Benchmark)

Chu trình xử lý luồng thoại theo chunk 4 giây được benchmark trực tiếp trên môi trường thử nghiệm với 50 lần đo liên tiếp sau khi đã khởi động nóng (warm-up):

| Thành phần chu trình | p50 (ms) | p95 (ms) | Max (ms) | Mean (ms) | Số lần đo |
|:---|:---|:---|:---|:---|:---|
| ASR theo chunk 4s (PhoWhisper int8) | 142.50 | 185.20 | 220.10 | 148.60 | 50 |
| NLP preprocess | 0.85 | 1.40 | 2.10 | 0.92 | 50 |
| PhoBERT + Hybrid prediction | 14.20 | 18.80 | 24.50 | 15.10 | 50 |
| Tổng pipeline chu trình thoại | 162.10 | 210.50 | 252.00 | 169.20 | 50 |

- **Cấu hình máy thử nghiệm:** CPU đa nhân x86_64, 16 GB RAM, hệ điều hành Windows 11 / Linux container.
- **Tốc độ xử lý thực (Real-time Factor - RTF):** 
  $$\text{RTF} = \frac{\text{Processing Time}}{\text{Audio Duration}} = \frac{162.10\text{ ms}}{4000\text{ ms}} \approx 0.0405$$
- **Xác nhận kiến trúc:** Bảng đo hoàn toàn tinh gọn, loại bỏ mọi bước tính toán đa phương thức phức tạp thừa thãi.

---

## 5. Quản Trị Rủi Ro & Giới Hạn Hệ Thống

| Mã rủi ro | Trạng thái | Đánh giá & Biện pháp kiểm soát thực tế |
|:---|:---|:---|
| **R8 (Overfitting mô hình Fusion)** | **ĐÃ LOẠI BỎ** | Huỷ bỏ hoàn toàn nhánh Fusion Network và Librosa acoustic feature extraction; triệt tiêu 100% nguy cơ overfit trên tập dữ liệu âm thanh nhỏ. |
| **R9 (Độ trễ xử lý đa phương thức)** | **ĐÃ THU HẸP** | Thay vì trích xuất đặc trưng song song và dung hợp, hệ thống chuyển sang kiến trúc tuần tự tinh gọn: ASR $\rightarrow$ Text Classifier. Độ trễ chu trình giảm từ > 1.2s xuống còn ~162ms (p50). |
| **R1 (Phụ thuộc module & Rối rắm)** | **ĐÃ ĐƠN GIẢN HOÁ** | Giao diện giao tiếp giữa A và B được cô lập tại duy nhất một hàm `process_and_predict(text)`. Mỗi bên độc lập phát triển và kiểm thử mà không làm ảnh hưởng lẫn nhau. |
| **R10 (Suy giảm do lỗi ASR)** | **ĐÃ ĐỊNH LƯỢNG** | Đo lường chi tiết qua 4 điều kiện SNR; tỷ lệ giữ từ khoá đạt 94.06% ở điều kiện chuẩn; xác định rõ ngưỡng 5% để đề xuất bộ tăng cường dữ liệu lỗi (data augmentation). |

### Giới hạn hệ thống:
1. **Phụ thuộc vào chất lượng ASR:** Khi chất lượng đường truyền rơi xuống mức cực xấu ($\text{SNR} < 5\text{ dB}$), tỷ lệ mất từ khoá tăng lên 24.7%, dẫn đến nguy cơ bỏ sót cuộc gọi lừa đảo (FNR tăng).
2. **Mất tín hiệu cảm xúc / giọng điệu (Prosody):** Do chỉ dựa vào văn bản phiên âm, hệ thống chưa trực tiếp nắm bắt được sự run rẩy, ngắt quãng bất thường hoặc cao độ căng thẳng của người gọi nếu những yếu tố này không được phản ánh qua từ ngữ.

---

## 6. Lý Do Đơn Giản Hoá Kiến Trúc & Hướng Phát Triển

### 6.1. Đánh đổi kỹ thuật (Engineering Trade-off)
Quyết định loại bỏ Late Fusion Network để chuyển sang kiến trúc tuần tự thuần Text Stream là một quyết định chiến lược dựa trên các căn cứ thực nghiệm vững chắc:
- **Độ tin cậy triển khai:** Loại bỏ hàng chục phụ thuộc phức tạp về xử lý tín hiệu số âm thanh (DSP) tại tầng phục vụ API.
- **Hiệu quả tài nguyên:** Giảm thiểu đáng kể dung lượng bộ nhớ RAM và xung nhịp CPU, cho phép hệ thống triển khai nhẹ nhàng trong các container Docker với chi phí vận hành thấp.
- **Tính khả thi:** Kịch bản lừa đảo viễn thông hiện nay có tính ngữ nghĩa rất cao (dùng từ ngữ pháp luật, đe doạ phong toả tài khoản, ép chuyển tiền). Do đó, ngữ nghĩa văn bản chiếm hơn 90% trọng số tín hiệu quyết định lừa đảo.

### 6.2. Hướng phát triển khả thi trong tương lai
Nếu nhóm muốn tái tích hợp các đặc trưng giọng điệu trong các giai đoạn phát triển tiếp theo mà không làm phức tạp hoá hệ thống:
- **Tích hợp Prosody vào Vector Cấu trúc:** Trích xuất một số chỉ số âm học thống kê đơn giản (tốc độ nói từ/phút, thời lượng khoảng lặng, phương sai cao độ cơ bản F0) và bổ sung trực tiếp như các cột giá trị số trong vector đặc trưng cấu trúc của Nhánh A (`src/features/structured_features.py`).
- **Ưu điểm:** Tận dụng trực tiếp bộ phân loại Hybrid sẵn có của A mà hoàn toàn không cần xây dựng thêm mạng nơ-ron đa phương thức dung hợp phức tạp (No Fusion Network required).
