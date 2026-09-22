# A1 — Vector đặc trưng cấu trúc (`extract_structured_features`)

**Người phụ trách:** A · **Tuần:** 1 · **Phụ thuộc:** không · **Nguồn:** kế hoạch mục 3.1–3.2

## Mục tiêu
Cài đặt hàm trích 11 đặc trưng diễn giải được từ văn bản đã tiền xử lý, dùng chung cho SMS và transcript ASR.

## Đầu vào
- `nlp_preprocess.run()` hiện có (teencode, bypass-filter, underthesea).
- `src/nlp/pii_anonymizer.py` (nếu có) — nguồn regex điện thoại / số tài khoản / URL / CMND-CCCD.
- `data/processed/text_train.csv`.

## Việc cần làm
1. **Kiểm tra repo:** tìm regex PII sẵn có. Nếu đã có trong `pii_anonymizer.py`, import lại — **không viết trùng**. Nếu chưa có, tạo `src/common/patterns.py` chứa `PHONE_PATTERN, BANK_ACCOUNT_PATTERN, URL_PATTERN, ID_NUMBER_PATTERN` theo kế hoạch mục 3.2 và báo B import từ đó.
2. **Từ điển từ khoá:** đặt trong `configs/scam_keywords.yaml` (4 nhóm: `urgency, authority, financial_action, reward`), khởi tạo bằng đúng danh sách ở kế hoạch mục 3.2. Code đọc file này, không hard-code.
3. **Tạo `src/features/structured_features.py`:**
   - `FEATURE_ORDER: list[str]` — thứ tự cố định 11 cột: `n_urgency_kw, n_authority_kw, n_financial_action_kw, n_reward_kw, has_phone_number, has_bank_account_like_number, has_url, has_id_number_request, message_length, uppercase_ratio, exclamation_count`.
   - `extract_structured_features(clean_text) -> dict` đúng như kế hoạch.
   - `features_to_vector(d) -> np.ndarray` (float32, theo `FEATURE_ORDER`).
   - `matched_keywords(clean_text) -> list[str]` trả các từ khoá khớp (dùng cho `flagged_keywords` của API).
4. **Đối chiếu từ khoá với dữ liệu thật:** chạy trên `text_train.csv`, xuất `results/keyword_coverage.csv` (mỗi từ khoá: số mẫu lừa đảo khớp, số mẫu bình thường khớp). Từ khoá nào khớp > 5% mẫu bình thường thì **đánh dấu trong báo cáo, không tự xoá**.
5. **Viết `tests/test_structured_features.py`** với ít nhất: một câu giả mạo công an (khớp authority + financial), một tin trúng thưởng, một tin bình thường (tất cả = 0), chuỗi rỗng (không chia cho 0), số CCCD 12 số, số điện thoại `+84 912 345 678`.

## Điểm cần lưu ý (từ phân tích kế hoạch)
- **Chồng lấn regex:** `BANK_ACCOUNT_PATTERN` (`\d{8,16}`) cũng khớp CCCD 12 số và nhiều số điện thoại viết liền. Giữ nguyên hành vi theo kế hoạch, nhưng ghi số lượng chồng lấn thực tế trên tập Train vào `docs/agent_notes/A1.md`.
- **Tên `has_id_number_request` gây hiểu nhầm:** thực chất chỉ phát hiện *có số dạng CMND/CCCD*, không phát hiện *yêu cầu cung cấp*. Không đổi tên (để khớp kế hoạch), nhưng ghi chú trong docstring.
- **Lệch phân phối giữa hai kênh:** transcript ASR không có dấu `!` và thường không có chữ hoa; nếu `nlp_preprocess` hạ chữ thường thì `uppercase_ratio` luôn = 0. Ghi nhận trong `docs/agent_notes/A1.md` để A4 kiểm tra ảnh hưởng (bỏ hai cột này khi đánh giá trên transcript).
- Từ khoá ASR: "otp" có thể bị phiên âm thành "ô ti pi" / "ô-tê-pê". Chỉ **ghi chú** ở bước này; xử lý ở B4 nếu số liệu robustness cho thấy cần thiết.

## Đầu ra
`src/features/structured_features.py`, `configs/scam_keywords.yaml`, `tests/test_structured_features.py`, `results/keyword_coverage.csv`, `docs/agent_notes/A1.md`.

## Nghiệm thu
- [ ] `pytest tests/test_structured_features.py` pass.
- [ ] `len(FEATURE_ORDER) == 11` và `features_to_vector("").shape == (11,)`.
- [ ] Không có regex bị định nghĩa trùng ở hai nơi (`grep -rn "PHONE_PATTERN" src/`).
- [ ] Từ khoá đọc từ YAML, không hard-code trong `.py`.
- [ ] Trích đặc trưng cho 10.000 mẫu < 5 giây (in thời gian ra log).
