# B5 — Báo cáo Chương 3: đánh giá ASR & độ bền

**Người phụ trách:** B · **Tuần:** 6 · **Phụ thuộc:** B2, B3, B4 · **Nguồn:** kế hoạch mục 5.2 (Tuần 6), mục 6, 8

## Mục tiêu
Viết `reports/chapter3_asr_robustness.md` từ số liệu thật trong `results/`, để gộp với báo cáo của A ở điểm S3.

## Việc cần làm
1. **Sinh bảng tự động** bằng `scripts/make_tables_b.py` (đọc `results/*.csv` → Markdown), không gõ số bằng tay.
2. **Nội dung báo cáo** (theo mục 8 của kế hoạch):
   1. *Mô tả dữ liệu âm thanh:* số file theo miền × sắc thái × nhãn, số người nói, cách thu, mô phỏng nhiễu; nêu rõ dữ liệu giờ dùng làm **tập test ASR/robustness**, không huấn luyện.
   2. *Đánh giá ASR:* WER PhoWhisper-base (và small/tiny nếu đo) vs Whisper gốc — trên VIVOS/mẫu con và tập tự thu; WER theo miền/sắc thái; ảnh hưởng của nhiễu.
   3. *Độ bền:* Accuracy/F1/FNR sạch vs ASR (mọi điều kiện), FN mới do ASR, keyword recall, kết quả theo chunk.
   4. *Latency:* bảng ms: NLP preprocess / PhoBERT+Hybrid / ASR theo chunk / tổng; không có dòng Fusion.
   5. *Rủi ro R10 & giới hạn:* trạng thái R8 (loại bỏ), R9 (thu hẹp), R1 (đơn giản hoá), R10 (số liệu đo được); giới hạn "ASR-then-classify"; mất tín hiệu giọng điệu.
   6. *Lý do đơn giản hoá:* đánh đổi phức tạp lấy ổn định/khả thi (kế hoạch mục 0).
3. **Đối chiếu số công bố:** nêu các số WER trên VIVOS của PhoWhisper (base ≈ 8,46%, small ≈ 6,33%, tiny ≈ 10,41%) là **số của tác giả**, tách riêng khỏi số nhóm tự đo.
4. **Phần nhỏ dữ liệu:** với ô n nhỏ, dùng ngôn từ thận trọng ("xu hướng", không "chứng minh"); không kết luận theo miền/sắc thái nếu n < 30.
5. **Đồng bộ với A (S3):** đọc `reports/chapter3_hybrid_ablation.md`, kiểm tra chéo các con số dùng chung (kích thước tập, cách tính FNR); ghi chênh lệch vào `docs/agent_notes/HANDOFF_B.md`.

## Đầu ra
`reports/chapter3_asr_robustness.md`, `scripts/make_tables_b.py`.

## Nghiệm thu
- [ ] Mọi con số trong báo cáo truy được về một file trong `results/`.
- [ ] Chạy lại `make_tables_b.py` cho bảng giống hệt.
- [ ] Có đủ 6 mục nội dung ở trên.
- [ ] Mục nào chưa đo được thì ghi "chưa đo" cùng lý do, không điền số.
