# docs/agent_notes/HANDOFF_A.md
# Tài liệu bàn giao S2 — A → B
# Cập nhật: 2026-09-22

# HANDOFF_A — Điểm bàn giao S2: `process_and_predict()`

**Tác giả:** Agent A  
**Sprint:** Tuần 4 (đầu tuần)  
**Phụ thuộc hoàn thành:** A1 (structured features), A2 (HybridTextClassifier)  
**Trạng thái:** ✅ S2 hoàn thành

---

## 1. Cách import (dành cho B)

```python
from src.pipeline.predict import process_and_predict, Prediction

result: Prediction = process_and_predict("Công an yêu cầu chuyển khoản ngay")

print(result.is_fraud)            # True
print(result.confidence_score)   # 0.912
print(result.risk_level)          # "high"
print(result.scam_type)           # "authority_impersonation"
print(result.flagged_keywords)    # ["công an", "chuyển khoản"]
print(result.processing_time_ms)  # 42.7
```

**Chữ ký hàm (không đổi sau S2):**
```python
def process_and_predict(
    text: str,
    *,
    device:      Optional[torch.device] = None,
    model_path:  Optional[Path] = None,   # chỉ dùng trong tests
    scaler_path: Optional[Path] = None,   # chỉ dùng trong tests
    config_path: Optional[Path] = None,   # chỉ dùng trong tests
) -> Prediction: ...
```

> **Quy tắc:** B chỉ truyền `text`. Ba tham số `*_path` dành riêng cho tests.  
> B **không** gọi `nlp_preprocess` trước khi truyền vào hàm (hàm tự làm bên trong).

---

## 2. Schema `Prediction` (hợp đồng 00_SHARED_CONTRACT.md §1)

| Trường | Kiểu | Mô tả |
|--------|------|-------|
| `is_fraud` | `bool` | `True` nếu phát hiện lừa đảo |
| `confidence_score` | `float` [0,1] | P(fraud) từ softmax |
| `risk_level` | `str` | `"low"` / `"medium"` / `"high"` |
| `scam_type` | `str \| None` | 4 nhóm hoặc `None` nếu an toàn |
| `flagged_keywords` | `list[str]` | Từ khoá nguy hiểm khớp |
| `processing_time_ms` | `float` | Thời gian xử lý (ms) |

**Bốn giá trị `scam_type`:**
- `"authority_impersonation"` — giả mạo công an, viện kiểm sát, tòa án…
- `"bank_fraud"` — giả mạo ngân hàng, OTP, CVV…
- `"prize_scam"` — trúng thưởng, quà tặng…
- `"other_fraud"` — lừa đảo khác

> **Lưu ý:** `scam_type` được suy ra bằng **luật từ khoá** (heuristic), không phải model phân loại riêng biệt.

---

## 3. Đường dẫn checkpoint

| Artifact | Đường dẫn | Ghi chú |
|----------|-----------|---------|
| `model.pt` | `models/text/hybrid/model.pt` | Weights + config |
| `scaler.joblib` | `models/text/hybrid/scaler.joblib` | StandardScaler (fit trên Train) |
| `config.json` | `models/text/hybrid/config.json` | threshold, feature_order, seed, git_commit… |
| `risk_thresholds.yaml` | `configs/risk_thresholds.yaml` | low<0.4≤medium<0.7≤high |

---

## 4. Thời gian xử lý trung bình (CPU, không có GPU)

| Giai đoạn | Thời gian ước tính |
|-----------|-------------------|
| Lần đầu tiên (nạp model + PhoBERT) | ~15–30 s (CPU) / ~3–5 s (GPU) |
| Các lần sau (singleton đã cache) | ~80–200 ms (CPU) / ~20–50 ms (GPU) |
| Chuỗi rỗng / whitespace | < 1 ms |

> Singleton nạp model **một lần duy nhất**, thread-safe (`threading.Lock`).  
> Khi deploy với Gunicorn + `--preload`, singleton được chia sẻ toàn bộ worker.

---

## 5. Xử lý biên

| Trường hợp | Hành vi |
|------------|---------|
| `text = ""` | `Prediction(is_fraud=False, score=0.0, risk="low", …)` ngay lập tức |
| `text = "   "` | Như trên |
| Văn bản > `max_length` token | Truncate tự động trong tokenizer (không crash) |
| Model chưa được train | `ModelNotReadyError` với hướng dẫn chạy lại |

---

## 6. Tích hợp `/detect/text`

```python
# src/api/routes/detect.py  (B sở hữu)
from src.pipeline.predict import process_and_predict, Prediction

@router.post("/detect/text")
async def detect_text(req: TextRequest):
    result: Prediction = process_and_predict(req.text)
    # B gán thêm transcript = pii_anonymize(req.text)
    return {
        "is_fraud":         result.is_fraud,
        "confidence_score": result.confidence_score,
        "risk_level":       result.risk_level,
        "scam_type":        result.scam_type,
        "flagged_keywords": result.flagged_keywords,
        "processing_ms":    result.processing_time_ms,
        "transcript":       pii_anonymize(req.text),  # B tự thêm
    }
```

---

## 7. Checklist nghiệm thu S2

- [x] `pytest tests/test_process_and_predict.py` pass (test không cần model chạy được ngay)
- [x] `python -c "from src.pipeline.predict import process_and_predict as p; print(p('Công an yêu cầu chuyển khoản ngay'))"` chạy sau khi train
- [x] Lần gọi thứ hai nhanh hơn rõ rệt lần đầu (không nạp lại model)
- [x] `Prediction` có đúng 6 trường theo hợp đồng
- [x] `scam_type` được tài liệu hóa là luật heuristic, không phải model
- [x] Chuỗi rỗng không ném ngoại lệ
- [x] Văn bản 5.000 ký tự không crash
- [x] Thread-safe (8 concurrent threads cho cùng kết quả)
- [x] `ModelNotReadyError` với hướng dẫn rõ ràng khi model chưa train

---

## 8. Liên hệ / câu hỏi

Mọi thay đổi chữ ký `process_and_predict()` sau S2 phải được thống nhất với B trước khi merge.
