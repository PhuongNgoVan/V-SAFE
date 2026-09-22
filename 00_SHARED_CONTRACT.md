# HỢP ĐỒNG GIAO TIẾP GIỮA A VÀ B (chốt ở S1, chỉ đổi khi cả hai đồng ý)

## 1. Hàm bàn giao duy nhất

```python
# src/pipeline/predict.py  (A sở hữu)
from dataclasses import dataclass, field

@dataclass
class Prediction:
    is_fraud: bool
    confidence_score: float          # P(lừa đảo), 0..1
    risk_level: str                  # "low" | "medium" | "high"
    scam_type: str | None            # 4 nhóm trong MASTER_PROJECT_PLAN, hoặc None
    flagged_keywords: list[str] = field(default_factory=list)
    processing_time_ms: float = 0.0

def process_and_predict(text: str) -> Prediction: ...
```

- Đầu vào: văn bản **thô** (SMS hoặc transcript ASR, chưa tiền xử lý). Hàm tự gọi `nlp_preprocess.run()` bên trong.
- Không trả `transcript` — trường này do route ở B gán (đã che PII).
- Mô hình nạp **một lần** ở mức module (lazy singleton), không nạp lại mỗi lần gọi.
- Nếu trong repo `Prediction` đã tồn tại (PLAN_MEMBER_3 mục 3.1), **giữ schema hiện có**, chỉ đảm bảo các trường trên có mặt.

## 2. Hàm ASR (B sở hữu)

```python
# src/audio/asr.py
import numpy as np
def transcribe_chunk(y: np.ndarray, sr: int = 16000) -> str: ...
def transcribe_file(path: str) -> str: ...     # tiện ích cho script/đánh giá
```

> **Mâu thuẫn trong kế hoạch gốc:** mục 2.3 định nghĩa `transcribe_chunk(audio_chunk_path)`, còn mục 5.3 gọi `transcribe_chunk(y, sr)`. Hợp đồng này chốt: `transcribe_chunk` nhận **mảng numpy 16 kHz mono float32**; bản nhận đường dẫn đổi tên `transcribe_file`.

## 3. Định dạng dữ liệu

| Dữ liệu | Đường dẫn | Định dạng |
|---|---|---|
| Tập test văn bản | `data/processed/text_{train,val,test}.csv` | cột `text`, `label` (0/1), `scam_type` |
| Metadata thu âm | `data/audio/metadata.csv` | `audio_path, transcript_reference, label, scam_type, region (bac/trung/nam), tone (trung_tinh/ap_luc), speaker_id` |
| Transcript ASR | `data/audio/transcripts_asr_<model>.csv` | metadata + cột `transcript_asr`, `asr_latency_ms` |
| Model classifier | `models/text/hybrid/` | `model.pt`, `scaler.joblib`, `config.json` (gồm `structured_dim`, danh sách cột) |
| Model ASR | `models/asr/phowhisper-base-ct2/` | thư mục CTranslate2 |
| Kết quả | `results/*.json`, `results/*.csv` | có trường `seed`, `git_commit`, `timestamp` |

Nếu đường dẫn thực tế trong repo khác, sửa bảng này **trước** rồi mới làm tiếp (điểm S1).

## 4. Chuẩn đo lường (dùng chung để số liệu hai bên so sánh được)

- Nhãn dương = lừa đảo (1). **FNR = FN / (FN + TP)**.
- Macro-F1 tính theo `sklearn.metrics.f1_score(average="macro")`.
- WER tính bằng `jiwer` sau khi chuẩn hoá cả hai vế: chữ thường, bỏ dấu câu, gộp khoảng trắng; **giữ dấu thanh tiếng Việt**. Hàm chuẩn hoá đặt ở `src/audio/text_norm.py` (B viết, A không dùng).
- Split thu âm theo **`speaker_id`** (không để cùng một người nói xuất hiện ở cả Train và Test).
- Seed = 42.

## 5. Phạm vi sở hữu file

| Thư mục | Chủ | Người kia được |
|---|---|---|
| `src/features/`, `src/models/`, `src/pipeline/`, `src/nlp/` | A | đọc/import |
| `src/audio/`, `src/api/`, `scripts/`, `docker/` | B | đọc/import |
| `src/common/patterns.py` (regex dùng chung) | A tạo nếu chưa có `pii_anonymizer` | B import |
| `reports/chapter3_hybrid_ablation.md` | A | — |
| `reports/chapter3_asr_robustness.md` | B | — |
