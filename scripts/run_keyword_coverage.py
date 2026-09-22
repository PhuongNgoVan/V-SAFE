"""
scripts/run_keyword_coverage.py
Đánh giá coverage từ khoá trên tập train.

Đầu vào : data/processed/text_train.csv  (cột: text, label, scam_type)
Đầu ra  : results/keyword_coverage.csv

Chạy từ thư mục gốc repo:
    python scripts/run_keyword_coverage.py

Từ khoá nào khớp > 5% mẫu bình thường sẽ được đánh dấu cột ``high_fp_risk``.
Không tự xoá từ khoá đó — chỉ báo cáo để thảo luận (theo A1_structured_features.md).
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

# Thêm root vào sys.path để import src.*
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src.features.structured_features import KEYWORDS  # noqa: E402
from src.features.structured_features import _ensure_keywords  # noqa: E402

DATA_PATH = ROOT / "data" / "processed" / "text_train.csv"
OUT_PATH = ROOT / "results" / "keyword_coverage.csv"
FP_THRESHOLD = 0.05  # 5%


def main() -> None:
    _ensure_keywords()  # đảm bảo KEYWORDS được nạp

    if not DATA_PATH.exists():
        print(f"[ERROR] Không tìm thấy {DATA_PATH}. Cần có dữ liệu trước.", file=sys.stderr)
        sys.exit(1)

    print(f"Đang đọc {DATA_PATH} …")
    df = pd.read_csv(DATA_PATH, encoding="utf-8")

    # Kiểm tra cột bắt buộc
    for col in ("text", "label"):
        if col not in df.columns:
            print(f"[ERROR] Thiếu cột '{col}' trong CSV.", file=sys.stderr)
            sys.exit(1)

    fraud_df = df[df["label"] == 1].copy()
    normal_df = df[df["label"] == 0].copy()
    n_fraud = len(fraud_df)
    n_normal = len(normal_df)
    print(f"  Lừa đảo : {n_fraud:,} mẫu | Bình thường: {n_normal:,} mẫu")

    fraud_texts = fraud_df["text"].fillna("").str.lower().tolist()
    normal_texts = normal_df["text"].fillna("").str.lower().tolist()

    rows: list[dict] = []
    t0 = time.perf_counter()

    for group_name, kw_list in KEYWORDS.items():
        for kw in kw_list:
            fraud_hits = sum(1 for t in fraud_texts if kw in t)
            normal_hits = sum(1 for t in normal_texts if kw in t)
            fraud_rate = fraud_hits / n_fraud if n_fraud > 0 else 0.0
            normal_rate = normal_hits / n_normal if n_normal > 0 else 0.0
            high_fp_risk = normal_rate > FP_THRESHOLD
            rows.append(
                {
                    "group": group_name,
                    "keyword": kw,
                    "fraud_hits": fraud_hits,
                    "fraud_rate": round(fraud_rate, 4),
                    "normal_hits": normal_hits,
                    "normal_rate": round(normal_rate, 4),
                    "high_fp_risk": high_fp_risk,
                }
            )

    elapsed = time.perf_counter() - t0
    print(f"  Quét xong trong {elapsed:.2f}s")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["group", "keyword", "fraud_hits", "fraud_rate", "normal_hits", "normal_rate", "high_fp_risk"]
    with open(OUT_PATH, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    high_fp = [r for r in rows if r["high_fp_risk"]]
    print(f"\n[Kết quả] Lưu tại: {OUT_PATH}")
    print(f"  Tổng từ khoá: {len(rows)}")
    print(f"  Từ khoá high FP risk (>{FP_THRESHOLD*100:.0f}% mẫu bình thường): {len(high_fp)}")
    if high_fp:
        print("  Danh sách cần đánh dấu trong báo cáo (KHÔNG tự xoá):")
        for r in high_fp:
            print(f"    [{r['group']}] \"{r['keyword']}\" — normal_rate={r['normal_rate']:.2%}")


if __name__ == "__main__":
    main()
