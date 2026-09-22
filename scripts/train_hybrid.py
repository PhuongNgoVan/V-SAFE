"""
scripts/train_hybrid.py
========================
Script huấn luyện ``HybridTextClassifier`` từ dòng lệnh.

Spec: A2_hybrid_classifier.md

Cách dùng
---------
Chạy cơ bản::

    python scripts/train_hybrid.py --seed 42

Với đối số tùy chỉnh::

    python scripts/train_hybrid.py \\
        --data_dir   data/processed \\
        --output_dir models/text/hybrid \\
        --cache_dir  data/cache \\
        --results_dir results \\
        --epochs 50 \\
        --patience 5 \\
        --lr 2e-4 \\
        --batch_size 64 \\
        --device cuda \\
        --seed 42

Đầu ra
------
``models/text/hybrid/``
    model.pt, scaler.joblib, config.json

``results/``
    hybrid_train_log.csv, hybrid_test_metrics.json

Nghiệm thu
----------
- scaler.fit CHỈ được gọi trên tập Train.
- Threshold chọn trên Val, không trên Test.
- Chạy lại với cùng seed → kết quả ±0.5% F1.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import logging
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Thêm thư mục gốc repo vào sys.path để import các module nội bộ
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.models.hybrid_classifier import (  # noqa: E402
    HybridTextClassifier,
    DROPOUT_P,
    HIDDEN_DIM,
    STRUCT_DIM,
    TEXT_DIM,
)
from src.features.structured_features import (  # noqa: E402
    FEATURE_ORDER,
    extract_structured_features,
    features_to_vector,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_hybrid")

# ---------------------------------------------------------------------------
# Hằng số
# ---------------------------------------------------------------------------
PHOBERT_MODEL_NAME = "vinai/phobert-base-v2"
EMBEDDING_SOURCE   = "phobert-base-v2-frozen-cls"
MAX_LEN            = 256
TEXT_COL           = "clean_text"
LABEL_COL          = "label"

# Cột số cần StandardScaler vs cột boolean giữ 0/1
NUMERIC_COLS = [
    "n_urgency_kw", "n_authority_kw", "n_financial_action_kw",
    "n_reward_kw", "message_length", "uppercase_ratio", "exclamation_count",
]
BOOLEAN_COLS = [
    "has_phone_number", "has_bank_account_like_number",
    "has_url", "has_id_number_request",
]
NUMERIC_IDX = [FEATURE_ORDER.index(c) for c in NUMERIC_COLS]


# ===========================================================================
# Argparse
# ===========================================================================

def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Huấn luyện HybridTextClassifier (A2)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data_dir",    type=Path, default=_REPO_ROOT / "data" / "processed",
                   help="Thư mục chứa text_{train,val,test}.csv")
    p.add_argument("--output_dir",  type=Path, default=_REPO_ROOT / "models" / "text" / "hybrid",
                   help="Thư mục lưu model.pt, scaler.joblib, config.json")
    p.add_argument("--cache_dir",   type=Path, default=_REPO_ROOT / "data" / "cache",
                   help="Thư mục cache embedding .npy")
    p.add_argument("--results_dir", type=Path, default=_REPO_ROOT / "results",
                   help="Thư mục lưu train_log.csv và test_metrics.json")
    p.add_argument("--phobert",     type=str,  default=PHOBERT_MODEL_NAME,
                   help="Tên hoặc đường dẫn PhoBERT model")
    p.add_argument("--epochs",      type=int,  default=50,    help="Số epoch tối đa")
    p.add_argument("--patience",    type=int,  default=5,     help="Early stopping patience")
    p.add_argument("--lr",          type=float,default=2e-4,  help="Learning rate AdamW")
    p.add_argument("--weight_decay",type=float,default=1e-2,  help="Weight decay AdamW")
    p.add_argument("--batch_size",  type=int,  default=64,    help="Batch size")
    p.add_argument("--emb_batch",   type=int,  default=32,    help="Batch size khi trích embedding")
    p.add_argument("--hidden",      type=int,  default=HIDDEN_DIM, help="Kích thước lớp ẩn")
    p.add_argument("--dropout",     type=float,default=DROPOUT_P,  help="Dropout probability")
    p.add_argument("--max_len",     type=int,  default=MAX_LEN,    help="Max token length PhoBERT")
    p.add_argument("--seed",        type=int,  default=42,    help="Random seed")
    p.add_argument("--device",      type=str,  default="auto",
                   help="Thiết bị: 'auto', 'cuda', 'cpu'")
    p.add_argument("--text_col",    type=str,  default=TEXT_COL,  help="Tên cột văn bản")
    p.add_argument("--label_col",   type=str,  default=LABEL_COL, help="Tên cột nhãn")
    p.add_argument("--no_cache",    action="store_true",
                   help="Bỏ qua cache embedding, luôn tính lại")
    return p.parse_args(argv)


# ===========================================================================
# Seed
# ===========================================================================

def set_seed(seed: int) -> None:
    """Đặt seed toàn bộ cho reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ===========================================================================
# Embedding
# ===========================================================================

def _text_hash(texts: list[str]) -> str:
    """SHA-256 của toàn bộ văn bản — dùng phát hiện cache cũ."""
    blob = "\x00".join(texts).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


@torch.no_grad()
def _encode_batch(
    texts: list[str],
    tokenizer,
    model,
    device: torch.device,
    batch_size: int,
    max_len: int,
) -> np.ndarray:
    """Trích [CLS] embedding 768-d cho danh sách texts (GPU-accelerated)."""
    all_embs: list[np.ndarray] = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding", leave=False):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch,
            max_length=max_len,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        out = model(**enc)
        cls_emb = out.last_hidden_state[:, 0, :].cpu().numpy()  # [CLS]
        all_embs.append(cls_emb)
    return np.vstack(all_embs).astype(np.float32)


def get_or_compute_embeddings(
    df: pd.DataFrame,
    split: str,
    cache_dir: Path,
    tokenizer,
    phobert,
    device: torch.device,
    emb_batch: int,
    max_len: int,
    text_col: str,
    no_cache: bool = False,
) -> np.ndarray:
    """Nạp từ cache (nếu hash khớp) hoặc tính lại và lưu cache."""
    texts = df[text_col].tolist()
    current_hash = _text_hash(texts)

    npy_path  = cache_dir / f"phobert_emb_{split}.npy"
    hash_path = cache_dir / f"phobert_emb_{split}.sha256"

    if not no_cache and npy_path.exists() and hash_path.exists():
        cached_hash = hash_path.read_text().strip()
        if cached_hash == current_hash:
            embs = np.load(npy_path)
            logger.info("[%s] Cache hit → %s (shape=%s)", split, npy_path.name, embs.shape)
            return embs
        logger.warning("[%s] Cache cũ (hash mismatch) → tính lại", split)

    logger.info("[%s] Đang trích embedding cho %d mẫu...", split, len(texts))
    embs = _encode_batch(texts, tokenizer, phobert, device, emb_batch, max_len)
    np.save(npy_path, embs)
    hash_path.write_text(current_hash)
    logger.info("[%s] Đã lưu cache → %s (shape=%s)", split, npy_path.name, embs.shape)
    return embs


# ===========================================================================
# Structured features
# ===========================================================================

def extract_feature_matrix(df: pd.DataFrame, text_col: str, desc: str = "") -> np.ndarray:
    """Trích ma trận (N, 11) float32 theo FEATURE_ORDER."""
    rows = []
    for text in tqdm(df[text_col].tolist(), desc=f"Structured [{desc}]", leave=False):
        d = extract_structured_features(str(text))
        rows.append(features_to_vector(d))
    return np.vstack(rows).astype(np.float32)


def apply_scaler(feat_raw: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    """Áp scaler CHỈ cho các cột số; giữ nguyên cột boolean 0/1."""
    feat = feat_raw.copy()
    feat[:, NUMERIC_IDX] = scaler.transform(feat_raw[:, NUMERIC_IDX])
    return feat


# ===========================================================================
# Dataset
# ===========================================================================

class HybridDataset(Dataset):
    """Dataset ghép embedding + structured features + nhãn."""

    def __init__(
        self,
        embeddings: np.ndarray,
        features:   np.ndarray,
        labels:     np.ndarray,
    ) -> None:
        assert len(embeddings) == len(features) == len(labels), \
            "embeddings, features, labels phải có cùng số mẫu"
        self.emb   = torch.tensor(embeddings, dtype=torch.float32)
        self.feat  = torch.tensor(features,   dtype=torch.float32)
        self.label = torch.tensor(labels,     dtype=torch.long)

    def __len__(self) -> int:
        return len(self.label)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.emb[idx], self.feat[idx], self.label[idx]


# ===========================================================================
# Evaluate
# ===========================================================================

def evaluate(
    model: HybridTextClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float]:
    """Tính (loss, accuracy, macro_f1) trên loader.

    Trả về
    ------
    (avg_loss, accuracy, macro_f1)
    """
    model.eval()
    all_preds:  list[int] = []
    all_labels: list[int] = []
    total_loss = 0.0

    with torch.no_grad():
        for emb, feat, lbl in loader:
            emb, feat, lbl = emb.to(device), feat.to(device), lbl.to(device)
            logits = model(emb, feat)
            loss   = criterion(logits, lbl)
            total_loss += loss.item() * len(lbl)
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(lbl.cpu().numpy().tolist())

    n        = len(all_labels)
    avg_loss = total_loss / n
    acc      = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, acc, macro_f1


# ===========================================================================
# Threshold selection
# ===========================================================================

def select_threshold(
    model: HybridTextClassifier,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    default: float = 0.5,
) -> float:
    """Chọn ngưỡng quyết định trên Val.

    Chỉ áp dụng cho nhị phân (num_classes == 2):
    - Nếu FNR @ 0.5 ≤ 5%  → giữ 0.5.
    - Nếu FNR @ 0.5 > 5%  → quét [0.30, 0.50) chọn threshold
      cho Macro-F1 cao nhất mà vẫn đáp ứng FNR ≤ 5%.
    Với đa lớp: trả về 0.5 (argmax).

    Ngưỡng chọn KHÔNG dùng dữ liệu Test.
    """
    if num_classes != 2:
        logger.info("Đa lớp → giữ threshold=0.5 (argmax)")
        return default

    model.eval()
    probs_list:  list[np.ndarray] = []
    labels_list: list[int]        = []

    with torch.no_grad():
        for emb, feat, lbl in loader:
            emb, feat = emb.to(device), feat.to(device)
            probs = model(emb, feat, return_probs=True).cpu().numpy()
            probs_list.append(probs)
            labels_list.extend(lbl.numpy().tolist())

    val_probs  = np.vstack(probs_list)
    val_labels = np.array(labels_list)

    def _metrics_at(t: float) -> tuple[float, float, float]:
        preds = (val_probs[:, 1] >= t).astype(int)
        tp = int(((preds == 1) & (val_labels == 1)).sum())
        fn = int(((preds == 0) & (val_labels == 1)).sum())
        fp = int(((preds == 1) & (val_labels == 0)).sum())
        tn = int(((preds == 0) & (val_labels == 0)).sum())
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        f1  = f1_score(val_labels, preds, average="macro", zero_division=0)
        return fnr, fpr, f1

    fnr_05, fpr_05, _ = _metrics_at(0.5)
    logger.info("Val FNR@0.50=%.4f, FPR@0.50=%.4f", fnr_05, fpr_05)

    if fnr_05 <= 0.05:
        logger.info("FNR ≤ 5%% → giữ threshold=0.50")
        return 0.5

    logger.warning("FNR=%.4f > 5%% → tìm threshold tối ưu trên Val", fnr_05)
    best_thresh, best_f1 = 0.5, -1.0
    header = f"{'Threshold':>10} | {'FNR':>7} | {'FPR':>7} | {'Macro-F1':>9}"
    logger.info(header)
    for t in np.arange(0.30, 0.52, 0.02):
        fnr_t, fpr_t, f1_t = _metrics_at(float(t))
        marker = " ← candidate" if fnr_t <= 0.05 else ""
        logger.info("%10.2f | %7.4f | %7.4f | %9.4f%s", t, fnr_t, fpr_t, f1_t, marker)
        if f1_t > best_f1 and fnr_t <= 0.05:
            best_f1     = f1_t
            best_thresh = float(t)

    logger.info("Threshold cuối chọn: %.2f (Val Macro-F1=%.4f)", best_thresh, best_f1)
    return best_thresh


# ===========================================================================
# Main
# ===========================================================================

def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)

    # ── Thiết lập ──
    for d in [args.output_dir, args.cache_dir, args.results_dir]:
        d.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info("Device: %s | Seed: %d", device, args.seed)

    # Git commit
    try:
        git_commit = subprocess.check_output(
            ["git", "-C", str(_REPO_ROOT), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        git_commit = "unknown"

    # ── Nạp dữ liệu ──
    logger.info("Nạp dữ liệu từ %s", args.data_dir)
    df_train = pd.read_csv(args.data_dir / "text_train.csv")
    df_val   = pd.read_csv(args.data_dir / "text_val.csv")
    df_test  = pd.read_csv(args.data_dir / "text_test.csv")

    for name, df in [("train", df_train), ("val", df_val), ("test", df_test)]:
        df[args.text_col].fillna("", inplace=True)
        logger.info("[%s] %d mẫu", name, len(df))

    classes     = sorted(df_train[args.label_col].unique().tolist())
    num_classes = len(classes)
    label2id    = {c: i for i, c in enumerate(classes)}
    id2label    = {i: c for c, i in label2id.items()}
    logger.info("Classes (%d): %s", num_classes, classes)

    y_train = np.array([label2id[l] for l in df_train[args.label_col]], dtype=np.int64)
    y_val   = np.array([label2id[l] for l in df_val[args.label_col]],   dtype=np.int64)
    y_test  = np.array([label2id[l] for l in df_test[args.label_col]],  dtype=np.int64)

    # ── PhoBERT embedding ──
    logger.info("Tải PhoBERT: %s", args.phobert)
    tokenizer = AutoTokenizer.from_pretrained(args.phobert)
    phobert   = AutoModel.from_pretrained(args.phobert).to(device)
    phobert.eval()
    for param in phobert.parameters():
        param.requires_grad = False

    emb_train = get_or_compute_embeddings(
        df_train, "train", args.cache_dir, tokenizer, phobert,
        device, args.emb_batch, args.max_len, args.text_col, args.no_cache,
    )
    emb_val = get_or_compute_embeddings(
        df_val, "val", args.cache_dir, tokenizer, phobert,
        device, args.emb_batch, args.max_len, args.text_col, args.no_cache,
    )
    emb_test = get_or_compute_embeddings(
        df_test, "test", args.cache_dir, tokenizer, phobert,
        device, args.emb_batch, args.max_len, args.text_col, args.no_cache,
    )
    del phobert
    torch.cuda.empty_cache()

    assert emb_train.shape[1] == TEXT_DIM, \
        f"Embedding dim phải là {TEXT_DIM}, nhận {emb_train.shape[1]}"

    # ── Structured features ──
    logger.info("Trích structured features...")
    feat_train_raw = extract_feature_matrix(df_train, args.text_col, "train")
    feat_val_raw   = extract_feature_matrix(df_val,   args.text_col, "val")
    feat_test_raw  = extract_feature_matrix(df_test,  args.text_col, "test")

    # StandardScaler — CHỈ fit trên Train
    scaler = StandardScaler()
    scaler.fit(feat_train_raw[:, NUMERIC_IDX])   # ← DUY NHẤT chỗ này
    logger.info("StandardScaler fit trên Train (numeric cols: %s)", NUMERIC_COLS)

    feat_train = apply_scaler(feat_train_raw, scaler)
    feat_val   = apply_scaler(feat_val_raw,   scaler)
    feat_test  = apply_scaler(feat_test_raw,  scaler)

    # ── DataLoader ──
    ds_train = HybridDataset(emb_train, feat_train, y_train)
    ds_val   = HybridDataset(emb_val,   feat_val,   y_val)
    ds_test  = HybridDataset(emb_test,  feat_test,  y_test)

    num_workers = min(4, 2)
    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=(device.type == "cuda"))
    dl_val   = DataLoader(ds_val,   batch_size=args.batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=(device.type == "cuda"))
    dl_test  = DataLoader(ds_test,  batch_size=args.batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=(device.type == "cuda"))

    # ── Model ──
    set_seed(args.seed)
    model = HybridTextClassifier(
        num_classes = num_classes,
        text_dim    = TEXT_DIM,
        struct_dim  = STRUCT_DIM,
        hidden      = args.hidden,
        dropout     = args.dropout,
    ).to(device)
    logger.info("Model: %s", model)

    # ── Class weights ──
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(num_classes),
        y=y_train,
    )
    weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    logger.info("Class weights: %s", dict(zip(classes, [f"{w:.3f}" for w in class_weights])))

    # ── Optimizer + Loss ──
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    # ── Training loop ──
    log_path   = args.results_dir / "hybrid_train_log.csv"
    log_fields = ["epoch", "train_loss", "val_loss", "val_acc", "val_macro_f1", "lr", "elapsed_s"]
    train_log:  list[dict] = []

    best_val_f1      = -1.0
    best_epoch       = 0
    patience_cnt     = 0
    best_model_state = None

    logger.info("Bắt đầu huấn luyện (max_epochs=%d, patience=%d)...", args.epochs, args.patience)
    print(f"\n{'='*65}")
    print(f"{'Epoch':>5} | {'TrainLoss':>9} | {'ValLoss':>7} | {'ValAcc':>6} | {'ValF1':>6} | {'LR':>8}")
    print(f"{'='*65}")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        epoch_loss = 0.0

        for emb, feat, lbl in dl_train:
            emb, feat, lbl = emb.to(device), feat.to(device), lbl.to(device)
            optimizer.zero_grad()
            logits = model(emb, feat)
            loss   = criterion(logits, lbl)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(lbl)

        train_loss = epoch_loss / len(ds_train)
        val_loss, val_acc, val_f1 = evaluate(model, dl_val, criterion, device)
        scheduler.step(val_f1)
        elapsed = time.time() - t0
        cur_lr  = optimizer.param_groups[0]["lr"]

        row = {
            "epoch":        epoch,
            "train_loss":   round(train_loss, 6),
            "val_loss":     round(val_loss,   6),
            "val_acc":      round(val_acc,    6),
            "val_macro_f1": round(val_f1,     6),
            "lr":           cur_lr,
            "elapsed_s":    round(elapsed,    2),
        }
        train_log.append(row)

        print(f"{epoch:>5} | {train_loss:>9.4f} | {val_loss:>7.4f} | {val_acc:>6.4f} | {val_f1:>6.4f} | {cur_lr:>8.2e}")

        if val_f1 > best_val_f1 + 1e-5:
            best_val_f1      = val_f1
            best_epoch       = epoch
            patience_cnt     = 0
            best_model_state = copy.deepcopy(model.state_dict())
            print(f"   ⭐ Best Val Macro-F1: {best_val_f1:.4f} (epoch {best_epoch})")
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                print(f"\n⏹  Early Stopping tại epoch {epoch} (patience={args.patience})")
                print(f"   Best epoch={best_epoch}, Val Macro-F1={best_val_f1:.4f}")
                break

    # Khôi phục best weights
    model.load_state_dict(best_model_state)
    logger.info("Huấn luyện xong. Best epoch=%d, Val Macro-F1=%.4f", best_epoch, best_val_f1)

    # Ghi CSV log
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=log_fields)
        writer.writeheader()
        writer.writerows(train_log)
    logger.info("Training log → %s", log_path)

    # ── Chọn threshold trên Val ──
    threshold = select_threshold(model, dl_val, device, num_classes)

    # ── Đánh giá Test (một lần duy nhất) ──
    logger.info("Đánh giá trên Test...")
    model.eval()
    test_probs_list:  list[np.ndarray] = []
    test_labels_list: list[int]        = []

    with torch.no_grad():
        for emb, feat, lbl in dl_test:
            emb, feat = emb.to(device), feat.to(device)
            probs = model(emb, feat, return_probs=True).cpu().numpy()
            test_probs_list.append(probs)
            test_labels_list.extend(lbl.numpy().tolist())

    test_probs_all  = np.vstack(test_probs_list)
    test_labels_all = np.array(test_labels_list)

    if num_classes == 2:
        test_preds = (test_probs_all[:, 1] >= threshold).astype(int)
    else:
        test_preds = test_probs_all.argmax(axis=-1)

    test_acc      = accuracy_score(test_labels_all, test_preds)
    test_macro_f1 = f1_score(test_labels_all, test_preds, average="macro",    zero_division=0)
    test_micro_f1 = f1_score(test_labels_all, test_preds, average="micro",    zero_division=0)
    cm            = confusion_matrix(test_labels_all, test_preds).tolist()

    if num_classes == 2:
        tn, fp, fn, tp = np.array(cm).ravel()
        test_fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        test_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    else:
        fnrs, fprs = [], []
        for i in range(num_classes):
            tp_i = int(((test_preds == i) & (test_labels_all == i)).sum())
            fn_i = int(((test_preds != i) & (test_labels_all == i)).sum())
            fp_i = int(((test_preds == i) & (test_labels_all != i)).sum())
            tn_i = int(((test_preds != i) & (test_labels_all != i)).sum())
            fnrs.append(fn_i / (fn_i + tp_i) if (fn_i + tp_i) > 0 else 0.0)
            fprs.append(fp_i / (fp_i + tn_i) if (fp_i + tn_i) > 0 else 0.0)
        test_fnr = float(np.mean(fnrs))
        test_fpr = float(np.mean(fprs))

    samples_per_class = {
        id2label[i]: int((test_labels_all == i).sum()) for i in range(num_classes)
    }

    test_metrics = {
        "accuracy":           round(float(test_acc),      4),
        "macro_f1":           round(float(test_macro_f1), 4),
        "micro_f1":           round(float(test_micro_f1), 4),
        "fnr":                round(float(test_fnr),      4),
        "fpr":                round(float(test_fpr),      4),
        "confusion_matrix":   cm,
        "samples_per_class":  samples_per_class,
        "threshold":          threshold,
        "best_val_macro_f1":  round(float(best_val_f1),  4),
        "best_epoch":         best_epoch,
        "n_test":             int(len(test_labels_all)),
    }

    metrics_path = args.results_dir / "hybrid_test_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, ensure_ascii=False, indent=2)
    logger.info("Test metrics → %s", metrics_path)

    print(f"\n{'═'*55}")
    print("  TEST METRICS")
    print(f"{'═'*55}")
    print(f"  Accuracy  : {test_acc:.4f}")
    print(f"  Macro-F1  : {test_macro_f1:.4f}")
    print(f"  FNR       : {test_fnr:.4f}")
    print(f"  FPR       : {test_fpr:.4f}")
    print(f"  Threshold : {threshold}")
    print(f"{'═'*55}")
    print(classification_report(
        test_labels_all, test_preds,
        target_names=[id2label[i] for i in range(num_classes)],
        zero_division=0,
    ))

    # ── Lưu artifacts ──
    # 1. model.pt
    model_path = args.output_dir / "model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": {
                "num_classes": num_classes,
                "text_dim":    TEXT_DIM,
                "struct_dim":  STRUCT_DIM,
                "hidden":      args.hidden,
                "dropout":     args.dropout,
            },
            "label2id": label2id,
            "id2label":  id2label,
        },
        model_path,
    )
    logger.info("model.pt → %s", model_path)

    # 2. scaler.joblib
    scaler_path = args.output_dir / "scaler.joblib"
    joblib.dump(scaler, scaler_path)
    logger.info("scaler.joblib → %s", scaler_path)

    # 3. config.json
    config = {
        "structured_dim":    STRUCT_DIM,
        "feature_order":     FEATURE_ORDER,
        "numeric_cols":      NUMERIC_COLS,
        "boolean_cols":      BOOLEAN_COLS,
        "hidden":            args.hidden,
        "dropout":           args.dropout,
        "threshold":         threshold,
        "embedding_source":  EMBEDDING_SOURCE,
        "embedding_dim":     TEXT_DIM,
        "pooling":           "CLS",
        "phobert_model":     args.phobert,
        "seed":              args.seed,
        "num_classes":       num_classes,
        "classes":           classes,
        "label2id":          label2id,
        "id2label":          {str(k): v for k, v in id2label.items()},
        "best_epoch":        best_epoch,
        "best_val_macro_f1": round(float(best_val_f1), 4),
        "lr":                args.lr,
        "weight_decay":      args.weight_decay,
        "max_len":           args.max_len,
        "batch_size":        args.batch_size,
        "git_commit":        git_commit,
    }
    config_path = args.output_dir / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    logger.info("config.json → %s", config_path)

    print(f"\n📦 Artifacts lưu tại: {args.output_dir}")
    print("   ├── model.pt")
    print("   ├── scaler.joblib")
    print("   └── config.json")


if __name__ == "__main__":
    main()
