"""
src/models/hybrid_classifier.py
================================
Định nghĩa lớp ``HybridTextClassifier`` — classifier lai ghép:

    PhoBERT [CLS] embedding (768-d)  ──┐
                                        ├──► Linear(779, 128) → ReLU → Dropout(0.3) → Linear(128, C)
    Structured features (11-d)      ────┘

Sở hữu: A — xem 00_SHARED_CONTRACT.md §5
Spec:    A2_hybrid_classifier.md

Hàm / lớp công khai
────────────────────
- ``HybridTextClassifier``      : nn.Module chính.
- ``load_from_checkpoint()``    : tiện ích nạp lại từ model.pt.

Hằng số
───────
- ``TEXT_DIM``    = 768   (PhoBERT base)
- ``STRUCT_DIM``  = 11    (số đặc trưng cấu trúc — xem FEATURE_ORDER trong structured_features.py)
- ``HIDDEN_DIM``  = 128
- ``DROPOUT_P``   = 0.3
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hằng số kiến trúc mặc định — phải khớp với A2_hybrid_classifier.md §4
# ---------------------------------------------------------------------------
TEXT_DIM:   int   = 768   # chiều embedding PhoBERT [CLS]
STRUCT_DIM: int   = 11    # số đặc trưng cấu trúc (xem FEATURE_ORDER)
HIDDEN_DIM: int   = 128   # kích thước lớp ẩn
DROPOUT_P:  float = 0.3   # xác suất dropout


class HybridTextClassifier(nn.Module):
    """Classifier lai: PhoBERT embedding + structured features.

    Kiến trúc
    ---------
    ::

        concat(text_emb [B,768], struct_feat [B,11])   →  [B, 779]
        Linear(779, hidden)                             →  [B, 128]
        ReLU()
        Dropout(p=dropout)
        Linear(hidden, num_classes)                     →  [B, C]

    Tham số khởi tạo
    ----------------
    num_classes : int
        Số lớp phân loại (C).
    text_dim : int, optional
        Chiều embedding văn bản đầu vào. Mặc định: 768 (PhoBERT base).
    struct_dim : int, optional
        Số đặc trưng cấu trúc đầu vào. Mặc định: 11.
    hidden : int, optional
        Kích thước lớp ẩn. Mặc định: 128.
    dropout : float, optional
        Xác suất dropout. Mặc định: 0.3.

    Ví dụ
    -----
    >>> model = HybridTextClassifier(num_classes=2)
    >>> emb  = torch.randn(8, 768)
    >>> feat = torch.randn(8, 11)
    >>> logits = model(emb, feat)           # (8, 2)
    >>> probs  = model(emb, feat, return_probs=True)  # (8, 2), tổng = 1
    """

    def __init__(
        self,
        num_classes: int,
        text_dim:    int   = TEXT_DIM,
        struct_dim:  int   = STRUCT_DIM,
        hidden:      int   = HIDDEN_DIM,
        dropout:     float = DROPOUT_P,
    ) -> None:
        super().__init__()

        if num_classes < 2:
            raise ValueError(f"num_classes phải ≥ 2, nhận {num_classes}")
        if text_dim <= 0 or struct_dim <= 0 or hidden <= 0:
            raise ValueError("text_dim, struct_dim, hidden phải là số dương")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout phải nằm trong [0, 1), nhận {dropout}")

        self.text_dim    = text_dim
        self.struct_dim  = struct_dim
        self.hidden      = hidden
        self.num_classes = num_classes
        self.dropout_p   = dropout

        input_dim = text_dim + struct_dim  # 768 + 11 = 779

        self.fc1     = nn.Linear(input_dim, hidden)
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.fc2     = nn.Linear(hidden, num_classes)

        self._init_weights()

        logger.debug(
            "HybridTextClassifier khởi tạo: %d+%d → %d → %d (dropout=%.2f)",
            text_dim, struct_dim, hidden, num_classes, dropout,
        )

    # ------------------------------------------------------------------
    # Khởi tạo trọng số
    # ------------------------------------------------------------------
    def _init_weights(self) -> None:
        """Xavier uniform cho Linear layers; bias = 0."""
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        text_emb:     torch.Tensor,
        struct_feat:  torch.Tensor,
        return_probs: bool = False,
    ) -> torch.Tensor:
        """Tính toán forward pass.

        Tham số
        -------
        text_emb : torch.Tensor
            Tensor embedding văn bản, shape ``(B, text_dim)``.
        struct_feat : torch.Tensor
            Tensor đặc trưng cấu trúc (đã chuẩn hoá), shape ``(B, struct_dim)``.
        return_probs : bool, optional
            - ``False`` (mặc định): trả về **logits** ``(B, C)`` — dùng với
              ``nn.CrossEntropyLoss``.
            - ``True``: trả về **xác suất softmax** ``(B, C)`` — dùng khi
              inference / đánh giá.

        Trả về
        ------
        torch.Tensor
            Shape ``(B, C)``. Logits hoặc xác suất tùy ``return_probs``.

        Raises
        ------
        ValueError
            Nếu chiều cuối của ``text_emb`` hoặc ``struct_feat`` không khớp
            với ``self.text_dim`` / ``self.struct_dim``.
        """
        if text_emb.shape[-1] != self.text_dim:
            raise ValueError(
                f"text_emb dim cuối phải là {self.text_dim}, nhận {text_emb.shape[-1]}"
            )
        if struct_feat.shape[-1] != self.struct_dim:
            raise ValueError(
                f"struct_feat dim cuối phải là {self.struct_dim}, nhận {struct_feat.shape[-1]}"
            )

        x = torch.cat([text_emb, struct_feat], dim=-1)  # (B, 779)
        x = self.fc1(x)                                  # (B, 128)
        x = self.relu(x)
        x = self.dropout(x)
        logits = self.fc2(x)                             # (B, C)

        if return_probs:
            return F.softmax(logits, dim=-1)
        return logits

    # ------------------------------------------------------------------
    # Tiện ích
    # ------------------------------------------------------------------
    def predict(
        self,
        text_emb:    torch.Tensor,
        struct_feat: torch.Tensor,
        threshold:   float = 0.5,
    ) -> torch.Tensor:
        """Dự đoán nhãn lớp.

        Với nhị phân (``num_classes == 2``): áp ``threshold`` lên xác suất
        lớp dương (index 1).  Với đa lớp: ``argmax``.

        Tham số
        -------
        text_emb : torch.Tensor
            Shape ``(B, text_dim)``.
        struct_feat : torch.Tensor
            Shape ``(B, struct_dim)``.
        threshold : float, optional
            Ngưỡng quyết định cho bài toán nhị phân. Mặc định: 0.5.

        Trả về
        ------
        torch.Tensor, dtype=torch.long
            Nhãn dự đoán, shape ``(B,)``.
        """
        with torch.no_grad():
            probs = self.forward(text_emb, struct_feat, return_probs=True)
        if self.num_classes == 2:
            return (probs[:, 1] >= threshold).long()
        return probs.argmax(dim=-1)

    def extra_repr(self) -> str:
        return (
            f"text_dim={self.text_dim}, struct_dim={self.struct_dim}, "
            f"hidden={self.hidden}, num_classes={self.num_classes}, "
            f"dropout={self.dropout_p}"
        )


# ---------------------------------------------------------------------------
# Tiện ích: nạp lại từ checkpoint
# ---------------------------------------------------------------------------

def load_from_checkpoint(
    checkpoint_path: str | Path,
    device: Optional[torch.device] = None,
) -> tuple[HybridTextClassifier, dict]:
    """Nạp lại ``HybridTextClassifier`` từ file ``model.pt``.

    File ``model.pt`` phải được lưu bởi ``torch.save`` với cấu trúc::

        {
            "model_state_dict": <OrderedDict>,
            "model_config": {
                "num_classes": int,
                "text_dim":    int,
                "struct_dim":  int,
                "hidden":      int,
                "dropout":     float,
            },
            "label2id": dict,
            "id2label":  dict,
        }

    Tham số
    -------
    checkpoint_path : str hoặc Path
        Đường dẫn đến file ``model.pt``.
    device : torch.device, optional
        Thiết bị để nạp model. Mặc định: tự động chọn (GPU nếu có).

    Trả về
    ------
    model : HybridTextClassifier
        Model ở chế độ ``eval()``.
    checkpoint : dict
        Toàn bộ nội dung checkpoint (chứa ``label2id``, ``id2label``, v.v.).

    Raises
    ------
    FileNotFoundError
        Nếu ``checkpoint_path`` không tồn tại.
    KeyError
        Nếu checkpoint thiếu key bắt buộc.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Không tìm thấy checkpoint: {checkpoint_path}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    required_keys = {"model_state_dict", "model_config"}
    missing = required_keys - checkpoint.keys()
    if missing:
        raise KeyError(f"Checkpoint thiếu các key: {missing}")

    cfg = checkpoint["model_config"]
    model = HybridTextClassifier(
        num_classes = cfg["num_classes"],
        text_dim    = cfg.get("text_dim",   TEXT_DIM),
        struct_dim  = cfg.get("struct_dim", STRUCT_DIM),
        hidden      = cfg.get("hidden",     HIDDEN_DIM),
        dropout     = cfg.get("dropout",    DROPOUT_P),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    logger.info(
        "Đã nạp HybridTextClassifier từ %s (device=%s, classes=%d)",
        checkpoint_path, device, cfg["num_classes"],
    )
    return model, checkpoint
