"""
src/pipeline/predict.py
========================
Điểm bàn giao S2 — hàm ``process_and_predict()`` là cầu nối duy nhất
giữa pipeline text của A và API của B.

Spec:  A3_process_and_predict.md  +  00_SHARED_CONTRACT.md §1
Sở hữu: A (xem 00_SHARED_CONTRACT.md §5)

Pipeline đầy đủ
---------------
::

    raw text
        │
        ▼ nlp_preprocess.run()          # tiền xử lý (teencode, bypass-filter…)
        │
        ▼ extract_text_embedding()      # PhoBERT [CLS] → 768-d  (singleton)
        │
        ▼ extract_structured_features() # 11 đặc trưng cấu trúc  (A1)
        │
        ▼ scaler.transform()            # StandardScaler (chỉ cột số)
        │
        ▼ HybridTextClassifier.forward()# 779 → 128 → C          (A2)
        │
        ▼ Prediction(…)                 # kết quả bàn giao cho B

Hàm / lớp công khai
────────────────────
- ``Prediction``           : dataclass kết quả theo hợp đồng.
- ``process_and_predict()`` : hàm chính, nhận raw text → Prediction.
- ``score_to_risk_level()`` : tiện ích chuyển confidence → risk label.

Singleton
---------
Model, scaler và tokenizer được nạp **một lần duy nhất** (lazy, thread-safe)
nhờ ``threading.Lock``.  Lần gọi đầu tiên chậm hơn; các lần sau dùng cache.

Xử lý biên
-----------
- Chuỗi rỗng / toàn khoảng trắng → ``Prediction(is_fraud=False, …)`` ngay lập tức.
- Văn bản quá dài → truncate theo ``max_length`` của tokenizer (không crash).
- Model chưa được huấn luyện (chưa có model.pt) → raise ``ModelNotReadyError``.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import torch
import yaml
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Đường dẫn mặc định (có thể ghi đè qua biến môi trường / tests)
# ---------------------------------------------------------------------------
_REPO_ROOT    = Path(__file__).resolve().parents[2]
_MODEL_DIR    = _REPO_ROOT / "models" / "text" / "hybrid"
_CONFIG_PATH  = _MODEL_DIR / "config.json"
_MODEL_PATH   = _MODEL_DIR / "model.pt"
_SCALER_PATH  = _MODEL_DIR / "scaler.joblib"
_RISK_CFG     = _REPO_ROOT / "configs" / "risk_thresholds.yaml"

# ---------------------------------------------------------------------------
# Import module nội bộ (với fallback khi chạy độc lập)
# ---------------------------------------------------------------------------
import sys
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.features.structured_features import (  # noqa: E402
    FEATURE_ORDER,
    extract_structured_features,
    features_to_vector,
    matched_keywords,
)
from src.models.hybrid_classifier import (  # noqa: E402
    HybridTextClassifier,
    TEXT_DIM,
    STRUCT_DIM,
)

# ---------------------------------------------------------------------------
# nlp_preprocess — dùng module thực nếu có, fallback noop nếu chưa implement
# ---------------------------------------------------------------------------
try:
    from src.nlp import nlp_preprocess as _nlp_mod  # type: ignore
    _NLP_RUN = _nlp_mod.run
    logger.debug("Dùng nlp_preprocess.run() từ src.nlp")
except (ImportError, AttributeError):
    logger.warning(
        "Không tìm thấy src.nlp.nlp_preprocess.run() — dùng fallback (lowercase + strip). "
        "Hãy implement src/nlp/nlp_preprocess.py để đạt chất lượng đầy đủ."
    )
    def _NLP_RUN(text: str) -> str:  # type: ignore[misc]
        """Fallback tối giản: strip + lowercase."""
        return text.strip().lower()


# ===========================================================================
# Exceptions
# ===========================================================================

class ModelNotReadyError(RuntimeError):
    """Raise khi model.pt / scaler.joblib chưa tồn tại."""


# ===========================================================================
# Dataclass Prediction — hợp đồng với B (00_SHARED_CONTRACT.md §1)
# ===========================================================================

@dataclass
class Prediction:
    """Kết quả phân loại văn bản — bàn giao cho B (xem 00_SHARED_CONTRACT.md §1).

    Trường
    ------
    is_fraud : bool
        ``True`` nếu văn bản bị phân loại là lừa đảo.
    confidence_score : float
        Xác suất lừa đảo P(fraud) ∈ [0, 1].  Đây là đầu ra softmax của model
        sau khi áp ngưỡng từ ``config.json``.
    risk_level : str
        ``"low"``     nếu score < 0.4  (cấu hình từ risk_thresholds.yaml)
        ``"medium"``  nếu 0.4 ≤ score < 0.7
        ``"high"``    nếu score ≥ 0.7
    scam_type : str | None
        Loại lừa đảo trong 4 nhóm: ``"authority_impersonation"``,
        ``"bank_fraud"``, ``"prize_scam"``, ``"other_fraud"``.
        ``None`` nếu không phát hiện lừa đảo.

        .. note::
            Trường này được suy ra bằng **luật từ khoá** (không phải model
            phân loại riêng biệt) dựa trên nhóm từ khoá nổi trội trong văn bản.
            Luật: authority → "authority_impersonation";
            financial_action + bank → "bank_fraud";
            reward → "prize_scam"; còn lại → "other_fraud".

    flagged_keywords : list[str]
        Danh sách từ khoá nguy hiểm khớp, từ ``matched_keywords()`` (A1).
    processing_time_ms : float
        Tổng thời gian xử lý tính từ lúc nhận raw text đến khi trả kết quả.
    """

    is_fraud:           bool
    confidence_score:   float
    risk_level:         str
    scam_type:          Optional[str]
    flagged_keywords:   list[str] = field(default_factory=list)
    processing_time_ms: float     = 0.0


# ===========================================================================
# Tiện ích risk_level
# ===========================================================================

_RISK_THRESHOLDS: dict[str, float] | None = None
_RISK_LOCK = threading.Lock()


def _load_risk_thresholds() -> dict[str, float]:
    global _RISK_THRESHOLDS
    if _RISK_THRESHOLDS is None:
        with _RISK_LOCK:
            if _RISK_THRESHOLDS is None:
                if _RISK_CFG.exists():
                    with open(_RISK_CFG, encoding="utf-8") as f:
                        cfg = yaml.safe_load(f)
                    _RISK_THRESHOLDS = {
                        "low_upper":    float(cfg.get("low_upper",    0.4)),
                        "medium_upper": float(cfg.get("medium_upper", 0.7)),
                    }
                else:
                    logger.warning(
                        "Không tìm thấy %s — dùng ngưỡng mặc định: low<0.4≤medium<0.7≤high",
                        _RISK_CFG,
                    )
                    _RISK_THRESHOLDS = {"low_upper": 0.4, "medium_upper": 0.7}
    return _RISK_THRESHOLDS


def score_to_risk_level(score: float) -> str:
    """Chuyển confidence score → risk label theo risk_thresholds.yaml.

    Tham số
    -------
    score : float
        Xác suất lừa đảo ∈ [0, 1].

    Trả về
    ------
    str
        ``"low"``, ``"medium"``, hoặc ``"high"``.
    """
    thr = _load_risk_thresholds()
    if score < thr["low_upper"]:
        return "low"
    if score < thr["medium_upper"]:
        return "medium"
    return "high"


# ===========================================================================
# scam_type — suy luận bằng luật từ khoá (A3 §3)
# ===========================================================================

def _infer_scam_type(clean_text: str, keywords_matched: list[str]) -> Optional[str]:
    """Suy ra loại lừa đảo từ nhóm từ khoá nổi trội.

    .. note::
        Đây là **luật heuristic**, không phải model phân loại riêng biệt.
        Thứ tự ưu tiên: authority > financial_action/bank > reward > other.

    Tham số
    -------
    clean_text : str
        Văn bản đã tiền xử lý (lowercase).
    keywords_matched : list[str]
        Danh sách từ khoá khớp từ ``matched_keywords()``.

    Trả về
    ------
    str | None
        Loại lừa đảo hoặc ``None`` nếu không đủ tín hiệu.
    """
    if not keywords_matched:
        return None

    tl = clean_text.lower()

    # Nhóm authority: công an, cảnh sát, viện kiểm sát, tòa án, cơ quan…
    _AUTHORITY_SIGNALS = [
        "công an", "cảnh sát", "viện kiểm sát", "tòa án", "toa an",
        "cơ quan điều tra", "cơ quan chức năng", "bộ công an", "thanh tra",
        "ủy ban nhân dân", "ubnd", "chính phủ", "bộ trưởng", "thứ trưởng",
        "cán bộ", "nhân viên điều tra",
    ]
    _BANK_SIGNALS = [
        "ngân hàng", "chuyển khoản", "chuyển tiền", "tài khoản",
        "số tài khoản", "otp", "mã otp", "mã xác thực", "internet banking",
        "atm", "thẻ ngân hàng", "cvv", "mã pin", "nhân viên ngân hàng",
        "đại diện ngân hàng", "ngân hàng nhà nước", "bộ tài chính",
    ]
    _REWARD_SIGNALS = [
        "trúng thưởng", "trúng giải", "giải thưởng", "nhận thưởng",
        "quà tặng", "khuyến mãi", "hoàn tiền", "cashback", "may mắn",
        "bốc thăm", "iphone", "xe máy", "du lịch miễn phí", "voucher",
    ]

    def _hit(signals: list[str]) -> bool:
        return any(sig in tl for sig in signals)

    if _hit(_AUTHORITY_SIGNALS):
        return "authority_impersonation"
    if _hit(_BANK_SIGNALS):
        return "bank_fraud"
    if _hit(_REWARD_SIGNALS):
        return "prize_scam"
    # Có từ khoá khớp nhưng không vào nhóm nào ở trên
    return "other_fraud"


# ===========================================================================
# Lazy singleton — Model + Scaler + Tokenizer
# ===========================================================================

class _ModelState:
    """Container thread-safe cho singleton model/scaler/tokenizer."""

    def __init__(self) -> None:
        self._lock        = threading.Lock()
        self._loaded      = False
        self.model:       Optional[HybridTextClassifier] = None
        self.scaler       = None
        self.tokenizer    = None
        self.config:      dict = {}
        self.device:      Optional[torch.device] = None
        self.threshold:   float = 0.5
        self.label2id:    dict = {}
        self.id2label:    dict = {}
        self.numeric_idx: list[int] = []
        self.max_len:     int = 256

    def load(
        self,
        model_path:  Path = _MODEL_PATH,
        scaler_path: Path = _SCALER_PATH,
        config_path: Path = _CONFIG_PATH,
        device:      Optional[torch.device] = None,
    ) -> None:
        """Nạp model, scaler, config — gọi tối đa một lần (thread-safe)."""
        if self._loaded:
            return
        with self._lock:
            if self._loaded:   # double-checked locking
                return
            self._do_load(model_path, scaler_path, config_path, device)
            self._loaded = True

    def _do_load(
        self,
        model_path:  Path,
        scaler_path: Path,
        config_path: Path,
        device:      Optional[torch.device],
    ) -> None:
        for p, label in [(model_path, "model.pt"), (scaler_path, "scaler.joblib"),
                         (config_path, "config.json")]:
            if not p.exists():
                raise ModelNotReadyError(
                    f"Không tìm thấy {label} tại {p}. "
                    "Hãy chạy `python scripts/train_hybrid.py --seed 42` trước."
                )

        # ── Config ──
        with open(config_path, encoding="utf-8") as f:
            cfg = json.load(f)
        self.config     = cfg
        self.threshold  = float(cfg.get("threshold", 0.5))
        self.max_len    = int(cfg.get("max_len", 256))
        phobert_name    = cfg.get("phobert_model", "vinai/phobert-base-v2")
        feature_order   = cfg.get("feature_order", FEATURE_ORDER)
        numeric_cols    = cfg.get("numeric_cols", [
            "n_urgency_kw", "n_authority_kw", "n_financial_action_kw",
            "n_reward_kw", "message_length", "uppercase_ratio", "exclamation_count",
        ])
        self.numeric_idx = [feature_order.index(c) for c in numeric_cols]

        label2id_raw = cfg.get("label2id", {})
        id2label_raw = cfg.get("id2label", {})
        self.label2id = {str(k): int(v) for k, v in label2id_raw.items()}
        self.id2label = {int(k): str(v) for k, v in id2label_raw.items()}

        # ── Device ──
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # ── Scaler ──
        self.scaler = joblib.load(scaler_path)
        logger.info("Đã nạp scaler từ %s", scaler_path)

        # ── Tokenizer + PhoBERT (đóng băng, chỉ dùng để embed) ──
        logger.info("Đang nạp tokenizer: %s", phobert_name)
        self.tokenizer = AutoTokenizer.from_pretrained(phobert_name)

        logger.info("Đang nạp PhoBERT encoder...")
        phobert_enc = AutoModel.from_pretrained(phobert_name).to(self.device)
        phobert_enc.eval()
        for p_ in phobert_enc.parameters():
            p_.requires_grad = False
        self._phobert = phobert_enc

        # ── Classifier ──
        ckpt = torch.load(model_path, map_location=self.device)
        mcfg = ckpt["model_config"]
        self.model = HybridTextClassifier(
            num_classes = mcfg["num_classes"],
            text_dim    = mcfg.get("text_dim",   TEXT_DIM),
            struct_dim  = mcfg.get("struct_dim",  STRUCT_DIM),
            hidden      = mcfg.get("hidden",      128),
            dropout     = mcfg.get("dropout",     0.3),
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        logger.info(
            "✅ Model nạp xong từ %s (device=%s, classes=%d, threshold=%.2f)",
            model_path, self.device, mcfg["num_classes"], self.threshold,
        )

    @torch.no_grad()
    def embed(self, text: str) -> np.ndarray:
        """Trích PhoBERT [CLS] embedding 768-d từ một đoạn văn bản."""
        enc = self.tokenizer(
            [text],
            max_length=self.max_len,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        out = self._phobert(**enc)
        return out.last_hidden_state[:, 0, :].cpu().numpy().astype(np.float32)  # (1, 768)

    def scale_features(self, feat_raw: np.ndarray) -> np.ndarray:
        """Áp StandardScaler chỉ cho các cột số; giữ nguyên cột boolean."""
        feat = feat_raw.copy()
        feat[:, self.numeric_idx] = self.scaler.transform(
            feat_raw[:, self.numeric_idx]
        )
        return feat


# Module-level singleton
_STATE = _ModelState()


def _ensure_loaded(
    model_path:  Optional[Path] = None,
    scaler_path: Optional[Path] = None,
    config_path: Optional[Path] = None,
    device:      Optional[torch.device] = None,
) -> _ModelState:
    """Đảm bảo model đã được nạp; nếu chưa thì nạp ngay."""
    _STATE.load(
        model_path  = model_path  or _MODEL_PATH,
        scaler_path = scaler_path or _SCALER_PATH,
        config_path = config_path or _CONFIG_PATH,
        device      = device,
    )
    return _STATE


# ===========================================================================
# Hàm chính — bàn giao S2
# ===========================================================================

def process_and_predict(
    text: str,
    *,
    device:      Optional[torch.device] = None,
    model_path:  Optional[Path] = None,
    scaler_path: Optional[Path] = None,
    config_path: Optional[Path] = None,
) -> Prediction:
    """Phân loại văn bản thô và trả về ``Prediction``.

    Đây là **hàm duy nhất** mà B được phép import từ ``src.pipeline.predict``
    (xem 00_SHARED_CONTRACT.md §1).

    Pipeline
    --------
    1. Kiểm tra biên (rỗng / khoảng trắng).
    2. ``nlp_preprocess.run(text)`` — tiền xử lý (tokenize, teencode…).
    3. ``extract_text_embedding()`` — PhoBERT [CLS] → 768-d.
    4. ``extract_structured_features()`` + ``features_to_vector()`` → 11-d.
    5. ``scaler.transform()`` (chỉ cột số).
    6. ``HybridTextClassifier.forward(return_probs=True)``.
    7. Áp ``threshold`` từ ``config.json`` → ``is_fraud``.
    8. ``score_to_risk_level()``, ``_infer_scam_type()``, ``matched_keywords()``.

    Tham số
    -------
    text : str
        Văn bản **thô** (SMS hoặc transcript ASR). Hàm tự tiền xử lý bên trong;
        **không** gọi ``nlp_preprocess`` trước khi truyền vào.
    device : torch.device, optional
        Thiết bị cho model (mặc định: tự chọn, ưu tiên CUDA).
    model_path, scaler_path, config_path : Path, optional
        Ghi đè đường dẫn mặc định — dùng chủ yếu trong tests.

    Trả về
    ------
    Prediction
        Kết quả phân loại đầy đủ.

    Raises
    ------
    ModelNotReadyError
        Nếu ``model.pt`` / ``scaler.joblib`` / ``config.json`` chưa tồn tại.

    Ghi chú
    -------
    ``scam_type`` được suy ra bằng **luật từ khoá** (không phải model riêng).
    Hàm này không tự gọi lại ``nlp_preprocess`` lần thứ hai — B không cần
    tiền xử lý trước khi truyền text vào.
    """
    _t0 = time.perf_counter()

    # ── Xử lý biên: chuỗi rỗng / toàn khoảng trắng ──────────────────────
    if not text or not text.strip():
        elapsed_ms = (time.perf_counter() - _t0) * 1000
        return Prediction(
            is_fraud          = False,
            confidence_score  = 0.0,
            risk_level        = "low",
            scam_type         = None,
            flagged_keywords  = [],
            processing_time_ms = elapsed_ms,
        )

    # ── Đảm bảo model đã nạp (lazy singleton) ────────────────────────────
    state = _ensure_loaded(model_path, scaler_path, config_path, device)

    # ── Bước 2: Tiền xử lý NLP ───────────────────────────────────────────
    clean_text: str = _NLP_RUN(text)

    # Xử lý biên: nếu sau khi tiền xử lý vẫn rỗng
    if not clean_text.strip():
        elapsed_ms = (time.perf_counter() - _t0) * 1000
        return Prediction(
            is_fraud          = False,
            confidence_score  = 0.0,
            risk_level        = "low",
            scam_type         = None,
            flagged_keywords  = [],
            processing_time_ms = elapsed_ms,
        )

    # ── Bước 3: PhoBERT embedding (768-d) ─────────────────────────────────
    # Văn bản quá dài sẽ tự động truncate bên trong tokenizer (max_length)
    emb_np = state.embed(clean_text)              # shape (1, 768)

    # ── Bước 4: Structured features (11-d) ───────────────────────────────
    feat_dict = extract_structured_features(clean_text)
    feat_raw  = features_to_vector(feat_dict).reshape(1, -1)  # (1, 11)

    # ── Bước 5: Scale features ────────────────────────────────────────────
    feat_scaled = state.scale_features(feat_raw)  # (1, 11)

    # ── Bước 6: Model inference ───────────────────────────────────────────
    emb_t  = torch.tensor(emb_np,      dtype=torch.float32).to(state.device)
    feat_t = torch.tensor(feat_scaled, dtype=torch.float32).to(state.device)

    with torch.no_grad():
        probs = state.model(emb_t, feat_t, return_probs=True)  # (1, C)
    probs_np = probs.cpu().numpy()[0]  # (C,)

    # ── Bước 7: Quyết định nhãn ───────────────────────────────────────────
    num_classes = len(probs_np)
    if num_classes == 2:
        confidence_score = float(probs_np[1])          # P(fraud)
        is_fraud         = confidence_score >= state.threshold
    else:
        # Đa lớp: lớp 0 = "safe", phần còn lại = các loại fraud
        predicted_class  = int(probs_np.argmax())
        confidence_score = float(probs_np[predicted_class])
        is_fraud         = predicted_class != 0   # giả sử 0 = an toàn

    # ── Bước 8: Metadata ──────────────────────────────────────────────────
    risk_level       = score_to_risk_level(confidence_score if is_fraud else 0.0)
    flagged_kws      = matched_keywords(clean_text)
    scam_type        = _infer_scam_type(clean_text, flagged_kws) if is_fraud else None

    elapsed_ms = (time.perf_counter() - _t0) * 1000

    return Prediction(
        is_fraud          = is_fraud,
        confidence_score  = round(confidence_score, 6),
        risk_level        = risk_level,
        scam_type         = scam_type,
        flagged_keywords  = flagged_kws,
        processing_time_ms = round(elapsed_ms, 3),
    )


# ===========================================================================
# CLI nhanh để kiểm tra nghiệm thu §26
# ===========================================================================

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Kiểm tra nhanh process_and_predict()")
    ap.add_argument("text", nargs="?",
                    default="Công an yêu cầu chuyển khoản ngay")
    args = ap.parse_args()

    print(f"Input: {args.text!r}")
    result = process_and_predict(args.text)
    print(result)

    # Lần 2 — kiểm tra không nạp lại
    _t1 = time.perf_counter()
    result2 = process_and_predict(args.text)
    _t2 = time.perf_counter()
    print(f"\nLần 2 (ms): {(_t2 - _t1) * 1000:.2f}  — model không nạp lại")
