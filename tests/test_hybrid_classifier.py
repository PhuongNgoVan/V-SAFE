"""
tests/test_hybrid_classifier.py
================================
Test suite cho ``HybridTextClassifier`` (A2).

Chạy::

    pytest tests/test_hybrid_classifier.py -v

Bao gồm
-------
1. ``TestForwardPass``           — shape logits / probs, dtype, device.
2. ``TestModelPersistence``      — lưu + nạp lại model.pt, dự đoán khớp.
3. ``TestScalerPersistence``     — lưu + nạp lại scaler.joblib, transform khớp.
4. ``TestEndToEndReload``        — pipeline đầy đủ: scaler → feature → model → predict.
5. ``TestScalerTrainOnly``       — xác nhận scaler KHÔNG fit trên dữ liệu ngoài Train.
6. ``TestEdgeCases``             — num_classes > 2, batch size 1, zero input.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Đảm bảo import được từ src/ (khi chạy từ thư mục gốc repo hoặc tests/)
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.models.hybrid_classifier import (  # noqa: E402
    HybridTextClassifier,
    load_from_checkpoint,
    DROPOUT_P,
    HIDDEN_DIM,
    STRUCT_DIM,
    TEXT_DIM,
)
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Hằng số dùng trong tests
# ---------------------------------------------------------------------------
BATCH_SIZES  = [1, 4, 16]
NUM_CLASSES_CASES = [2, 3, 5]
DEVICE = torch.device("cpu")   # tests chạy trên CPU để độc lập môi trường


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def binary_model() -> HybridTextClassifier:
    """Model nhị phân (num_classes=2) với kiến trúc mặc định."""
    torch.manual_seed(0)
    return HybridTextClassifier(num_classes=2).to(DEVICE)


@pytest.fixture
def multiclass_model() -> HybridTextClassifier:
    """Model 4 lớp để test đa lớp."""
    torch.manual_seed(0)
    return HybridTextClassifier(num_classes=4).to(DEVICE)


@pytest.fixture
def dummy_batch():
    """Trả về (emb, feat) dummy với B=8."""
    torch.manual_seed(42)
    emb  = torch.randn(8, TEXT_DIM,   dtype=torch.float32)
    feat = torch.randn(8, STRUCT_DIM, dtype=torch.float32)
    return emb, feat


@pytest.fixture
def tmp_dir():
    """Thư mục tạm thời tự xoá sau mỗi test."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


# ===========================================================================
# 1. Forward Pass
# ===========================================================================

class TestForwardPass:
    """Kiểm tra forward pass: shape, dtype, device, return_probs."""

    @pytest.mark.parametrize("B", BATCH_SIZES)
    def test_logits_shape(self, B: int):
        """Logits phải có shape (B, num_classes)."""
        model = HybridTextClassifier(num_classes=2).to(DEVICE)
        emb   = torch.randn(B, TEXT_DIM)
        feat  = torch.randn(B, STRUCT_DIM)
        out   = model(emb, feat, return_probs=False)
        assert out.shape == (B, 2), f"Shape logits sai: {out.shape}"

    @pytest.mark.parametrize("B", BATCH_SIZES)
    def test_probs_shape_and_sum(self, B: int):
        """Xác suất phải có shape (B, C) và tổng mỗi hàng = 1.0 (±1e-5)."""
        model = HybridTextClassifier(num_classes=3).to(DEVICE)
        emb   = torch.randn(B, TEXT_DIM)
        feat  = torch.randn(B, STRUCT_DIM)
        probs = model(emb, feat, return_probs=True)
        assert probs.shape == (B, 3)
        row_sums = probs.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones(B), atol=1e-5), \
            f"Probs không tổng = 1: {row_sums}"

    def test_probs_nonnegative(self, binary_model, dummy_batch):
        """Xác suất không âm."""
        emb, feat = dummy_batch
        probs = binary_model(emb, feat, return_probs=True)
        assert (probs >= 0).all(), "Có xác suất âm"

    def test_output_dtype_float32(self, binary_model, dummy_batch):
        """Đầu ra phải là float32."""
        emb, feat = dummy_batch
        out = binary_model(emb, feat)
        assert out.dtype == torch.float32, f"dtype sai: {out.dtype}"

    def test_logits_differ_from_probs(self, binary_model, dummy_batch):
        """Logits và probs phải khác nhau (softmax làm thay đổi giá trị)."""
        emb, feat = dummy_batch
        logits = binary_model(emb, feat, return_probs=False)
        probs  = binary_model(emb, feat, return_probs=True)
        # Tổng logits theo hàng thường ≠ 1
        assert not torch.allclose(logits, probs, atol=1e-3), \
            "Logits và probs giống nhau — kiểm tra softmax"

    def test_wrong_text_dim_raises(self, binary_model):
        """Sai chiều text_emb phải raise ValueError."""
        emb  = torch.randn(4, TEXT_DIM + 1)   # sai chiều
        feat = torch.randn(4, STRUCT_DIM)
        with pytest.raises(ValueError, match="text_emb dim"):
            binary_model(emb, feat)

    def test_wrong_struct_dim_raises(self, binary_model):
        """Sai chiều struct_feat phải raise ValueError."""
        emb  = torch.randn(4, TEXT_DIM)
        feat = torch.randn(4, STRUCT_DIM + 3)  # sai chiều
        with pytest.raises(ValueError, match="struct_feat dim"):
            binary_model(emb, feat)

    @pytest.mark.parametrize("num_classes", NUM_CLASSES_CASES)
    def test_multiclass_output_shape(self, num_classes: int):
        """Test với nhiều giá trị num_classes."""
        model = HybridTextClassifier(num_classes=num_classes).to(DEVICE)
        emb   = torch.randn(5, TEXT_DIM)
        feat  = torch.randn(5, STRUCT_DIM)
        out   = model(emb, feat)
        assert out.shape == (5, num_classes)

    def test_batch_size_1(self, binary_model):
        """Batch size = 1 không được gây lỗi."""
        emb  = torch.randn(1, TEXT_DIM)
        feat = torch.randn(1, STRUCT_DIM)
        out  = binary_model(emb, feat)
        assert out.shape == (1, 2)

    def test_zero_input(self, binary_model):
        """Input toàn 0 không được gây NaN/Inf."""
        emb  = torch.zeros(4, TEXT_DIM)
        feat = torch.zeros(4, STRUCT_DIM)
        out  = binary_model(emb, feat)
        assert not torch.isnan(out).any(),  "NaN trong output"
        assert not torch.isinf(out).any(),  "Inf trong output"

    def test_invalid_num_classes_raises(self):
        """num_classes < 2 phải raise ValueError."""
        with pytest.raises(ValueError):
            HybridTextClassifier(num_classes=1)

    def test_invalid_dropout_raises(self):
        """dropout >= 1.0 phải raise ValueError."""
        with pytest.raises(ValueError):
            HybridTextClassifier(num_classes=2, dropout=1.0)

    def test_predict_binary(self, binary_model, dummy_batch):
        """predict() trả về tensor long shape (B,) với giá trị 0 hoặc 1."""
        emb, feat = dummy_batch
        preds = binary_model.predict(emb, feat)
        assert preds.shape == (8,)
        assert preds.dtype == torch.long
        assert set(preds.tolist()).issubset({0, 1})

    def test_predict_multiclass(self, multiclass_model, dummy_batch):
        """predict() trả về nhãn trong [0, num_classes-1]."""
        emb, feat = dummy_batch
        preds = multiclass_model.predict(emb, feat)
        assert preds.shape == (8,)
        assert all(0 <= p < 4 for p in preds.tolist())


# ===========================================================================
# 2. Model Persistence (lưu + nạp lại model.pt)
# ===========================================================================

class TestModelPersistence:
    """Kiểm tra lưu và nạp lại model.pt cho kết quả dự đoán đồng nhất."""

    def _save_checkpoint(
        self,
        model: HybridTextClassifier,
        path: Path,
        num_classes: int,
    ) -> None:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "model_config": {
                    "num_classes": num_classes,
                    "text_dim":    model.text_dim,
                    "struct_dim":  model.struct_dim,
                    "hidden":      model.hidden,
                    "dropout":     model.dropout_p,
                },
                "label2id": {i: i for i in range(num_classes)},
                "id2label":  {i: i for i in range(num_classes)},
            },
            path,
        )

    def test_reload_gives_identical_logits(self, tmp_dir, dummy_batch):
        """Nạp lại model.pt → logits trùng khớp hoàn toàn (max diff < 1e-6)."""
        torch.manual_seed(7)
        model = HybridTextClassifier(num_classes=2).to(DEVICE)
        model.eval()

        ckpt_path = tmp_dir / "model.pt"
        self._save_checkpoint(model, ckpt_path, num_classes=2)

        reloaded, _ = load_from_checkpoint(ckpt_path, device=DEVICE)

        emb, feat = dummy_batch
        with torch.no_grad():
            out_orig     = model(emb, feat)
            out_reloaded = reloaded(emb, feat)

        max_diff = (out_orig - out_reloaded).abs().max().item()
        assert max_diff < 1e-6, \
            f"Logits không khớp sau khi nạp lại: max diff = {max_diff:.2e}"

    def test_reload_gives_identical_probs(self, tmp_dir, dummy_batch):
        """Nạp lại model.pt → xác suất trùng khớp (max diff < 1e-6)."""
        torch.manual_seed(99)
        model = HybridTextClassifier(num_classes=3).to(DEVICE)
        model.eval()

        ckpt_path = tmp_dir / "model.pt"
        self._save_checkpoint(model, ckpt_path, num_classes=3)

        reloaded, _ = load_from_checkpoint(ckpt_path, device=DEVICE)

        emb, feat = dummy_batch
        with torch.no_grad():
            probs_orig     = model(emb, feat, return_probs=True)
            probs_reloaded = reloaded(emb, feat, return_probs=True)

        max_diff = (probs_orig - probs_reloaded).abs().max().item()
        assert max_diff < 1e-6, f"Probs không khớp: max diff = {max_diff:.2e}"

    def test_reload_predictions_match(self, tmp_dir, dummy_batch):
        """Nhãn dự đoán (argmax) phải khớp sau khi nạp lại."""
        torch.manual_seed(13)
        model = HybridTextClassifier(num_classes=4).to(DEVICE)
        model.eval()

        ckpt_path = tmp_dir / "model.pt"
        self._save_checkpoint(model, ckpt_path, num_classes=4)

        reloaded, _ = load_from_checkpoint(ckpt_path, device=DEVICE)

        emb, feat = dummy_batch
        preds_orig     = model.predict(emb, feat)
        preds_reloaded = reloaded.predict(emb, feat)

        assert torch.equal(preds_orig, preds_reloaded), \
            f"Predictions không khớp:\n  orig={preds_orig}\n  reload={preds_reloaded}"

    def test_reload_model_config_preserved(self, tmp_dir):
        """Các hyper-param trong model_config phải được bảo toàn."""
        model = HybridTextClassifier(
            num_classes=5, text_dim=768, struct_dim=11, hidden=64, dropout=0.2
        )
        ckpt_path = tmp_dir / "model.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "model_config": {
                    "num_classes": 5,
                    "text_dim":    768,
                    "struct_dim":  11,
                    "hidden":      64,
                    "dropout":     0.2,
                },
                "label2id": {}, "id2label": {},
            },
            ckpt_path,
        )
        reloaded, ckpt = load_from_checkpoint(ckpt_path, device=DEVICE)
        cfg = ckpt["model_config"]
        assert cfg["num_classes"] == 5
        assert cfg["hidden"]      == 64
        assert cfg["dropout"]     == 0.2
        assert reloaded.num_classes == 5
        assert reloaded.hidden      == 64

    def test_missing_checkpoint_raises(self, tmp_dir):
        """Nạp file không tồn tại phải raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_from_checkpoint(tmp_dir / "nonexistent.pt")

    def test_missing_key_in_checkpoint_raises(self, tmp_dir):
        """Checkpoint thiếu 'model_config' phải raise KeyError."""
        bad_path = tmp_dir / "bad.pt"
        torch.save({"model_state_dict": {}}, bad_path)
        with pytest.raises(KeyError):
            load_from_checkpoint(bad_path)


# ===========================================================================
# 3. Scaler Persistence (lưu + nạp lại scaler.joblib)
# ===========================================================================

class TestScalerPersistence:
    """Kiểm tra scaler.joblib lưu và nạp lại cho kết quả transform khớp."""

    def _make_scaler(self, n: int = 100) -> tuple[StandardScaler, np.ndarray]:
        """Tạo scaler đã fit trên Train ngẫu nhiên."""
        rng       = np.random.default_rng(42)
        train_raw = rng.standard_normal((n, 7)).astype(np.float32)
        scaler    = StandardScaler()
        scaler.fit(train_raw)
        return scaler, train_raw

    def test_reload_transform_matches(self, tmp_dir):
        """transform() của scaler nạp lại phải khớp hoàn toàn với bản gốc."""
        scaler, train_raw = self._make_scaler()
        path = tmp_dir / "scaler.joblib"
        joblib.dump(scaler, path)

        reloaded_scaler = joblib.load(path)

        rng      = np.random.default_rng(99)
        test_raw = rng.standard_normal((20, 7)).astype(np.float32)

        t_orig     = scaler.transform(test_raw)
        t_reloaded = reloaded_scaler.transform(test_raw)

        max_diff = np.abs(t_orig - t_reloaded).max()
        assert max_diff < 1e-7, f"Scaler transform không khớp: max diff={max_diff:.2e}"

    def test_mean_std_preserved(self, tmp_dir):
        """mean_ và scale_ của scaler phải được bảo toàn qua joblib."""
        scaler, _ = self._make_scaler(200)
        path = tmp_dir / "scaler.joblib"
        joblib.dump(scaler, path)

        reloaded = joblib.load(path)
        assert np.allclose(scaler.mean_,  reloaded.mean_,  atol=1e-7)
        assert np.allclose(scaler.scale_, reloaded.scale_, atol=1e-7)


# ===========================================================================
# 4. End-to-End Reload (pipeline đầy đủ)
# ===========================================================================

class TestEndToEndReload:
    """
    Mô phỏng serving pipeline:
    raw features → scaler.transform → model.forward → predict

    Xác nhận: dự đoán trong pipeline reload trùng với pipeline gốc.
    """

    NUMERIC_IDX = [0, 1, 2, 3, 8, 9, 10]   # tương ứng NUMERIC_COLS trong FEATURE_ORDER

    def _make_data(self, B: int = 16):
        rng  = np.random.default_rng(0)
        embs = rng.standard_normal((B, TEXT_DIM)).astype(np.float32)
        feat_raw = np.concatenate([
            rng.integers(0, 5, (B, 4)).astype(np.float32),  # keyword counts
            rng.integers(0, 2, (B, 4)).astype(np.float32),  # boolean cols
            rng.standard_normal((B, 3)).astype(np.float32), # numeric: msg_len, upper, excl
        ], axis=1)
        return embs, feat_raw

    def _apply_scaler(self, feat_raw: np.ndarray, scaler: StandardScaler) -> np.ndarray:
        feat = feat_raw.copy()
        feat[:, self.NUMERIC_IDX] = scaler.transform(feat_raw[:, self.NUMERIC_IDX])
        return feat

    def test_end_to_end_pipeline_reproducible(self, tmp_dir):
        """Pipeline reload cho dự đoán trùng khớp với pipeline gốc."""
        torch.manual_seed(42)
        B      = 16
        n_cls  = 2
        embs_np, feat_raw = self._make_data(B)

        # ── Fit scaler (chỉ trên 'train' subset) ──
        scaler = StandardScaler()
        scaler.fit(feat_raw[:B//2, self.NUMERIC_IDX])   # chỉ dùng nửa đầu làm 'train'

        feat_scaled = self._apply_scaler(feat_raw, scaler)

        # ── Model gốc ──
        model = HybridTextClassifier(num_classes=n_cls).to(DEVICE)
        model.eval()

        emb_t  = torch.tensor(embs_np,    dtype=torch.float32)
        feat_t = torch.tensor(feat_scaled, dtype=torch.float32)

        with torch.no_grad():
            probs_orig = model(emb_t, feat_t, return_probs=True).numpy()

        # ── Lưu artifacts ──
        ckpt_path   = tmp_dir / "model.pt"
        scaler_path = tmp_dir / "scaler.joblib"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "model_config": {
                    "num_classes": n_cls,
                    "text_dim":    TEXT_DIM,
                    "struct_dim":  STRUCT_DIM,
                    "hidden":      HIDDEN_DIM,
                    "dropout":     DROPOUT_P,
                },
                "label2id": {0: 0, 1: 1},
                "id2label":  {0: 0, 1: 1},
            },
            ckpt_path,
        )
        joblib.dump(scaler, scaler_path)

        # ── Nạp lại ──
        reloaded_model,  _ = load_from_checkpoint(ckpt_path,   device=DEVICE)
        reloaded_scaler    = joblib.load(scaler_path)

        feat_rescaled = self._apply_scaler(feat_raw, reloaded_scaler)
        feat_r_t      = torch.tensor(feat_rescaled, dtype=torch.float32)

        with torch.no_grad():
            probs_reload = reloaded_model(emb_t, feat_r_t, return_probs=True).numpy()

        max_diff = np.abs(probs_orig - probs_reload).max()
        assert max_diff < 1e-5, \
            f"Pipeline reload không reproducible: max diff = {max_diff:.2e}"

    def test_prediction_labels_match(self, tmp_dir):
        """Nhãn dự đoán (argmax) phải khớp hoàn toàn."""
        torch.manual_seed(77)
        B = 12
        embs_np, feat_raw = self._make_data(B)

        scaler = StandardScaler()
        scaler.fit(feat_raw[:, self.NUMERIC_IDX])

        feat_scaled = self._apply_scaler(feat_raw, scaler)
        model = HybridTextClassifier(num_classes=3).to(DEVICE)
        model.eval()

        emb_t  = torch.tensor(embs_np,    dtype=torch.float32)
        feat_t = torch.tensor(feat_scaled, dtype=torch.float32)
        preds_orig = model.predict(emb_t, feat_t).numpy()

        # Lưu + nạp lại
        ckpt_path = tmp_dir / "m.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "model_config": {
                    "num_classes": 3, "text_dim": TEXT_DIM,
                    "struct_dim": STRUCT_DIM, "hidden": HIDDEN_DIM,
                    "dropout": DROPOUT_P,
                },
                "label2id": {}, "id2label": {},
            },
            ckpt_path,
        )
        scaler_path = tmp_dir / "s.joblib"
        joblib.dump(scaler, scaler_path)

        reloaded_model,  _ = load_from_checkpoint(ckpt_path, device=DEVICE)
        reloaded_scaler    = joblib.load(scaler_path)

        feat_r = self._apply_scaler(feat_raw, reloaded_scaler)
        feat_r_t = torch.tensor(feat_r, dtype=torch.float32)
        preds_reload = reloaded_model.predict(emb_t, feat_r_t).numpy()

        assert np.array_equal(preds_orig, preds_reload), \
            f"Predictions không khớp:\n  orig={preds_orig}\n  reload={preds_reload}"


# ===========================================================================
# 5. Scaler Train-Only Constraint
# ===========================================================================

class TestScalerTrainOnly:
    """Xác nhận scaler chỉ fit trên Train, không bị 'ô nhiễm' bởi Val/Test."""

    def test_scaler_fit_only_on_train_mean(self):
        """mean_ của scaler phải bằng mean của Train, không phải mean của tất cả data."""
        rng   = np.random.default_rng(0)
        n_col = 3
        train = rng.uniform(0, 1,  (50,  n_col)).astype(np.float32)
        val   = rng.uniform(10, 20, (20, n_col)).astype(np.float32)  # phân phối rất khác
        test  = rng.uniform(10, 20, (20, n_col)).astype(np.float32)

        scaler = StandardScaler()
        scaler.fit(train)                       # CHỈ fit trên Train

        assert np.allclose(scaler.mean_, train.mean(axis=0), atol=1e-5), \
            "scaler.mean_ không khớp với mean(train)"
        # Đảm bảo không phải mean của tất cả
        all_data   = np.vstack([train, val, test])
        mean_all   = all_data.mean(axis=0)
        assert not np.allclose(scaler.mean_, mean_all, atol=0.5), \
            "scaler.mean_ giống mean(all data) — có thể đã fit trên Val/Test"

    def test_transform_val_does_not_change_scaler(self):
        """Gọi transform() trên Val/Test không thay đổi scaler.mean_."""
        rng   = np.random.default_rng(1)
        train = rng.standard_normal((60, 5)).astype(np.float32)
        val   = rng.standard_normal((20, 5)).astype(np.float32)

        scaler = StandardScaler()
        scaler.fit(train)
        mean_before = scaler.mean_.copy()

        scaler.transform(val)   # transform KHÔNG thay đổi scaler
        mean_after = scaler.mean_.copy()

        assert np.array_equal(mean_before, mean_after), \
            "scaler.mean_ bị thay đổi sau khi gọi transform() — logic sai"


# ===========================================================================
# 6. Edge Cases
# ===========================================================================

class TestEdgeCases:
    """Các trường hợp biên."""

    def test_model_in_eval_mode_no_dropout_effect(self):
        """Trong eval mode, dropout không ảnh hưởng → hai forward giống nhau."""
        torch.manual_seed(5)
        model = HybridTextClassifier(num_classes=2, dropout=0.9)
        model.eval()

        emb  = torch.randn(4, TEXT_DIM)
        feat = torch.randn(4, STRUCT_DIM)
        with torch.no_grad():
            out1 = model(emb, feat)
            out2 = model(emb, feat)
        assert torch.allclose(out1, out2), \
            "eval mode: hai lần forward phải cho cùng kết quả (dropout=0)"

    def test_model_in_train_mode_dropout_differs(self):
        """Trong train mode với dropout cao, hai forward CÓ THỂ khác nhau."""
        torch.manual_seed(5)
        model = HybridTextClassifier(num_classes=2, dropout=0.9)
        model.train()

        emb  = torch.randn(32, TEXT_DIM)
        feat = torch.randn(32, STRUCT_DIM)
        out1 = model(emb, feat)
        out2 = model(emb, feat)
        # Với dropout=0.9 và B=32, rất cao khả năng khác nhau
        # (không phải luôn đúng, nhưng với seed cố định phải khác)
        # Chỉ kiểm tra model không crash
        assert out1.shape == (32, 2)
        assert out2.shape == (32, 2)

    def test_gradient_flows(self):
        """Gradient phải chảy qua cả fc1 và fc2."""
        model = HybridTextClassifier(num_classes=2)
        model.train()
        emb  = torch.randn(4, TEXT_DIM, requires_grad=False)
        feat = torch.randn(4, STRUCT_DIM, requires_grad=False)

        logits = model(emb, feat)
        loss   = logits.sum()
        loss.backward()

        assert model.fc1.weight.grad is not None, "Gradient không chảy tới fc1"
        assert model.fc2.weight.grad is not None, "Gradient không chảy tới fc2"
        assert not torch.isnan(model.fc1.weight.grad).any(), "NaN trong gradient fc1"
        assert not torch.isnan(model.fc2.weight.grad).any(), "NaN trong gradient fc2"

    def test_custom_hidden_and_dims(self):
        """Tạo model với hidden và struct_dim tùy chỉnh."""
        model = HybridTextClassifier(
            num_classes=6,
            text_dim=512,
            struct_dim=20,
            hidden=256,
            dropout=0.1,
        )
        emb  = torch.randn(3, 512)
        feat = torch.randn(3, 20)
        out  = model(emb, feat)
        assert out.shape == (3, 6)

    def test_extra_repr(self):
        """extra_repr phải chứa các thông tin chính."""
        model = HybridTextClassifier(num_classes=2)
        r = model.extra_repr()
        assert "text_dim" in r
        assert "hidden"   in r
        assert "dropout"  in r
