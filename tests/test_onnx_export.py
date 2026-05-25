from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from morseformer.models.acoustic import AcousticConfig  # noqa: E402
from morseformer.models.rnnt import RnntConfig, RnntModel  # noqa: E402
from morseformer.onnx_export import (  # noqa: E402
    RnntEncoderOnnx,
    RnntJointOnnx,
    RnntPredictorStepOnnx,
    export_rnnt_onnx,
)


def _tiny_model() -> RnntModel:
    cfg = RnntConfig(
        encoder=AcousticConfig(
            d_model=32,
            n_heads=4,
            n_layers=1,
            ff_expansion=2,
            conv_kernel=7,
            dropout=0.0,
        ),
        d_pred=32,
        pred_lstm_layers=1,
        d_joint=32,
        dropout=0.0,
    )
    return RnntModel(cfg).eval()


def test_onnx_wrappers_have_runtime_shapes() -> None:
    model = _tiny_model()
    features = torch.randn(1, 200, 1)
    lengths = torch.tensor([200], dtype=torch.long)

    enc_out, enc_lengths, ctc_log_probs = RnntEncoderOnnx(model)(features, lengths)
    assert enc_out.shape == (1, 50, 32)
    assert enc_lengths.tolist() == [50]
    assert ctc_log_probs.shape == (1, 50, model.cfg.vocab_size)

    token = torch.tensor([[model.cfg.blank_index]], dtype=torch.long)
    h = torch.zeros(model.cfg.pred_lstm_layers, 1, model.cfg.d_pred)
    c = torch.zeros(model.cfg.pred_lstm_layers, 1, model.cfg.d_pred)
    pred_out, h_out, c_out = RnntPredictorStepOnnx(model)(token, h, c)
    assert pred_out.shape == (1, 1, model.cfg.d_pred)
    assert h_out.shape == h.shape
    assert c_out.shape == c.shape

    logits = RnntJointOnnx(model)(enc_out[:, :1, :], pred_out)
    assert logits.shape == (1, 1, 1, model.cfg.vocab_size)


def test_export_rnnt_onnx_writes_manifest(tmp_path) -> None:
    pytest.importorskip("onnx")

    manifest = export_rnnt_onnx(_tiny_model(), tmp_path, sample_frames=200)

    assert (tmp_path / "rnnt_encoder.onnx").exists()
    assert (tmp_path / "rnnt_predictor_step.onnx").exists()
    assert (tmp_path / "rnnt_joint.onnx").exists()
    saved = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert saved == manifest
    assert saved["model"]["vocab_size"] == _tiny_model().cfg.vocab_size
