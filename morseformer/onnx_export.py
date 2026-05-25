"""ONNX export helpers for the RNN-T runtime.

The Rust runtime should drive decoding explicitly, so we export the
model as three small graphs instead of trying to trace the Python greedy
loop:

* encoder: features -> encoder frames + CTC log-probs
* predictor_step: previous token + LSTM state -> prediction output + state
* joint: one encoder frame + prediction output -> vocab logits
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from morseformer.models.acoustic import AcousticConfig
from morseformer.models.rnnt import RnntConfig, RnntModel


class RnntEncoderOnnx(nn.Module):
    """Export wrapper for the acoustic encoder and CTC head."""

    def __init__(self, model: RnntModel) -> None:
        super().__init__()
        self.model = model

    def forward(
        self, features: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        enc_out, enc_lengths = self.model.acoustic.encode(features, lengths)
        if enc_lengths is None:
            enc_lengths = torch.full(
                (features.size(0),),
                enc_out.size(1),
                dtype=torch.long,
                device=features.device,
            )
        ctc_log_probs = torch.log_softmax(self.model.acoustic.head(enc_out), dim=-1)
        return enc_out, enc_lengths, ctc_log_probs


class RnntPredictorStepOnnx(nn.Module):
    """Export wrapper for one prediction-network LSTM step."""

    def __init__(self, model: RnntModel) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        token: torch.Tensor,
        h_in: torch.Tensor,
        c_in: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pred_out, (h_out, c_out) = self.model.pred.step(token, (h_in, c_in))
        return pred_out, h_out, c_out


class RnntJointOnnx(nn.Module):
    """Export wrapper for the RNN-T joint network."""

    def __init__(self, model: RnntModel) -> None:
        super().__init__()
        self.model = model

    def forward(self, enc_frame: torch.Tensor, pred_out: torch.Tensor) -> torch.Tensor:
        return self.model.joint(enc_frame, pred_out)


def load_rnnt_checkpoint(path: Path, device: torch.device | str = "cpu") -> RnntModel:
    """Load a stripped release/dev RNN-T checkpoint."""
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    state = dict(ckpt["model"])
    for key, value in (ckpt.get("ema") or {}).items():
        if key in state:
            state[key] = value

    model_cfg = ckpt["config"]["model"]
    enc = model_cfg["encoder"]
    vocab_size = model_cfg.get("vocab_size")
    extra = {"vocab_size": vocab_size} if vocab_size is not None else {}
    encoder_cfg = AcousticConfig(
        d_model=enc["d_model"],
        n_heads=enc["n_heads"],
        n_layers=enc["n_layers"],
        ff_expansion=enc["ff_expansion"],
        conv_kernel=enc["conv_kernel"],
        dropout=enc["dropout"],
        **extra,
    )
    rnnt_cfg = RnntConfig(
        encoder=encoder_cfg,
        d_pred=model_cfg["d_pred"],
        pred_lstm_layers=model_cfg["pred_lstm_layers"],
        d_joint=model_cfg["d_joint"],
        dropout=model_cfg["dropout"],
        **extra,
    )
    model = RnntModel(rnnt_cfg).to(device).eval()
    model.load_state_dict(state)
    return model


def export_rnnt_onnx(
    model: RnntModel,
    out_dir: Path,
    *,
    sample_frames: int = 3000,
    batch_size: int = 1,
    opset: int = 18,
) -> dict[str, Any]:
    """Export the RNN-T runtime graphs and return the manifest."""
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    cfg = model.cfg
    input_dim = cfg.encoder.input_dim
    d_model = cfg.encoder.d_model
    d_pred = cfg.d_pred
    num_layers = cfg.pred_lstm_layers
    vocab_size = cfg.vocab_size

    features = torch.randn(batch_size, sample_frames, input_dim, dtype=torch.float32)
    lengths = torch.full((batch_size,), sample_frames, dtype=torch.long)
    token = torch.full((batch_size, 1), cfg.blank_index, dtype=torch.long)
    h = torch.zeros(num_layers, batch_size, d_pred, dtype=torch.float32)
    c = torch.zeros(num_layers, batch_size, d_pred, dtype=torch.float32)
    enc_frame = torch.randn(batch_size, 1, d_model, dtype=torch.float32)
    pred_out = torch.randn(batch_size, 1, d_pred, dtype=torch.float32)

    encoder_path = out_dir / "rnnt_encoder.onnx"
    predictor_path = out_dir / "rnnt_predictor_step.onnx"
    joint_path = out_dir / "rnnt_joint.onnx"

    encoder = RnntEncoderOnnx(model).eval()
    predictor = RnntPredictorStepOnnx(model).eval()
    joint = RnntJointOnnx(model).eval()

    torch.onnx.export(
        encoder,
        (features, lengths),
        str(encoder_path),
        input_names=["features", "lengths"],
        output_names=["enc_out", "enc_lengths", "ctc_log_probs"],
        dynamic_axes={
            "features": {0: "batch", 1: "frames"},
            "lengths": {0: "batch"},
            "enc_out": {0: "batch", 1: "enc_frames"},
            "enc_lengths": {0: "batch"},
            "ctc_log_probs": {0: "batch", 1: "enc_frames"},
        },
        opset_version=opset,
    )
    torch.onnx.export(
        predictor,
        (token, h, c),
        str(predictor_path),
        input_names=["token", "h_in", "c_in"],
        output_names=["pred_out", "h_out", "c_out"],
        dynamic_axes={
            "token": {0: "batch"},
            "h_in": {1: "batch"},
            "c_in": {1: "batch"},
            "pred_out": {0: "batch"},
            "h_out": {1: "batch"},
            "c_out": {1: "batch"},
        },
        opset_version=opset,
    )
    torch.onnx.export(
        joint,
        (enc_frame, pred_out),
        str(joint_path),
        input_names=["enc_frame", "pred_out"],
        output_names=["logits"],
        dynamic_axes={
            "enc_frame": {0: "batch"},
            "pred_out": {0: "batch"},
            "logits": {0: "batch"},
        },
        opset_version=opset,
    )

    manifest: dict[str, Any] = {
        "format": "morseformer-rnnt-onnx",
        "opset": opset,
        "graphs": {
            "encoder": encoder_path.name,
            "predictor_step": predictor_path.name,
            "joint": joint_path.name,
        },
        "model": {
            "input_dim": input_dim,
            "d_model": d_model,
            "d_pred": d_pred,
            "pred_lstm_layers": num_layers,
            "d_joint": cfg.d_joint,
            "vocab_size": vocab_size,
            "blank_index": cfg.blank_index,
        },
        "runtime": {
            "sample_frames": sample_frames,
            "subsample": 4,
        },
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest
