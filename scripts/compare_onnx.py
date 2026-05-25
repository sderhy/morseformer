"""Compare exported ONNX RNN-T graphs against the PyTorch checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from morseformer.cli.registry import RECOMMENDED_ACOUSTIC, resolve_model
from morseformer.onnx_export import (
    RnntEncoderOnnx,
    RnntJointOnnx,
    RnntPredictorStepOnnx,
    load_rnnt_checkpoint,
)


def _max_abs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)))


def _assert_close(name: str, got: np.ndarray, expected: np.ndarray, atol: float) -> float:
    diff = _max_abs(got, expected)
    if diff > atol:
        raise SystemExit(f"{name}: max_abs_diff={diff:.6g} exceeds atol={atol}")
    return diff


def _session(path: Path) -> ort.InferenceSession:
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def _resolve_checkpoint(args: argparse.Namespace) -> Path:
    if args.ckpt is not None:
        return args.ckpt
    return resolve_model(args.model)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate ONNX runtime outputs against PyTorch wrappers.",
    )
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--ckpt", type=Path, help="path to a local .pt checkpoint")
    src.add_argument("--model", default=RECOMMENDED_ACOUSTIC)
    parser.add_argument(
        "--onnx-dir",
        type=Path,
        default=Path("build/onnx") / RECOMMENDED_ACOUSTIC,
    )
    parser.add_argument("--frames", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--atol", type=float, default=2e-4)
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    ckpt = _resolve_checkpoint(args)
    model = load_rnnt_checkpoint(ckpt, device=torch.device("cpu"))

    features_np = rng.standard_normal((1, args.frames, model.cfg.encoder.input_dim)).astype(
        np.float32
    )
    lengths_np = np.array([args.frames], dtype=np.int64)
    token_np = np.array([[model.cfg.blank_index]], dtype=np.int64)
    h_np = np.zeros((model.cfg.pred_lstm_layers, 1, model.cfg.d_pred), dtype=np.float32)
    c_np = np.zeros((model.cfg.pred_lstm_layers, 1, model.cfg.d_pred), dtype=np.float32)

    with torch.no_grad():
        features = torch.from_numpy(features_np)
        lengths = torch.from_numpy(lengths_np)
        enc_pt, enc_lengths_pt, ctc_pt = RnntEncoderOnnx(model)(features, lengths)
        pred_pt, h_pt, c_pt = RnntPredictorStepOnnx(model)(
            torch.from_numpy(token_np),
            torch.from_numpy(h_np),
            torch.from_numpy(c_np),
        )
        logits_pt = RnntJointOnnx(model)(enc_pt[:, :1, :], pred_pt)

    encoder = _session(args.onnx_dir / "rnnt_encoder.onnx")
    predictor = _session(args.onnx_dir / "rnnt_predictor_step.onnx")
    joint = _session(args.onnx_dir / "rnnt_joint.onnx")

    enc_out, enc_lengths, ctc_log_probs = encoder.run(
        None, {"features": features_np, "lengths": lengths_np}
    )
    pred_out, h_out, c_out = predictor.run(
        None, {"token": token_np, "h_in": h_np, "c_in": c_np}
    )
    logits = joint.run(
        None,
        {"enc_frame": enc_out[:, :1, :].astype(np.float32), "pred_out": pred_out},
    )[0]

    diffs = {
        "encoder.enc_out": _assert_close("encoder.enc_out", enc_out, enc_pt.numpy(), args.atol),
        "encoder.enc_lengths": _assert_close(
            "encoder.enc_lengths", enc_lengths, enc_lengths_pt.numpy(), 0.0
        ),
        "encoder.ctc_log_probs": _assert_close(
            "encoder.ctc_log_probs", ctc_log_probs, ctc_pt.numpy(), args.atol
        ),
        "predictor.pred_out": _assert_close(
            "predictor.pred_out", pred_out, pred_pt.numpy(), args.atol
        ),
        "predictor.h_out": _assert_close("predictor.h_out", h_out, h_pt.numpy(), args.atol),
        "predictor.c_out": _assert_close("predictor.c_out", c_out, c_pt.numpy(), args.atol),
        "joint.logits": _assert_close("joint.logits", logits, logits_pt.numpy(), args.atol),
    }
    for name, diff in diffs.items():
        print(f"{name}: max_abs_diff={diff:.6g}")
    print("ONNX comparison passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
