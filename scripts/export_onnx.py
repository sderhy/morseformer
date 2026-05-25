"""Export a morseformer RNN-T checkpoint to ONNX runtime graphs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from morseformer.cli.registry import RECOMMENDED_ACOUSTIC, resolve_model
from morseformer.onnx_export import export_rnnt_onnx, load_rnnt_checkpoint


def _resolve_checkpoint(args: argparse.Namespace) -> Path:
    if args.ckpt is not None:
        return args.ckpt
    return resolve_model(args.model)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export the RNN-T runtime graphs used by the Rust decoder.",
    )
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--ckpt", type=Path, help="path to a local .pt checkpoint")
    src.add_argument(
        "--model",
        default=RECOMMENDED_ACOUSTIC,
        help=f"registry model name to resolve/download (default: {RECOMMENDED_ACOUSTIC})",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("build/onnx") / RECOMMENDED_ACOUSTIC,
        help="directory that will receive rnnt_encoder.onnx, "
             "rnnt_predictor_step.onnx, rnnt_joint.onnx, and manifest.json",
    )
    parser.add_argument("--sample-frames", type=int, default=3000)
    parser.add_argument("--opset", type=int, default=18)
    args = parser.parse_args(argv)

    ckpt = _resolve_checkpoint(args)
    model = load_rnnt_checkpoint(ckpt, device=torch.device("cpu"))
    manifest = export_rnnt_onnx(
        model,
        args.out_dir,
        sample_frames=args.sample_frames,
        opset=args.opset,
    )
    print(json.dumps({"checkpoint": str(ckpt), "out_dir": str(args.out_dir)}, indent=2))
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
