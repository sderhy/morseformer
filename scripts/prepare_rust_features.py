"""Prepare .npy features and a PyTorch reference for the Rust decoder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from morseformer.cli.registry import RECOMMENDED_ACOUSTIC, resolve_model
from morseformer.core.tokenizer import decode
from morseformer.features import FrontendConfig, extract_features
from morseformer.onnx_export import load_rnnt_checkpoint


def _load_audio(path: Path, target_sr: int) -> np.ndarray:
    from scipy.io import wavfile

    sr, audio = wavfile.read(str(path))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if np.issubdtype(audio.dtype, np.integer):
        audio = audio.astype(np.float32) / float(np.iinfo(audio.dtype).max)
    else:
        audio = audio.astype(np.float32)
    if sr != target_sr:
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(sr, target_sr)
        audio = resample_poly(audio, target_sr // g, sr // g).astype(np.float32)
    return audio


def _features_from_args(args: argparse.Namespace) -> np.ndarray:
    if args.zero_frames is not None:
        return np.zeros((args.zero_frames, 1), dtype=np.float32)
    if args.wav is None:
        raise SystemExit("pass either --wav or --zero-frames")
    audio = _load_audio(args.wav, args.sample_rate)
    return extract_features(
        audio,
        args.sample_rate,
        FrontendConfig(
            tone_freq=args.freq,
            bandwidth=args.bandwidth,
            frame_rate=args.frame_rate,
        ),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wav", type=Path)
    parser.add_argument("--zero-frames", type=int)
    parser.add_argument("--model", default=RECOMMENDED_ACOUSTIC)
    parser.add_argument("--ckpt", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference-json", type=Path)
    parser.add_argument("--sample-rate", type=int, default=8000)
    parser.add_argument("--frame-rate", type=int, default=500)
    parser.add_argument("--freq", type=float, default=600.0)
    parser.add_argument("--bandwidth", type=float, default=200.0)
    parser.add_argument("--max-emit-per-frame", type=int, default=5)
    args = parser.parse_args(argv)

    features = _features_from_args(args).astype(np.float32, copy=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, features)

    ckpt = args.ckpt or resolve_model(args.model)
    model = load_rnnt_checkpoint(ckpt, device=torch.device("cpu"))
    x = torch.from_numpy(features).unsqueeze(0)
    lengths = torch.tensor([features.shape[0]], dtype=torch.long)
    with torch.no_grad():
        tokens = model.greedy_rnnt_decode(
            x,
            lengths,
            max_emit_per_frame=args.max_emit_per_frame,
        )[0]
    reference = {
        "features": str(args.out),
        "shape": list(features.shape),
        "tokens": tokens,
        "text": decode(tokens),
    }
    ref_path = args.reference_json or args.out.with_suffix(".json")
    ref_path.write_text(json.dumps(reference, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(reference, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
