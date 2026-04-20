"""Decode a CW audio file with a trained morseformer checkpoint.

Usage::

    python -m scripts.decode_audio <wav_or_npy> \
        --ckpt checkpoints/phase3_0/best_rnnt.pt \
        [--freq 600] [--sample-rate 8000]

Reports both the CTC-greedy and (when the checkpoint is an RNN-T model)
the RNN-T-greedy hypothesis. Defaults match the training config
(8 kHz sample rate, 600 Hz carrier). Input audio is converted to mono
and resampled to ``--sample-rate`` if needed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from morseformer.core.tokenizer import BLANK_INDEX, ctc_greedy_decode, decode
from morseformer.features import FrontendConfig, extract_features
from morseformer.models.acoustic import AcousticConfig, AcousticModel
from morseformer.models.rnnt import RnntConfig, RnntModel


def _load_audio(path: Path, target_sr: int) -> np.ndarray:
    from scipy.io import wavfile

    sr, audio = wavfile.read(str(path))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    # Normalize integer PCM to [-1, 1].
    if np.issubdtype(audio.dtype, np.integer):
        audio = audio / float(np.iinfo(audio.dtype).max)
    max_abs = float(np.max(np.abs(audio)))
    if max_abs > 1.5:  # int that slipped through as float
        audio = audio / max_abs

    if sr != target_sr:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(sr, target_sr)
        up = target_sr // g
        down = sr // g
        audio = resample_poly(audio, up, down).astype(np.float32)
    return audio


def _is_rnnt_checkpoint(ckpt: dict) -> bool:
    # RNN-T checkpoints save best_rnnt_cer + state keys prefixed with "acoustic."
    if "best_rnnt_cer" in ckpt:
        return True
    state = ckpt.get("model", {})
    return any(k.startswith("acoustic.") for k in state)


def _rnnt_cfg_from_state(state: dict) -> RnntConfig:
    """Reconstruct RnntConfig from state-dict shapes (in case the saved
    config is empty / lossy)."""
    d_model = state["acoustic.head.weight"].shape[1]
    d_pred = state["pred.embed.weight"].shape[1]
    d_joint = state["joint.enc_proj.weight"].shape[0]
    # Count Conformer blocks by scanning keys.
    block_ids = set()
    for k in state:
        if k.startswith("acoustic.blocks."):
            block_ids.add(int(k.split(".")[2]))
    n_layers = len(block_ids) if block_ids else 8
    # Heads: inferred from attention weight shape.
    attn_w = state.get("acoustic.blocks.0.self_attn.in_proj_weight")
    n_heads = 4
    if attn_w is not None and attn_w.shape[0] == 3 * d_model:
        n_heads = 4  # default — true value is not recoverable from shape
    return RnntConfig(
        encoder=AcousticConfig(d_model=d_model, n_layers=n_layers, n_heads=n_heads),
        d_pred=d_pred,
        d_joint=d_joint,
    )


def _acoustic_cfg_from_state(state: dict) -> AcousticConfig:
    d_model = state["head.weight"].shape[1]
    block_ids = {int(k.split(".")[1]) for k in state if k.startswith("blocks.")}
    return AcousticConfig(d_model=d_model, n_layers=len(block_ids) or 8, n_heads=4)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("audio", type=Path, help="path to a .wav file")
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--freq", type=float, default=600.0,
                   help="carrier tone frequency (Hz) the front-end assumes")
    p.add_argument("--bandwidth", type=float, default=200.0)
    p.add_argument("--sample-rate", type=int, default=8000,
                   help="audio is resampled to this rate before the front-end")
    p.add_argument("--frame-rate", type=int, default=500)
    p.add_argument("--device", default=None,
                   help="cpu / cuda / mps (default: auto)")
    p.add_argument("--use-ema", action="store_true", default=True,
                   help="use EMA weights when available (default: on)")
    p.add_argument("--no-ema", dest="use_ema", action="store_false")
    return p


def _auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = torch.device(args.device or _auto_device())

    # Load checkpoint.
    ckpt = torch.load(str(args.ckpt), map_location="cpu", weights_only=False)
    is_rnnt = _is_rnnt_checkpoint(ckpt)
    state = dict(ckpt["model"])
    if args.use_ema and "ema" in ckpt and ckpt["ema"]:
        # Overlay EMA weights onto the model state for inference.
        for k, v in ckpt["ema"].items():
            if k in state:
                state[k] = v

    if is_rnnt:
        cfg = _rnnt_cfg_from_state(state)
        model = RnntModel(cfg).to(device).eval()
        model.load_state_dict(state)
    else:
        cfg = _acoustic_cfg_from_state(state)
        model = AcousticModel(cfg).to(device).eval()
        model.load_state_dict(state)

    # Load + featurize audio.
    audio = _load_audio(args.audio, args.sample_rate)
    fcfg = FrontendConfig(
        tone_freq=args.freq,
        bandwidth=args.bandwidth,
        frame_rate=args.frame_rate,
    )
    feats = extract_features(audio, args.sample_rate, fcfg)   # [T, 1]
    x = torch.from_numpy(feats).unsqueeze(0).to(device)        # [1, T, 1]
    lengths = torch.tensor([feats.shape[0]], dtype=torch.long, device=device)

    # Decode.
    duration_s = audio.size / args.sample_rate
    print(f"[decode] audio: {args.audio} ({duration_s:.2f} s "
          f"@ {args.sample_rate} Hz, carrier {args.freq:.0f} Hz)")
    print(f"[decode] model: {args.ckpt} ({'RNN-T' if is_rnnt else 'CTC-only'}, "
          f"{sum(p.numel() for p in model.parameters()):,} params, "
          f"EMA {'on' if args.use_ema and ckpt.get('ema') else 'off'})")

    with torch.no_grad():
        if is_rnnt:
            enc_out, enc_lengths = model.acoustic.encode(x, lengths)
            ctc_logits = model.acoustic.head(enc_out)
            ctc_argmax = ctc_logits.argmax(dim=-1)[0].cpu().tolist()
            ctc_len = int(enc_lengths[0].item())
            ctc_hyp = ctc_greedy_decode(ctc_argmax[:ctc_len])

            rnnt_tokens = model.greedy_rnnt_decode(x, lengths)[0]
            rnnt_hyp = decode(rnnt_tokens)

            print(f"\nCTC  : {ctc_hyp!r}")
            print(f"RNN-T: {rnnt_hyp!r}")
        else:
            log_probs, lengths_out = model(x, lengths)
            argmax = log_probs.argmax(dim=-1)[0].cpu().tolist()
            out_len = int(lengths_out[0].item())
            hyp = ctc_greedy_decode(argmax[:out_len])
            print(f"\nCTC  : {hyp!r}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
