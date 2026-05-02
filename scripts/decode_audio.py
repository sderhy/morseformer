"""Decode a CW audio file with a trained morseformer checkpoint.

Usage::

    python -m scripts.decode_audio <wav_or_npy> \
        --ckpt checkpoints/phase3_0/best_rnnt.pt \
        [--freq 600] [--sample-rate 8000] [--chunk-seconds 6.0]

Reports both the CTC-greedy and (when the checkpoint is an RNN-T model)
the RNN-T-greedy hypothesis. Defaults match the training config
(8 kHz sample rate, 600 Hz carrier). Input audio is converted to mono
and resampled to ``--sample-rate`` if needed.

Long-audio handling
-------------------
The Phase 3.0 model is trained on 6-second clips. On much longer
recordings, the RNN-T greedy decoder (which carries a stateful
prediction-network LSTM and sees a long encoder sequence in a single
pass) often collapses to emitting almost nothing. To decode arbitrary
length audio, ``decode_audio`` splits the input into non-overlapping
``--chunk-seconds`` windows (default 6.0 s, matching training) and
concatenates the per-chunk hypotheses. Set ``--chunk-seconds 0`` to
disable chunking and run the model on the whole clip in one shot
(useful for research / debugging).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from morseformer.core.tokenizer import BLANK_INDEX, ctc_greedy_decode, decode
from morseformer.features import FrontendConfig, extract_features
from morseformer.models.acoustic import AcousticConfig, AcousticModel
from morseformer.models.fusion import FusionConfig, greedy_rnnt_decode_with_lm
from morseformer.models.lm import GptLM, LmConfig
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
    # The output dim of the CTC head is the vocabulary size — read it
    # off the state so that legacy 46-vocab checkpoints (Phase 3.0–3.3)
    # keep loading after the tokenizer was extended to 49 in Phase 3.4.
    vocab_size = state["acoustic.head.weight"].shape[0]
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
        encoder=AcousticConfig(
            d_model=d_model, n_layers=n_layers, n_heads=n_heads,
            vocab_size=vocab_size,
        ),
        d_pred=d_pred,
        d_joint=d_joint,
        vocab_size=vocab_size,
    )


def _acoustic_cfg_from_state(state: dict) -> AcousticConfig:
    d_model = state["head.weight"].shape[1]
    vocab_size = state["head.weight"].shape[0]
    block_ids = {int(k.split(".")[1]) for k in state if k.startswith("blocks.")}
    return AcousticConfig(
        d_model=d_model, n_layers=len(block_ids) or 8, n_heads=4,
        vocab_size=vocab_size,
    )


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
    p.add_argument("--chunk-seconds", type=float, default=6.0,
                   help="chunk window length in seconds. Matches the "
                        "training-clip length; set to 0 to disable "
                        "chunking.")
    p.add_argument("--lm-ckpt", type=Path, default=None,
                   help="optional LM checkpoint (e.g. "
                        "checkpoints/lm_phase5_2/best.pt) for shallow-"
                        "fusion RNN-T decoding. Recommended pairing: "
                        "λ_lm=0.7 with the phase_3_5-mix LM.")
    p.add_argument("--fusion-weight", type=float, default=0.0,
                   help="λ_lm for shallow fusion. 0.0 disables fusion; "
                        "0.7 was tuned on the Alice-real-audio bench "
                        "(2026-05-01) — peak −11.4 %% CER vs greedy.")
    p.add_argument("--confidence-threshold", type=float, default=0.0,
                   help="acoustic-only emission gate. 0.6 mirrors the "
                        "decode_live default (FP-suppression on noise). "
                        "Stacks with --fusion-weight: gating still "
                        "happens on the acoustic head, so the LM cannot "
                        "rescue noise-driven low-confidence emissions.")
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

    # Optional LM for shallow-fusion RNN-T decoding. Only meaningful on
    # RNN-T checkpoints; CTC fusion is implemented in
    # morseformer.models.fusion.greedy_ctc_decode_with_lm but not wired
    # in here yet — `decode_audio` is the offline RNN-T inference entry
    # point.
    lm_model: GptLM | None = None
    if args.lm_ckpt is not None and args.fusion_weight > 0.0:
        if not is_rnnt:
            raise SystemExit(
                "--lm-ckpt only supported for RNN-T checkpoints right now."
            )
        lm_ckpt = torch.load(
            str(args.lm_ckpt), map_location="cpu", weights_only=False
        )
        lm_mcfg = lm_ckpt["config"]["model"]
        lm_cfg = LmConfig(
            vocab_size=lm_mcfg["vocab_size"],
            d_model=lm_mcfg["d_model"], n_heads=lm_mcfg["n_heads"],
            n_layers=lm_mcfg["n_layers"], dropout=lm_mcfg["dropout"],
        )
        lm_model = GptLM(lm_cfg).to(device).eval()
        lm_state = dict(lm_ckpt["model"])
        if args.use_ema and "ema" in lm_ckpt and lm_ckpt["ema"]:
            for k, v in lm_ckpt["ema"].items():
                if k in lm_state:
                    lm_state[k] = v
        lm_model.load_state_dict(lm_state)

    # Load audio and decide on chunk layout.
    audio = _load_audio(args.audio, args.sample_rate)
    duration_s = audio.size / args.sample_rate
    fcfg = FrontendConfig(
        tone_freq=args.freq,
        bandwidth=args.bandwidth,
        frame_rate=args.frame_rate,
    )

    chunk_samples = (
        int(round(args.chunk_seconds * args.sample_rate))
        if args.chunk_seconds > 0
        else 0
    )
    if chunk_samples > 0 and audio.size > chunk_samples:
        # Split into non-overlapping chunks, padding the last one with
        # zeros so every chunk is exactly ``chunk_samples`` long.
        n_chunks = (audio.size + chunk_samples - 1) // chunk_samples
        padded = np.zeros(n_chunks * chunk_samples, dtype=audio.dtype)
        padded[: audio.size] = audio
        chunks = padded.reshape(n_chunks, chunk_samples)
    else:
        chunks = audio[None, :]

    print(f"[decode] audio: {args.audio} ({duration_s:.2f} s "
          f"@ {args.sample_rate} Hz, carrier {args.freq:.0f} Hz)")
    print(f"[decode] model: {args.ckpt} ({'RNN-T' if is_rnnt else 'CTC-only'}, "
          f"{sum(p.numel() for p in model.parameters()):,} params, "
          f"EMA {'on' if args.use_ema and ckpt.get('ema') else 'off'})")
    if lm_model is not None:
        print(f"[decode] lm:    {args.lm_ckpt} "
              f"(λ_lm={args.fusion_weight}, "
              f"thr={args.confidence_threshold})")
    print(f"[decode] chunks: {len(chunks)} × "
          f"{(chunk_samples or audio.size) / args.sample_rate:.2f} s")

    ctc_parts: list[str] = []
    rnnt_parts: list[str] = []

    with torch.no_grad():
        for chunk in chunks:
            feats = extract_features(chunk, args.sample_rate, fcfg)       # [T, 1]
            x = torch.from_numpy(feats).unsqueeze(0).to(device)            # [1, T, 1]
            lengths = torch.tensor(
                [feats.shape[0]], dtype=torch.long, device=device
            )
            if is_rnnt:
                enc_out, enc_lengths = model.acoustic.encode(x, lengths)
                ctc_logits = model.acoustic.head(enc_out)
                ctc_argmax = ctc_logits.argmax(dim=-1)[0].cpu().tolist()
                ctc_len = int(enc_lengths[0].item())
                ctc_parts.append(ctc_greedy_decode(ctc_argmax[:ctc_len]))

                if lm_model is not None:
                    fusion_cfg = FusionConfig(
                        fusion_weight=args.fusion_weight,
                        ilm_weight=0.0,
                        confidence_threshold=args.confidence_threshold,
                    )
                    rnnt_tokens = greedy_rnnt_decode_with_lm(
                        model, lm_model, x, lengths, fusion_cfg
                    )[0]
                else:
                    rnnt_tokens = model.greedy_rnnt_decode(
                        x, lengths,
                        confidence_threshold=args.confidence_threshold,
                    )[0]
                rnnt_parts.append(decode(rnnt_tokens))
            else:
                log_probs, lengths_out = model(x, lengths)
                argmax = log_probs.argmax(dim=-1)[0].cpu().tolist()
                out_len = int(lengths_out[0].item())
                ctc_parts.append(ctc_greedy_decode(argmax[:out_len]))

    # Join adjacent chunks with a single space — the tokenizer also uses
    # ``' '`` for inter-element gap, so this preserves readability even
    # if a word happens to straddle the chunk boundary.
    ctc_hyp = " ".join(p for p in ctc_parts if p)
    if is_rnnt:
        rnnt_hyp = " ".join(p for p in rnnt_parts if p)
        print(f"\nCTC  : {ctc_hyp!r}")
        print(f"RNN-T: {rnnt_hyp!r}")
    else:
        print(f"\nCTC  : {ctc_hyp!r}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
