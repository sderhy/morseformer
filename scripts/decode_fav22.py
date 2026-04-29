"""Sliding-window decode of the FAV22 reference audio.

Decodes a FAV22 wav file (the long, real-CW recording paired with
``data/corpus/fav22_blocks.jsonl``) into per-chunk hypotheses with
absolute timestamps. The output JSONL is the input to
``scripts/align_fav22.py``, which then matches it against the reference
text to recover (audio_start, audio_end, label) training pairs.

The decoder runs the model in non-overlapping chunks of ``--chunk-seconds``
seconds — same semantics as ``scripts/decode_audio.py`` — but writes one
JSONL line per chunk instead of concatenating to stdout. That keeps the
mapping char_idx → time_s exact and survives long-running decodes.

Usage::

    python -m scripts.decode_fav22 \
        --audio /home/serge/Bureau/wav/fav22-lent.wav \
        --ckpt checkpoints/phase3_6/best_rnnt.pt \
        --output data/corpus/fav22_lent_decoded.jsonl

Each output line has the schema::

    {
      "chunk_idx": 7,
      "start_s": 42.0,
      "end_s": 48.0,
      "rnnt_hyp": "QUAND LE PAQUEBOT GÉANT...",
      "ctc_hyp":  "QUAND LE PAQUEBOT GÉANT...",
    }
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from morseformer.core.tokenizer import ctc_greedy_decode, decode
from morseformer.features import FrontendConfig, extract_features
from morseformer.models.acoustic import AcousticConfig
from morseformer.models.rnnt import RnntConfig, RnntModel
from scripts.decode_audio import (
    _acoustic_cfg_from_state,
    _auto_device,
    _is_rnnt_checkpoint,
    _load_audio,
    _rnnt_cfg_from_state,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--audio", type=Path, required=True)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--freq", type=float, default=600.0)
    p.add_argument("--bandwidth", type=float, default=200.0)
    p.add_argument("--sample-rate", type=int, default=8000)
    p.add_argument("--frame-rate", type=int, default=500)
    p.add_argument("--chunk-seconds", type=float, default=6.0)
    p.add_argument("--device", default=None)
    p.add_argument("--use-ema", action="store_true", default=True)
    p.add_argument("--no-ema", dest="use_ema", action="store_false")
    p.add_argument("--max-seconds", type=float, default=0.0,
                   help="If > 0, stop after decoding this many seconds. "
                        "Useful for quick smoke-tests.")
    p.add_argument("--progress-every", type=int, default=50,
                   help="Print a progress line every N chunks.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = torch.device(args.device or _auto_device())

    print(f"[decode_fav22] loading {args.ckpt}")
    ckpt = torch.load(str(args.ckpt), map_location="cpu", weights_only=False)
    is_rnnt = _is_rnnt_checkpoint(ckpt)
    state = dict(ckpt["model"])
    if args.use_ema and ckpt.get("ema"):
        for k, v in ckpt["ema"].items():
            if k in state:
                state[k] = v
    if not is_rnnt:
        raise RuntimeError(
            "decode_fav22 expects an RNN-T checkpoint; got CTC-only model."
        )
    cfg = _rnnt_cfg_from_state(state)
    model = RnntModel(cfg).to(device).eval()
    model.load_state_dict(state)
    print(f"[decode_fav22] model loaded "
          f"({sum(p.numel() for p in model.parameters()):,} params, "
          f"vocab={cfg.vocab_size}, EMA={'on' if args.use_ema and ckpt.get('ema') else 'off'})")

    print(f"[decode_fav22] loading audio {args.audio}")
    t0 = time.time()
    audio = _load_audio(args.audio, args.sample_rate)
    duration_s = audio.size / args.sample_rate
    if args.max_seconds > 0:
        duration_s = min(duration_s, args.max_seconds)
        audio = audio[: int(duration_s * args.sample_rate)]
    print(f"[decode_fav22] audio: {duration_s:.1f} s = {duration_s/60:.2f} min "
          f"(loaded in {time.time()-t0:.1f} s)")

    chunk_samples = int(round(args.chunk_seconds * args.sample_rate))
    n_chunks = (audio.size + chunk_samples - 1) // chunk_samples
    padded = np.zeros(n_chunks * chunk_samples, dtype=audio.dtype)
    padded[: audio.size] = audio
    chunks = padded.reshape(n_chunks, chunk_samples)
    print(f"[decode_fav22] decoding {n_chunks} chunks of "
          f"{args.chunk_seconds:.1f} s — eta ~{n_chunks * 0.05/60:.1f} min on GPU")

    fcfg = FrontendConfig(
        tone_freq=args.freq,
        bandwidth=args.bandwidth,
        frame_rate=args.frame_rate,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    with args.output.open("w", encoding="utf-8") as f, torch.no_grad():
        for i, chunk in enumerate(chunks):
            feats = extract_features(chunk, args.sample_rate, fcfg)
            x = torch.from_numpy(feats).unsqueeze(0).to(device)
            lengths = torch.tensor([feats.shape[0]], dtype=torch.long, device=device)

            enc_out, enc_lengths = model.acoustic.encode(x, lengths)
            ctc_logits = model.acoustic.head(enc_out)
            ctc_argmax = ctc_logits.argmax(dim=-1)[0].cpu().tolist()
            ctc_len = int(enc_lengths[0].item())
            ctc_hyp = ctc_greedy_decode(ctc_argmax[:ctc_len])

            rnnt_tokens = model.greedy_rnnt_decode(x, lengths)[0]
            rnnt_hyp = decode(rnnt_tokens)

            start_s = i * args.chunk_seconds
            end_s = start_s + args.chunk_seconds
            f.write(json.dumps({
                "chunk_idx": i,
                "start_s": start_s,
                "end_s": end_s,
                "rnnt_hyp": rnnt_hyp,
                "ctc_hyp": ctc_hyp,
            }, ensure_ascii=False) + "\n")
            if (i + 1) % args.progress_every == 0:
                elapsed = time.time() - t_start
                rate = (i + 1) / elapsed
                eta = (n_chunks - i - 1) / rate
                print(f"[decode_fav22] chunk {i+1:>5d}/{n_chunks} "
                      f"({100*(i+1)/n_chunks:5.1f}%) "
                      f"rate={rate:.1f} chunks/s eta={eta/60:.1f} min")

    elapsed = time.time() - t_start
    print(f"[decode_fav22] done. {n_chunks} chunks in {elapsed/60:.1f} min "
          f"({n_chunks/elapsed:.1f} chunks/s) → {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
