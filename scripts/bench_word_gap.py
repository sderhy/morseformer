"""Long-inter-word-silence regression benchmark.

Renders short multi-word phrases at increasing ``word_gap_inflation``
and measures whether the model still emits a SPACE between words.
Targets the v0.5.0 live failure mode: prolonged silence between two
words sometimes did not trigger a SPACE token because no synthetic
training sample had ever shown an inflated word gap.

Usage::

    python -m scripts.bench_word_gap --ckpt checkpoints/phase5_4/last.pt
    python -m scripts.bench_word_gap --ckpt checkpoints/phase5_5/best_rnnt.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from morseformer.core.tokenizer import decode
from morseformer.features import FrontendConfig, extract_features
from morse_synth.core import render
from morse_synth.keying import KeyingConfig
from morse_synth.operator import OperatorConfig
from scripts.decode_audio import (
    _auto_device,
    _is_rnnt_checkpoint,
    _rnnt_cfg_from_state,
)
from morseformer.models.rnnt import RnntModel


# Two short common words, each 3-5 chars: keeps the rendered audio
# under 6 s even at inflation 6× (~2.5 s of inter-word silence).
PAIRS = [
    ("HELLO", "WORLD"), ("CQ", "TEST"), ("GOOD", "DAY"), ("RIG", "NEW"),
    ("WX", "FINE"), ("NAME", "JOHN"), ("FROM", "HERE"), ("BACK", "SOON"),
    ("THE", "QUICK"), ("OVER", "THERE"), ("READY", "GO"), ("CALL", "ME"),
    ("WORK", "DONE"), ("GLAD", "QSO"), ("BEAM", "WEST"), ("FIRST", "TIME"),
    ("TNX", "FER"), ("BEST", "WISH"), ("OPEN", "BAND"), ("BIG", "GUN"),
    ("CHECK", "AGN"), ("HARD", "COPY"), ("LOUD", "ENUF"), ("VERY", "WEAK"),
    ("LONG", "PATH"), ("SHORT", "SKIP"), ("PRESS", "KEY"), ("TURN", "LEFT"),
    ("CLEAR", "NOW"), ("STAND", "BY"),
]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--wpm", type=float, default=25.0)
    p.add_argument("--inflations", default="1.0,2.0,4.0,6.0")
    p.add_argument("--snr-db", type=float, default=20.0)
    p.add_argument("--freq", type=float, default=600.0)
    p.add_argument("--sample-rate", type=int, default=8000)
    p.add_argument("--target-seconds", type=float, default=6.0)
    p.add_argument("--device", default=None)
    p.add_argument("--use-ema", action="store_true", default=True)
    p.add_argument("--no-ema", dest="use_ema", action="store_false")
    p.add_argument("--seed", type=int, default=20260503)
    return p


def _pad_or_truncate(x: np.ndarray, n: int) -> np.ndarray:
    if x.size >= n:
        return x[:n]
    out = np.zeros(n, dtype=x.dtype)
    out[: x.size] = x
    return out


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = torch.device(args.device or _auto_device())
    inflations = [float(x) for x in args.inflations.split(",") if x]

    print(f"[bench_word_gap] loading {args.ckpt}")
    ckpt = torch.load(str(args.ckpt), map_location="cpu", weights_only=False)
    if not _is_rnnt_checkpoint(ckpt):
        raise SystemExit("expected RNN-T checkpoint")
    state = dict(ckpt["model"])
    if args.use_ema and ckpt.get("ema"):
        for k, v in ckpt["ema"].items():
            if k in state:
                state[k] = v
    cfg = _rnnt_cfg_from_state(state)
    model = RnntModel(cfg).to(device).eval()
    model.load_state_dict(state)

    fcfg = FrontendConfig(tone_freq=args.freq, bandwidth=200.0, frame_rate=500)
    keying = KeyingConfig(shape="raised_cosine", rise_ms=5.0)
    target_samples = int(round(args.target_seconds * args.sample_rate))

    print(f"[bench_word_gap] {len(PAIRS)} pairs × {len(inflations)} inflations "
          f"@ {args.wpm} wpm, SNR {args.snr_db} dB")
    print()
    print(f"{'inflation':>10} {'gap_s':>7} {'word_acc':>9} {'space_acc':>10} "
          f"{'fusion':>8} {'CER':>6}")
    print("-" * 60)

    rng = np.random.default_rng(args.seed)
    with torch.no_grad():
        for inflation in inflations:
            n_word_ok = 0
            n_space_ok = 0
            n_fusion = 0
            cer_num = 0
            cer_den = 0
            gap_seconds = (7.0 * inflation) / (1.2 * args.wpm)  # crude print
            for w1, w2 in PAIRS:
                gt = f"{w1} {w2}"
                op = OperatorConfig(
                    wpm=args.wpm,
                    word_gap_inflation=inflation,
                    seed=int(rng.integers(0, 2**31 - 1)),
                )
                clean = render(
                    gt, operator=op, keying=keying, channel=None,
                    freq=args.freq, sample_rate=args.sample_rate,
                ).astype(np.float32)
                clean = _pad_or_truncate(clean, target_samples)
                # Add light AWGN at fixed SNR
                if np.isfinite(args.snr_db):
                    sig_power = float(np.mean(clean ** 2)) + 1e-12
                    noise_power = sig_power / (10.0 ** (args.snr_db / 10.0))
                    noise = rng.standard_normal(clean.size).astype(np.float32) * np.sqrt(noise_power)
                    audio = clean + noise
                else:
                    audio = clean
                feats = extract_features(audio, args.sample_rate, fcfg)
                x = torch.from_numpy(feats).unsqueeze(0).to(device)
                lengths = torch.tensor([feats.shape[0]], dtype=torch.long, device=device)
                tokens = model.greedy_rnnt_decode(x, lengths)[0]
                hyp = decode(tokens).strip()

                # SPACE recall
                if " " in hyp:
                    n_space_ok += 1
                # Word recall (both words present in correct order)
                hyp_words = hyp.split()
                if len(hyp_words) >= 2 and hyp_words[0] == w1 and hyp_words[1] == w2:
                    n_word_ok += 1
                # Fusion: hyp contains the literal "W1W2" without space
                if (w1 + w2) in hyp.replace(" ", ""):
                    if " " not in hyp:
                        n_fusion += 1
                # CER (Levenshtein-free: just char count diff via difflib ratio)
                # cheap: edit distance
                import difflib
                sm = difflib.SequenceMatcher(autojunk=False, a=hyp, b=gt)
                # ops
                ops = sm.get_opcodes()
                err = 0
                for tag, i1, i2, j1, j2 in ops:
                    if tag != "equal":
                        err += max(i2 - i1, j2 - j1)
                cer_num += err
                cer_den += len(gt)

            n = len(PAIRS)
            print(f"{inflation:>10.1f} {gap_seconds:>7.2f} "
                  f"{n_word_ok:>4d}/{n:>3d} "
                  f"{n_space_ok:>4d}/{n:>3d}    "
                  f"{n_fusion:>4d}/{n:>3d} "
                  f"{cer_num/max(1,cer_den):>5.1%}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
