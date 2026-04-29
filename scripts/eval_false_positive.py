"""False-positive bench — measure how many characters a model emits on
noise-only / weak-signal audio that should be ignored.

The 2026-04-25 London↔French live test surfaced the model's worst real-
world failure mode: long stretches of "letter soup" (E I S T sequences)
emitted on band noise or signals well below the receiver's intended
filter. Phase 3.2's training curriculum (3-mode empty samples + 30 %
random text) is designed to push this rate down. This bench is the
metric.

Layout: 3 modes × ``--n-per-mode`` samples each (default 50 → 150 total),
matching the empty-audio sub-modes used in training:

    1. Pure AWGN — quiet band, no signal.
    2. AWGN + QRN bursts — atmospheric clicks.
    3. Distant weak CW (SNR -35 to -25 dB) — a real signal too faint to
       decode. Labelled empty: the model must ignore it.

Each model is scored on:
    * mean characters emitted per sample (target: ≈ 0)
    * median + max
    * fraction of "letter-soup" samples (> 5 chars emitted)

Usage::

    python -m scripts.eval_false_positive \
        --baseline-ckpt   checkpoints/phase3_1/best_rnnt.pt \
        --candidate-ckpt  checkpoints/phase3_2/best_rnnt.pt \
        --n-per-mode 50
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

import torch

from morseformer.core.tokenizer import decode
from morseformer.data.synthetic import DatasetConfig
from morseformer.data.validation import (
    ValidationConfig,
    ValidationSample,
    build_noise_only_validation,
)
from morseformer.models.acoustic import AcousticConfig
from morseformer.models.rnnt import RnntConfig, RnntModel


_MODE_NAMES = ("pure-AWGN", "AWGN+QRN", "distant-weak-CW")


def _auto_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_rnnt(path: Path, device: torch.device) -> RnntModel:
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    enc = cfg["model"]["encoder"]
    ckpt_vocab = cfg["model"].get("vocab_size")
    encoder_cfg = AcousticConfig(
        d_model=enc["d_model"], n_heads=enc["n_heads"], n_layers=enc["n_layers"],
        ff_expansion=enc["ff_expansion"], conv_kernel=enc["conv_kernel"],
        dropout=enc["dropout"],
        **({"vocab_size": ckpt_vocab} if ckpt_vocab is not None else {}),
    )
    rnnt_cfg = RnntConfig(
        encoder=encoder_cfg,
        d_pred=cfg["model"]["d_pred"],
        pred_lstm_layers=cfg["model"]["pred_lstm_layers"],
        d_joint=cfg["model"]["d_joint"],
        dropout=cfg["model"]["dropout"],
        **({"vocab_size": ckpt_vocab} if ckpt_vocab is not None else {}),
    )
    model = RnntModel(rnnt_cfg).to(device)
    state = dict(ckpt["model"])
    ema = ckpt.get("ema")
    if ema:
        for k, v in ema.items():
            if k in state:
                state[k] = v
    model.load_state_dict(state)
    model.eval()
    return model


def _score(
    model: RnntModel,
    samples: list[ValidationSample],
    device: torch.device,
    batch_size: int,
    n_per_mode: int,
    confidence_threshold: float,
) -> list[list[int]]:
    """Return a list (3 modes) of per-sample emission counts."""
    counts: list[list[int]] = [[], [], []]
    for i in range(0, len(samples), batch_size):
        batch_samples = samples[i : i + batch_size]
        features = torch.stack([s.features for s in batch_samples]).to(device)
        lengths = torch.tensor(
            [s.n_frames for s in batch_samples], dtype=torch.long, device=device
        )
        with torch.no_grad():
            hyps = model.greedy_rnnt_decode(
                features, lengths, confidence_threshold=confidence_threshold,
            )
        for j, hyp in enumerate(hyps):
            mode = (i + j) // n_per_mode
            text = decode(hyp)
            counts[mode].append(len(text))
    return counts


def _summary(name: str, per_mode: list[list[int]]) -> str:
    lines = [f"=== {name} ==="]
    lines.append(f"  {'mode':<20} | {'n':>3} | {'mean':>6} | {'median':>6} | {'max':>4} | {'pct>5':>6}")
    lines.append("  " + "-" * 60)
    overall: list[int] = []
    for mode_idx, lengths in enumerate(per_mode):
        if not lengths:
            continue
        mean = statistics.mean(lengths)
        median = statistics.median(lengths)
        mx = max(lengths)
        pct_soup = 100.0 * sum(1 for n in lengths if n > 5) / len(lengths)
        lines.append(
            f"  {_MODE_NAMES[mode_idx]:<20} | {len(lengths):>3d} | "
            f"{mean:>6.2f} | {median:>6.1f} | {mx:>4d} | {pct_soup:>5.1f}%"
        )
        overall.extend(lengths)
    if overall:
        mean = statistics.mean(overall)
        median = statistics.median(overall)
        mx = max(overall)
        pct_soup = 100.0 * sum(1 for n in overall if n > 5) / len(overall)
        lines.append(
            f"  {'overall':<20} | {len(overall):>3d} | "
            f"{mean:>6.2f} | {median:>6.1f} | {mx:>4d} | {pct_soup:>5.1f}%"
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--baseline-ckpt", type=Path, default=None,
                   help="optional baseline checkpoint for side-by-side comparison")
    p.add_argument("--candidate-ckpt", type=Path, required=True,
                   help="checkpoint to evaluate (e.g. checkpoints/phase3_2/best_rnnt.pt)")
    p.add_argument("--n-per-mode", type=int, default=50,
                   help="samples per mode (3 modes total)")
    p.add_argument("--seed", type=int, default=20_260_425)
    p.add_argument("--rx-filter-bw", type=float, default=500.0)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", default=None)
    p.add_argument("--confidence-threshold", type=float, default=0.0,
                   help="optional inference-time RNN-T conf threshold")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = torch.device(args.device or _auto_device())

    val_cfg = ValidationConfig.matching(
        DatasetConfig.phase_3_2(), seed=args.seed,
    )
    samples = build_noise_only_validation(
        cfg=val_cfg, rx_filter_bw=args.rx_filter_bw, n_per_mode=args.n_per_mode,
    )
    print(f"[fp-bench] {len(samples)} samples — "
          f"{args.n_per_mode} per mode × 3 modes")

    candidate = _load_rnnt(args.candidate_ckpt, device)
    print(f"[fp-bench] candidate: {args.candidate_ckpt} "
          f"({sum(p.numel() for p in candidate.parameters()):,} params)")
    cand_per_mode = _score(
        candidate, samples, device, args.batch_size,
        args.n_per_mode, args.confidence_threshold,
    )
    print()
    print(_summary(f"candidate ({args.candidate_ckpt.name})", cand_per_mode))

    if args.baseline_ckpt is not None:
        baseline = _load_rnnt(args.baseline_ckpt, device)
        print(f"\n[fp-bench] baseline: {args.baseline_ckpt}")
        base_per_mode = _score(
            baseline, samples, device, args.batch_size,
            args.n_per_mode, args.confidence_threshold,
        )
        print()
        print(_summary(f"baseline ({args.baseline_ckpt.name})", base_per_mode))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
