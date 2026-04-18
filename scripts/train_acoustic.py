"""Command-line entry point for Phase 2.0 acoustic-model training.

Usage::

    python -m scripts.train_acoustic \
        --total-steps 100000 --batch-size 32 --peak-lr 3e-4 \
        --checkpoint-dir checkpoints/phase2_0

All flags have sensible defaults matching ``TrainConfig``. Only the
most commonly overridden ones are exposed on the CLI; the full config
can still be edited in ``morseformer/train/acoustic.py`` for
reproducibility-critical runs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from morseformer.models.acoustic import AcousticConfig
from morseformer.train.acoustic import TrainConfig, train


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # Optimisation
    p.add_argument("--peak-lr", type=float, default=3e-4)
    p.add_argument("--warmup-steps", type=int, default=2_000)
    p.add_argument("--total-steps", type=int, default=100_000)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip-norm", type=float, default=1.0)
    # Runtime
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default=None,
                   help="cpu / cuda / mps (default: auto)")
    p.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    # Model width (keep n_layers/d_model on CLI for quick sweeps)
    p.add_argument("--d-model", type=int, default=144)
    p.add_argument("--n-layers", type=int, default=8)
    p.add_argument("--n-heads", type=int, default=4)
    # Bookkeeping
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=1_000)
    p.add_argument("--checkpoint-dir", type=Path,
                   default=Path("checkpoints/phase2_0"))
    p.add_argument("--seed", type=int, default=0)
    return p


def _auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = args.device or _auto_device()

    model_cfg = AcousticConfig(
        d_model=args.d_model, n_layers=args.n_layers, n_heads=args.n_heads
    )
    cfg = TrainConfig(
        model=model_cfg,
        peak_lr=args.peak_lr,
        warmup_steps=args.warmup_steps,
        total_steps=args.total_steps,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        dtype=args.dtype,
        log_every=args.log_every,
        eval_every=args.eval_every,
        checkpoint_dir=args.checkpoint_dir,
        jsonl_log=args.checkpoint_dir / "train.jsonl",
    )
    cfg.dataset.seed = args.seed

    print(f"[train_acoustic] device={device} dtype={args.dtype} "
          f"total_steps={args.total_steps}")
    print(f"[train_acoustic] checkpoints → {args.checkpoint_dir}")
    result = train(cfg)
    print(f"[train_acoustic] done. best_cer={result['best_cer']:.4f} "
          f"after {result['steps']} steps")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
