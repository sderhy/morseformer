"""Command-line entry point for Phase 4 LM training.

Usage::

    python -m scripts.train_lm \
        --total-steps 20000 --batch-size 128 --peak-lr 3e-4 \
        --checkpoint-dir checkpoints/lm_phase4_0

The LM is tiny (~5 M params) so defaults target a single run that
fits on a 6 GB GPU in minutes to low-hours depending on step budget.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from morseformer.data.lm_dataset import LmDatasetConfig
from morseformer.models.lm import LmConfig
from morseformer.train.lm_loop import LmTrainConfig, train


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # Optimisation
    p.add_argument("--peak-lr", type=float, default=3e-4)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--total-steps", type=int, default=20_000)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip-norm", type=float, default=1.0)
    # Runtime
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    # Model
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-layers", type=int, default=6)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    # Dataset
    p.add_argument("--context-length", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    # Bookkeeping
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--val-batches", type=int, default=50)
    p.add_argument("--checkpoint-dir", type=Path,
                   default=Path("checkpoints/lm_phase4_0"))
    p.add_argument("--resume-from", type=Path, default=None)
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

    model_cfg = LmConfig(
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
    )
    dataset_cfg = LmDatasetConfig(
        context_length=args.context_length,
        seed=args.seed,
    )
    cfg = LmTrainConfig(
        model=model_cfg,
        dataset=dataset_cfg,
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
        save_every=args.save_every,
        val_batches=args.val_batches,
        checkpoint_dir=args.checkpoint_dir,
        jsonl_log=args.checkpoint_dir / "train.jsonl",
        resume_from=args.resume_from,
    )

    print(f"[train_lm] device={device} dtype={args.dtype} "
          f"total_steps={args.total_steps} batch_size={args.batch_size}")
    print(f"[train_lm] d_model={args.d_model} n_layers={args.n_layers} "
          f"n_heads={args.n_heads} context={args.context_length}")
    print(f"[train_lm] checkpoints → {args.checkpoint_dir}")
    result = train(cfg)
    print(f"[train_lm] done. best_val_ppl={result['best_val_ppl']:.3f} "
          f"after {result['steps']} steps")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
