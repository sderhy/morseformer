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

from morseformer.data.synthetic import DatasetConfig
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
    # Curriculum preset — picks the DatasetConfig factory used.
    p.add_argument("--curriculum",
                   choices=("phase2_0", "phase2_1", "phase2_2"),
                   default="phase2_0",
                   help="Dataset preset: clean (phase2_0), moderate noise + "
                        "mild jitter (phase2_1), or moderate noise + wider "
                        "jitter matching the benchmark operator profile "
                        "(phase2_2).")
    # SNR-laddered validation. Empty → clean validation.
    p.add_argument("--validation-snrs", default="",
                   help="Comma-separated SNR list for SNR-ladder validation, "
                        "e.g. '+20,+10,+5,0,-5,-10'. Empty = clean val.")
    p.add_argument("--validation-rx-filter-bw", type=float, default=500.0,
                   help="RX bandpass BW (Hz) applied to validation ladder "
                        "samples. 0 or None disables.")
    # Bookkeeping
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=1_000)
    p.add_argument("--save-every", type=int, default=500,
                   help="Write last.pt every N steps between evals so "
                        "a crash never loses more than this many steps. "
                        "0 disables.")
    p.add_argument("--checkpoint-dir", type=Path,
                   default=Path("checkpoints/phase2_0"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume-from", type=Path, default=None,
                   help="Path to a checkpoint (e.g. checkpoints/.../last.pt) "
                        "to resume training from. Restores model, EMA, "
                        "optimizer, scheduler, step counter, best-CER.")
    return p


def _parse_snrs(spec: str) -> tuple[float, ...]:
    out: list[float] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        out.append(float(token))
    return tuple(out)


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
    if args.curriculum == "phase2_2":
        dataset_cfg = DatasetConfig.phase_2_2(seed=args.seed)
    elif args.curriculum == "phase2_1":
        dataset_cfg = DatasetConfig.phase_2_1(seed=args.seed)
    else:
        dataset_cfg = DatasetConfig.phase_2_0(seed=args.seed)
    validation_snrs = _parse_snrs(args.validation_snrs)
    rx_bw = args.validation_rx_filter_bw if args.validation_rx_filter_bw else None

    cfg = TrainConfig(
        model=model_cfg,
        dataset=dataset_cfg,
        validation_snrs=validation_snrs,
        validation_rx_filter_bw=rx_bw,
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
        checkpoint_dir=args.checkpoint_dir,
        jsonl_log=args.checkpoint_dir / "train.jsonl",
        resume_from=args.resume_from,
    )

    print(f"[train_acoustic] device={device} dtype={args.dtype} "
          f"total_steps={args.total_steps}")
    print(f"[train_acoustic] checkpoints → {args.checkpoint_dir}")
    result = train(cfg)
    print(f"[train_acoustic] done. best_cer={result['best_cer']:.4f} "
          f"after {result['steps']} steps")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
