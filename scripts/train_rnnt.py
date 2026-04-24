"""Command-line entry point for Phase 3 RNN-T training.

Usage::

    python -m scripts.train_rnnt \
        --pretrained-encoder checkpoints/phase2_1/best_cer.pt \
        --total-steps 80000 --batch-size 12 --peak-lr 2e-4 \
        --curriculum phase2_1 \
        --checkpoint-dir checkpoints/phase3_0

All flags have sensible defaults matching ``RnntTrainConfig``. The
encoder width / depth flags (``--d-model``, ``--n-layers``, ``--n-heads``)
must match the bootstrap checkpoint's encoder when ``--pretrained-encoder``
is set, otherwise the state-dict load will fail.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from morseformer.data.synthetic import DatasetConfig
from morseformer.models.acoustic import AcousticConfig
from morseformer.models.rnnt import RnntConfig
from morseformer.train.rnnt_loop import RnntTrainConfig, train


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # Optimisation
    p.add_argument("--peak-lr", type=float, default=2e-4)
    p.add_argument("--warmup-steps", type=int, default=2_000)
    p.add_argument("--total-steps", type=int, default=80_000)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip-norm", type=float, default=1.0)
    # Multi-task weights
    p.add_argument("--ctc-weight", type=float, default=0.3)
    p.add_argument("--rnnt-weight", type=float, default=0.7)
    # Runtime
    p.add_argument("--batch-size", type=int, default=12)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default=None,
                   help="cpu / cuda / mps (default: auto)")
    p.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    # Encoder width (must match pretrained checkpoint if bootstrapping)
    p.add_argument("--d-model", type=int, default=144)
    p.add_argument("--n-layers", type=int, default=8)
    p.add_argument("--n-heads", type=int, default=4)
    # RNN-T head sizes
    p.add_argument("--d-pred", type=int, default=128)
    p.add_argument("--pred-lstm-layers", type=int, default=1)
    p.add_argument("--d-joint", type=int, default=256)
    # Bootstrap
    p.add_argument("--pretrained-encoder", type=Path, default=None,
                   help="Path to a Phase 2 checkpoint (e.g. "
                        "checkpoints/phase2_1/best_cer.pt). EMA weights "
                        "are applied when available.")
    p.add_argument("--pretrained-rnnt", type=Path, default=None,
                   help="Path to a Phase 3 RnntModel checkpoint (e.g. "
                        "checkpoints/phase3_0/best_rnnt.pt). Loaded with "
                        "strict=False, so a deeper encoder inherits the "
                        "first N layers and re-inits the rest.")
    # Curriculum
    p.add_argument("--curriculum",
                   choices=("phase2_0", "phase2_1", "phase2_2", "phase3_1"),
                   default="phase2_1",
                   help="Dataset preset. phase2_1 = Phase 3.0 clean "
                        "ablation. phase3_1 = realistic HF channel "
                        "(carrier jitter, QSB, QRN, QRM, empty samples).")
    # SNR-laddered validation
    p.add_argument("--validation-snrs", default="",
                   help="Comma-separated SNR list for SNR-ladder validation. "
                        "Empty = clean val.")
    p.add_argument("--validation-rx-filter-bw", type=float, default=500.0)
    # Bookkeeping
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=1_000)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--checkpoint-dir", type=Path,
                   default=Path("checkpoints/phase3_0"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume-from", type=Path, default=None)
    return p


def _parse_snrs(spec: str) -> tuple[float, ...]:
    out: list[float] = []
    for token in spec.split(","):
        token = token.strip()
        if token:
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

    encoder_cfg = AcousticConfig(
        d_model=args.d_model, n_layers=args.n_layers, n_heads=args.n_heads
    )
    model_cfg = RnntConfig(
        encoder=encoder_cfg,
        d_pred=args.d_pred,
        pred_lstm_layers=args.pred_lstm_layers,
        d_joint=args.d_joint,
    )

    if args.curriculum == "phase3_1":
        dataset_cfg = DatasetConfig.phase_3_1(seed=args.seed)
    elif args.curriculum == "phase2_2":
        dataset_cfg = DatasetConfig.phase_2_2(seed=args.seed)
    elif args.curriculum == "phase2_1":
        dataset_cfg = DatasetConfig.phase_2_1(seed=args.seed)
    else:
        dataset_cfg = DatasetConfig.phase_2_0(seed=args.seed)

    validation_snrs = _parse_snrs(args.validation_snrs)
    rx_bw = args.validation_rx_filter_bw if args.validation_rx_filter_bw else None

    cfg = RnntTrainConfig(
        model=model_cfg,
        dataset=dataset_cfg,
        validation_snrs=validation_snrs,
        validation_rx_filter_bw=rx_bw,
        ctc_weight=args.ctc_weight,
        rnnt_weight=args.rnnt_weight,
        pretrained_encoder=args.pretrained_encoder,
        pretrained_rnnt=args.pretrained_rnnt,
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

    print(f"[train_rnnt] device={device} dtype={args.dtype} "
          f"total_steps={args.total_steps}")
    print(f"[train_rnnt] ctc_weight={args.ctc_weight} "
          f"rnnt_weight={args.rnnt_weight}")
    if args.pretrained_encoder is not None:
        print(f"[train_rnnt] bootstrap encoder from {args.pretrained_encoder}")
    if args.pretrained_rnnt is not None:
        print(f"[train_rnnt] bootstrap full RNN-T from {args.pretrained_rnnt}")
    print(f"[train_rnnt] checkpoints → {args.checkpoint_dir}")
    result = train(cfg)
    print(f"[train_rnnt] done. best_ctc_cer={result['best_ctc_cer']:.4f} "
          f"best_rnnt_cer={result['best_rnnt_cer']:.4f} "
          f"after {result['steps']} steps")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
