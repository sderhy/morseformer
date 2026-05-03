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
from morseformer.data.validation import ValidationConfig
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
                   choices=("phase2_0", "phase2_1", "phase2_2",
                            "phase3_1", "phase3_2", "phase3_3",
                            "phase3_4", "phase3_5", "phase3_6",
                            "phase4_0_a", "phase4_0_b", "phase4_0_c",
                            "phase5_3"),
                   default="phase2_1",
                   help="Dataset preset. phase2_1 = Phase 3.0 clean "
                        "ablation. phase3_1 = realistic HF channel. "
                        "phase3_2 = phase3_1 channel + 30 %% random "
                        "sequences + 20 %% 3-mode empty samples "
                        "(anti-hallucination curriculum). phase3_3 = "
                        "phase3_2 channel + 12 %% multilingual prose "
                        "(FR/DE/ES/EN, normalised to ASCII) to fight "
                        "the English-prior bias seen on real French QSOs. "
                        "phase3_4 = phase3_3 channel + 24 %% prose "
                        "(8 %% multilingual + 16 %% FR-only) to train "
                        "the new É / À / apostrophe tokens added in "
                        "the 49-vocab tokenizer. "
                        "phase3_5 = phase3_4 mix with widened operator "
                        "jitter (0-0.15 element, 0-0.25 gap) to fix "
                        "the morning-keying É / À false positives "
                        "observed in the post-Phase-3.4 live test. "
                        "phase3_6 = phase3_5 + 6 %% adversarial-FR "
                        "(WA/WI/WO/QU + vowel patterns from FR prose + "
                        "FAV22-clair) + 10 %% post-emission-silence "
                        "samples to close the residual É / À false "
                        "positives observed at the end of the Phase 3.5 "
                        "live evaluation. "
                        "phase4_0_a/b/c = Phase 4.0 architectural pivot "
                        "to pure char-level acoustics (100 %% random "
                        "chars + accent boost, quiet-zone padded). "
                        "4.0a clean, 4.0b + jitter, 4.0c + full HF "
                        "channel; chain via --pretrained-rnnt from the "
                        "previous best, bootstrap from "
                        "checkpoints/phase3_5/best_rnnt.pt for 4.0a.")
    # SNR-laddered validation
    p.add_argument("--validation-snrs", default="",
                   help="Comma-separated SNR list for SNR-ladder validation. "
                        "Empty = clean val.")
    p.add_argument("--validation-rx-filter-bw", type=float, default=500.0)
    p.add_argument("--validation-wpm-bins", default="",
                   help="Comma-separated WPM bins for the validation grid "
                        "(e.g. '18,22,26,30,32'). Empty = curriculum-aware "
                        "default: phase4_0_* uses (18, 22, 26, 30, 32) to "
                        "match the [18, 32] training range; everything else "
                        "uses (16, 20, 22, 25, 28) to match the legacy "
                        "[16, 28] range.")
    # Bookkeeping
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=1_000)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--checkpoint-dir", type=Path,
                   default=Path("checkpoints/phase3_0"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume-from", type=Path, default=None)
    # Real-audio mix (Phase 3.7+)
    p.add_argument("--real-audio-jsonl", type=Path, default=None,
                   help="Path to an aligned real-audio JSONL "
                        "(scripts/align_ebook_cw.py output). When set, "
                        "the training stream is a mix of synthetic and "
                        "real-audio samples controlled by "
                        "--real-audio-probability.")
    p.add_argument("--real-audio-probability", type=float, default=0.20,
                   help="Fraction of batch items drawn from the real-audio "
                        "source. Ignored when --real-audio-jsonl is unset.")
    p.add_argument("--real-audio-score-threshold", type=float, default=0.7,
                   help="Drop aligned chunks below this difflib score "
                        "(alignment confidence). 0.7 keeps ≈98 %% of the "
                        "Alice/ToL ebook2cw dataset.")
    return p


def _parse_snrs(spec: str) -> tuple[float, ...]:
    out: list[float] = []
    for token in spec.split(","):
        token = token.strip()
        if token:
            out.append(float(token))
    return tuple(out)


def _default_wpm_bins(curriculum: str) -> tuple[float, ...]:
    if curriculum.startswith("phase4_0"):
        return (18.0, 22.0, 26.0, 30.0, 32.0)
    return (16.0, 20.0, 22.0, 25.0, 28.0)


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

    if args.curriculum == "phase5_3":
        dataset_cfg = DatasetConfig.phase_5_3(seed=args.seed)
    elif args.curriculum == "phase4_0_c":
        dataset_cfg = DatasetConfig.phase_4_0_c(seed=args.seed)
    elif args.curriculum == "phase4_0_b":
        dataset_cfg = DatasetConfig.phase_4_0_b(seed=args.seed)
    elif args.curriculum == "phase4_0_a":
        dataset_cfg = DatasetConfig.phase_4_0_a(seed=args.seed)
    elif args.curriculum == "phase3_6":
        dataset_cfg = DatasetConfig.phase_3_6(seed=args.seed)
    elif args.curriculum == "phase3_5":
        dataset_cfg = DatasetConfig.phase_3_5(seed=args.seed)
    elif args.curriculum == "phase3_4":
        dataset_cfg = DatasetConfig.phase_3_4(seed=args.seed)
    elif args.curriculum == "phase3_3":
        dataset_cfg = DatasetConfig.phase_3_3(seed=args.seed)
    elif args.curriculum == "phase3_2":
        dataset_cfg = DatasetConfig.phase_3_2(seed=args.seed)
    elif args.curriculum == "phase3_1":
        dataset_cfg = DatasetConfig.phase_3_1(seed=args.seed)
    elif args.curriculum == "phase2_2":
        dataset_cfg = DatasetConfig.phase_2_2(seed=args.seed)
    elif args.curriculum == "phase2_1":
        dataset_cfg = DatasetConfig.phase_2_1(seed=args.seed)
    else:
        dataset_cfg = DatasetConfig.phase_2_0(seed=args.seed)

    validation_snrs = _parse_snrs(args.validation_snrs)
    rx_bw = args.validation_rx_filter_bw if args.validation_rx_filter_bw else None

    # Build a curriculum-matched validation set so the val pipeline
    # uses the same text mix, target_duration, quiet zones and frontend
    # as the training stream. Without this, the legacy default
    # ValidationConfig (DEFAULT_MIX, 6 s window, no quiet zones) would
    # silently mismatch on Phase 4.0 — the model would be ranked on
    # Q-codes / callsigns while training on random_phase4 sequences.
    if args.validation_wpm_bins.strip():
        wpm_bins = tuple(
            float(x.strip()) for x in args.validation_wpm_bins.split(",")
            if x.strip()
        )
    else:
        wpm_bins = _default_wpm_bins(args.curriculum)
    validation_cfg = ValidationConfig.matching(
        dataset_cfg, n_per_wpm=40, wpm_bins=wpm_bins,
    )

    cfg = RnntTrainConfig(
        model=model_cfg,
        dataset=dataset_cfg,
        validation=validation_cfg,
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
        real_audio_jsonl=args.real_audio_jsonl,
        real_audio_probability=args.real_audio_probability,
        real_audio_score_threshold=args.real_audio_score_threshold,
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
