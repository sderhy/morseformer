"""End-to-end smoke test for a morseformer release checkpoint.

Generates a small deterministic SNR-ladder validation set (fixed seed),
decodes every sample through the RNN-T greedy head of the loaded
checkpoint, and compares per-SNR CER to the thresholds documented for
the v0.1.0 release. Exits with code 0 if every bin is within the
threshold; non-zero on the first regression.

Uses:

1. As a **user-side installation smoke test**: after ``pip install`` /
   ``hf download``, run this to verify the full pipeline
   (synth → front-end → encoder → RNN-T head) works on the user's
   hardware.

2. As a **regression guard before a future release**: thresholds are
   margined only wide enough to absorb sample-to-sample noise at the
   small n used here — a genuine model regression will flip a bin to
   FAIL.

Example::

    python -m scripts.test_release
    python -m scripts.test_release --ckpt release/rnnt_phase3_0.pt

Both on-disk locations (release/ and checkpoints/phase3_0/) are tried
automatically if ``--ckpt`` is not given.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

from eval.metrics import character_error_rate
from morseformer.core.tokenizer import decode
from morseformer.data.synthetic import collate
from morseformer.data.validation import (
    ValidationConfig,
    ValidationSample,
    build_snr_ladder_validation,
)
from morseformer.models.acoustic import AcousticConfig
from morseformer.models.rnnt import RnntConfig, RnntModel


# CER thresholds the checkpoint must stay within on each SNR bin. Set
# with reference to the v0.1.0 benchmark (see README.md):
#
#   +20 dB : observed 0.0000 — threshold 0.02 is effectively "no errors"
#     0 dB : observed 0.0000 — threshold 0.05 has some slack for n = 25
#   −10 dB : observed 0.7620 — threshold 0.85 catches a real regression
#                             without flaking on n = 25 sampling noise
_THRESHOLDS: dict[float, float] = {
    20.0: 0.02,
    0.0: 0.05,
    -10.0: 0.85,
}

# Small and fast: 5 samples/wpm × 5 wpm × 3 snr = 75 samples.
_SNRS: tuple[float, ...] = (20.0, 0.0, -10.0)
_N_PER_WPM: int = 5


def _candidate_paths() -> tuple[Path, ...]:
    return (
        Path("release/rnnt_phase3_0.pt"),
        Path("checkpoints/phase3_0/best_rnnt.pt"),
    )


def _resolve_ckpt(explicit: Path | None) -> Path:
    if explicit is not None:
        if not explicit.exists():
            raise SystemExit(f"[test_release] --ckpt {explicit} not found.")
        return explicit
    for p in _candidate_paths():
        if p.exists():
            return p
    raise SystemExit(
        "[test_release] no checkpoint found. Tried:\n  - "
        + "\n  - ".join(str(p) for p in _candidate_paths())
        + "\nDownload the release weights with:\n"
        "  pip install huggingface_hub\n"
        "  hf download sderhy/morseformer rnnt_phase3_0.pt "
        "--local-dir checkpoints/phase3_0"
    )


def _load_rnnt(path: Path, device: torch.device) -> RnntModel:
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    enc = cfg["model"]["encoder"]
    encoder_cfg = AcousticConfig(
        d_model=enc["d_model"], n_heads=enc["n_heads"], n_layers=enc["n_layers"],
        ff_expansion=enc["ff_expansion"], conv_kernel=enc["conv_kernel"],
        dropout=enc["dropout"],
    )
    rnnt_cfg = RnntConfig(
        encoder=encoder_cfg,
        d_pred=cfg["model"]["d_pred"],
        pred_lstm_layers=cfg["model"]["pred_lstm_layers"],
        d_joint=cfg["model"]["d_joint"],
        dropout=cfg["model"]["dropout"],
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


def _decode_and_score(
    model: RnntModel,
    samples: list[ValidationSample],
    device: torch.device,
    batch_size: int,
) -> dict[float, list[float]]:
    per_snr: dict[float, list[float]] = {}
    for start in range(0, len(samples), batch_size):
        chunk = samples[start : start + batch_size]
        batch = collate([s.as_batch_item() for s in chunk])
        features = batch["features"].to(device)
        lengths = batch["n_frames"].to(device)
        hyps = model.greedy_rnnt_decode(features, lengths)
        for i, s in enumerate(chunk):
            hyp_text = decode(hyps[i])
            cer = character_error_rate(s.text, hyp_text)
            per_snr.setdefault(s.snr_db, []).append(cer)
    return per_snr


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ckpt", type=Path, default=None,
                   help="path to a Phase 3 RNN-T checkpoint; auto-detected "
                        "from release/ or checkpoints/phase3_0/ if omitted.")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", default=None,
                   help="cpu / cuda (default: cuda if available)")
    return p


def _auto_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = torch.device(args.device or _auto_device())

    ckpt_path = _resolve_ckpt(args.ckpt)
    print(f"[test_release] device={device}")
    print(f"[test_release] ckpt  ={ckpt_path}")

    model = _load_rnnt(ckpt_path, device)
    print(f"[test_release] params={sum(p.numel() for p in model.parameters()):,}")

    val_cfg = ValidationConfig(n_per_wpm=_N_PER_WPM)
    samples = build_snr_ladder_validation(_SNRS, cfg=val_cfg)
    print(f"[test_release] val-set: {len(samples)} samples "
          f"({_N_PER_WPM}/wpm × {len(val_cfg.wpm_bins)} wpm × {len(_SNRS)} snr)")

    per_snr = _decode_and_score(model, samples, device, args.batch_size)

    print()
    print(f"  {'SNR (dB)':>8} | {'n':>3} | {'CER':>7} | {'threshold':>9} | result")
    print("  ---------+-----+---------+-----------+-------")
    all_pass = True
    for snr in _SNRS:
        cers = per_snr.get(snr, [])
        key = snr if math.isfinite(snr) else math.inf
        mean = sum(cers) / len(cers) if cers else float("nan")
        threshold = _THRESHOLDS[snr]
        ok = mean <= threshold
        all_pass = all_pass and ok
        mark = "PASS" if ok else "FAIL"
        print(f"  {snr:>+8.1f} | {len(cers):>3d} | {mean:>7.4f} | "
              f"{threshold:>9.4f} | {mark}")

    print()
    print("[test_release] " + ("all bins PASS" if all_pass else "REGRESSION"))
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
