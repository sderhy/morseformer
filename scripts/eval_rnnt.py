"""Side-by-side eval of two RNN-T checkpoints on both validation benches.

Reports CER / WER on:

* the **guard** bench (``build_snr_ladder_validation`` — AWGN + 500 Hz
  RX filter only, the Phase 3.0 validation conditions). Phase 3.1 must
  not regress here.
* the **realistic** bench (``build_realistic_ladder_validation`` —
  guard + Phase 3.1 augs: carrier-frequency jitter, QSB, QRN, carrier
  drift, 25 % QRM). Phase 3.1 must win here.

Usage::

    python -m scripts.eval_rnnt \
        --baseline-ckpt checkpoints/phase3_0/best_rnnt.pt \
        --candidate-ckpt checkpoints/phase3_1/best_rnnt.pt \
        --snrs "+20,+10,+5,0,-5,-10" \
        --n-per-wpm 40

Omit ``--candidate-ckpt`` to evaluate only the baseline.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

from eval.metrics import character_error_rate, word_error_rate
from morseformer.core.tokenizer import decode
from morseformer.data.synthetic import collate
from morseformer.data.validation import (
    ValidationConfig,
    ValidationSample,
    build_realistic_ladder_validation,
    build_snr_ladder_validation,
)
from morseformer.models.acoustic import AcousticConfig
from morseformer.models.rnnt import RnntConfig, RnntModel


def _parse_floats(spec: str) -> tuple[float, ...]:
    return tuple(float(t) for t in spec.split(",") if t.strip())


def _auto_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


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


def _val_batches(samples: list[ValidationSample], batch_size: int):
    for i in range(0, len(samples), batch_size):
        yield collate([s.as_batch_item() for s in samples[i : i + batch_size]])


def _score_model(
    model: RnntModel,
    samples: list[ValidationSample],
    device: torch.device,
    batch_size: int,
) -> dict[float, dict[str, float]]:
    """Return CER / WER / n per SNR bucket."""
    per_snr: dict[float, list[tuple[float, float]]] = {}
    count = 0
    for batch in _val_batches(samples, batch_size):
        features = batch["features"].to(device)
        lengths = batch["n_frames"].to(device)
        hyps = model.greedy_rnnt_decode(features, lengths)
        for j in range(features.size(0)):
            ref = samples[count]
            hyp_text = decode(hyps[j])
            cer = character_error_rate(ref.text, hyp_text)
            wer = word_error_rate(ref.text, hyp_text)
            per_snr.setdefault(ref.snr_db, []).append((cer, wer))
            count += 1
    out: dict[float, dict[str, float]] = {}
    for snr, pairs in per_snr.items():
        cer_m = sum(c for c, _ in pairs) / len(pairs)
        wer_m = sum(w for _, w in pairs) / len(pairs)
        out[snr] = {"cer": cer_m, "wer": wer_m, "n": len(pairs)}
    return out


def _format_row(
    label: str,
    snrs: tuple[float, ...],
    baseline: dict[float, dict[str, float]],
    candidate: dict[float, dict[str, float]] | None,
) -> str:
    out = [f"\n=== {label} ==="]
    header = "  SNR (dB) |   n |  base-CER"
    if candidate is not None:
        header += " | cand-CER | Δ (pp)"
    out.append(header)
    out.append("  " + "-" * (len(header) - 2))
    base_mean = cand_mean = 0.0
    base_n = cand_n = 0
    for snr in snrs:
        b = baseline.get(snr)
        if b is None:
            continue
        base_mean += b["cer"] * b["n"]
        base_n += b["n"]
        if candidate is not None:
            c = candidate.get(snr, {"cer": float("nan"), "wer": float("nan"), "n": 0})
            cand_mean += c["cer"] * c["n"]
            cand_n += c["n"]
            delta = (c["cer"] - b["cer"]) * 100
            out.append(
                f"  {snr:>+8.1f} | {b['n']:>3d} | {b['cer']:>9.4f} | "
                f"{c['cer']:>8.4f} | {delta:+6.2f}"
            )
        else:
            out.append(
                f"  {snr:>+8.1f} | {b['n']:>3d} | {b['cer']:>9.4f}"
            )
    # Weighted overall line.
    base_overall = base_mean / base_n if base_n else float("nan")
    if candidate is not None:
        cand_overall = cand_mean / cand_n if cand_n else float("nan")
        delta = (cand_overall - base_overall) * 100
        out.append(
            f"  {'overall':>8} | {base_n:>3d} | {base_overall:>9.4f} | "
            f"{cand_overall:>8.4f} | {delta:+6.2f}"
        )
    else:
        out.append(
            f"  {'overall':>8} | {base_n:>3d} | {base_overall:>9.4f}"
        )
    return "\n".join(out)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--baseline-ckpt", type=Path,
                   default=Path("checkpoints/phase3_0/best_rnnt.pt"))
    p.add_argument("--candidate-ckpt", type=Path, default=None,
                   help="Second checkpoint (e.g. Phase 3.1). Omit to "
                        "evaluate only the baseline.")
    p.add_argument("--snrs", default="+20,+10,+5,0,-5,-10")
    p.add_argument("--n-per-wpm", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = torch.device(args.device or _auto_device())
    snrs = _parse_floats(args.snrs)

    val_cfg = ValidationConfig(n_per_wpm=args.n_per_wpm)
    guard = build_snr_ladder_validation(snrs, cfg=val_cfg)
    realistic = build_realistic_ladder_validation(snrs, cfg=val_cfg)
    print(f"[eval_rnnt] device={device}")
    print(f"[eval_rnnt] guard:     {len(guard)} samples "
          f"({args.n_per_wpm}/wpm × {len(val_cfg.wpm_bins)} wpm × {len(snrs)} snr)")
    print(f"[eval_rnnt] realistic: {len(realistic)} samples "
          f"(same grid + carrier jitter + QSB + QRN + 25 % QRM)")

    print(f"[eval_rnnt] baseline: {args.baseline_ckpt}")
    baseline_model = _load_rnnt(args.baseline_ckpt, device)
    base_guard = _score_model(baseline_model, guard, device, args.batch_size)
    base_realistic = _score_model(baseline_model, realistic, device, args.batch_size)
    del baseline_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if args.candidate_ckpt is not None:
        print(f"[eval_rnnt] candidate: {args.candidate_ckpt}")
        candidate_model = _load_rnnt(args.candidate_ckpt, device)
        cand_guard = _score_model(
            candidate_model, guard, device, args.batch_size,
        )
        cand_realistic = _score_model(
            candidate_model, realistic, device, args.batch_size,
        )
    else:
        cand_guard = cand_realistic = None

    print(_format_row("guard (AWGN only)", snrs, base_guard, cand_guard))
    print(_format_row("realistic (Phase 3.1 channel)", snrs,
                      base_realistic, cand_realistic))

    if args.candidate_ckpt is not None:
        # Overall headline: did the candidate win on realistic without
        # regressing on guard?
        guard_reg = sum(
            (cand_guard[s]["cer"] - base_guard[s]["cer"]) for s in snrs if s in cand_guard
        ) / len(snrs)
        real_gain = sum(
            (base_realistic[s]["cer"] - cand_realistic[s]["cer"]) for s in snrs if s in cand_realistic
        ) / len(snrs)
        print()
        print(f"[eval_rnnt] guard regression (cand - base, mean Δ): "
              f"{guard_reg*100:+.2f} pp (negative = improvement, "
              f"positive = regression)")
        print(f"[eval_rnnt] realistic gain    (base - cand, mean Δ): "
              f"{real_gain*100:+.2f} pp (positive = improvement)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
