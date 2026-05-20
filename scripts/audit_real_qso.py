"""One-off audit: decode the real-QSO corpus at ``../testlive/`` and
report CER / WER per clip + per operator.

The corpus lives outside the repo so the script takes its root via
``--root`` (default ``../testlive``). For each operator directory it
finds, pairs every ``*.wav`` with the matching ``*.txt`` and runs the
acoustic + preset specified through the model registry.

This is intentionally *not* a release-gate category yet. The point is
to see the numbers on real over-the-air audio before deciding whether
to:

* lock the corpus as a new release-gate category (extends
  ``release_gate_v1.json`` → ``v2.json``)
* split it into train / held-out and fine-tune a Phase 8 acoustic
* both / neither.

Run::

    python -m scripts.audit_real_qso
    python -m scripts.audit_real_qso --root ../testlive --operator g3ses
    python -m scripts.audit_real_qso --acoustic rnnt_phase5_5 --preset live
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from eval.bench_lcwo import (
    _load_audio,
    _load_lm,
    _load_rnnt,
    _normalize_prosign_brackets,
    make_morseformer_decoder,
)
from eval.metrics import character_error_rate, word_error_rate
from morseformer.cli.presets import get_preset
from morseformer.cli.registry import resolve_model
from morseformer.data.text import _normalize_prose

_DEFAULT_ROOT = Path("../testlive")
_DEFAULT_OPERATORS = ("g3ses", "g6pz")
_SAMPLE_RATE = 8000
_CARRIER_HZ = 600.0


def _pairs_for_operator(root: Path, op: str) -> list[tuple[Path, Path]]:
    """Find every ``*.wav`` in ``root/op`` with a matching ``*.txt``."""
    op_dir = root / op
    if not op_dir.is_dir():
        return []
    pairs: list[tuple[Path, Path]] = []
    for wav in sorted(op_dir.glob("*.wav")):
        txt = wav.with_suffix(".txt")
        if txt.exists():
            pairs.append((wav, txt))
    return pairs


def _auto_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", type=Path, default=_DEFAULT_ROOT)
    p.add_argument(
        "--operator", action="append", default=None,
        help="Operator subdir to audit. Repeatable. "
             f"Default: {', '.join(_DEFAULT_OPERATORS)}",
    )
    p.add_argument("--acoustic", default="rnnt_phase5_5",
                   help="Registry name of the acoustic checkpoint. "
                        "Ignored when --ckpt is given.")
    p.add_argument("--ckpt", type=Path, default=None,
                   help="Direct path to an RNN-T checkpoint, bypassing "
                        "the model registry. Useful for newly-trained "
                        "candidates (e.g. checkpoints/phase8/best_rnnt.pt) "
                        "before they get a registry entry.")
    p.add_argument(
        "--preset", default="live",
        help="Decode preset (live / prose / contest / conservative).",
    )
    p.add_argument("--device", default=None)
    p.add_argument(
        "--lang", default="en",
        help="Language for _normalize_prose (only affects DE umlauts).",
    )
    p.add_argument(
        "--show-hyp-chars", type=int, default=0,
        help="Print the first N chars of each hypothesis (debugging).",
    )
    p.add_argument(
        "--post-segment", action="store_true",
        help="Run the dictionary-based word splitter on the decoder "
             "output (see morseformer.decoding.word_splitter). Closes "
             "the run-on word failure mode observed on real OTA where "
             "operators collapse inter-word gaps to zero "
             "(DROMCHRIS → DR OM CHRIS, MYWXIS → MY WX IS).",
    )
    args = p.parse_args(argv)

    root = args.root
    if not root.is_dir():
        print(f"[audit] root not found: {root}", file=sys.stderr)
        return 2
    operators = args.operator or list(_DEFAULT_OPERATORS)

    device = torch.device(args.device or _auto_device())
    preset = get_preset(args.preset)
    print(
        f"[audit] root={root} operators={operators} "
        f"acoustic={args.acoustic} preset={args.preset} device={device}"
    )

    ckpt_path = args.ckpt if args.ckpt is not None else resolve_model(args.acoustic)
    acoustic_label = str(args.ckpt) if args.ckpt is not None else args.acoustic
    print(f"[audit] acoustic checkpoint: {ckpt_path} (label={acoustic_label})")
    acoustic = _load_rnnt(ckpt_path, device)
    lm = None
    fusion_weight = 0.0
    if preset.lm and preset.fusion_weight > 0:
        lm = _load_lm(resolve_model(preset.lm), device)
        fusion_weight = preset.fusion_weight
        print(f"[audit] LM={preset.lm} fusion_weight={fusion_weight}")

    decoder = make_morseformer_decoder(
        acoustic, device=device,
        confidence_threshold=preset.confidence_threshold,
        digit_threshold=preset.digit_threshold,
        lm=lm, fusion_weight=fusion_weight,
        sample_rate=_SAMPLE_RATE, carrier_hz=_CARRIER_HZ,
        post_segment=args.post_segment,
    )

    per_clip: list[dict] = []
    per_operator: dict[str, list[tuple[float, float, float]]] = {}

    for op in operators:
        pairs = _pairs_for_operator(root, op)
        if not pairs:
            print(f"[audit] no pairs found for operator '{op}'.")
            continue
        print(f"\n[audit] {op}: {len(pairs)} clips")
        for wav, txt in pairs:
            audio = _load_audio(wav, _SAMPLE_RATE)
            duration_s = len(audio) / _SAMPLE_RATE
            gt_raw = _normalize_prosign_brackets(txt.read_text(encoding="utf-8"))
            gt = _normalize_prose(gt_raw, args.lang)
            hyp = decoder(audio, _SAMPLE_RATE)
            cer = character_error_rate(gt, hyp)
            wer = word_error_rate(gt, hyp)
            per_clip.append({
                "operator": op,
                "clip": wav.stem,
                "dur_s": duration_s,
                "gt_chars": len(gt),
                "hyp_chars": len(hyp),
                "cer": cer,
                "wer": wer,
                "hyp_head": hyp[: args.show_hyp_chars] if args.show_hyp_chars else "",
            })
            per_operator.setdefault(op, []).append((duration_s, cer, wer))

    # --------------------------------------------------------------------- #
    # Per-clip table
    # --------------------------------------------------------------------- #
    print("\n## Per-clip results\n")
    print(
        f"  {'operator':<8} {'clip':<10} {'dur':>6} {'gt':>5} {'hyp':>5} "
        f"{'CER%':>7} {'WER%':>7}"
    )
    for r in per_clip:
        print(
            f"  {r['operator']:<8} {r['clip']:<10} "
            f"{r['dur_s']:>6.1f} {r['gt_chars']:>5} {r['hyp_chars']:>5} "
            f"{r['cer']*100:>7.2f} {r['wer']*100:>7.2f}"
        )
        if args.show_hyp_chars > 0:
            print(f"    hyp: {r['hyp_head']!r}")

    # --------------------------------------------------------------------- #
    # Per-operator + overall summary
    # --------------------------------------------------------------------- #
    print("\n## Summary\n")
    print(
        f"  {'operator':<8} {'n':>3} {'total_dur':>10} "
        f"{'mean_CER%':>10} {'mean_WER%':>10}"
    )
    all_cer: list[float] = []
    all_wer: list[float] = []
    for op, rows in per_operator.items():
        cer = float(np.mean([c for _, c, _ in rows])) * 100
        wer = float(np.mean([w for _, _, w in rows])) * 100
        dur = float(sum(d for d, _, _ in rows))
        print(
            f"  {op:<8} {len(rows):>3} {dur:>10.1f} "
            f"{cer:>10.2f} {wer:>10.2f}"
        )
        all_cer.extend(c for _, c, _ in rows)
        all_wer.extend(w for _, _, w in rows)
    if all_cer:
        print(
            f"  {'ALL':<8} {len(all_cer):>3} "
            f"{sum(d for op in per_operator.values() for d, _, _ in op):>10.1f} "
            f"{np.mean(all_cer)*100:>10.2f} {np.mean(all_wer)*100:>10.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
