"""``morseformer decode`` — offline file decoding."""

from __future__ import annotations

import argparse
from pathlib import Path

from .presets import DEFAULT_PRESET, PRESETS, get_preset
from .registry import RECOMMENDED_ACOUSTIC, resolve_model


def add_decode_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "decode",
        help="decode a .wav file offline",
        description="Decode a CW audio file with a preset or explicit knobs.",
    )
    p.add_argument("audio", type=Path, help="path to a .wav file")
    p.add_argument("--preset", choices=tuple(PRESETS), default=DEFAULT_PRESET,
                   help=f"decode preset (default: {DEFAULT_PRESET}). "
                        f"Use `morseformer models` to see what each preset "
                        f"loads.")
    p.add_argument("--model", default=None,
                   help="override the preset's acoustic model. Default: "
                        "the preset's recommended acoustic.")
    p.add_argument("--lm", default=None,
                   help="override the preset's LM. 'none' disables fusion.")
    p.add_argument("--device", default=None,
                   help="cpu / cuda / mps (default: auto)")


def run_decode(args: argparse.Namespace) -> int:
    preset = get_preset(args.preset)
    acoustic_name = args.model or preset.acoustic or RECOMMENDED_ACOUSTIC
    if args.lm is not None:
        lm_name = None if args.lm.lower() == "none" else args.lm
        fusion_weight = 0.0 if lm_name is None else preset.fusion_weight
    else:
        lm_name = preset.lm
        fusion_weight = preset.fusion_weight

    if fusion_weight > 0 and lm_name is None:
        # Defensive — preset should never end up with weight > 0 + lm None.
        fusion_weight = 0.0

    acoustic_path = resolve_model(acoustic_name)
    lm_path: Path | None = resolve_model(lm_name) if lm_name else None

    print(f"[morseformer] preset='{preset.name}'  acoustic={acoustic_name}"
          + (f"  lm={lm_name} (λ={fusion_weight})" if lm_name else "  lm=off"))

    forwarded = [
        str(args.audio),
        "--ckpt", str(acoustic_path),
        "--confidence-threshold", str(preset.confidence_threshold),
        "--digit-threshold", str(preset.digit_threshold),
    ]
    if lm_path is not None and fusion_weight > 0:
        forwarded += ["--lm-ckpt", str(lm_path),
                      "--fusion-weight", str(fusion_weight)]
    if args.device:
        forwarded += ["--device", args.device]

    # Local import so that `morseformer --help` does not pull torch.
    from scripts.decode_audio import main as decode_main
    return decode_main(forwarded)
