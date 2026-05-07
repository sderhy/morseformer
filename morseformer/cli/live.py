"""``morseformer live`` — real-time PulseAudio streaming decode."""

from __future__ import annotations

import argparse

from .presets import DEFAULT_PRESET, PRESETS, get_preset
from .registry import RECOMMENDED_ACOUSTIC, resolve_model


def add_live_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "live",
        help="real-time decode from the default audio input",
        description="Stream-decode audio from the default PulseAudio input.",
    )
    p.add_argument("--preset", choices=tuple(PRESETS), default=DEFAULT_PRESET,
                   help=f"streaming preset (default: {DEFAULT_PRESET}).")
    p.add_argument("--model", default=None,
                   help="override the preset's acoustic model.")
    p.add_argument("--carrier", type=float, default=None,
                   help="CW tone frequency in Hz (default 600).")
    p.add_argument("--audio-device", default=None,
                   help="PulseAudio source name (default: system default).")
    p.add_argument("--device", default=None,
                   help="cpu / cuda for inference (default: auto)")


def run_live(args: argparse.Namespace) -> int:
    preset = get_preset(args.preset)
    acoustic_name = args.model or preset.acoustic or RECOMMENDED_ACOUSTIC

    if preset.lm is not None:
        # Streaming fusion regresses CER on the v0.5.x acoustic
        # (project_streaming_fusion_failed). Drop the LM silently with a
        # one-liner notice — the acoustic-only path is what live actually
        # uses today.
        print(f"[morseformer] note: '{preset.name}' preset's LM is offline-"
              f"only; live decode runs acoustic-only.")

    acoustic_path = resolve_model(acoustic_name)

    print(f"[morseformer] preset='{preset.name}'  acoustic={acoustic_name}  "
          f"thr={preset.confidence_threshold}  digit_thr={preset.digit_threshold}")

    forwarded = [
        "--ckpt", str(acoustic_path),
        "--confidence-threshold", str(preset.confidence_threshold),
        "--digit-threshold", str(preset.digit_threshold),
    ]
    if args.carrier is not None:
        forwarded += ["--carrier", str(args.carrier)]
    if args.audio_device:
        forwarded += ["--audio-device", args.audio_device]
    if args.device:
        forwarded += ["--device", args.device]

    from scripts.decode_live import main as live_main
    return live_main(forwarded)
