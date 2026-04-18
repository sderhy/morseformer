"""Command-line entry-point: `python -m eval.cli`.

Examples:

    # Phase-0 sanity (harness invariant)
    python -m eval.cli --decoder rule_based --dataset sanity

    # Single-SNR noisy benchmark
    python -m eval.cli --decoder rule_based --dataset noisy --snr-db 0

    # Full SNR ladder (the real Phase-1 benchmark)
    python -m eval.cli --decoder rule_based --dataset snr_ladder \\
        --snrs="+20,+10,+5,0,-5,-10,-15" --n-per-snr 20
"""

from __future__ import annotations

import argparse
import json
import math
import sys

from eval.benchmark import Decoder, run
from eval.datasets import generate_noisy, generate_sanity
from eval.snr_ladder import run_snr_ladder
from morseformer.baselines import rule_based

# Decoders that need no configuration beyond the audio/sample_rate pair
# are registered directly. The neural decoder needs a checkpoint path,
# so it is handled specially below.
DECODERS: dict[str, Decoder] = {
    "rule_based": rule_based.decode,
}

_DECODER_CHOICES = sorted(list(DECODERS) + ["neural"])


def _parse_snrs(spec: str) -> list[float]:
    """Parse a comma-separated SNR list like '+20,+10,0,-5,-10'."""
    out: list[float] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if token.lower() in ("inf", "+inf", "clean"):
            out.append(float("inf"))
        else:
            out.append(float(token))
    return out


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="morseformer-eval",
        description="Run a morseformer decoder against a dataset.",
    )
    parser.add_argument(
        "--decoder", choices=_DECODER_CHOICES, default="rule_based",
        help="Which decoder to benchmark.",
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="Path to a trained checkpoint (required when --decoder neural).",
    )
    parser.add_argument(
        "--device", default="cpu",
        help="Device for the neural decoder (cpu / cuda).",
    )
    parser.add_argument(
        "--no-ema", action="store_true",
        help="Neural decoder: load raw model weights instead of EMA ones.",
    )
    parser.add_argument(
        "--dataset", choices=("sanity", "noisy", "snr_ladder"), default="sanity",
        help="Which dataset / evaluation mode.",
    )
    parser.add_argument(
        "--n", type=int, default=20,
        help="Number of samples for sanity/noisy datasets.",
    )
    parser.add_argument(
        "--snr-db", type=float, default=10.0,
        help="Target SNR in dB for the 'noisy' dataset.",
    )
    parser.add_argument(
        "--snrs", default="+20,+10,+5,0,-5,-10,-15",
        help="Comma-separated SNR list for 'snr_ladder' mode (in dB, or 'inf').",
    )
    parser.add_argument(
        "--n-per-snr", type=int, default=20,
        help="Samples per SNR step in 'snr_ladder' mode.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Master seed for dataset generation.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Print per-sample reference/hypothesis lines.",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Emit machine-readable JSON instead of a human summary.",
    )
    return parser.parse_args(argv)


def _print_single(result, *, decoder: str, dataset_label: str, verbose: bool) -> None:
    print(f"decoder={decoder}  dataset={dataset_label}  n={result.n_samples}")
    print(f"  mean CER         = {result.mean_cer:.4f}")
    print(f"  mean WER         = {result.mean_wer:.4f}")
    print(f"  mean Callsign F1 = {result.mean_callsign_f1:.4f}")
    if verbose:
        print()
        for s in result.per_sample:
            mark = "ok " if s.cer < 0.05 else "BAD"
            print(f"  [{mark}] {s.sample_id}  CER={s.cer:.3f}")
            print(f"        ref: {s.reference!r}")
            print(f"        hyp: {s.hypothesis!r}")


def _build_decoder(args: argparse.Namespace) -> Decoder:
    if args.decoder == "neural":
        if not args.checkpoint:
            print("--checkpoint is required when --decoder neural",
                  file=sys.stderr)
            sys.exit(2)
        # Deferred import so that users on machines without torch can
        # still run `--decoder rule_based`.
        from morseformer.baselines.neural import (
            NeuralDecoder, NeuralDecoderConfig,
        )
        return NeuralDecoder.from_checkpoint(
            args.checkpoint,
            NeuralDecoderConfig(device=args.device, use_ema=not args.no_ema),
        )
    return DECODERS[args.decoder]


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    decoder = _build_decoder(args)

    if args.dataset == "sanity":
        ds = generate_sanity(n=args.n)
        result = run(decoder, ds)
        if args.json:
            json.dump(
                {
                    "decoder": args.decoder, "dataset": "sanity",
                    "n_samples": result.n_samples,
                    "mean_cer": result.mean_cer, "mean_wer": result.mean_wer,
                    "mean_callsign_f1": result.mean_callsign_f1,
                },
                sys.stdout, indent=2,
            )
            sys.stdout.write("\n")
        else:
            _print_single(result, decoder=args.decoder, dataset_label="sanity",
                          verbose=args.verbose)
        return 0

    if args.dataset == "noisy":
        ds = generate_noisy(n=args.n, snr_db=args.snr_db, seed=args.seed)
        result = run(decoder, ds)
        label = (
            f"noisy (SNR=+inf)" if math.isinf(args.snr_db)
            else f"noisy (SNR={args.snr_db:+.1f} dB)"
        )
        if args.json:
            json.dump(
                {
                    "decoder": args.decoder, "dataset": "noisy",
                    "snr_db": args.snr_db,
                    "n_samples": result.n_samples,
                    "mean_cer": result.mean_cer, "mean_wer": result.mean_wer,
                    "mean_callsign_f1": result.mean_callsign_f1,
                },
                sys.stdout, indent=2,
            )
            sys.stdout.write("\n")
        else:
            _print_single(result, decoder=args.decoder, dataset_label=label,
                          verbose=args.verbose)
        return 0

    if args.dataset == "snr_ladder":
        snrs = _parse_snrs(args.snrs)
        ladder = run_snr_ladder(
            decoder, snrs, n_per_snr=args.n_per_snr, seed=args.seed
        )
        if args.json:
            json.dump(
                {
                    "decoder": args.decoder, "dataset": "snr_ladder",
                    "n_per_snr": args.n_per_snr,
                    "rows": ladder.as_rows(),
                },
                sys.stdout, indent=2,
            )
            sys.stdout.write("\n")
        else:
            print(f"decoder={args.decoder}  dataset=snr_ladder  n_per_snr={args.n_per_snr}")
            print()
            print(ladder.format_table())
        return 0

    # argparse should have caught this already.
    print(f"Unknown dataset: {args.dataset!r}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
