"""Command-line entry-point: `python -m eval.cli`.

Example:
    python -m eval.cli --decoder rule_based --dataset sanity -v
"""

from __future__ import annotations

import argparse
import json
import sys

from eval.benchmark import run
from eval.datasets import generate_sanity
from morseformer.baselines import rule_based

DECODERS = {
    "rule_based": rule_based.decode,
}

DATASET_FACTORIES = {
    "sanity": lambda: generate_sanity(n=20),
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="morseformer-eval",
        description="Run a morseformer decoder against a dataset.",
    )
    parser.add_argument(
        "--decoder",
        choices=sorted(DECODERS),
        default="rule_based",
        help="Which decoder to benchmark.",
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASET_FACTORIES),
        default="sanity",
        help="Which dataset to score against.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print per-sample reference/hypothesis lines.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of a human summary.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    decoder = DECODERS[args.decoder]
    dataset = DATASET_FACTORIES[args.dataset]()
    result = run(decoder, dataset)

    if args.json:
        payload = {
            "decoder": args.decoder,
            "dataset": args.dataset,
            "n_samples": result.n_samples,
            "mean_cer": result.mean_cer,
            "mean_wer": result.mean_wer,
            "mean_callsign_f1": result.mean_callsign_f1,
            "per_sample": [
                {
                    "id": s.sample_id,
                    "ref": s.reference,
                    "hyp": s.hypothesis,
                    "cer": s.cer,
                    "wer": s.wer,
                    "callsign_f1": s.callsign_f1,
                }
                for s in result.per_sample
            ],
        }
        json.dump(payload, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0

    print(f"decoder={args.decoder}  dataset={args.dataset}  n={result.n_samples}")
    print(f"  mean CER         = {result.mean_cer:.4f}")
    print(f"  mean WER         = {result.mean_wer:.4f}")
    print(f"  mean Callsign F1 = {result.mean_callsign_f1:.4f}")
    if args.verbose:
        print()
        for s in result.per_sample:
            mark = "ok " if s.cer < 0.05 else "BAD"
            print(f"  [{mark}] {s.sample_id}  CER={s.cer:.3f}")
            print(f"        ref: {s.reference!r}")
            print(f"        hyp: {s.hypothesis!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
