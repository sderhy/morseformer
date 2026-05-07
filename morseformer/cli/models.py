"""``morseformer models`` — list and download model checkpoints."""

from __future__ import annotations

import argparse

from .registry import REGISTRY, known_names, resolve_model


def add_models_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "models",
        help="list and download model checkpoints",
        description="Inspect the available morseformer model checkpoints "
                    "and pre-fetch them from the HuggingFace Hub.",
    )
    msub = p.add_subparsers(dest="action", metavar="{list,download}")

    pl = msub.add_parser("list", help="list known models")
    pl.add_argument("--advanced", action="store_true",
                    help="show all checkpoints including legacy 46-vocab "
                         "models. Default: recommended only.")

    pd = msub.add_parser("download", help="pre-fetch a model from the Hub")
    pd.add_argument("name", help="registry name (e.g. rnnt_phase5_7)")


def run_models(args: argparse.Namespace) -> int:
    if args.action == "list" or args.action is None:
        advanced = bool(getattr(args, "advanced", False))
        names = known_names(advanced=advanced)
        print(f"morseformer models ({'all' if advanced else 'recommended'}):")
        for name in names:
            info = REGISTRY[name]
            if info.recommended:
                tag = "★ "
            elif info.legacy:
                tag = "L "
            else:
                tag = "  "
            print(f"  {tag} {name:<20}  vocab={info.vocab}  ({info.kind})")
            print(f"          {info.description}")
        print("\n  ★ = recommended  ·  L = legacy 46-vocab (no FR accents)")
        if not advanced:
            print("\nUse `morseformer models list --advanced` for the full set.")
        return 0
    if args.action == "download":
        path = resolve_model(args.name)
        print(f"[morseformer] {args.name} → {path}")
        return 0
    print("[morseformer] models: pass `list` or `download <name>`.")
    return 2
