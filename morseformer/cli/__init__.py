"""morseformer CLI — `morseformer {decode,live,models}` console entrypoint."""

from __future__ import annotations

import argparse
import sys

from morseformer import __version__

from .decode import add_decode_parser, run_decode
from .live import add_live_parser, run_live
from .models import add_models_parser, run_models


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="morseformer",
        description="Open-source transformer Morse / CW decoder.",
    )
    p.add_argument("--version", action="version",
                   version=f"morseformer {__version__}")
    sub = p.add_subparsers(dest="command", required=True, metavar="{decode,live,models}")
    add_decode_parser(sub)
    add_live_parser(sub)
    add_models_parser(sub)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "decode":
        return run_decode(args)
    if args.command == "live":
        return run_live(args)
    if args.command == "models":
        return run_models(args)
    print(f"[morseformer] unknown command: {args.command}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
