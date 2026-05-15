"""``morseformer gui`` — launch the PySide6 desktop app."""

from __future__ import annotations

import argparse


def add_gui_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "gui",
        help="launch the PySide6 desktop app",
        description="Launch the PySide6 GUI (live mic decode + file decode). "
                    "Requires the [gui] extra: `pip install \"morseformer[gui]\"`.",
    )
    p.set_defaults(_run=run_gui)


def run_gui(args: argparse.Namespace) -> int:    # noqa: ARG001
    try:
        from morseformer.gui.app import main as gui_main
    except ImportError as e:
        print(
            f"[morseformer] gui dependencies are missing: {e}\n"
            "Install them with: pip install \"morseformer[gui]\"",
        )
        return 2
    return gui_main([])
