"""morseformer — open-source transformer-based Morse/CW decoder."""

from __future__ import annotations

import sys

_MIN_PYTHON = (3, 10)
_MAX_PYTHON_EXCLUSIVE = (3, 14)


def _unsupported_python_message(version_info: tuple[int, int, int]) -> str:
    major, minor, micro = version_info[:3]
    return (
        f"morseformer supports Python 3.10-3.13; you are running "
        f"Python {major}.{minor}.{micro}. Create a supported virtual "
        f"environment, then reinstall morseformer. For development:\n"
        f"  uv venv .venv --python 3.12\n"
        f"  source .venv/bin/activate\n"
        f"  uv pip install -e \".[dev,live,gui,demo]\"\n"
        f"For a regular install:\n"
        f"  python3.12 -m venv venv\n"
        f"  source venv/bin/activate\n"
        f"  python -m pip install --upgrade pip\n"
        f"  pip install morseformer"
    )


if not (_MIN_PYTHON <= sys.version_info[:2] < _MAX_PYTHON_EXCLUSIVE):
    raise RuntimeError(_unsupported_python_message(sys.version_info[:3]))

__version__ = "0.6.4"
