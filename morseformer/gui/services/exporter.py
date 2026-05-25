"""Transcript export.

UI-free service. Today it writes a plain-text transcript with a small
header; the ``.adif`` seam is stubbed so a future logbook export drops in
without touching the call sites.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from morseformer import __version__


def export_text(text: str, path: str | Path, *, header: bool = True) -> Path:
    """Write ``text`` to ``path`` as UTF-8, optionally with a header."""
    p = Path(path)
    body = text if text.endswith("\n") or not text else text + "\n"
    if header:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
        body = f"# morseformer v{__version__} transcript — {ts}\n\n{body}"
    p.write_text(body, encoding="utf-8")
    return p


def export_adif(qsos: list[dict], path: str | Path) -> Path:  # pragma: no cover
    """Placeholder for ADIF logbook export (see :mod:`.logbook`)."""
    raise NotImplementedError(
        "ADIF export is not implemented yet; wire it to the logbook service."
    )
