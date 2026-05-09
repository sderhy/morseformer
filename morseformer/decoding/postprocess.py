"""Cosmetic post-processing for decoder output.

The model emits run-on prosigns as their literal multi-character form
(``BK``, ``SK``, ``AR``, ``KN``…). For display we substitute the
conventional one-character glyph where it improves readability:

* ``BK`` → ``=``  — both signal a break / hand-off; the single-char ``=``
  is much easier to spot in continuous prose than the digram ``BK`` that
  reads like part of a callsign.

Word boundaries are required on both sides so we never touch ``BK``
inside a callsign or word (``K1BK``, ``BKR…``).
"""

from __future__ import annotations

import re

_PROSIGN_SUBS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bBK\b"), "="),
)


def format_output(text: str) -> str:
    """Apply display-only prosign substitutions to a finished string."""
    for pat, rep in _PROSIGN_SUBS:
        text = pat.sub(rep, text)
    return text


class StreamFormatter:
    """Apply :func:`format_output` to an incrementally produced stream.

    Holds back a short trailing alphanumeric run between calls so a
    prosign that happens to straddle a fragment boundary is still
    rewritten correctly. The hold is bounded (2 chars) so latency stays
    flat — at most the last 2 letters of each fragment are deferred to
    the next ``feed()`` or ``flush()`` call.
    """

    _MAX_HOLD = 2

    def __init__(self) -> None:
        self._pending = ""

    def feed(self, fragment: str) -> str:
        if not fragment:
            return ""
        text = self._pending + fragment
        cut = len(text)
        held = 0
        while cut > 0 and text[cut - 1].isalnum() and held < self._MAX_HOLD:
            cut -= 1
            held += 1
        self._pending = text[cut:]
        return format_output(text[:cut])

    def flush(self) -> str:
        out = format_output(self._pending)
        self._pending = ""
        return out
