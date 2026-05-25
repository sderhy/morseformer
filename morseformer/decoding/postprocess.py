"""Cosmetic post-processing for decoder output.

The model emits run-on prosigns as their literal multi-character form
(``BK``, ``SK``, ``AR``, ``KN``…). For display we substitute the
conventional one-character glyph where it improves readability and
break the output into logical lines so the user is not staring at a
wall of upper-case prose:

* ``BK`` → ``=``  — both signal a break / hand-off; the single-char ``=``
  is much easier to spot in continuous prose than the digram ``BK`` that
  reads like part of a callsign.
* After every ``=`` or standalone ``KN``, append a newline. These are
  the conventional amateur over / break / hand-off markers, so a line
  break there matches how operators read a QSO transcript.

Word boundaries are required so we never touch ``BK`` / ``KN`` inside a
callsign or other word (``K1BK``, ``KN4...``).
"""

from __future__ import annotations

import re

_PROSIGN_SUBS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bBK\b"), "="),
)

# Tokens that mark a logical break in the QSO and get a trailing newline
# in display output. After ``_PROSIGN_SUBS`` has run, ``BK`` is already
# ``=``; ``KN`` stays as a digram (the user can opt to read it that way
# or treat it as the line break).
_BREAK_TOKEN_RE = re.compile(r"(=|\bKN\b)[ \t]*")

# Standalone ``K`` (invitation to transmit) — opt-in line break, since
# many operators end an over with a bare ``K``. Word boundaries keep us
# off the ``K`` inside callsigns / words.
_K_BREAK_RE = re.compile(r"(\bK\b)[ \t]*")


def format_output(
    text: str,
    *,
    break_tokens: bool = True,
    break_after_k: bool = False,
    lowercase: bool = False,
) -> str:
    """Apply display-only prosign substitutions to a finished string.

    By default adds a newline after every ``=`` or standalone ``KN`` so a
    long decoded transcript reads as a series of over / break segments
    instead of one continuous upper-case run. The defaults reproduce the
    historical behaviour; the keyword flags let display surfaces (the GUI
    Text menu) toggle each transform independently.

    Args:
        break_tokens: newline after ``=`` / ``KN`` (the default break).
        break_after_k: also newline after a standalone ``K``.
        lowercase: render the result in lower case.
    """
    for pat, rep in _PROSIGN_SUBS:
        text = pat.sub(rep, text)
    if break_tokens:
        text = _BREAK_TOKEN_RE.sub(lambda m: m.group(1) + "\n", text)
    if break_after_k:
        text = _K_BREAK_RE.sub(lambda m: m.group(1) + "\n", text)
    if lowercase:
        text = text.lower()
    return text.rstrip("\n")


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
