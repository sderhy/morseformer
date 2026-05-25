"""Amateur-radio callsign detection and QRZ.com linking.

UI-free service. Given a chunk of decoded text it returns the spans that
look like callsigns, a QRZ.com lookup URL for each, and a best-effort
DXCC country guess derived from :mod:`morseformer.data.itu_prefixes`.

The matcher is deliberately conservative: it requires the canonical
``<prefix><digit><suffix>`` shape with word boundaries, so common CW
artefacts (``599``, ``5NN``, bare prosigns) do not light up as calls.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from morseformer.data.itu_prefixes import ENTRIES

# Canonical callsign shape, with optional portable prefix (``F/...``) and
# optional portable/secondary suffix (``/P``, ``/MM``, ``/QRP`` ...):
#
#   [prefix/]  <0-1 digit> <1-2 letters> <digit> <1-4 letters>  [/suffix]
#
# Examples matched: F4HYY, G3ZRJ, MW0BGL, 2E0ABC, 9A1AA, HB9XYZ, IK6ZKD,
# DL1ABC/P, F/DL1ABC. Not matched: 599, 5NN, CQ, K (bare prosign).
_CALLSIGN_RE = re.compile(
    r"\b(?:[A-Z0-9]{1,3}/)?"      # optional portable prefix, e.g. F/
    r"\d?[A-Z]{1,2}\d[A-Z]{1,4}"  # core body
    r"(?:/[A-Z0-9]{1,4})?\b"      # optional portable suffix, e.g. /P
)

QRZ_BASE_URL = "https://www.qrz.com/db/"

# Longest-first prefix → DXCC entity, so "EA8" wins over "EA".
_PREFIX_TO_COUNTRY: tuple[tuple[str, str], ...] = tuple(
    sorted(
        ((e.root, e.name) for e in ENTRIES),
        key=lambda kv: len(kv[0]),
        reverse=True,
    )
)


@dataclass(frozen=True)
class CallsignMatch:
    """One detected callsign and its position in the source text."""

    call: str
    start: int
    end: int

    @property
    def qrz_url(self) -> str:
        return qrz_url(self.call)

    @property
    def country(self) -> str | None:
        return country_for(self.call)


def _core(call: str) -> str:
    """Strip an optional portable prefix/suffix to the licensed core.

    ``F/DL1ABC/P`` → ``DL1ABC``. We pick the longest ``/``-delimited part
    that contains a digit, which is the actual home call.
    """
    parts = [p for p in call.split("/") if p]
    if len(parts) == 1:
        return parts[0]
    with_digit = [p for p in parts if any(c.isdigit() for c in p)]
    candidates = with_digit or parts
    return max(candidates, key=len)


def qrz_url(call: str) -> str:
    """QRZ.com lookup URL for a callsign (uses the licensed core)."""
    return QRZ_BASE_URL + _core(call).upper()


def country_for(call: str) -> str | None:
    """Best-effort DXCC entity name from the callsign prefix, or ``None``."""
    core = _core(call).upper()
    for prefix, name in _PREFIX_TO_COUNTRY:
        if core.startswith(prefix):
            return name
    return None


def find_callsigns(text: str) -> list[CallsignMatch]:
    """Return every callsign-shaped span in ``text``, left to right."""
    return [
        CallsignMatch(call=m.group(0), start=m.start(), end=m.end())
        for m in _CALLSIGN_RE.finditer(text.upper())
    ]
