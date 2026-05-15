"""Unit tests for the ITU/DXCC callsign-shape prior."""

from __future__ import annotations

import pytest

from morseformer.decoding.callsign_prior import (
    KNOWN_ROOTS,
    is_callsign_shape,
    score_callsign,
)

_VALID_SHAPES = [
    "F4HYY",       # letter + digit + 3 letters
    "DL5WW",       # 2 letters + digit + 2 letters
    "DL5WWW",      # 2 letters + digit + 3 letters
    "DL5WWWW",     # 2 letters + digit + 4 letters (max suffix)
    "K1ABC",       # 1 letter + digit + 3 letters
    "G4LIOW",      # 1 letter + digit + 4 letters
    "HB9DNX",      # 2 letters + digit + 3 letters (digit-bearing root)
    "EA8DAQ",      # 2 letters + digit + 3 letters
    "KL7AB",       # 2 letters + digit + 2 letters
    "9A2BC",       # digit + letter + digit + 2 letters
    "KH6ABC",      # 2 letters + digit + 3 letters (Hawaii root)
]


@pytest.mark.parametrize("word", _VALID_SHAPES)
def test_is_callsign_shape_accepts_valid(word: str) -> None:
    assert is_callsign_shape(word) is True


_INVALID_SHAPES = [
    "",            # empty
    "HELLO",       # all letters, no digit
    "CQ",          # too short, no digit
    "599",         # all digits
    "73",          # all digits
    "K",           # bare prefix
    "K1",          # prefix + digit but no suffix
    "K1ABCDE",     # suffix > 4 letters
    "QRZ",         # all letters
    "KN",          # bare prosign letters
    "F4HYY1",      # trailing digit after suffix
    "f4hyy",       # lowercase: regex is uppercase-only
    "F4 HYY",      # space mid-call
]


@pytest.mark.parametrize("word", _INVALID_SHAPES)
def test_is_callsign_shape_rejects_invalid(word: str) -> None:
    assert is_callsign_shape(word) is False


@pytest.mark.parametrize(
    "word",
    ["F4HYY/P", "DL5WW/M", "HB9DNX/QRP", "K1ABC/MM", "K1ABC/AM", "K1ABC/A"],
)
def test_is_callsign_shape_accepts_portable_suffix(word: str) -> None:
    assert is_callsign_shape(word) is True


@pytest.mark.parametrize("word", ["F4HYY/X", "F4HYY/123", "F4HYY/"])
def test_is_callsign_shape_rejects_invalid_portable(word: str) -> None:
    assert is_callsign_shape(word) is False


def test_score_known_root_returns_full_weight() -> None:
    # F is in the DXCC table (France); F4HYY parses to root F4 → F.
    assert score_callsign("F4HYY", weight=1.0) == pytest.approx(1.0)
    # HB9 is in the table as a digit-bearing root and must match directly.
    assert "HB9" in KNOWN_ROOTS
    assert score_callsign("HB9DNX", weight=1.0) == pytest.approx(1.0)


def test_score_unknown_root_returns_half_weight() -> None:
    unknown = _pick_unknown_root_callsign()
    assert score_callsign(unknown, weight=1.0) == pytest.approx(0.5)


def test_score_unknown_root_fraction_zero_drops_unknown() -> None:
    unknown = _pick_unknown_root_callsign()
    assert score_callsign(unknown, weight=1.0, unknown_root_fraction=0.0) == 0.0


def test_score_non_callsign_returns_zero() -> None:
    for word in ["HELLO", "CQ", "599", "K", "K1ABCDE", ""]:
        assert score_callsign(word) == 0.0


def test_score_scales_linearly_with_weight() -> None:
    base = score_callsign("F4HYY", weight=1.0)
    scaled = score_callsign("F4HYY", weight=2.5)
    assert scaled == pytest.approx(base * 2.5)


def test_score_zero_weight_returns_zero_even_for_known_root() -> None:
    assert score_callsign("F4HYY", weight=0.0) == 0.0


def _pick_unknown_root_callsign() -> str:
    """Find a structurally valid callsign whose root is NOT in the DXCC
    table. Returned dynamically so the test does not silently break when
    the DXCC list grows."""
    # 2-letter-digit roots: easy to enumerate, almost all are unused.
    for first in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        for second in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            for digit in "1234567890":
                root = f"{first}{second}{digit}"
                candidate = f"{root}XYZ"
                if not is_callsign_shape(candidate):
                    continue
                if root in KNOWN_ROOTS:
                    continue
                if root[:-1] in KNOWN_ROOTS:
                    continue
                return candidate
    raise AssertionError("could not find an unknown structural callsign")
