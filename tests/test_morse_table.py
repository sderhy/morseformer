"""Sanity checks on the Morse code table."""

from __future__ import annotations

import pytest

from morseformer.core.morse_table import (
    INVERSE_TABLE,
    MORSE_TABLE,
    decode_code,
    encode,
    unit_seconds,
)


def test_full_alphabet_present() -> None:
    for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789":
        assert ch in MORSE_TABLE, f"missing {ch!r}"


def test_inverse_table_is_unique() -> None:
    # Each dit/dah pattern maps to exactly one character.
    assert len(INVERSE_TABLE) == len(MORSE_TABLE)


def test_known_codes() -> None:
    assert MORSE_TABLE["S"] == "..."
    assert MORSE_TABLE["O"] == "---"
    assert MORSE_TABLE["K"] == "-.-"
    assert MORSE_TABLE["5"] == "....."


def test_encode_word_boundary() -> None:
    codes = encode("SOS HELP")
    assert codes == ["...", "---", "...", "", "....", ".", ".-..", ".--."]


def test_encode_skips_unknown() -> None:
    codes = encode("A~B")
    # Tilde is not in the table; it is silently dropped.
    assert codes == [".-", "-..."]


def test_decode_code_roundtrip() -> None:
    for ch, code in MORSE_TABLE.items():
        assert decode_code(code) == ch


def test_unit_seconds_paris() -> None:
    # 20 WPM → 1 dot-unit = 60 ms.
    assert unit_seconds(20) == pytest.approx(0.06)
    # 60 WPM → 1 dot-unit = 20 ms.
    assert unit_seconds(60) == pytest.approx(0.02)


def test_unit_seconds_invalid() -> None:
    with pytest.raises(ValueError):
        unit_seconds(0)
    with pytest.raises(ValueError):
        unit_seconds(-5)
