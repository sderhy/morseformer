"""Unit tests for CER / WER / callsign scoring."""

from __future__ import annotations

import pytest

from eval.metrics import (
    callsign_scores,
    character_error_rate,
    word_error_rate,
)


def test_cer_perfect() -> None:
    assert character_error_rate("HELLO", "HELLO") == 0.0


def test_cer_case_insensitive() -> None:
    assert character_error_rate("Hello", "HELLO") == 0.0


def test_cer_single_substitution() -> None:
    # 1 edit / 5 characters
    assert character_error_rate("HELLO", "HELXO") == pytest.approx(0.2)


def test_cer_empty_ref() -> None:
    assert character_error_rate("", "") == 0.0
    assert character_error_rate("", "ANYTHING") == 1.0


def test_cer_empty_hyp() -> None:
    assert character_error_rate("ABC", "") == 1.0


def test_wer_perfect() -> None:
    assert word_error_rate("HELLO WORLD", "HELLO WORLD") == 0.0


def test_wer_one_word_wrong() -> None:
    assert word_error_rate("HELLO WORLD", "HELLO EARTH") == pytest.approx(0.5)


def test_callsign_perfect() -> None:
    r = callsign_scores("CQ DE F5ABC K", "CQ DE F5ABC K")
    assert r.f1 == 1.0
    assert r.precision == 1.0
    assert r.recall == 1.0


def test_callsign_miss() -> None:
    r = callsign_scores("CQ DE F5ABC K", "CQ DE K")
    assert r.recall == 0.0
    assert r.f1 == 0.0


def test_callsign_spurious() -> None:
    r = callsign_scores("CQ DE K", "CQ DE F5ABC K")
    assert r.precision == 0.0
    assert r.f1 == 0.0


def test_callsign_partial() -> None:
    # One right, one spurious.
    r = callsign_scores("DE F5ABC K W1ABC", "DE F5ABC K G0XYZ")
    # ref = {F5ABC, W1ABC}, hyp = {F5ABC, G0XYZ}; tp = 1.
    assert r.precision == pytest.approx(0.5)
    assert r.recall == pytest.approx(0.5)
    assert r.f1 == pytest.approx(0.5)


def test_callsign_both_empty() -> None:
    r = callsign_scores("HELLO WORLD", "HELLO WORLD")
    assert r.f1 == 1.0
