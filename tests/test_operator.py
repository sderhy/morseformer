"""Tests for the parametric operator model."""

from __future__ import annotations

import numpy as np
import pytest

from morse_synth.operator import OperatorConfig, build_events


def _total_duration(events) -> float:
    return sum(d for _, d in events)


def test_zero_jitter_matches_deterministic_timing() -> None:
    cfg = OperatorConfig(wpm=20.0)
    ev = build_events("E", cfg)
    # Single dit: one ON event, one dot-unit long.
    assert len(ev) == 1
    assert ev[0][0] is True
    assert ev[0][1] == pytest.approx(0.06)  # 1.2 / 20


def test_word_boundary_is_seven_units() -> None:
    cfg = OperatorConfig(wpm=20.0)
    ev = build_events("E E", cfg)
    unit = 0.06
    # Expected sequence: ON 1u, OFF 7u, ON 1u
    assert len(ev) == 3
    assert ev[1] == (False, pytest.approx(7 * unit))


def test_seed_is_reproducible() -> None:
    ev_a = build_events("PARIS", OperatorConfig(wpm=20, element_jitter=0.3, gap_jitter=0.3, seed=7))
    ev_b = build_events("PARIS", OperatorConfig(wpm=20, element_jitter=0.3, gap_jitter=0.3, seed=7))
    assert ev_a == ev_b


def test_different_seeds_give_different_timings() -> None:
    ev_a = build_events("PARIS", OperatorConfig(wpm=20, element_jitter=0.3, seed=1))
    ev_b = build_events("PARIS", OperatorConfig(wpm=20, element_jitter=0.3, seed=2))
    assert ev_a != ev_b


def test_jitter_preserves_event_count() -> None:
    # Timing noise must never drop or insert events — only move them in time.
    ref = build_events("HELLO WORLD", OperatorConfig(wpm=20))
    jittered = build_events("HELLO WORLD", OperatorConfig(wpm=20, element_jitter=0.3, gap_jitter=0.3, seed=42))
    assert len(ref) == len(jittered)
    # Same ON/OFF pattern.
    assert [e[0] for e in ref] == [e[0] for e in jittered]


def test_farnsworth_slows_message() -> None:
    base = build_events("HELLO", OperatorConfig(wpm=20))
    slow = build_events("HELLO", OperatorConfig(wpm=20, farnsworth_char_gap=6.0))
    assert _total_duration(slow) > _total_duration(base)


def test_unknown_chars_dropped() -> None:
    ev = build_events("A~B", OperatorConfig(wpm=20))
    # "A" (.-) has 3 events (on-off-on), "B" (-...) has 7 events.
    # Plus one inter-char gap between them => 3 + 1 + 7 = 11 events.
    assert len(ev) == 11


def test_empty_text_returns_empty() -> None:
    assert build_events("", OperatorConfig()) == []
    assert build_events("   ", OperatorConfig()) == []


def test_invalid_wpm() -> None:
    with pytest.raises(ValueError):
        build_events("E", OperatorConfig(wpm=0))
