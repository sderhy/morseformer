"""Unit tests for the warmup+cosine LR schedule."""

from __future__ import annotations

import math

import pytest

from morseformer.train.scheduler import WarmupCosineSchedule


def test_warmup_is_linear() -> None:
    s = WarmupCosineSchedule(warmup_steps=100, total_steps=1000)
    assert s(0) == 0.0
    assert s(50) == pytest.approx(0.5)
    assert s(100) == pytest.approx(1.0)


def test_cosine_decay_reaches_floor() -> None:
    s = WarmupCosineSchedule(warmup_steps=100, total_steps=1000, min_lr_ratio=0.1)
    assert s(1000) == pytest.approx(0.1)
    assert s(1500) == pytest.approx(0.1)  # clamped after total_steps


def test_cosine_midpoint() -> None:
    # At the halfway point of the cosine phase (relative progress 0.5),
    # the multiplier should be (1 + min_ratio) / 2.
    s = WarmupCosineSchedule(warmup_steps=0, total_steps=1000, min_lr_ratio=0.1)
    mid = s(500)
    expected = 0.1 + (1.0 - 0.1) * 0.5 * (1 + math.cos(math.pi * 0.5))
    assert mid == pytest.approx(expected)


def test_schedule_is_monotone_decreasing_after_warmup() -> None:
    s = WarmupCosineSchedule(warmup_steps=100, total_steps=1000, min_lr_ratio=0.0)
    last = s(100)
    for step in range(101, 1001, 10):
        cur = s(step)
        assert cur <= last + 1e-9, f"LR increased at step {step}: {last}→{cur}"
        last = cur


def test_zero_warmup_starts_at_peak() -> None:
    s = WarmupCosineSchedule(warmup_steps=0, total_steps=100)
    assert s(0) == pytest.approx(1.0)


def test_validation_errors() -> None:
    with pytest.raises(ValueError):
        WarmupCosineSchedule(warmup_steps=-1, total_steps=100)
    with pytest.raises(ValueError):
        WarmupCosineSchedule(warmup_steps=50, total_steps=50)
    with pytest.raises(ValueError):
        WarmupCosineSchedule(warmup_steps=10, total_steps=100, min_lr_ratio=-0.1)
    with pytest.raises(ValueError):
        WarmupCosineSchedule(warmup_steps=10, total_steps=100, min_lr_ratio=1.5)
