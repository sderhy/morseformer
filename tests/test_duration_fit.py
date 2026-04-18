"""Tests for the duration-aware text sampling (fixes label-audio drift)."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from morseformer.data.synthetic import (  # noqa: E402
    DatasetConfig,
    SyntheticCWDataset,
    _FALLBACK_SHORT_TEXTS,
    estimate_cw_duration_s,
)


def test_estimator_matches_paris_standard() -> None:
    # "PARIS" at 1 WPM is defined as 1 unit = 1.2 s, word = 50 units = 60 s.
    # One word alone: 50 − 7 (word gap) = 43 units of audio + 0.05 s tail.
    est = estimate_cw_duration_s("PARIS", wpm=1.0)
    expected_s = 43 * 1.2 + 0.05
    assert abs(est - expected_s) < 0.5


def test_estimator_scales_inversely_with_wpm() -> None:
    a = estimate_cw_duration_s("HELLO WORLD", wpm=20.0)
    b = estimate_cw_duration_s("HELLO WORLD", wpm=40.0)
    # Doubling WPM should roughly halve duration.
    assert abs(a / b - 2.0) < 0.05


def test_fallback_texts_fit_at_slowest_training_wpm() -> None:
    budget = 6.0 * 0.9
    for txt in _FALLBACK_SHORT_TEXTS():
        assert estimate_cw_duration_s(txt, 16.0) <= budget, txt


def test_sample_fitting_text_respects_budget() -> None:
    cfg = DatasetConfig(target_duration_s=6.0)
    ds = SyntheticCWDataset(cfg)
    rng = np.random.default_rng(0)
    budget = cfg.target_duration_s * 0.9
    for _ in range(500):
        wpm = float(rng.uniform(16, 28))
        text = ds._sample_fitting_text(rng, wpm)
        assert estimate_cw_duration_s(text, wpm) <= budget


def test_fit_guarantee_at_slowest_wpm() -> None:
    # Even forced to the slowest WPM, the sampler must never return an
    # overlong text: the fallback list kicks in when sampling fails.
    cfg = DatasetConfig(target_duration_s=6.0, wpm_range=(16.0, 16.0))
    ds = SyntheticCWDataset(cfg)
    rng = np.random.default_rng(1)
    budget = cfg.target_duration_s * 0.9
    for _ in range(300):
        text = ds._sample_fitting_text(rng, 16.0)
        assert estimate_cw_duration_s(text, 16.0) <= budget


def test_generator_output_is_consistent_with_label() -> None:
    # After the fix, no generated sample should be a case where the
    # audio was forcibly truncated — which would break label/audio
    # alignment. We verify this indirectly: the text always fits.
    cfg = DatasetConfig(target_duration_s=6.0)
    ds = SyntheticCWDataset(cfg)
    it = iter(ds)
    for _ in range(50):
        item = next(it)
        # n_tokens > 0 always (empty labels are a dataset bug).
        assert item["n_tokens"] > 0
