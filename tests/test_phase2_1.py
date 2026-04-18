"""Tests for the Phase 2.1 dataset extensions: channel noise,
operator jitter, and SNR-laddered validation."""

from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from morseformer.data.synthetic import (  # noqa: E402
    DatasetConfig,
    SyntheticCWDataset,
)
from morseformer.data.validation import (  # noqa: E402
    ValidationConfig,
    build_clean_validation,
    build_snr_ladder_validation,
)


# --------------------------------------------------------------------- #
# DatasetConfig presets
# --------------------------------------------------------------------- #


def test_phase_2_0_preset_is_clean() -> None:
    cfg = DatasetConfig.phase_2_0()
    assert cfg.channel_probability == 0.0
    assert cfg.operator_element_jitter_range == (0.0, 0.0)
    assert cfg.operator_gap_jitter_range == (0.0, 0.0)
    assert cfg.rx_filter_bw is None


def test_phase_2_1_preset_enables_channel_and_jitter() -> None:
    cfg = DatasetConfig.phase_2_1()
    assert cfg.channel_probability == 1.0
    assert cfg.snr_db_range == (0.0, 30.0)
    assert cfg.rx_filter_bw == 500.0
    assert cfg.operator_element_jitter_range[1] > 0.0
    assert cfg.operator_gap_jitter_range[1] > 0.0


def test_phase_2_1_overrides_respected() -> None:
    cfg = DatasetConfig.phase_2_1(snr_db_range=(-5.0, 5.0), rx_filter_bw=300.0)
    assert cfg.snr_db_range == (-5.0, 5.0)
    assert cfg.rx_filter_bw == 300.0


# --------------------------------------------------------------------- #
# Noise / jitter actually affects the audio
# --------------------------------------------------------------------- #


def _first_features(cfg: DatasetConfig) -> torch.Tensor:
    ds = SyntheticCWDataset(cfg)
    return next(iter(ds))["features"]


def test_channel_changes_features() -> None:
    # Same seed, same WPM range, but one run is clean and one is noisy.
    # The feature tensors must differ.
    clean_cfg = DatasetConfig(seed=123)
    noisy_cfg = DatasetConfig(
        seed=123,
        channel_probability=1.0,
        snr_db_range=(5.0, 5.0),   # deterministic moderate SNR
    )
    clean = _first_features(clean_cfg)
    noisy = _first_features(noisy_cfg)
    assert clean.shape == noisy.shape
    assert not torch.allclose(clean, noisy, atol=1e-3)


def test_channel_probability_zero_matches_clean() -> None:
    # With channel_probability=0.0 the stream must match the Phase 2.0
    # default bit-for-bit, even if snr_db_range is set.
    a = DatasetConfig(seed=7)
    b = DatasetConfig(seed=7, channel_probability=0.0, snr_db_range=(-20.0, -20.0))
    fa = _first_features(a)
    fb = _first_features(b)
    torch.testing.assert_close(fa, fb)


def test_jitter_introduces_timing_variance() -> None:
    # With zero jitter, repeatedly rendering the same text+wpm should
    # yield identical audio. With non-zero jitter it should not.
    from morse_synth.core import render
    from morse_synth.operator import OperatorConfig

    def _render_with_seed(op_seed: int, element_jitter: float) -> np.ndarray:
        return render(
            "HELLO CQ", operator=OperatorConfig(
                wpm=20.0, element_jitter=element_jitter, seed=op_seed,
            ), channel=None, sample_rate=8000,
        )

    a = _render_with_seed(0, 0.0)
    b = _render_with_seed(1, 0.0)
    # Zero jitter: seed doesn't matter, outputs identical.
    assert a.shape == b.shape and np.array_equal(a, b)

    c = _render_with_seed(0, 0.05)
    d = _render_with_seed(1, 0.05)
    # Non-zero jitter: different seeds → different audio.
    assert c.shape[0] != d.shape[0] or not np.array_equal(c, d)


def test_mixed_probability_produces_both_kinds() -> None:
    # At 50% channel probability the per-sample Bernoulli should produce
    # roughly equal numbers of clean (None) and channel-applied
    # (ChannelConfig) outcomes. Tested directly on the helper so the
    # result is independent of the per-utterance feature normalisation.
    cfg = DatasetConfig(
        seed=0,
        channel_probability=0.5,
        snr_db_range=(0.0, 0.0),
    )
    ds = SyntheticCWDataset(cfg)
    rng = np.random.default_rng(0)
    n = 400
    applied = sum(ds._sample_channel(rng) is not None for _ in range(n))
    frac = applied / n
    assert 0.4 <= frac <= 0.6, f"50% channel prob gave {frac:.2f}"


def test_zero_probability_never_applies_channel() -> None:
    ds = SyntheticCWDataset(DatasetConfig(seed=0, channel_probability=0.0))
    rng = np.random.default_rng(0)
    assert all(ds._sample_channel(rng) is None for _ in range(100))


def test_full_probability_always_applies_channel() -> None:
    ds = SyntheticCWDataset(DatasetConfig(
        seed=0, channel_probability=1.0, snr_db_range=(5.0, 25.0)
    ))
    rng = np.random.default_rng(0)
    for _ in range(100):
        ch = ds._sample_channel(rng)
        assert ch is not None
        assert 5.0 <= ch.snr_db <= 25.0


# --------------------------------------------------------------------- #
# SNR-laddered validation builder
# --------------------------------------------------------------------- #


def test_snr_ladder_covers_full_grid() -> None:
    cfg = ValidationConfig(n_per_wpm=2, wpm_bins=(20.0, 25.0))
    snrs = (10.0, 0.0, -5.0)
    samples = build_snr_ladder_validation(snrs, cfg=cfg)
    assert len(samples) == 2 * 2 * 3
    cells = Counter((s.wpm, s.snr_db) for s in samples)
    for wpm in (20.0, 25.0):
        for snr in snrs:
            assert cells[(wpm, snr)] == 2


def test_snr_ladder_is_deterministic() -> None:
    snrs = (10.0, 0.0)
    cfg = ValidationConfig(n_per_wpm=3, seed=42, wpm_bins=(20.0,))
    a = build_snr_ladder_validation(snrs, cfg=cfg)
    b = build_snr_ladder_validation(snrs, cfg=cfg)
    for sa, sb in zip(a, b):
        torch.testing.assert_close(sa.features, sb.features)
        assert sa.text == sb.text
        assert sa.snr_db == sb.snr_db


def test_snr_ladder_samples_really_include_noise() -> None:
    # At a very low SNR the audio must differ from the clean equivalent.
    cfg = ValidationConfig(n_per_wpm=1, wpm_bins=(22.0,), seed=0)
    clean = build_clean_validation(cfg)
    ladder = build_snr_ladder_validation((-10.0,), cfg=cfg, rx_filter_bw=None)
    # Same seed/bin → same text; features must diverge due to noise.
    assert clean[0].text == ladder[0].text
    assert not torch.allclose(clean[0].features, ladder[0].features, atol=1e-3)


def test_clean_validation_snr_field_is_inf() -> None:
    for s in build_clean_validation(ValidationConfig(n_per_wpm=1)):
        assert math.isinf(s.snr_db)


def test_n_per_cell_override() -> None:
    cfg = ValidationConfig(n_per_wpm=3, wpm_bins=(20.0, 25.0))
    samples = build_snr_ladder_validation(
        (10.0,), cfg=cfg, n_per_cell=5
    )
    assert len(samples) == 2 * 1 * 5
