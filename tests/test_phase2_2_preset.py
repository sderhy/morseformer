"""Test the phase_2_2 preset — Phase 2.1 with widened operator jitter."""

from __future__ import annotations

from morseformer.data.synthetic import DatasetConfig


def test_phase_2_2_widens_jitter() -> None:
    cfg = DatasetConfig.phase_2_2()
    # Keeps Phase 2.1's channel settings.
    assert cfg.channel_probability == 1.0
    assert cfg.snr_db_range == (0.0, 30.0)
    assert cfg.rx_filter_bw == 500.0
    # Widens jitter ranges vs Phase 2.1.
    assert cfg.operator_element_jitter_range == (0.0, 0.12)
    assert cfg.operator_gap_jitter_range == (0.0, 0.20)


def test_phase_2_2_spans_benchmark_operator_profile() -> None:
    # eval.datasets.generate_noisy uses element_jitter=0.08, gap_jitter=0.15.
    # Our training range must cover these values.
    cfg = DatasetConfig.phase_2_2()
    assert cfg.operator_element_jitter_range[0] <= 0.08 <= cfg.operator_element_jitter_range[1]
    assert cfg.operator_gap_jitter_range[0] <= 0.15 <= cfg.operator_gap_jitter_range[1]


def test_phase_2_2_overrides_respected() -> None:
    cfg = DatasetConfig.phase_2_2(
        snr_db_range=(-5.0, 15.0),
        operator_element_jitter_range=(0.02, 0.10),
    )
    assert cfg.snr_db_range == (-5.0, 15.0)
    assert cfg.operator_element_jitter_range == (0.02, 0.10)
    # Non-overridden fields keep Phase 2.2 defaults.
    assert cfg.operator_gap_jitter_range == (0.0, 0.20)
