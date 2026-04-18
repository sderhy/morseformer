"""Tests for the SNR-ladder benchmark harness."""

from __future__ import annotations

from eval.snr_ladder import run_snr_ladder
from morseformer.baselines import rule_based


def test_ladder_populates_all_snrs() -> None:
    snrs = [float("inf"), 10.0, 0.0]
    ladder = run_snr_ladder(rule_based.decode, snrs, n_per_snr=4, seed=1)
    assert ladder.snrs_db == snrs
    for snr in snrs:
        assert snr in ladder.per_snr
        assert ladder.per_snr[snr].n_samples == 4


def test_ladder_cer_degrades_with_lower_snr() -> None:
    snrs = [float("inf"), 20.0, 0.0, -15.0]
    ladder = run_snr_ladder(rule_based.decode, snrs, n_per_snr=6, seed=7)
    cers = [ladder.per_snr[snr].mean_cer for snr in snrs]
    # Monotonic non-decrease in CER as SNR falls. Allow a small slack
    # between adjacent bins (noisy finite-sample estimates) but require
    # the extremes to differ.
    assert cers[0] <= cers[-1]
    assert cers[-1] > cers[0] + 0.05


def test_ladder_rows_and_format() -> None:
    ladder = run_snr_ladder(rule_based.decode, [10.0, 0.0], n_per_snr=3, seed=2)
    rows = ladder.as_rows()
    assert len(rows) == 2
    assert {row["snr_db"] for row in rows} == {10.0, 0.0}
    table = ladder.format_table()
    assert "SNR (dB)" in table
    assert "CER" in table
