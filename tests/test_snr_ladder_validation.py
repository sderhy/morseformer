"""Tests for the SNR-ladder validation hook in the training loop."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from morseformer.data.synthetic import DatasetConfig  # noqa: E402
from morseformer.data.validation import ValidationConfig  # noqa: E402
from morseformer.models.acoustic import AcousticConfig  # noqa: E402
from morseformer.train.acoustic import TrainConfig, train  # noqa: E402


def test_empty_validation_snrs_uses_clean_val(tmp_path: Path) -> None:
    cfg = TrainConfig(
        model=AcousticConfig(d_model=16, n_heads=2, n_layers=1,
                              ff_expansion=2, conv_kernel=7, dropout=0.0),
        dataset=DatasetConfig(seed=0),
        validation=ValidationConfig(n_per_wpm=1, seed=1),
        warmup_steps=5, total_steps=15, batch_size=4,
        log_every=5, eval_every=15,
        checkpoint_dir=tmp_path, jsonl_log=tmp_path / "train.jsonl",
    )
    assert cfg.validation_snrs == ()
    train(cfg)

    with (tmp_path / "train.jsonl").open() as f:
        eval_events = [
            json.loads(line) for line in f
            if json.loads(line).get("event") == "eval"
        ]
    assert len(eval_events) >= 1
    # Clean validation: one SNR key "inf", n_samples = n_per_wpm * n_wpm_bins.
    ev = eval_events[0]
    assert list(ev["per_snr_cer"]) == ["inf"]
    assert ev["n_samples"] == 1 * len(ValidationConfig().wpm_bins)


def test_nonempty_validation_snrs_uses_ladder(tmp_path: Path) -> None:
    snrs = (10.0, 0.0)
    cfg = TrainConfig(
        model=AcousticConfig(d_model=16, n_heads=2, n_layers=1,
                              ff_expansion=2, conv_kernel=7, dropout=0.0),
        dataset=DatasetConfig.phase_2_1(seed=0),
        validation=ValidationConfig(n_per_wpm=1, seed=1,
                                    wpm_bins=(20.0, 25.0)),
        validation_snrs=snrs,
        validation_rx_filter_bw=500.0,
        warmup_steps=5, total_steps=15, batch_size=4,
        log_every=5, eval_every=15,
        checkpoint_dir=tmp_path, jsonl_log=tmp_path / "train.jsonl",
    )
    train(cfg)

    with (tmp_path / "train.jsonl").open() as f:
        eval_events = [
            json.loads(line) for line in f
            if json.loads(line).get("event") == "eval"
        ]
    ev = eval_events[0]
    # WPM bins × SNR bins × n_per_wpm = 2 × 2 × 1 = 4 samples.
    assert ev["n_samples"] == 4
    # JSON keys are strings; compare the numeric parse of each.
    snrs_reported = sorted(float(k) for k in ev["per_snr_cer"])
    assert snrs_reported == sorted(snrs)
    wpms_reported = sorted(float(k) for k in ev["per_wpm_cer"])
    assert wpms_reported == [20.0, 25.0]
