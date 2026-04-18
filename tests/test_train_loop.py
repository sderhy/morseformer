"""Smoke test for the training loop: a short run must reduce the CTC loss."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from morseformer.models.acoustic import AcousticConfig  # noqa: E402
from morseformer.data.synthetic import DatasetConfig  # noqa: E402
from morseformer.data.validation import ValidationConfig  # noqa: E402
from morseformer.train.acoustic import TrainConfig, train  # noqa: E402


def test_loss_descends_on_short_run(tmp_path: Path) -> None:
    # Tiny model + tiny val set + short schedule — this is a sanity check
    # that loss descends, not a quality benchmark.
    cfg = TrainConfig(
        model=AcousticConfig(
            d_model=32, n_heads=4, n_layers=2,
            ff_expansion=2, conv_kernel=7, dropout=0.0,
        ),
        dataset=DatasetConfig(seed=0),
        validation=ValidationConfig(n_per_wpm=2, seed=1),
        peak_lr=1e-3,
        warmup_steps=10,
        total_steps=80,
        batch_size=8,
        num_workers=0,
        log_every=10,
        eval_every=40,
        checkpoint_dir=tmp_path,
        jsonl_log=tmp_path / "train.jsonl",
        ema_decay=0.99,
    )
    result = train(cfg)
    assert result["steps"] == 80

    # Pull the loss trace from the JSONL and confirm it trends down.
    losses: list[float] = []
    with (tmp_path / "train.jsonl").open() as f:
        for line in f:
            evt = json.loads(line)
            if evt.get("event") == "step":
                losses.append(evt["loss"])
    assert len(losses) >= 6
    assert losses[-1] < losses[0], (
        f"loss did not descend: start={losses[0]:.3f} end={losses[-1]:.3f}"
    )

    # At least one checkpoint was written.
    assert (tmp_path / "last.pt").exists()
    # best_cer.pt is written only if val improved; on this tiny run it
    # should always write at least one eval.
    assert (tmp_path / "best_cer.pt").exists()


def test_checkpoint_roundtrip(tmp_path: Path) -> None:
    cfg = TrainConfig(
        model=AcousticConfig(
            d_model=16, n_heads=2, n_layers=1,
            ff_expansion=2, conv_kernel=7, dropout=0.0,
        ),
        dataset=DatasetConfig(seed=2),
        validation=ValidationConfig(n_per_wpm=1, seed=3),
        warmup_steps=5,
        total_steps=20,
        batch_size=4,
        log_every=5,
        eval_every=20,
        checkpoint_dir=tmp_path,
        jsonl_log=tmp_path / "train.jsonl",
    )
    train(cfg)
    ckpt = torch.load(tmp_path / "last.pt", map_location="cpu", weights_only=False)
    assert ckpt["step"] == 20
    assert "model" in ckpt and "ema" in ckpt
    assert "metrics" in ckpt
    # CER is finite and non-negative. Magnitude is meaningless for a
    # 20-step tiny model — it often hallucinates many extra characters.
    import math
    assert math.isfinite(ckpt["metrics"]["cer"])
    assert ckpt["metrics"]["cer"] >= 0.0
