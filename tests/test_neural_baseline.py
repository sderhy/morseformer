"""Tests for the neural-decoder baseline (load checkpoint → decode audio)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from eval.datasets import generate_sanity  # noqa: E402
from morseformer.baselines.neural import (  # noqa: E402
    NeuralDecoder,
    NeuralDecoderConfig,
)
from morseformer.data.synthetic import DatasetConfig  # noqa: E402
from morseformer.data.validation import ValidationConfig  # noqa: E402
from morseformer.models.acoustic import AcousticConfig  # noqa: E402
from morseformer.train.acoustic import TrainConfig, train  # noqa: E402


def _train_tiny_checkpoint(tmp_path: Path) -> Path:
    """Run a very short training to write a real checkpoint we can load.

    The model is tiny so this finishes in seconds on CPU. We do not
    assert any quality on the resulting decoder — only that it loads
    and returns strings without crashing.
    """
    cfg = TrainConfig(
        model=AcousticConfig(
            d_model=16, n_heads=2, n_layers=1,
            ff_expansion=2, conv_kernel=7, dropout=0.0,
        ),
        dataset=DatasetConfig(seed=0),
        validation=ValidationConfig(n_per_wpm=1, seed=1),
        warmup_steps=5,
        total_steps=15,
        batch_size=4,
        log_every=5,
        eval_every=15,
        checkpoint_dir=tmp_path,
        jsonl_log=tmp_path / "train.jsonl",
    )
    train(cfg)
    return tmp_path / "best_cer.pt"


def test_from_checkpoint_loads_cleanly(tmp_path: Path) -> None:
    ckpt = _train_tiny_checkpoint(tmp_path)
    decoder = NeuralDecoder.from_checkpoint(ckpt)
    assert decoder.train_sample_rate == 8000
    # Model is in eval mode.
    assert not decoder.model.training


def test_decoder_returns_string_on_audio(tmp_path: Path) -> None:
    ckpt = _train_tiny_checkpoint(tmp_path)
    decoder = NeuralDecoder.from_checkpoint(ckpt)
    samples = generate_sanity(n=3)
    for s in samples:
        out = decoder(s.audio, s.sample_rate)
        assert isinstance(out, str)  # may be empty or garbage — tiny model


def test_decoder_respects_sample_rate(tmp_path: Path) -> None:
    ckpt = _train_tiny_checkpoint(tmp_path)
    decoder = NeuralDecoder.from_checkpoint(ckpt)
    audio = np.zeros(8000, dtype=np.float32)
    with pytest.raises(ValueError):
        decoder(audio, sample_rate=16000)


def test_decoder_handles_empty_audio(tmp_path: Path) -> None:
    ckpt = _train_tiny_checkpoint(tmp_path)
    decoder = NeuralDecoder.from_checkpoint(ckpt)
    assert decoder(np.zeros(0, dtype=np.float32), sample_rate=8000) == ""


def test_no_ema_toggle_changes_weights(tmp_path: Path) -> None:
    # With/without EMA should load *different* weights (EMA weights
    # diverge from raw weights once `update` has been called).
    ckpt = _train_tiny_checkpoint(tmp_path)
    d_ema = NeuralDecoder.from_checkpoint(ckpt, NeuralDecoderConfig(use_ema=True))
    d_raw = NeuralDecoder.from_checkpoint(ckpt, NeuralDecoderConfig(use_ema=False))
    any_diff = False
    for (_, p_ema), (_, p_raw) in zip(
        d_ema.model.named_parameters(), d_raw.model.named_parameters()
    ):
        if not torch.allclose(p_ema, p_raw, atol=1e-8):
            any_diff = True
            break
    assert any_diff, "EMA and raw weights should differ after any training"


def test_decode_batch_matches_per_sample(tmp_path: Path) -> None:
    ckpt = _train_tiny_checkpoint(tmp_path)
    decoder = NeuralDecoder.from_checkpoint(ckpt)
    samples = generate_sanity(n=4)
    single = [decoder(s.audio, s.sample_rate) for s in samples]
    batched = decoder.decode_batch([s.audio for s in samples], sample_rate=8000)
    assert single == batched


def test_cli_smoke_end_to_end(tmp_path: Path) -> None:
    """Full CLI path: `python -m eval.cli --decoder neural --checkpoint ...`.

    Uses --json so we can parse the output structurally.
    """
    import json
    from eval.cli import main as cli_main

    ckpt = _train_tiny_checkpoint(tmp_path)
    # Redirect stdout to a file to capture JSON.
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = cli_main([
            "--decoder", "neural",
            "--checkpoint", str(ckpt),
            "--dataset", "sanity",
            "--n", "3",
            "--json",
        ])
    assert rc == 0
    payload = json.loads(buf.getvalue())
    assert payload["decoder"] == "neural"
    assert payload["n_samples"] == 3
    assert "mean_cer" in payload
