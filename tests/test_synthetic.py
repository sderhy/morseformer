"""Unit tests for the synthetic PyTorch dataset."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import numpy as np  # noqa: E402

from morseformer.core.tokenizer import BLANK_INDEX, decode  # noqa: E402
from morseformer.data.synthetic import (  # noqa: E402
    DatasetConfig,
    SyntheticCWDataset,
    _pad_or_truncate,
    _worker_seed,
    collate,
)
from morseformer.features import FrontendConfig  # noqa: E402


# --------------------------------------------------------------------- #
# DatasetConfig
# --------------------------------------------------------------------- #


def test_target_samples_computation() -> None:
    cfg = DatasetConfig(target_duration_s=6.0, sample_rate=8000)
    assert cfg.target_samples == 48_000


def test_incompatible_sample_rate_rejected() -> None:
    bad = DatasetConfig(
        sample_rate=7999,
        frontend=FrontendConfig(frame_rate=500),
    )
    with pytest.raises(ValueError):
        SyntheticCWDataset(bad)


# --------------------------------------------------------------------- #
# Single-item shape / dtype
# --------------------------------------------------------------------- #


def test_item_shapes_and_dtypes() -> None:
    cfg = DatasetConfig(seed=0)
    ds = SyntheticCWDataset(cfg)
    it = iter(ds)
    for _ in range(10):
        item = next(it)
        assert item["features"].shape == (3000, 1)
        assert item["features"].dtype == torch.float32
        assert item["tokens"].dtype == torch.int64
        assert item["tokens"].ndim == 1
        assert item["n_frames"] == 3000
        assert item["n_tokens"] == item["tokens"].shape[0]


def test_tokens_are_in_valid_range() -> None:
    cfg = DatasetConfig(seed=1)
    ds = SyntheticCWDataset(cfg)
    it = iter(ds)
    for _ in range(20):
        item = next(it)
        t = item["tokens"]
        # Non-blank, in [0, vocab_size) — the tokenizer never emits blank.
        assert (t >= 0).all()
        assert (t < 46).all()
        # encode() specifically never emits the blank token.
        assert (t != BLANK_INDEX).all()


def test_decoded_tokens_are_nonempty_text() -> None:
    cfg = DatasetConfig(seed=2)
    ds = SyntheticCWDataset(cfg)
    it = iter(ds)
    for _ in range(20):
        item = next(it)
        txt = decode(item["tokens"].tolist())
        assert len(txt) > 0


# --------------------------------------------------------------------- #
# Determinism / worker-safety
# --------------------------------------------------------------------- #


def test_same_seed_same_stream() -> None:
    cfg = DatasetConfig(seed=123)
    ds_a = SyntheticCWDataset(cfg)
    ds_b = SyntheticCWDataset(cfg)
    it_a, it_b = iter(ds_a), iter(ds_b)
    for _ in range(20):
        a = next(it_a)
        b = next(it_b)
        torch.testing.assert_close(a["features"], b["features"])
        assert a["tokens"].tolist() == b["tokens"].tolist()


def test_different_seed_different_stream() -> None:
    it_a = iter(SyntheticCWDataset(DatasetConfig(seed=0)))
    it_b = iter(SyntheticCWDataset(DatasetConfig(seed=1)))
    a_tokens = [tuple(next(it_a)["tokens"].tolist()) for _ in range(30)]
    b_tokens = [tuple(next(it_b)["tokens"].tolist()) for _ in range(30)]
    assert a_tokens != b_tokens


def test_worker_seeds_diverge() -> None:
    # Different worker ids should yield different seeds; same (base, id)
    # should be stable.
    s0 = _worker_seed(42, 0)
    s1 = _worker_seed(42, 1)
    s2 = _worker_seed(42, 2)
    assert s0 != s1 != s2 and s0 != s2
    assert _worker_seed(42, 0) == s0


# --------------------------------------------------------------------- #
# Duration handling
# --------------------------------------------------------------------- #


def test_pad_or_truncate_shorter() -> None:
    x = np.arange(10, dtype=np.float32)
    out = _pad_or_truncate(x, 16)
    assert out.shape == (16,)
    assert np.array_equal(out[:10], x)
    assert (out[10:] == 0).all()


def test_pad_or_truncate_longer() -> None:
    x = np.arange(20, dtype=np.float32)
    out = _pad_or_truncate(x, 16)
    assert np.array_equal(out, x[:16])


def test_pad_or_truncate_exact() -> None:
    x = np.arange(16, dtype=np.float32)
    out = _pad_or_truncate(x, 16)
    assert out is x  # no copy when already the right size


def test_fixed_duration_invariant_across_wpm() -> None:
    # Even at slow WPM (text often longer than audio window), the yielded
    # features always have the target frame count — audio was padded
    # or truncated to hit the duration exactly.
    cfg = DatasetConfig(seed=7, wpm_range=(16.0, 16.0))
    ds = SyntheticCWDataset(cfg)
    it = iter(ds)
    for _ in range(20):
        assert next(it)["features"].shape == (3000, 1)


# --------------------------------------------------------------------- #
# Collation
# --------------------------------------------------------------------- #


def test_collate_batch_shapes() -> None:
    cfg = DatasetConfig(seed=3)
    ds = SyntheticCWDataset(cfg)
    it = iter(ds)
    batch = collate([next(it) for _ in range(8)])
    assert batch["features"].shape == (8, 3000, 1)
    assert batch["features"].dtype == torch.float32
    assert batch["tokens"].shape[0] == 8
    assert batch["tokens"].dtype == torch.int64
    assert batch["n_frames"].shape == (8,)
    assert batch["n_tokens"].shape == (8,)
    assert (batch["n_frames"] == 3000).all()


def test_collate_pads_to_max_tokens() -> None:
    cfg = DatasetConfig(seed=4)
    ds = SyntheticCWDataset(cfg)
    it = iter(ds)
    items = [next(it) for _ in range(12)]
    max_l = max(i["n_tokens"] for i in items)
    batch = collate(items)
    assert batch["tokens"].shape == (12, max_l)
    # Every valid prefix matches the original tokens.
    for i, item in enumerate(items):
        n = item["n_tokens"]
        assert (batch["tokens"][i, :n] == item["tokens"]).all()
        # Padding is the blank index.
        if n < max_l:
            assert (batch["tokens"][i, n:] == BLANK_INDEX).all()


def test_collate_empty_batch_raises() -> None:
    with pytest.raises(ValueError):
        collate([])


# --------------------------------------------------------------------- #
# End-to-end: dataset → model → CTC loss
# --------------------------------------------------------------------- #


def test_batch_runs_through_model_and_ctc() -> None:
    from morseformer.models.acoustic import AcousticConfig, AcousticModel

    cfg = DatasetConfig(seed=5)
    ds = SyntheticCWDataset(cfg)
    it = iter(ds)
    batch = collate([next(it) for _ in range(4)])

    mcfg = AcousticConfig(
        d_model=32, n_heads=4, n_layers=2, ff_expansion=2, conv_kernel=7
    )
    model = AcousticModel(mcfg).train()
    log_probs, lengths_out = model(batch["features"], lengths=batch["n_frames"])
    assert lengths_out is not None
    assert log_probs.shape == (4, 750, 46)

    flat_targets = torch.cat([
        batch["tokens"][i, : batch["n_tokens"][i]] for i in range(4)
    ])
    loss = torch.nn.functional.ctc_loss(
        log_probs.transpose(0, 1),
        flat_targets,
        lengths_out,
        batch["n_tokens"],
        blank=mcfg.blank_index,
        zero_infinity=True,
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert model.head.weight.grad is not None
    assert torch.any(model.head.weight.grad != 0)
