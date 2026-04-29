"""Unit tests for the synthetic PyTorch dataset."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import numpy as np  # noqa: E402

from morseformer.core.tokenizer import BLANK_INDEX, VOCAB_SIZE, decode  # noqa: E402
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
        assert (t < VOCAB_SIZE).all()
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
    assert log_probs.shape == (4, 750, VOCAB_SIZE)

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


# --------------------------------------------------------------------- #
# Phase 3.2 preset + 3-mode empty samples
# --------------------------------------------------------------------- #


def test_phase_3_2_preset_runs_and_yields_valid_items() -> None:
    cfg = DatasetConfig.phase_3_2(seed=7)
    # Sanity: preset values are wired.
    assert cfg.empty_sample_probability == 0.20
    assert cfg.text_mix.random == 0.30
    assert cfg.channel_probability == 1.0
    ds = SyntheticCWDataset(cfg)
    it = iter(ds)
    n_empty = n_full = 0
    for _ in range(40):
        item = next(it)
        # Audio shape is identical for empty and non-empty samples.
        assert item["features"].shape == (3000, 1)
        assert item["features"].dtype == torch.float32
        if item["n_tokens"] == 0:
            n_empty += 1
        else:
            n_full += 1
            t = item["tokens"]
            assert (t > 0).all()
            assert (t < VOCAB_SIZE).all()
    # Both branches reachable.
    assert n_empty > 0, "no empty samples drawn from phase_3_2 preset"
    assert n_full > 0


def test_phase_3_2_empty_modes_all_reachable() -> None:
    """The 3-mode empty sampler should hit each branch within a fixed
    seed budget. Modes are uniform 1/3 each, so 60 empties should cover
    all three with overwhelming probability."""
    cfg = DatasetConfig.phase_3_2(seed=11)
    ds = SyntheticCWDataset(cfg)
    rng = np.random.default_rng(0)
    audios: list[np.ndarray] = []
    for _ in range(60):
        a, toks = ds._empty_sample_features(rng)
        assert toks == []
        assert a.shape == (cfg.target_samples,)
        assert a.dtype == np.float32
        audios.append(a)
    # Modes can be loosely separated by RMS — pure AWGN ≈ 0.1, AWGN+QRN
    # higher (clicks add power), distant CW similar to AWGN. Just sanity
    # check that we get a spread of envelope statistics.
    rms = np.array([float(np.sqrt(np.mean(a**2))) for a in audios])
    # All finite, none zero (signal is at least the RX-filtered noise).
    assert np.all(np.isfinite(rms))
    assert np.all(rms > 0)
    # Spread > some threshold proves the modes differ.
    assert rms.std() > 0.005, f"RMS spread too tight: std={rms.std():.5f}"


def test_phase_3_2_empty_samples_have_empty_tokens() -> None:
    cfg = DatasetConfig.phase_3_2(
        seed=13, empty_sample_probability=1.0,  # force empty branch
    )
    ds = SyntheticCWDataset(cfg)
    it = iter(ds)
    for _ in range(20):
        item = next(it)
        assert item["n_tokens"] == 0
        assert item["tokens"].numel() == 0
        # Audio shape unaffected.
        assert item["features"].shape == (3000, 1)


# --------------------------------------------------------------------- #
# Phase 3.6 preset + post-emission silence
# --------------------------------------------------------------------- #


def test_phase_3_6_preset_wires_new_knobs() -> None:
    cfg = DatasetConfig.phase_3_6(seed=7)
    assert cfg.empty_sample_probability == 0.20
    assert cfg.post_emission_silence_probability == 0.10
    assert cfg.text_mix.adversarial_fr == 0.06
    assert cfg.operator_element_jitter_range == (0.0, 0.15)
    assert cfg.operator_gap_jitter_range == (0.0, 0.25)


def test_phase_3_6_preset_yields_valid_items() -> None:
    cfg = DatasetConfig.phase_3_6(seed=9)
    ds = SyntheticCWDataset(cfg)
    it = iter(ds)
    n_empty = n_full = 0
    for _ in range(60):
        item = next(it)
        assert item["features"].shape == (3000, 1)
        assert item["features"].dtype == torch.float32
        if item["n_tokens"] == 0:
            n_empty += 1
        else:
            n_full += 1
            t = item["tokens"]
            assert (t >= 0).all()
            assert (t < VOCAB_SIZE).all()
    assert n_empty > 0
    assert n_full > 0


def test_post_emission_silence_branch_returns_short_non_empty() -> None:
    """Forcing post_emission_silence_probability=1.0 should yield only
    short, non-empty utterances — the distinguishing feature of the
    branch vs the empty-audio branch."""
    cfg = DatasetConfig.phase_3_6(
        seed=11,
        empty_sample_probability=0.0,
        post_emission_silence_probability=1.0,
    )
    ds = SyntheticCWDataset(cfg)
    it = iter(ds)
    for _ in range(20):
        item = next(it)
        assert item["features"].shape == (3000, 1)
        # Real tokens (non-empty), and short by construction.
        assert item["n_tokens"] > 0
        lo, hi = cfg.post_emission_silence_text_chars
        # Token count should be close to char count; allow some slack
        # because the fallback short-text pool can include up to 5 chars.
        assert item["n_tokens"] <= hi + 1
