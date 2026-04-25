"""Unit tests for the deterministic validation-set builder."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from morseformer.core.tokenizer import decode  # noqa: E402
from morseformer.data.synthetic import DatasetConfig, collate  # noqa: E402
from morseformer.data.validation import (  # noqa: E402
    ValidationConfig,
    build_clean_validation,
    build_noise_only_validation,
)


def test_total_size_matches_bins() -> None:
    cfg = ValidationConfig(n_per_wpm=5, wpm_bins=(16.0, 22.0, 28.0))
    samples = build_clean_validation(cfg)
    assert len(samples) == 15
    assert cfg.total_size == 15


def test_all_samples_have_expected_shape() -> None:
    cfg = ValidationConfig(n_per_wpm=3)
    for s in build_clean_validation(cfg):
        assert s.features.shape == (3000, 1)
        assert s.features.dtype == torch.float32
        assert s.tokens.dtype == torch.int64
        assert s.n_frames == 3000
        assert s.n_tokens == s.tokens.shape[0] > 0
        assert len(s.text) > 0


def test_wpm_bins_are_covered_evenly() -> None:
    cfg = ValidationConfig(n_per_wpm=7, wpm_bins=(16.0, 20.0, 28.0))
    from collections import Counter
    counts = Counter(s.wpm for s in build_clean_validation(cfg))
    assert counts == {16.0: 7, 20.0: 7, 28.0: 7}


def test_set_is_deterministic_across_calls() -> None:
    cfg = ValidationConfig(n_per_wpm=4, seed=42)
    a = build_clean_validation(cfg)
    b = build_clean_validation(cfg)
    assert len(a) == len(b)
    for sa, sb in zip(a, b):
        torch.testing.assert_close(sa.features, sb.features)
        assert sa.tokens.tolist() == sb.tokens.tolist()
        assert sa.text == sb.text
        assert sa.wpm == sb.wpm


def test_different_seeds_give_different_sets() -> None:
    a = build_clean_validation(ValidationConfig(n_per_wpm=6, seed=1))
    b = build_clean_validation(ValidationConfig(n_per_wpm=6, seed=2))
    assert [s.text for s in a] != [s.text for s in b]


def test_text_and_tokens_are_consistent() -> None:
    for s in build_clean_validation(ValidationConfig(n_per_wpm=3)):
        assert decode(s.tokens.tolist()) == s.text


def test_samples_collate_cleanly() -> None:
    samples = build_clean_validation(ValidationConfig(n_per_wpm=3))
    batch = collate([s.as_batch_item() for s in samples[:8]])
    assert batch["features"].shape == (8, 3000, 1)
    assert batch["tokens"].shape[0] == 8
    assert (batch["n_frames"] == 3000).all()


def test_matching_inherits_dataset_cfg() -> None:
    from morseformer.features import FrontendConfig
    ds_cfg = DatasetConfig(
        target_duration_s=4.0,
        sample_rate=16000,
        freq_hz=700.0,
        frontend=FrontendConfig(frame_rate=1000),
    )
    val_cfg = ValidationConfig.matching(ds_cfg, n_per_wpm=2)
    assert val_cfg.target_duration_s == 4.0
    assert val_cfg.sample_rate == 16000
    assert val_cfg.freq_hz == 700.0
    assert val_cfg.frontend.frame_rate == 1000
    assert val_cfg.n_per_wpm == 2


def test_noise_only_validation_has_empty_labels() -> None:
    """The false-positive bench must label every sample with an empty
    text + zero tokens, regardless of mode."""
    cfg = ValidationConfig(n_per_wpm=4, wpm_bins=(20.0, 24.0), seed=1)
    samples = build_noise_only_validation(cfg=cfg, n_per_mode=8)
    # 3 modes × 8 = 24 samples.
    assert len(samples) == 24
    for s in samples:
        assert s.text == ""
        assert s.n_tokens == 0
        assert s.tokens.numel() == 0
        # Audio shape matches the regular validation pipeline.
        assert s.features.shape == (3000, 1)
        assert s.features.dtype == torch.float32


def test_noise_only_validation_modes_produce_distinct_audio() -> None:
    """The 3 sub-modes must produce different feature sequences — no
    mode silently collapsed to the same content as another."""
    cfg = ValidationConfig(n_per_wpm=4, wpm_bins=(22.0,), seed=2)
    samples = build_noise_only_validation(cfg=cfg, n_per_mode=4)
    # samples[0..3] = mode 0, [4..7] = mode 1, [8..11] = mode 2.
    assert not torch.equal(samples[0].features, samples[4].features)
    assert not torch.equal(samples[0].features, samples[8].features)
    assert not torch.equal(samples[4].features, samples[8].features)


def test_noise_only_validation_is_deterministic() -> None:
    cfg = ValidationConfig(n_per_wpm=4, seed=42)
    a = build_noise_only_validation(cfg=cfg, n_per_mode=5)
    b = build_noise_only_validation(cfg=cfg, n_per_mode=5)
    assert len(a) == len(b)
    for sa, sb in zip(a, b):
        torch.testing.assert_close(sa.features, sb.features)


def test_end_to_end_validation_batch() -> None:
    from morseformer.models.acoustic import AcousticConfig, AcousticModel

    samples = build_clean_validation(ValidationConfig(n_per_wpm=2))
    batch = collate([s.as_batch_item() for s in samples[:4]])
    model = AcousticModel(
        AcousticConfig(d_model=32, n_heads=4, n_layers=2,
                       ff_expansion=2, conv_kernel=7)
    ).eval()
    with torch.no_grad():
        log_probs, lengths_out = model(batch["features"],
                                       lengths=batch["n_frames"])
    assert log_probs.shape == (4, 750, 46)
    assert lengths_out.tolist() == [750] * 4
