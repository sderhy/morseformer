"""Unit tests for the deterministic validation-set builder."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from morseformer.core.tokenizer import VOCAB_SIZE, decode  # noqa: E402
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


def test_matching_propagates_operator_envelope() -> None:
    """``matching`` must copy every operator-timing field so the val
    distribution sees the same jitter / dash-ratio / gap-inflation as
    the training stream (fixes ``project_phase4_0b_result``)."""
    ds_cfg = DatasetConfig(
        operator_element_jitter_range=(0.0, 0.30),
        operator_gap_jitter_range=(0.0, 0.50),
        operator_dash_dot_ratio_range=(2.5, 4.5),
        operator_gap_inflation_range=(0.8, 1.6),
        operator_word_gap_inflation_range=(1.0, 8.0),
        operator_run_on_pairs=(("S", "K", 0.5),),
    )
    val_cfg = ValidationConfig.matching(ds_cfg)
    assert val_cfg.operator_element_jitter_range == (0.0, 0.30)
    assert val_cfg.operator_gap_jitter_range == (0.0, 0.50)
    assert val_cfg.operator_dash_dot_ratio_range == (2.5, 4.5)
    assert val_cfg.operator_gap_inflation_range == (0.8, 1.6)
    assert val_cfg.operator_word_gap_inflation_range == (1.0, 8.0)
    assert val_cfg.operator_run_on_pairs == (("S", "K", 0.5),)


def test_matching_propagates_realistic_channel_envelope() -> None:
    """``matching`` must copy every channel-envelope field so the
    realistic ladder uses the training distribution instead of the
    Phase 3.1 numbers historically hardcoded in ``_render_one_realistic``."""
    ds_cfg = DatasetConfig(
        freq_offset_range_hz=(-80.0, 80.0),
        qsb_rate_range_hz=(0.1, 2.0),
        qsb_depth_range_db=(0.0, 25.0),
        qrn_rate_range_per_sec=(0.0, 3.0),
        carrier_drift_sigma_range_hz_per_s=(0.0, 2.5),
        qrm_probability=0.40,
        qrm_offset_range_hz=(-500.0, 500.0),
        qrm_rel_db_range=(-25.0, -5.0),
    )
    val_cfg = ValidationConfig.matching(ds_cfg)
    assert val_cfg.freq_offset_range_hz == (-80.0, 80.0)
    assert val_cfg.qsb_rate_range_hz == (0.1, 2.0)
    assert val_cfg.qsb_depth_range_db == (0.0, 25.0)
    assert val_cfg.qrn_rate_range_per_sec == (0.0, 3.0)
    assert val_cfg.carrier_drift_sigma_range_hz_per_s == (0.0, 2.5)
    assert val_cfg.qrm_probability == 0.40
    assert val_cfg.qrm_offset_range_hz == (-500.0, 500.0)
    assert val_cfg.qrm_rel_db_range == (-25.0, -5.0)


def test_matching_propagates_empty_and_post_silence_knobs() -> None:
    """``matching`` must copy the empty-sample / post-emission-silence
    fields so noise-only val builders reflect the Phase 5.6 sub-mode
    when it is enabled in training."""
    ds_cfg = DatasetConfig(
        empty_sample_probability=0.40,
        empty_sample_pseudo_morse_enabled=True,
        post_emission_silence_probability=0.10,
        post_emission_silence_text_chars=(2, 4),
    )
    val_cfg = ValidationConfig.matching(ds_cfg)
    assert val_cfg.empty_sample_probability == 0.40
    assert val_cfg.empty_sample_pseudo_morse_enabled is True
    assert val_cfg.post_emission_silence_probability == 0.10
    assert val_cfg.post_emission_silence_text_chars == (2, 4)


def test_matching_overrides_win_over_dataset_cfg() -> None:
    """Explicit kwargs to ``matching`` must override the inherited
    ``DatasetConfig`` value — this is how callers force e.g.
    ``n_per_wpm`` without changing the audio envelope."""
    ds_cfg = DatasetConfig(operator_element_jitter_range=(0.0, 0.30))
    val_cfg = ValidationConfig.matching(
        ds_cfg, operator_element_jitter_range=(0.0, 0.05)
    )
    assert val_cfg.operator_element_jitter_range == (0.0, 0.05)


def test_jittered_dataset_cfg_produces_visibly_different_val_audio() -> None:
    """Acid test for the P0-A fix: a Phase-5.5 training config and a
    bare clean config must produce visibly different val audio when
    rendered with the same seed. Before this fix the val ignored every
    jitter / channel knob, so both configs produced bit-identical
    samples (operator was always ideal, channel was always hardcoded).

    The front-end does per-utterance zero-mean / unit-variance
    normalisation, which masks global energy differences — so the
    comparison is per-sample feature tensors rather than aggregate
    statistics.
    """
    from morseformer.data.validation import build_realistic_ladder_validation

    clean_cfg = ValidationConfig(n_per_wpm=4, wpm_bins=(20.0,), seed=7)
    jittered_cfg = ValidationConfig.matching(
        DatasetConfig.phase_5_5(),
        n_per_wpm=4,
        wpm_bins=(20.0,),
        seed=7,
    )

    clean = build_realistic_ladder_validation(
        snrs_db=(10.0,), cfg=clean_cfg, n_per_cell=4
    )
    jittered = build_realistic_ladder_validation(
        snrs_db=(10.0,), cfg=jittered_cfg, n_per_cell=4
    )

    assert len(clean) == len(jittered) > 0
    differing = sum(
        not torch.equal(a.features, b.features)
        for a, b in zip(clean, jittered)
    )
    # Most samples must differ. We allow a small slack because two
    # samples *could* coincidentally render identically when the
    # random draws happen to land on near-default values, but if the
    # propagation is wired the great majority of samples will differ.
    assert differing >= len(clean) - 1, (
        f"Only {differing}/{len(clean)} val samples differ between clean "
        "and jittered configs — ValidationConfig.matching propagation is "
        "not wired into the renderer."
    )


def test_clean_path_propagates_operator_envelope() -> None:
    """The clean (non-realistic) render path must also respect the
    operator envelope. Otherwise ``build_clean_validation`` and
    ``build_snr_ladder_validation`` would silently keep the
    ideal-timing assumption even when matching() was used."""
    from morseformer.data.validation import build_snr_ladder_validation

    ideal_cfg = ValidationConfig(n_per_wpm=4, wpm_bins=(20.0,), seed=11)
    jittered_cfg = ValidationConfig.matching(
        DatasetConfig.phase_5_5(),
        n_per_wpm=4,
        wpm_bins=(20.0,),
        seed=11,
    )

    ideal = build_snr_ladder_validation(
        snrs_db=(20.0,), cfg=ideal_cfg, n_per_cell=4
    )
    jittered = build_snr_ladder_validation(
        snrs_db=(20.0,), cfg=jittered_cfg, n_per_cell=4
    )

    differing = sum(
        not torch.equal(a.features, b.features)
        for a, b in zip(ideal, jittered)
    )
    assert differing >= len(ideal) - 1, (
        f"Only {differing}/{len(ideal)} samples differ on the AWGN ladder — "
        "operator envelope is not wired into _render_one."
    )


def test_matching_default_cfg_keeps_phase_3_1_channel_numbers() -> None:
    """``ValidationConfig()`` with defaults must keep the historical
    Phase-3.1 channel numbers — required for backward compatibility
    with bench scripts that build a bare ``ValidationConfig`` and call
    ``build_realistic_ladder_validation`` directly."""
    cfg = ValidationConfig()
    assert cfg.freq_offset_range_hz == (-50.0, 50.0)
    assert cfg.qsb_rate_range_hz == (0.05, 1.0)
    assert cfg.qsb_depth_range_db == (0.0, 15.0)
    assert cfg.qrn_rate_range_per_sec == (0.0, 1.0)
    assert cfg.carrier_drift_sigma_range_hz_per_s == (0.0, 1.0)
    assert cfg.qrm_probability == 0.25
    assert cfg.qrm_offset_range_hz == (-300.0, 300.0)
    assert cfg.qrm_rel_db_range == (-18.0, -8.0)


def test_pseudo_morse_mode_adds_fourth_noise_bench_slice() -> None:
    """When ``empty_sample_pseudo_morse_enabled`` is True the noise-only
    builder must emit a 4th mode — same Phase 5.6 distribution the
    training pipeline uses for digit-hallucination suppression."""
    base = ValidationConfig(wpm_bins=(20.0,), seed=5)
    on = ValidationConfig(
        wpm_bins=(20.0,), seed=5, empty_sample_pseudo_morse_enabled=True
    )
    samples_off = build_noise_only_validation(cfg=base, n_per_mode=3)
    samples_on = build_noise_only_validation(cfg=on, n_per_mode=3)
    # Default (off): 3 modes × 3 = 9 samples.
    assert len(samples_off) == 9
    # Pseudo-morse on: 4 modes × 3 = 12 samples.
    assert len(samples_on) == 12
    # All samples are still labelled empty.
    for s in samples_on:
        assert s.text == ""
        assert s.n_tokens == 0


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
    assert log_probs.shape == (4, 750, VOCAB_SIZE)
    assert lengths_out.tolist() == [750] * 4
