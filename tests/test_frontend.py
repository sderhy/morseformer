"""Unit tests for the DSP front-end (features/frontend.py)."""

from __future__ import annotations

import numpy as np
import pytest

from morse_synth.core import synthesize
from morseformer.features import FrontendConfig, extract_features


def _tone_burst(
    sample_rate: int, duration_s: float, freq: float = 600.0, amplitude: float = 0.5
) -> np.ndarray:
    t = np.arange(int(duration_s * sample_rate), dtype=np.float32) / sample_rate
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_output_shape_and_dtype() -> None:
    sr = 8000
    cfg = FrontendConfig(frame_rate=500)
    audio = _tone_burst(sr, duration_s=1.0)
    feats = extract_features(audio, sr, cfg)
    assert feats.dtype == np.float32
    assert feats.ndim == 2
    assert feats.shape[1] == 1


def test_frame_rate_is_respected() -> None:
    sr = 8000
    cfg = FrontendConfig(frame_rate=500)  # 16 samples per frame
    audio = _tone_burst(sr, duration_s=1.0)  # 8000 samples
    feats = extract_features(audio, sr, cfg)
    # ceil or floor of 8000/16 = 500; allow ±1 frame for boundary handling.
    assert abs(feats.shape[0] - 500) <= 1


def test_amplitude_invariance_after_normalisation() -> None:
    sr = 8000
    cfg = FrontendConfig(frame_rate=500)
    audio = synthesize("PARIS", wpm=20, sample_rate=sr)
    loud = audio * 10.0
    feats_soft = extract_features(audio, sr, cfg)
    feats_loud = extract_features(loud, sr, cfg)
    assert feats_soft.shape == feats_loud.shape
    # Per-utterance zero-mean / unit-variance makes features near-invariant
    # to amplitude scaling. Not exactly equal because `log_floor` inside
    # the log creates a mild non-linearity that shifts with signal energy.
    corr = float(
        np.corrcoef(feats_soft.squeeze(-1), feats_loud.squeeze(-1))[0, 1]
    )
    assert corr > 0.999


def test_normalised_features_are_centered() -> None:
    sr = 8000
    cfg = FrontendConfig(frame_rate=500)
    audio = synthesize("CQ CQ DE F6ABC", wpm=22, sample_rate=sr)
    feats = extract_features(audio, sr, cfg)
    assert feats.shape[0] > 0
    assert abs(float(feats.mean())) < 1e-4
    assert abs(float(feats.std()) - 1.0) < 1e-3


def test_empty_input_returns_empty() -> None:
    sr = 8000
    cfg = FrontendConfig(frame_rate=500)
    feats = extract_features(np.zeros(0, dtype=np.float32), sr, cfg)
    assert feats.shape == (0, 1)


def test_very_short_input_returns_empty() -> None:
    sr = 8000
    cfg = FrontendConfig(frame_rate=500)  # hop = 16
    feats = extract_features(np.zeros(4, dtype=np.float32), sr, cfg)
    assert feats.shape == (0, 1)


def test_incompatible_sample_rate_raises() -> None:
    cfg = FrontendConfig(frame_rate=500)
    with pytest.raises(ValueError):
        extract_features(np.zeros(1000, dtype=np.float32), sample_rate=7777, cfg=cfg)


def test_default_config_works() -> None:
    sr = 8000
    audio = _tone_burst(sr, duration_s=0.5)
    feats = extract_features(audio, sr)  # no cfg → defaults
    assert feats.shape[1] == 1
    assert feats.shape[0] > 0


def test_frontend_responds_to_signal_presence() -> None:
    # A synthesised 'T' (one dah) should produce higher envelope values
    # during the ON portion than during the tail silence.
    sr = 8000
    cfg = FrontendConfig(frame_rate=500)
    audio = synthesize("T", wpm=20, sample_rate=sr, tail_ms=200)
    feats = extract_features(audio, sr, cfg).squeeze(-1)
    assert feats.size > 20
    # First third (ON) should be well above last third (silent tail).
    third = feats.size // 3
    on_mean = float(feats[:third].mean())
    tail_mean = float(feats[-third:].mean())
    assert on_mean > tail_mean + 1.0  # standardised units
