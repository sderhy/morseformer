"""Tests for the HF-channel simulation."""

from __future__ import annotations

import math

import numpy as np
import pytest

from morse_synth.channel import ChannelConfig, apply_channel
from morse_synth.core import synthesize


@pytest.fixture
def clean_audio():
    return synthesize("HELLO WORLD", wpm=20.0, sample_rate=8000, freq=600.0)


def _measured_snr_db(clean: np.ndarray, noisy: np.ndarray) -> float:
    """Compute measured SNR from the clean signal and the resulting noisy signal."""
    noise = noisy - clean[: noisy.size]
    s_rms = float(np.sqrt(np.mean(clean[np.abs(clean) > np.percentile(np.abs(clean), 70)] ** 2)))
    n_rms = float(np.sqrt(np.mean(noise ** 2)))
    if n_rms <= 0:
        return float("inf")
    return 20.0 * math.log10(s_rms / n_rms)


def test_inf_snr_is_noiseless(clean_audio):
    out = apply_channel(clean_audio, 8000, ChannelConfig(snr_db=float("inf")))
    np.testing.assert_array_equal(out, clean_audio)


@pytest.mark.parametrize("target_snr", [20.0, 10.0, 0.0, -10.0])
def test_snr_matches_target(clean_audio, target_snr):
    out = apply_channel(clean_audio, 8000, ChannelConfig(snr_db=target_snr, seed=42))
    measured = _measured_snr_db(clean_audio, out)
    # Allow ±1.5 dB of error: the RMS-based measurement is only approximate
    # because of the envelope-threshold definition of "signal RMS".
    assert abs(measured - target_snr) < 1.5, f"target={target_snr}, got {measured:.2f}"


def test_seed_is_reproducible(clean_audio):
    cfg = ChannelConfig(snr_db=0.0, seed=123)
    a = apply_channel(clean_audio, 8000, cfg)
    b = apply_channel(clean_audio, 8000, cfg)
    np.testing.assert_array_equal(a, b)


def test_different_seeds_differ(clean_audio):
    a = apply_channel(clean_audio, 8000, ChannelConfig(snr_db=0.0, seed=1))
    b = apply_channel(clean_audio, 8000, ChannelConfig(snr_db=0.0, seed=2))
    # At 0 dB SNR two independent noise realisations must not be identical.
    assert not np.array_equal(a, b)


def test_qrn_adds_outliers(clean_audio):
    cfg_clean = ChannelConfig(snr_db=float("inf"), qrn_rate_per_sec=0.0)
    cfg_noisy = ChannelConfig(
        snr_db=float("inf"),
        qrn_rate_per_sec=100.0,
        qrn_amplitude_db=10.0,  # 10 dB above signal peak
        seed=99,
    )
    clean_out = apply_channel(clean_audio, 8000, cfg_clean)
    noisy_out = apply_channel(clean_audio, 8000, cfg_noisy)
    # QRN should produce samples significantly larger than any sample in the clean pass.
    assert np.max(np.abs(noisy_out)) > np.max(np.abs(clean_out)) * 2


def test_qsb_modulates_amplitude():
    # Use a continuous tone (no CW gaps) so the envelope measurement is
    # unambiguous — inter-character silences in CW would confound the ratio.
    sr = 8000
    t = np.arange(4 * sr).astype(np.float32) / sr
    tone = np.sin(2 * np.pi * 600 * t).astype(np.float32)
    cfg = ChannelConfig(snr_db=float("inf"), qsb_rate_hz=0.5, qsb_depth_db=20.0, seed=0)
    faded = apply_channel(tone, sr, cfg)
    # Envelope via short-window RMS. Ratio peak/trough should be close to
    # 10**(20/20) = 10, but allow wide margin because the fade trajectory
    # is continuous and may not reach the full depth inside the window.
    win = sr // 10  # 0.1 s
    rms = np.array([
        np.sqrt(np.mean(faded[i * win : (i + 1) * win] ** 2))
        for i in range(len(faded) // win)
    ])
    assert rms.max() / (rms.min() + 1e-9) > 3.0


def test_output_shape_preserved(clean_audio):
    cfg = ChannelConfig(snr_db=-5.0, qrn_rate_per_sec=10.0, qsb_rate_hz=0.5,
                        qsb_depth_db=10.0, seed=0)
    out = apply_channel(clean_audio, 8000, cfg)
    assert out.shape == clean_audio.shape
    assert out.dtype == clean_audio.dtype


def test_rx_filter_attenuates_far_tones():
    sr = 8000
    # Pure sinusoid at 2000 Hz — should be cut by a 600 Hz ±100 Hz filter.
    t = np.arange(sr).astype(np.float32) / sr
    tone = np.sin(2 * np.pi * 2000 * t).astype(np.float32)
    cfg = ChannelConfig(snr_db=float("inf"), rx_filter_bw=200.0, rx_filter_centre=600.0)
    filtered = apply_channel(tone, sr, cfg)
    # > 20 dB rejection expected.
    assert np.max(np.abs(filtered)) < np.max(np.abs(tone)) * 0.1
