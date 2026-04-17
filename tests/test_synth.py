"""Unit tests for the CW synthesizer."""

from __future__ import annotations

import numpy as np
import pytest

from morse_synth.core import synthesize


def test_returns_float32() -> None:
    audio = synthesize("E", wpm=20, sample_rate=8000)
    assert audio.dtype == np.float32


def test_silent_for_empty_text() -> None:
    audio = synthesize("   ", wpm=20, sample_rate=8000)
    assert audio.size >= 0
    assert np.max(np.abs(audio)) == 0.0


def test_duration_matches_paris() -> None:
    # "E" = one dit = 1 unit = 1.2/wpm seconds of ON, plus ~50 ms silent tail.
    wpm = 20
    sr = 8000
    tail_ms = 50
    audio = synthesize("E", wpm=wpm, sample_rate=sr, tail_ms=tail_ms, rise_ms=1.0)
    expected = int((1.2 / wpm) * sr) + int(tail_ms / 1000 * sr)
    # Allow a few-sample slack for rounding in the convolution window.
    assert abs(audio.size - expected) <= 8


def test_carrier_is_near_target_freq() -> None:
    sr = 8000
    freq = 600.0
    # Long enough signal for a clean FFT peak.
    audio = synthesize("OOO", wpm=15, sample_rate=sr, freq=freq)
    window = np.hanning(len(audio)).astype(np.float32)
    spectrum = np.abs(np.fft.rfft(audio * window))
    peak_bin = int(np.argmax(spectrum))
    peak_freq = peak_bin * sr / len(audio)
    assert abs(peak_freq - freq) < 10.0  # within 10 Hz


def test_amplitude_respected() -> None:
    audio = synthesize("T", wpm=20, sample_rate=8000, amplitude=0.3, rise_ms=1.0)
    assert np.max(np.abs(audio)) <= 0.31  # small margin for envelope smoothing


def test_invalid_wpm() -> None:
    with pytest.raises(ValueError):
        synthesize("E", wpm=0, sample_rate=8000)


def test_sample_rate_too_low() -> None:
    # At 20 WPM one unit is 60 ms; at 20 Hz sample rate one unit is 1.2 samples.
    with pytest.raises(ValueError):
        synthesize("E", wpm=20, sample_rate=20)
