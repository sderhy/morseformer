"""Keying shape: how the on/off envelope transitions at each edge.

The shape of a CW keying edge controls the spectral occupancy of the
transmitted signal. A perfectly rectangular keying produces wide-band
"key clicks"; a smoothly-shaped edge (raised-cosine, Gaussian) gives a
narrow spectrum that matches modern transmitter behaviour. We also
support a simple per-element chirp — a linear frequency sweep inside
each keydown — that mimics certain older rigs and frequency-pulling
under strong keying.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

from morse_synth.operator import Event

Shape = Literal["rect", "raised_cosine", "gauss"]


@dataclass
class KeyingConfig:
    """Parameters for the keying-edge shape.

    Attributes:
        shape:               Edge shape — "rect" (hard keying; wide clicks),
                             "raised_cosine" (default, clean), or "gauss".
        rise_ms:             Edge rise/fall time in ms. Ignored for "rect".
        chirp_hz_per_unit:   Linear frequency sweep during each keydown,
                             in Hz per dot-unit. 0 = no chirp.
    """

    shape: Shape = "raised_cosine"
    rise_ms: float = 5.0
    chirp_hz_per_unit: float = 0.0


def _edge_kernel(rise_samples: int, shape: Shape) -> np.ndarray:
    """Return a low-pass kernel used to shape rectangular keying edges."""
    if shape == "rect":
        # Zero-width smoothing: no shaping at all.
        k = np.array([1.0], dtype=np.float32)
        return k
    if shape == "raised_cosine":
        k = np.hanning(2 * rise_samples + 1).astype(np.float32)
    elif shape == "gauss":
        # Match the effective width of the Hann window at its -3dB points.
        sigma = rise_samples / 2.0
        x = np.arange(-3 * rise_samples, 3 * rise_samples + 1, dtype=np.float32)
        k = np.exp(-0.5 * (x / sigma) ** 2).astype(np.float32)
    else:
        raise ValueError(f"Unknown keying shape: {shape!r}")
    return k / k.sum()


def render_events(
    events: Sequence[Event],
    *,
    keying: KeyingConfig | None = None,
    freq: float = 600.0,
    sample_rate: int = 8000,
    amplitude: float = 0.5,
    tail_ms: float = 50.0,
    wpm: float | None = None,
) -> np.ndarray:
    """Render a sequence of (is_on, duration_seconds) events to audio.

    Args:
        events:         Output of morse_synth.operator.build_events.
        keying:         KeyingConfig; defaults to a clean raised-cosine 5 ms.
        freq:           Carrier frequency in Hz.
        sample_rate:    Audio sample rate in Hz.
        amplitude:      Peak amplitude in [0, 1].
        tail_ms:        Silent tail after the last event, in ms.
        wpm:            Used only when chirp is enabled, to translate
                        "Hz per unit" into "Hz per second". If None and
                        chirp != 0, the first keydown's duration is taken
                        as 1 unit for the purposes of the chirp slope.

    Returns:
        float32 1-D numpy array.
    """
    keying = keying or KeyingConfig()

    if not events:
        return np.zeros(int(round(tail_ms / 1000.0 * sample_rate)), dtype=np.float32)

    # Running sample cursor per event.
    cumulative_samples: list[int] = [0]
    for _, dur in events:
        cumulative_samples.append(cumulative_samples[-1] + int(round(dur * sample_rate)))
    n_body = cumulative_samples[-1]
    n_tail = int(round(tail_ms / 1000.0 * sample_rate))
    n_total = n_body + n_tail

    envelope = np.zeros(n_total, dtype=np.float32)
    for (is_on, _dur), start, end in zip(
        events, cumulative_samples[:-1], cumulative_samples[1:]
    ):
        if is_on:
            envelope[start:end] = 1.0

    # Shape the edges (skip for "rect").
    if keying.shape != "rect" and keying.rise_ms > 0:
        rise_n = max(1, int(round(keying.rise_ms / 1000.0 * sample_rate)))
        kernel = _edge_kernel(rise_n, keying.shape)
        envelope = np.convolve(envelope, kernel, mode="same")

    t = np.arange(n_total, dtype=np.float64) / sample_rate

    # Baseline carrier.
    phase = 2.0 * np.pi * freq * t

    # Optional per-element chirp: adds a ramp to the phase during each keydown.
    if keying.chirp_hz_per_unit != 0.0:
        # Rate in Hz/s: use operator's nominal unit duration. Fall back to
        # first keydown's duration.
        if wpm is not None and wpm > 0:
            unit_s = 1.2 / wpm
        else:
            unit_s = next((d for on, d in events if on), 1.0)
        hz_per_s = keying.chirp_hz_per_unit / max(unit_s, 1e-6)
        chirp_phase = np.zeros(n_total, dtype=np.float64)
        for (is_on, _), start, end in zip(
            events, cumulative_samples[:-1], cumulative_samples[1:]
        ):
            if not is_on:
                continue
            seg_n = end - start
            if seg_n == 0:
                continue
            local_t = np.arange(seg_n) / sample_rate
            # ∫ 2π·hz_per_s·τ dτ from 0 to local_t
            chirp_phase[start:end] = np.pi * hz_per_s * local_t ** 2
        phase = phase + chirp_phase

    carrier = np.sin(phase).astype(np.float32)
    return (amplitude * envelope * carrier).astype(np.float32)
