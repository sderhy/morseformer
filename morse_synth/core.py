"""Top-level synthesis API.

The full Phase-1 pipeline factors cleanly into three stages, each with
its own configuration object:

    operator  (morse_synth.operator)   text  →  timed events
    keying    (morse_synth.keying)     events → clean audio
    channel   (morse_synth.channel)    clean  → noisy audio

`render()` is the composite entry-point. `synthesize()` is a thin,
backward-compatible wrapper that produces clean audio with the same
signature and output as Phase 0.
"""

from __future__ import annotations

import numpy as np

from morse_synth.channel import ChannelConfig, apply_channel
from morse_synth.keying import KeyingConfig, render_events
from morse_synth.operator import OperatorConfig, build_events


def render(
    text: str,
    *,
    operator: OperatorConfig | None = None,
    keying: KeyingConfig | None = None,
    channel: ChannelConfig | None = None,
    freq: float = 600.0,
    sample_rate: int = 8000,
    amplitude: float = 0.5,
    tail_ms: float = 50.0,
) -> np.ndarray:
    """Render a message through the full operator + keying + channel pipeline."""
    operator = operator or OperatorConfig()
    keying = keying or KeyingConfig()
    channel = channel or ChannelConfig()

    events = build_events(text, operator)
    clean = render_events(
        events,
        keying=keying,
        freq=freq,
        sample_rate=sample_rate,
        amplitude=amplitude,
        tail_ms=tail_ms,
        wpm=operator.wpm,
    )
    return apply_channel(clean, sample_rate, channel)


def synthesize(
    text: str,
    *,
    wpm: float = 20.0,
    freq: float = 600.0,
    sample_rate: int = 8000,
    rise_ms: float = 5.0,
    amplitude: float = 0.5,
    tail_ms: float = 50.0,
) -> np.ndarray:
    """Clean-only convenience wrapper preserved from Phase 0.

    Equivalent to `render()` with default (ideal) operator, a raised-cosine
    keying shape, and no channel effects. Kept so that Phase-0 tests and
    external callers that only want clean CW do not have to know about
    the configuration dataclasses.
    """
    if wpm <= 0:
        raise ValueError(f"WPM must be positive, got {wpm}")

    unit_samples = int(round(1.2 / wpm * sample_rate))
    if unit_samples < 2:
        raise ValueError(
            f"WPM too fast for sample_rate={sample_rate}: "
            f"one unit would be {unit_samples} samples"
        )

    return render(
        text,
        operator=OperatorConfig(wpm=wpm),
        keying=KeyingConfig(shape="raised_cosine", rise_ms=rise_ms),
        channel=None,
        freq=freq,
        sample_rate=sample_rate,
        amplitude=amplitude,
        tail_ms=tail_ms,
    )
