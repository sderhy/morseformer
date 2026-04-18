"""Parametric operator model: imperfect human timing.

An `OperatorConfig` captures the deviations from ideal Morse timing that
real senders show: per-element length jitter, gap-length jitter, Farnsworth
stretching of inter-character and inter-word gaps. `build_events` converts
a text message into a sequence of (is_on, duration_seconds) pairs according
to the configuration; a fixed seed makes each sample reproducible.

The more subtle effects — operator-specific "fist" signatures, fatigue
drift over a message, left-versus-right paddle bias — will be added in a
later iteration; Phase 1's goal is only to exit the "perfectly-clean
mechanical timing" regime so that weak-signal training data looks like
real traffic.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from morseformer.core.morse_table import MORSE_TABLE, unit_seconds


@dataclass
class OperatorConfig:
    """Parameters for the parametric operator model.

    Attributes:
        wpm:                 Nominal speed in words per minute (PARIS).
        element_jitter:      Stddev of per-element length noise, measured in
                             dot-units. Applied multiplicatively around the
                             nominal element duration (1 unit for dit, 3 for dah).
        gap_jitter:          Stddev of per-gap length noise, in dot-units. Applied
                             to every inter-element, inter-character, inter-word gap.
        farnsworth_char_gap: Character-gap length in dot-units. 3.0 = ideal; larger
                             values slow down the character spacing while keeping
                             element speed high (Farnsworth).
        farnsworth_word_gap: Word-gap length in dot-units. 7.0 = ideal.
        seed:                Optional integer seed for reproducibility.
    """

    wpm: float = 20.0
    element_jitter: float = 0.0
    gap_jitter: float = 0.0
    farnsworth_char_gap: float = 3.0
    farnsworth_word_gap: float = 7.0
    seed: int | None = None


# An event is a (is_on, duration_seconds) pair. The keying stage converts a
# list of events into an audio waveform.
Event = tuple[bool, float]


def build_events(text: str, cfg: OperatorConfig | None = None) -> list[Event]:
    """Turn a text message into a timed (on/off) event sequence.

    Unknown characters are silently dropped. Returns an empty list for an
    empty or whitespace-only input.
    """
    cfg = cfg or OperatorConfig()
    if cfg.wpm <= 0:
        raise ValueError(f"WPM must be positive, got {cfg.wpm}")

    rng = np.random.default_rng(cfg.seed)
    u = unit_seconds(cfg.wpm)

    def jitter_units(nominal: float, sigma: float) -> float:
        if sigma <= 0:
            return nominal
        # Clip to avoid negative or pathologically-long durations.
        factor = nominal + rng.normal(0.0, sigma)
        return max(0.1, factor)

    events: list[Event] = []
    words = text.upper().split()
    for word_i, word in enumerate(words):
        if word_i > 0:
            gap_units = jitter_units(cfg.farnsworth_word_gap, cfg.gap_jitter)
            events.append((False, gap_units * u))
        for char_i, ch in enumerate(word):
            code = MORSE_TABLE.get(ch)
            if code is None:
                continue
            if char_i > 0:
                gap_units = jitter_units(cfg.farnsworth_char_gap, cfg.gap_jitter)
                events.append((False, gap_units * u))
            for elem_i, elem in enumerate(code):
                if elem_i > 0:
                    gap_units = jitter_units(1.0, cfg.gap_jitter)
                    events.append((False, gap_units * u))
                nominal = 1.0 if elem == "." else 3.0
                elem_units = jitter_units(nominal, cfg.element_jitter)
                events.append((True, elem_units * u))

    return events
