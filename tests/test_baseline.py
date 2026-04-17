"""End-to-end test: synthesize clean CW, then decode it, expect perfect match."""

from __future__ import annotations

import pytest

from eval.metrics import character_error_rate
from morseformer.baselines.rule_based import decode
from morse_synth.core import synthesize


@pytest.mark.parametrize(
    "text",
    [
        "PARIS",
        "HELLO WORLD",
        "CQ CQ CQ DE F5ABC F5ABC K",
        "TEST DE W1ABC",
        "73",
    ],
)
def test_rule_based_perfect_on_clean_synth(text: str) -> None:
    sr = 8000
    audio = synthesize(text, wpm=20.0, sample_rate=sr, freq=600.0)
    hyp = decode(audio, sr, tone_freq=600.0)
    cer = character_error_rate(text, hyp)
    # On clean, noise-free synthetic audio the rule-based baseline should be
    # essentially perfect; allow a one-character slack for edge effects.
    assert cer <= 1.0 / len(text.replace(" ", "")), f"{text!r} -> {hyp!r}, CER={cer}"


def test_rule_based_empty_audio() -> None:
    import numpy as np

    assert decode(np.zeros(0, dtype=np.float32), 8000) == ""


def test_rule_based_silence() -> None:
    import numpy as np

    assert decode(np.zeros(4000, dtype=np.float32), 8000) == ""
