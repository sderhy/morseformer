"""Synthetic datasets for Phase-0 evaluation.

The `sanity` dataset is intentionally easy: clean carrier, fixed 20 WPM,
no noise, no timing variability. A working decoder must hit near-zero CER
on this set — it is a harness sanity check, not a benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from morse_synth.core import synthesize

# QSO-typical material: CQ calls, callsigns, contest exchanges, Q-codes,
# 73s. Designed to exercise the decoder on the token distribution it will
# see in real traffic, not generic prose.
SANITY_TEXTS: list[str] = [
    "CQ CQ CQ DE F5ABC F5ABC K",
    "CQ DX DE W1ABC K",
    "QRZ DE K1AB K",
    "HELLO OM",
    "TEST DE G0XYZ",
    "GM OM UR 599 IN FRANCE",
    "GA OM UR RST 579 579",
    "TNX FER QSO 73",
    "QTH NICE NICE",
    "NAME SERGE SERGE",
    "WX SUNNY HR TEMP 20",
    "RIG IC 7300 ANT DIPOLE",
    "599 001 FRANCE",
    "CQ TEST DE F5ABC",
    "PSE K",
    "AGN AGN AGN",
    "E E",
    "73 73",
    "DE F5ABC K",
    "THE QUICK BROWN FOX",
]


@dataclass
class Sample:
    sample_id: str
    text: str
    audio: np.ndarray
    sample_rate: int


def generate_sanity(
    n: int = 20,
    *,
    sample_rate: int = 8000,
    wpm: float = 20.0,
    freq: float = 600.0,
) -> list[Sample]:
    """Generate a deterministic sanity dataset.

    Args:
        n: Number of samples (repeats the template list if n > len(SANITY_TEXTS)).
        sample_rate: Audio sample rate in Hz.
        wpm: Fixed speed for every sample.
        freq: Carrier frequency in Hz.

    Returns:
        List of `Sample` records.
    """
    texts = (SANITY_TEXTS * ((n // len(SANITY_TEXTS)) + 1))[:n]
    samples: list[Sample] = []
    for i, text in enumerate(texts):
        audio = synthesize(text, wpm=wpm, sample_rate=sample_rate, freq=freq)
        samples.append(
            Sample(
                sample_id=f"sanity_{i:03d}",
                text=text,
                audio=audio,
                sample_rate=sample_rate,
            )
        )
    return samples
