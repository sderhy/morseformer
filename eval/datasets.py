"""Synthetic datasets for evaluation.

Two families:

    * `sanity` — clean carrier, fixed 20 WPM, no noise, no timing variability.
      Near-zero CER is a mandatory harness invariant, not a benchmark.
    * `noisy`  — parametric operator jitter + HF-channel effects at a
      configurable target SNR. The SNR ladder composes this into a
      CER-vs-SNR curve, which is the real benchmark.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from morse_synth.channel import ChannelConfig
from morse_synth.core import render, synthesize
from morse_synth.keying import KeyingConfig
from morse_synth.operator import OperatorConfig

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


# Defaults for the noisy dataset. Intentionally mild: this is where the
# eval harness spends most of its time, and the configured values should
# correspond to "typical, decent-quality HF CW copy" — not worst-case HF.
# The SNR argument of generate_noisy is what sweeps the benchmark; the
# other knobs characterise channel quality at a given SNR.
_DEFAULT_OPERATOR = OperatorConfig(
    wpm=22.0,
    element_jitter=0.08,
    gap_jitter=0.15,
)
_DEFAULT_KEYING = KeyingConfig(shape="raised_cosine", rise_ms=5.0)
_DEFAULT_CHANNEL_BASE = ChannelConfig(
    qrn_rate_per_sec=0.0,
    qsb_rate_hz=0.0,
    carrier_drift_hz_per_s=0.0,
    rx_filter_bw=None,
)


def generate_noisy(
    n: int = 20,
    snr_db: float = 10.0,
    *,
    sample_rate: int = 8000,
    freq: float = 600.0,
    wpm_range: tuple[float, float] = (18.0, 28.0),
    operator_template: OperatorConfig | None = None,
    keying: KeyingConfig | None = None,
    channel_base: ChannelConfig | None = None,
    seed: int = 42,
) -> list[Sample]:
    """Generate a deterministic noisy dataset at the given target SNR.

    Each sample uses an independently-seeded operator (random WPM drawn
    uniformly from `wpm_range`, independent timing jitter seed) and
    channel instance, so adjacent samples are statistically independent.

    Args:
        n:                 Number of samples.
        snr_db:            Target SNR applied uniformly across the set.
        sample_rate:       Audio sample rate in Hz.
        freq:              Carrier frequency in Hz.
        wpm_range:         Inclusive (low, high) range for uniform WPM sampling.
        operator_template: Base OperatorConfig; WPM and seed are overridden
                           per sample but the jitter parameters are preserved.
        keying:            KeyingConfig for all samples.
        channel_base:      Base ChannelConfig; SNR and seed are overridden
                           per sample, but QRN/QSB/drift/filter parameters
                           are preserved.
        seed:              Master seed; derived per-sample seeds are stable.

    Returns:
        List of `Sample` records.
    """
    texts = (SANITY_TEXTS * ((n // len(SANITY_TEXTS)) + 1))[:n]
    op_template = operator_template or _DEFAULT_OPERATOR
    keying = keying or _DEFAULT_KEYING
    channel_template = channel_base or _DEFAULT_CHANNEL_BASE
    rng = np.random.default_rng(seed)

    samples: list[Sample] = []
    for i, text in enumerate(texts):
        sample_seed = int(rng.integers(0, 2**31 - 1))
        wpm = float(rng.uniform(*wpm_range))
        op = OperatorConfig(
            wpm=wpm,
            element_jitter=op_template.element_jitter,
            gap_jitter=op_template.gap_jitter,
            farnsworth_char_gap=op_template.farnsworth_char_gap,
            farnsworth_word_gap=op_template.farnsworth_word_gap,
            seed=sample_seed,
        )
        channel = ChannelConfig(
            snr_db=snr_db,
            qrn_rate_per_sec=channel_template.qrn_rate_per_sec,
            qrn_amplitude_db=channel_template.qrn_amplitude_db,
            qrn_decay_ms=channel_template.qrn_decay_ms,
            qsb_rate_hz=channel_template.qsb_rate_hz,
            qsb_depth_db=channel_template.qsb_depth_db,
            carrier_drift_hz_per_s=channel_template.carrier_drift_hz_per_s,
            rx_filter_bw=channel_template.rx_filter_bw,
            rx_filter_centre=channel_template.rx_filter_centre,
            seed=sample_seed,
        )
        audio = render(
            text,
            operator=op,
            keying=keying,
            channel=channel,
            freq=freq,
            sample_rate=sample_rate,
        )
        snr_tag = "inf" if not math.isfinite(snr_db) else f"{snr_db:+.0f}"
        samples.append(
            Sample(
                sample_id=f"noisy_{snr_tag}dB_{i:03d}",
                text=text,
                audio=audio,
                sample_rate=sample_rate,
            )
        )
    return samples
