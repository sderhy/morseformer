"""HF channel simulation applied to a clean CW signal.

Phase 1 implements the effects that dominate weak-signal HF copy:

    * Additive white Gaussian noise calibrated to a target SNR in dB.
    * Atmospheric noise (QRN): random impulses with short exponential
      decay, Poisson-distributed in time.
    * Slow fading (QSB): a multiplicative, sinusoidal+random envelope.
    * Carrier drift: a slow frequency walk via analytic-signal rotation.
    * Receiver-side bandpass filter approximating IC-7300-style CW
      filter cascades.

Co-channel QRM (other CW signals, SSB splatter, data bursts) and
full multipath are deferred to Phase 1.5 — they are important but not
on the critical path for getting the acoustic model off the ground.

The target-SNR convention: SNR is measured as
    10·log10( rms(signal)^2 / rms(awgn_noise)^2 )
with signal RMS computed over the keydown segments only (envelope above
the 70th percentile), which is how an operator perceives "signal above
the background hiss". A report of "−10 dB" here is a few dB worse than
a report based on unweighted full-signal RMS — intentionally
pessimistic so that a model trained at a given SNR will over-match real
ham "S-meter SNR" at inference time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt


@dataclass
class ChannelConfig:
    """Parameters for the HF-channel simulation.

    Attributes:
        snr_db:                   Target signal-to-noise ratio in dB. Use
                                  float('inf') for a noiseless output.
        qrn_rate_per_sec:         Mean Poisson rate of atmospheric impulses.
        qrn_amplitude_db:         Impulse peak amplitude, in dB relative to
                                  the clean-signal peak. 0 dB = same peak.
        qrn_decay_ms:             Exponential-decay time constant of each impulse.
        qsb_rate_hz:              Sinusoidal fading frequency. 0 disables.
        qsb_depth_db:             Fading depth in dB (e.g. 20 = fades down to
                                  1/100 of full signal at the trough).
        carrier_drift_hz_per_s:   Stddev of the carrier random-walk step,
                                  in Hz per second of signal. 0 disables.
        rx_filter_bw:             Optional RX filter bandwidth in Hz.
                                  None disables the filter.
        rx_filter_centre:         RX filter centre frequency in Hz.
        seed:                     Integer seed for reproducibility.
    """

    snr_db: float = math.inf
    qrn_rate_per_sec: float = 0.0
    qrn_amplitude_db: float = -6.0
    qrn_decay_ms: float = 1.0
    qsb_rate_hz: float = 0.0
    qsb_depth_db: float = 0.0
    carrier_drift_hz_per_s: float = 0.0
    rx_filter_bw: float | None = None
    rx_filter_centre: float = 600.0
    seed: int | None = None


def _signal_rms(x: np.ndarray) -> float:
    """RMS over the keydown segments (envelope above 70th percentile)."""
    env = np.abs(x)
    threshold = float(np.percentile(env, 70.0))
    if threshold <= 0:
        return float(np.sqrt(np.mean(x ** 2)))
    mask = env > threshold
    if not mask.any():
        return float(np.sqrt(np.mean(x ** 2)))
    return float(np.sqrt(np.mean(x[mask] ** 2)))


def _add_awgn(x: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    if not np.isfinite(snr_db):
        return x
    s_rms = _signal_rms(x)
    if s_rms <= 0:
        return x
    n_rms = s_rms / (10 ** (snr_db / 20.0))
    noise = rng.normal(0.0, n_rms, size=x.shape).astype(x.dtype)
    return x + noise


def _add_qrn(
    x: np.ndarray,
    sample_rate: int,
    rate: float,
    amp_db: float,
    decay_ms: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if rate <= 0:
        return x
    n = x.size
    duration = n / sample_rate
    n_imp = int(rng.poisson(rate * duration))
    if n_imp == 0:
        return x
    peak = float(np.max(np.abs(x))) or 1.0
    amp_linear = peak * 10 ** (amp_db / 20.0)
    decay_samples = max(2, int(round(decay_ms / 1000.0 * sample_rate)))
    tail = np.exp(-np.arange(decay_samples) / (decay_ms / 1000.0 * sample_rate))
    qrn = np.zeros_like(x)
    positions = rng.integers(0, n, size=n_imp)
    signs = rng.choice([-1.0, 1.0], size=n_imp)
    # Per-impulse amplitude: half-normal distribution so most are small, a few loud.
    amps = np.abs(rng.normal(0, amp_linear, size=n_imp))
    for p, s, a in zip(positions, signs, amps):
        end = min(p + decay_samples, n)
        qrn[p:end] += (s * a * tail[: end - p]).astype(x.dtype)
    return x + qrn


def _apply_qsb(
    x: np.ndarray,
    sample_rate: int,
    rate_hz: float,
    depth_db: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if rate_hz <= 0 or depth_db <= 0:
        return x
    n = x.size
    t = np.arange(n, dtype=np.float64) / sample_rate
    # Fading envelope: base + half-sine with random phase, scaled so the
    # trough reaches exactly the target depth.
    trough = 10 ** (-depth_db / 20.0)
    phase = rng.uniform(0, 2 * np.pi)
    envelope = trough + (1 - trough) * (0.5 + 0.5 * np.cos(2 * np.pi * rate_hz * t + phase))
    return (x * envelope).astype(x.dtype)


def _apply_carrier_drift(
    x: np.ndarray,
    sample_rate: int,
    drift_sigma_hz_per_s: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if drift_sigma_hz_per_s <= 0:
        return x
    n = x.size
    dt = 1.0 / sample_rate
    # Random walk in frequency: stddev grows as sqrt(t).
    steps = rng.normal(0.0, drift_sigma_hz_per_s * math.sqrt(dt), size=n)
    freq_offset = np.cumsum(steps)
    phase = 2 * np.pi * np.cumsum(freq_offset) * dt
    analytic = hilbert(x)
    return np.real(analytic * np.exp(1j * phase)).astype(x.dtype)


def _apply_rx_filter(
    x: np.ndarray,
    sample_rate: int,
    bw: float,
    centre: float,
) -> np.ndarray:
    low = max(1.0, centre - bw / 2)
    high = min(sample_rate / 2 - 1, centre + bw / 2)
    sos = butter(6, [low, high], btype="band", fs=sample_rate, output="sos")
    return sosfiltfilt(sos, x).astype(x.dtype)


def apply_channel(
    x: np.ndarray, sample_rate: int, cfg: ChannelConfig | None = None
) -> np.ndarray:
    """Apply the configured HF-channel effects to a clean audio signal.

    Order of effects matters: fading and drift are applied to the clean
    signal first (they represent ionospheric propagation), then noise and
    QRN are added (receiver-side), and finally the receiver filter.
    """
    cfg = cfg or ChannelConfig()
    rng = np.random.default_rng(cfg.seed)

    y = x.astype(np.float32, copy=True)
    y = _apply_qsb(y, sample_rate, cfg.qsb_rate_hz, cfg.qsb_depth_db, rng)
    y = _apply_carrier_drift(y, sample_rate, cfg.carrier_drift_hz_per_s, rng)
    y = _add_awgn(y, cfg.snr_db, rng)
    y = _add_qrn(
        y,
        sample_rate,
        cfg.qrn_rate_per_sec,
        cfg.qrn_amplitude_db,
        cfg.qrn_decay_ms,
        rng,
    )
    if cfg.rx_filter_bw is not None and cfg.rx_filter_bw > 0:
        y = _apply_rx_filter(y, sample_rate, cfg.rx_filter_bw, cfg.rx_filter_centre)
    return y
