"""Deterministic DSP front-end: raw audio → frame-rate features.

CW is a single amplitude-modulated sinusoid, so we do not need a generic
speech front-end (log-mel with 80 channels). Instead we exploit the
domain: bandpass around the carrier, take the analytic envelope, and
decimate to the target frame rate. One feature channel suffices for
Phase 2.1; extra channels (instantaneous frequency, SNR estimate) will
be added in Phase 3 when carrier drift matters.

The output is a `[T, F]` float32 array, where T = ceil(N * frame_rate /
sample_rate) and F is the number of feature channels (currently 1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt


@dataclass
class FrontendConfig:
    """Hyperparameters for the front-end.

    Attributes:
        tone_freq:     Assumed carrier frequency in Hz.
        bandwidth:     BPF bandwidth in Hz (centred on tone_freq).
        frame_rate:    Target feature rate in frames per second.
        log_floor:     Small floor added before log to avoid log(0).
    """

    tone_freq: float = 600.0
    bandwidth: float = 200.0
    frame_rate: int = 500
    log_floor: float = 1e-6


def _bandpass(x: np.ndarray, sample_rate: int, cfg: FrontendConfig) -> np.ndarray:
    low = max(1.0, cfg.tone_freq - cfg.bandwidth / 2.0)
    high = min(sample_rate / 2 - 1.0, cfg.tone_freq + cfg.bandwidth / 2.0)
    sos = butter(4, [low, high], btype="band", fs=sample_rate, output="sos")
    return sosfiltfilt(sos, x).astype(np.float32)


def _envelope(x: np.ndarray) -> np.ndarray:
    return np.abs(hilbert(x)).astype(np.float32)


def _decimate_mean(x: np.ndarray, hop: int) -> np.ndarray:
    """Box-average decimation: average hop-length windows.

    Acts as a mild anti-alias filter + decimator in one step. We use box
    averaging rather than `scipy.signal.decimate` because the features
    are slowly-varying envelope values, not full-bandwidth audio, so the
    phase distortion of a higher-order IIR is unnecessary.
    """
    if hop <= 1:
        return x
    n_frames = x.size // hop
    trimmed = x[: n_frames * hop]
    return trimmed.reshape(n_frames, hop).mean(axis=1).astype(np.float32)


def _normalise(env_log: np.ndarray) -> np.ndarray:
    # Per-utterance zero-mean / unit-variance. Robust to amplitude scaling of
    # the input signal.
    mu = float(np.mean(env_log))
    sigma = float(np.std(env_log))
    if sigma < 1e-8:
        return (env_log - mu).astype(np.float32)
    return ((env_log - mu) / sigma).astype(np.float32)


def extract_features(
    audio: np.ndarray, sample_rate: int, cfg: FrontendConfig | None = None
) -> np.ndarray:
    """Extract frame-rate features from a raw audio signal.

    Returns a `[T, 1]` float32 array at `cfg.frame_rate` Hz.
    """
    cfg = cfg or FrontendConfig()
    if sample_rate % cfg.frame_rate != 0:
        raise ValueError(
            f"sample_rate={sample_rate} must be a multiple of "
            f"frame_rate={cfg.frame_rate}"
        )
    hop = sample_rate // cfg.frame_rate
    if audio.size < hop * 2:
        return np.zeros((0, 1), dtype=np.float32)

    x = np.asarray(audio, dtype=np.float32)
    x = _bandpass(x, sample_rate, cfg)
    env = _envelope(x)
    log_env = np.log(env + cfg.log_floor).astype(np.float32)
    frames = _decimate_mean(log_env, hop)
    frames = _normalise(frames)
    return frames[:, None]  # [T, 1]
