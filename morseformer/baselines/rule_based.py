"""Classical rule-based CW decoder — simple DSP pipeline, no ML.

Pipeline:
    1. Bandpass around the carrier frequency (default 600 Hz ± 100 Hz).
    2. Hilbert envelope, then short moving-average smoothing.
    3. Percentile-based adaptive threshold (noise floor vs peak).
    4. Run-length encode the binary mask.
    5. Estimate one dot-unit duration as the lower-percentile of ON runs.
    6. Classify runs into {dit, dah} and gaps into {intra, inter-char,
       inter-word}; look up each accumulated code in the Morse table.

This is a faithful reference baseline, comparable to what `fldigi` or
`cwdecoder` do on a single channel. It is intended to give the eval
harness a non-trivial number to beat, not to be state of the art.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt

from morseformer.core.morse_table import decode_code


def _bandpass(x: np.ndarray, sample_rate: int, centre: float, bw: float) -> np.ndarray:
    low = max(1.0, centre - bw / 2)
    high = min(sample_rate / 2 - 1, centre + bw / 2)
    sos = butter(4, [low, high], btype="band", fs=sample_rate, output="sos")
    return sosfiltfilt(sos, x).astype(np.float32)


def _envelope(x: np.ndarray, sample_rate: int, smooth_ms: float = 3.0) -> np.ndarray:
    env = np.abs(hilbert(x)).astype(np.float32)
    n = max(3, int(round(smooth_ms / 1000.0 * sample_rate)))
    if n > 1:
        env = np.convolve(env, np.ones(n, dtype=np.float32) / n, mode="same")
    return env


def _run_length_encode(mask: np.ndarray) -> list[tuple[bool, int]]:
    if mask.size == 0:
        return []
    changes = np.flatnonzero(np.diff(mask.astype(np.int8))) + 1
    starts = np.concatenate(([0], changes))
    ends = np.concatenate((changes, [mask.size]))
    return [(bool(mask[s]), int(e - s)) for s, e in zip(starts, ends)]


def decode(
    audio: np.ndarray,
    sample_rate: int,
    *,
    tone_freq: float = 600.0,
    bandwidth: float = 200.0,
    smooth_ms: float = 3.0,
    noise_pct: float = 20.0,
    peak_pct: float = 95.0,
) -> str:
    """Decode a complete CW audio clip to text."""
    if audio.size == 0:
        return ""

    audio = np.asarray(audio, dtype=np.float32)
    filtered = _bandpass(audio, sample_rate, tone_freq, bandwidth)
    env = _envelope(filtered, sample_rate, smooth_ms=smooth_ms)

    noise = float(np.percentile(env, noise_pct))
    peak = float(np.percentile(env, peak_pct))
    if peak <= noise * 1.5:
        # No discernible signal above noise.
        return ""
    threshold = (noise + peak) / 2.0
    mask = env > threshold

    runs = _run_length_encode(mask)
    if not runs:
        return ""

    # Trim leading/trailing silence.
    while runs and not runs[0][0]:
        runs.pop(0)
    while runs and not runs[-1][0]:
        runs.pop()
    if not runs:
        return ""

    on_durs = np.array([d for v, d in runs if v], dtype=np.float32)
    if on_durs.size == 0:
        return ""

    # Estimate one dot-unit as the lower-quartile of ON run lengths: dits are
    # ~1 unit, dahs ~3 units, so the lower tail of the ON-length distribution
    # is dominated by dits on any message longer than a few letters.
    unit = float(np.percentile(on_durs, 25))
    if unit <= 0:
        return ""

    output: list[str] = []
    current_code: list[str] = []

    def flush_char() -> None:
        if current_code:
            ch = decode_code("".join(current_code))
            if ch:
                output.append(ch)
            current_code.clear()

    for is_on, dur in runs:
        units = dur / unit
        if is_on:
            current_code.append("." if units < 2.0 else "-")
        else:
            if units < 2.0:
                pass  # inter-element, keep building the same char
            elif units < 5.0:
                flush_char()
            else:
                flush_char()
                if output and output[-1] != " ":
                    output.append(" ")
    flush_char()

    return "".join(output).strip()
