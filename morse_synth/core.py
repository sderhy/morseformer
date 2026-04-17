"""Phase-0 Morse/CW synthesizer.

Clean output only: pure carrier sine at a fixed frequency, raised-cosine
envelope at element edges, deterministic PARIS-standard timing. Noise
injection, operator-timing variability, and HF-channel simulation (QRN,
QRM, QSB, multipath, drift) land in later phases under this same package.
"""

from __future__ import annotations

import numpy as np

from morseformer.core.morse_table import MORSE_TABLE, unit_seconds


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
    """Render a text message as clean CW audio.

    Args:
        text: Message (case-insensitive). Spaces separate words. Characters
            absent from the Morse table are silently dropped.
        wpm: Speed in words per minute (PARIS standard).
        freq: Carrier frequency in Hz (typical ham sidetones: 400-800 Hz).
        sample_rate: Audio sample rate in Hz.
        rise_ms: Raised-cosine rise/fall time at element edges, in ms.
            Shapes the keying click bandwidth.
        amplitude: Peak amplitude in [0, 1].
        tail_ms: Silent tail appended after the last element, in ms.

    Returns:
        float32 1-D numpy array of audio samples.
    """
    if not text.strip():
        return np.zeros(int(tail_ms / 1000.0 * sample_rate), dtype=np.float32)

    u_sec = unit_seconds(wpm)
    u_samples = int(round(u_sec * sample_rate))
    if u_samples < 2:
        raise ValueError(
            f"WPM too fast for sample_rate={sample_rate}: "
            f"one unit would be {u_samples} samples"
        )

    # Build the keying mask at unit resolution: a flat list where each entry is
    # either True (key down) or False (key up) for exactly one dot-unit.
    mask_units: list[bool] = []
    for word_i, word in enumerate(text.upper().split()):
        if word_i > 0:
            mask_units.extend([False] * 7)  # inter-word gap
        for char_i, ch in enumerate(word):
            code = MORSE_TABLE.get(ch)
            if code is None:
                continue
            if char_i > 0:
                mask_units.extend([False] * 3)  # inter-character gap
            for elem_i, elem in enumerate(code):
                if elem_i > 0:
                    mask_units.append(False)  # inter-element gap
                elem_units = 1 if elem == "." else 3
                mask_units.extend([True] * elem_units)

    if not mask_units:
        return np.zeros(int(tail_ms / 1000.0 * sample_rate), dtype=np.float32)

    # Expand mask to sample resolution, plus silent tail.
    n_body = len(mask_units) * u_samples
    n_tail = int(round(tail_ms / 1000.0 * sample_rate))
    envelope = np.zeros(n_body + n_tail, dtype=np.float32)
    mask = np.array(mask_units, dtype=bool)
    # Vectorised fill: each unit contributes `u_samples` samples.
    expanded = np.repeat(mask, u_samples).astype(np.float32)
    envelope[: len(expanded)] = expanded

    # Smooth edges with a Hann window of length 2*rise_n+1 (acts as a
    # raised-cosine low-pass on the rectangular keying mask).
    rise_n = max(1, int(round(rise_ms / 1000.0 * sample_rate)))
    kernel = np.hanning(2 * rise_n + 1).astype(np.float32)
    kernel /= kernel.sum()
    envelope = np.convolve(envelope, kernel, mode="same")

    # Carrier.
    t = np.arange(envelope.size, dtype=np.float32) / sample_rate
    carrier = np.sin(2.0 * np.pi * freq * t).astype(np.float32)
    return (amplitude * envelope * carrier).astype(np.float32)
