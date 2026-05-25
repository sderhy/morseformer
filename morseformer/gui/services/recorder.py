"""Incremental WAV recorder for the live decode stream.

UI-free service owned by the decoder worker. It captures exactly the
audio the model sees — mono float32 already resampled to the streaming
rate (8 kHz by default) — and writes it to a 16-bit PCM ``.wav`` as it
arrives, so a long session never buffers the whole recording in memory.

The worker calls :meth:`feed` from a single thread (the queue drainer),
so no locking is needed.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np


class WavRecorder:
    """Streams float32 mono audio to a 16-bit PCM WAV file."""

    def __init__(self, path: str | Path, sample_rate: int) -> None:
        self._path = Path(path)
        self._sample_rate = int(sample_rate)
        self._wav: wave.Wave_write | None = None
        self._frames_written = 0

    @property
    def path(self) -> Path:
        return self._path

    @property
    def seconds_written(self) -> float:
        return self._frames_written / self._sample_rate if self._sample_rate else 0.0

    def start(self) -> None:
        self._wav = wave.open(str(self._path), "wb")
        self._wav.setnchannels(1)
        self._wav.setsampwidth(2)  # int16
        self._wav.setframerate(self._sample_rate)
        self._frames_written = 0

    def feed(self, audio: np.ndarray) -> None:
        if self._wav is None or audio.size == 0:
            return
        clipped = np.clip(audio.astype(np.float32), -1.0, 1.0)
        pcm = (clipped * 32767.0).astype("<i2")
        self._wav.writeframes(pcm.tobytes())
        self._frames_written += pcm.size

    def stop(self) -> Path | None:
        """Finalise the file. Returns the path, or ``None`` if nothing
        was recorded (the empty file is removed)."""
        if self._wav is None:
            return None
        try:
            self._wav.close()
        finally:
            self._wav = None
        if self._frames_written == 0:
            self._path.unlink(missing_ok=True)
            return None
        return self._path
