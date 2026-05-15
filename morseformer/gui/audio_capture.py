"""Sounddevice-backed audio capture.

Wraps ``sounddevice.InputStream`` so the GUI can:

* enumerate input devices (USB mic, built-in, etc.) and pick one by index,
* capture mono float32 audio in the background,
* deliver it to the rest of the app as ``np.float32`` chunks resampled to
  the model's target sample rate (8 kHz).

The class is intentionally framework-agnostic — it exposes a plain Python
callback rather than Qt signals so it can be unit-tested headlessly. The
Qt worker in :mod:`morseformer.gui.decoder_worker` wraps this in a
``QObject`` and re-emits the audio chunks on the Qt event loop.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class InputDevice:
    """A capturable audio input device."""

    index: int
    name: str
    default_sample_rate: float
    max_channels: int

    def label(self) -> str:
        return f"{self.name} ({int(self.default_sample_rate)} Hz)"


def list_input_devices() -> list[InputDevice]:
    """Return every device that can be opened for input.

    Imports ``sounddevice`` lazily so importing this module does not require
    PortAudio to be installed (useful for headless smoke tests).
    """
    import sounddevice as sd

    out: list[InputDevice] = []
    for idx, info in enumerate(sd.query_devices()):
        if int(info.get("max_input_channels", 0)) <= 0:
            continue
        out.append(
            InputDevice(
                index=idx,
                name=str(info.get("name", f"device {idx}")),
                default_sample_rate=float(info.get("default_samplerate", 44100.0)),
                max_channels=int(info["max_input_channels"]),
            )
        )
    return out


def default_input_device() -> InputDevice | None:
    """Return the OS-default input, if any. None on headless systems."""
    devices = list_input_devices()
    if not devices:
        return None
    import sounddevice as sd

    try:
        default = sd.default.device
        idx = default[0] if isinstance(default, (tuple, list)) else default
    except Exception:
        idx = None
    if idx is not None:
        for d in devices:
            if d.index == idx:
                return d
    return devices[0]


class AudioCapture:
    """Background mic capture with on-the-fly resampling to the model rate.

    Usage::

        cap = AudioCapture(device_index=2, on_audio=my_callback)
        cap.start()
        ...
        cap.stop()

    ``on_audio`` receives ``np.ndarray`` mono float32 chunks at
    ``target_sample_rate``. The callback runs on a sounddevice thread,
    NOT the main thread — the Qt worker re-marshals onto the event loop.
    """

    def __init__(
        self,
        *,
        device_index: int,
        target_sample_rate: int = 8000,
        block_seconds: float = 0.1,
        on_audio: Callable[[np.ndarray], None],
    ) -> None:
        import sounddevice as sd  # noqa: F401  — fail fast if missing

        self._device_index = device_index
        self._target_sample_rate = int(target_sample_rate)
        self._block_seconds = float(block_seconds)
        self._on_audio = on_audio
        self._stream = None  # type: ignore[assignment]
        self._device_sample_rate: int = 0

    @property
    def device_sample_rate(self) -> int:
        return self._device_sample_rate

    def start(self) -> None:
        import sounddevice as sd

        info = sd.query_devices(self._device_index)
        native_rate = int(info["default_samplerate"])
        self._device_sample_rate = native_rate
        blocksize = max(1, int(round(self._block_seconds * native_rate)))

        def _cb(indata, frames, time_info, status):  # noqa: ARG001
            if status:
                # XRuns / overflows: keep going, the user will see VU dips.
                pass
            mono = indata[:, 0] if indata.ndim > 1 else indata
            chunk = self._resample(np.asarray(mono, dtype=np.float32), native_rate)
            if chunk.size:
                self._on_audio(chunk)

        self._stream = sd.InputStream(
            device=self._device_index,
            samplerate=native_rate,
            channels=1,
            dtype="float32",
            blocksize=blocksize,
            callback=_cb,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is None:
            return
        try:
            self._stream.stop()
            self._stream.close()
        finally:
            self._stream = None

    def _resample(self, audio: np.ndarray, src_rate: int) -> np.ndarray:
        if src_rate == self._target_sample_rate:
            return audio
        # scipy's polyphase resampler is cheap, anti-aliased, and works
        # block-by-block without context — perfect for short chunks.
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(src_rate, self._target_sample_rate)
        up = self._target_sample_rate // g
        down = src_rate // g
        return resample_poly(audio, up, down).astype(np.float32)
