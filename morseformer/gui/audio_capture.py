"""Audio capture with two backends: sounddevice (default) and PulseAudio.

sounddevice's PortAudio works everywhere on Linux native, macOS and
Windows. Under WSL2 it falls flat — PortAudio enumerates ALSA devices
via ``/proc/asound/cards``, which is empty since WSL2 has no real sound
hardware, and PulseAudio appears only through an ALSA plugin that can
*open* ``device="pulse"`` but never shows up in PortAudio's device list.

To stay one binary across all three platforms we add a second backend
that talks to PulseAudio directly (``pasimple`` for capture, ``pactl``
for enumeration). When ``PULSE_SERVER`` is set or a Pulse socket is
visible, we list its sources as additional input devices. The GUI does
not need to know which backend a given device uses — the factory
``AudioCapture(device=...)`` picks the right one.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

_TARGET_SAMPLE_RATE_DEFAULT = 8000


@dataclass(frozen=True)
class InputDevice:
    """A capturable audio input device.

    ``backend == "sounddevice"`` → ``index`` is a PortAudio device index.
    ``backend == "pulse"``       → ``pulse_source`` is the source name
    passed to ``pasimple``; ``index`` is the Pulse source idx (purely
    informational).
    """

    index: int
    name: str
    default_sample_rate: float
    max_channels: int
    backend: str = "sounddevice"
    pulse_source: str | None = None

    def label(self) -> str:
        suffix = " (pulse)" if self.backend == "pulse" else ""
        return f"[{self.index}] {self.name} ({int(self.default_sample_rate)} Hz){suffix}"


# ---------------------------------------------------------------------- #
# Enumeration
# ---------------------------------------------------------------------- #


def _list_sounddevice_inputs() -> list[InputDevice]:
    try:
        import sounddevice as sd
    except Exception:
        return []
    try:
        devices = sd.query_devices()
    except Exception:
        return []
    out: list[InputDevice] = []
    for idx, info in enumerate(devices):
        if int(info.get("max_input_channels", 0)) <= 0:
            continue
        out.append(
            InputDevice(
                index=idx,
                name=str(info.get("name", f"device {idx}")),
                default_sample_rate=float(info.get("default_samplerate", 44100.0)),
                max_channels=int(info["max_input_channels"]),
                backend="sounddevice",
            )
        )
    return out


def _pulse_available() -> bool:
    if shutil.which("pactl") is None:
        return False
    if os.environ.get("PULSE_SERVER"):
        return True
    return os.path.exists("/mnt/wslg/PulseServer") or os.path.exists(
        f"/run/user/{os.getuid()}/pulse/native"
    )


def _list_pulse_sources() -> list[InputDevice]:
    if not _pulse_available():
        return []
    try:
        out = subprocess.run(
            ["pactl", "--format=json", "list", "short", "sources"],
            capture_output=True, text=True, timeout=2.0, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if out.returncode != 0:
        return []
    try:
        data = json.loads(out.stdout)
    except json.JSONDecodeError:
        return []
    devices: list[InputDevice] = []
    for entry in data:
        name = str(entry.get("name", ""))
        if not name or name.endswith(".monitor"):
            # Skip monitors of output sinks — they replay what's *played*.
            continue
        spec = str(entry.get("sample_specification", "16000Hz"))
        rate = _parse_sample_rate(spec)
        channels = _parse_channels(spec)
        devices.append(
            InputDevice(
                index=int(entry.get("index", -1)),
                name=name,
                default_sample_rate=rate,
                max_channels=channels,
                backend="pulse",
                pulse_source=name,
            )
        )
    return devices


def _parse_sample_rate(spec: str) -> float:
    # spec looks like "s16le 1ch 44100Hz".
    for tok in spec.split():
        if tok.endswith("Hz"):
            try:
                return float(tok[:-2])
            except ValueError:
                pass
    return 44100.0


def _parse_channels(spec: str) -> int:
    for tok in spec.split():
        if tok.endswith("ch"):
            try:
                return int(tok[:-2])
            except ValueError:
                pass
    return 1


def list_input_devices() -> list[InputDevice]:
    """Return every audio input device the host exposes.

    Combines sounddevice (PortAudio) and PulseAudio enumerations. Pulse
    sources only appear when PortAudio comes up empty *or* when running
    under WSL2 — on a normal Linux desktop the PortAudio list already
    contains the Pulse default, and re-listing them would be confusing.
    """
    sd_devices = _list_sounddevice_inputs()
    pulse_devices = _list_pulse_sources()
    if not sd_devices and pulse_devices:
        return pulse_devices
    # Show Pulse alongside PortAudio only when PortAudio failed to find
    # anything useful but Pulse has sources.
    return sd_devices


def default_input_device() -> InputDevice | None:
    devices = list_input_devices()
    if not devices:
        return None
    if devices[0].backend == "sounddevice":
        try:
            import sounddevice as sd

            default = sd.default.device
            idx = default[0] if isinstance(default, (tuple, list)) else default
        except Exception:
            idx = None
        if idx is not None:
            for d in devices:
                if d.backend == "sounddevice" and d.index == idx:
                    return d
    return devices[0]


# ---------------------------------------------------------------------- #
# Capture
# ---------------------------------------------------------------------- #


class AudioCapture:
    """Background mic capture with on-the-fly resampling to the model rate.

    Pass either an ``InputDevice`` (preferred) or a sounddevice integer
    index for backward compatibility.
    """

    def __init__(
        self,
        *,
        device: InputDevice | int,
        device_index: int | None = None,
        target_sample_rate: int = _TARGET_SAMPLE_RATE_DEFAULT,
        block_seconds: float = 0.1,
        on_audio: Callable[[np.ndarray], None],
    ) -> None:
        if isinstance(device, InputDevice):
            self._device = device
        else:
            # Legacy / sounddevice index path.
            self._device = InputDevice(
                index=int(device),
                name=f"sd:{device}",
                default_sample_rate=44100.0,
                max_channels=1,
                backend="sounddevice",
            )
        # Backward-compat: some callers still pass device_index=N as a
        # standalone kw. Honour it if device wasn't explicit.
        if device_index is not None and not isinstance(device, InputDevice):
            self._device = InputDevice(
                index=int(device_index),
                name=f"sd:{device_index}",
                default_sample_rate=44100.0,
                max_channels=1,
                backend="sounddevice",
            )
        self._target_sample_rate = int(target_sample_rate)
        self._block_seconds = float(block_seconds)
        self._on_audio = on_audio
        self._device_sample_rate: int = 0
        self._impl: _SounddeviceImpl | _PulseImpl | None = None

    @property
    def device_sample_rate(self) -> int:
        return self._device_sample_rate

    def start(self) -> None:
        if self._device.backend == "pulse":
            self._impl = _PulseImpl(
                source_name=self._device.pulse_source or self._device.name,
                target_rate=self._target_sample_rate,
                block_seconds=self._block_seconds,
                on_audio=self._on_audio,
            )
        else:
            self._impl = _SounddeviceImpl(
                device_index=self._device.index,
                target_rate=self._target_sample_rate,
                block_seconds=self._block_seconds,
                on_audio=self._on_audio,
            )
        self._impl.start()
        self._device_sample_rate = self._impl.native_rate

    def stop(self) -> None:
        if self._impl is None:
            return
        try:
            self._impl.stop()
        finally:
            self._impl = None


class _SounddeviceImpl:
    def __init__(
        self,
        *,
        device_index: int,
        target_rate: int,
        block_seconds: float,
        on_audio: Callable[[np.ndarray], None],
    ) -> None:
        import sounddevice as sd  # noqa: F401 — fail fast if missing

        self._device_index = device_index
        self._target_rate = target_rate
        self._block_seconds = block_seconds
        self._on_audio = on_audio
        self._stream = None
        self.native_rate = 0

    def start(self) -> None:
        import sounddevice as sd

        info = sd.query_devices(self._device_index)
        self.native_rate = int(info["default_samplerate"])
        blocksize = max(1, int(round(self._block_seconds * self.native_rate)))

        def _cb(indata, frames, time_info, status):  # noqa: ARG001
            mono = indata[:, 0] if indata.ndim > 1 else indata
            chunk = _resample(np.asarray(mono, dtype=np.float32),
                              self.native_rate, self._target_rate)
            if chunk.size:
                self._on_audio(chunk)

        self._stream = sd.InputStream(
            device=self._device_index,
            samplerate=self.native_rate,
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


class _PulseImpl:
    """Capture via the PulseAudio simple API.

    pasimple's reads are blocking, so we run a background thread that
    loops on ``read()`` and dispatches chunks through the user callback.
    """

    def __init__(
        self,
        *,
        source_name: str,
        target_rate: int,
        block_seconds: float,
        on_audio: Callable[[np.ndarray], None],
    ) -> None:
        import pasimple  # noqa: F401 — fail fast if missing

        self._source_name = source_name
        self._target_rate = target_rate
        self._block_seconds = block_seconds
        self._on_audio = on_audio
        self._stream = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        # Pulse exposes a sample format per source; we standardise on
        # float32 LE at 44100 Hz mono. PA will resample as needed.
        self.native_rate = 44100

    def start(self) -> None:
        import pasimple

        self._stop_event.clear()
        self._stream = pasimple.PaSimple(
            direction=pasimple.PA_STREAM_RECORD,
            format=pasimple.PA_SAMPLE_FLOAT32LE,
            channels=1,
            rate=self.native_rate,
            app_name="morseformer",
            stream_name=f"capture:{self._source_name}",
            device_name=self._source_name,
        )
        self._thread = threading.Thread(
            target=self._read_loop, name="pulse-capture", daemon=True,
        )
        self._thread.start()

    def _read_loop(self) -> None:
        assert self._stream is not None
        bytes_per_sample = 4  # float32
        block_bytes = max(1, int(self._block_seconds * self.native_rate)) * bytes_per_sample
        while not self._stop_event.is_set():
            try:
                raw = self._stream.read(block_bytes)
            except Exception:
                break
            audio = np.frombuffer(raw, dtype=np.float32).copy()
            if audio.size == 0:
                continue
            chunk = _resample(audio, self.native_rate, self._target_rate)
            if chunk.size:
                try:
                    self._on_audio(chunk)
                except Exception:
                    # Never let a UI callback failure kill the thread.
                    pass

    def stop(self) -> None:
        self._stop_event.set()
        thread = self._thread
        self._thread = None
        if self._stream is not None:
            try:
                self._stream.close()
            finally:
                self._stream = None
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)


def _resample(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate:
        return audio
    from math import gcd

    from scipy.signal import resample_poly

    g = gcd(src_rate, dst_rate)
    up = dst_rate // g
    down = src_rate // g
    return resample_poly(audio, up, down).astype(np.float32)
