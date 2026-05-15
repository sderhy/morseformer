"""QObject worker that owns the model + decoder, off the UI thread.

The worker is meant to be moved into a dedicated ``QThread``. It exposes
slots for ``start_live`` / ``stop_live`` / ``decode_file`` and signals for
transcript fragments, VU level, and state changes. The UI never touches
torch or sounddevice directly.

Live audio flow:

* :class:`audio_capture.AudioCapture` opens the chosen device on a
  PortAudio thread.
* Each block is pushed into a ``queue.Queue`` (thread-safe).
* A ``QTimer`` running on the worker's QThread drains the queue, feeds
  the bytes into :class:`StreamingDecoder.feed`, and emits any returned
  text fragments + the RMS level via signals.
"""

from __future__ import annotations

import math
import queue
import time
import wave
from pathlib import Path

import numpy as np
import torch
from PySide6.QtCore import QObject, QTimer, Signal, Slot

from morseformer.cli.presets import PRESETS, Preset
from morseformer.cli.registry import resolve_model
from morseformer.decoding.streaming import (
    StreamingConfig,
    StreamingDecoder,
    decode_offline,
)
from morseformer.gui.audio_capture import AudioCapture
from morseformer.models.acoustic import AcousticConfig
from morseformer.models.rnnt import RnntConfig, RnntModel

_DRAIN_INTERVAL_MS = 50  # how often the worker pulls from the audio queue


def _load_rnnt(path: Path, device: torch.device) -> RnntModel:
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    state = dict(ckpt["model"])
    for k, v in (ckpt.get("ema") or {}).items():
        if k in state:
            state[k] = v
    enc = ckpt["config"]["model"]["encoder"]
    vocab_size = ckpt["config"]["model"].get("vocab_size")
    extra = {"vocab_size": vocab_size} if vocab_size is not None else {}
    encoder_cfg = AcousticConfig(
        d_model=enc["d_model"], n_heads=enc["n_heads"], n_layers=enc["n_layers"],
        ff_expansion=enc["ff_expansion"], conv_kernel=enc["conv_kernel"],
        dropout=enc["dropout"], **extra,
    )
    rnnt_cfg = RnntConfig(
        encoder=encoder_cfg,
        d_pred=ckpt["config"]["model"]["d_pred"],
        pred_lstm_layers=ckpt["config"]["model"]["pred_lstm_layers"],
        d_joint=ckpt["config"]["model"]["d_joint"],
        dropout=ckpt["config"]["model"]["dropout"],
        **extra,
    )
    model = RnntModel(rnnt_cfg).to(device).eval()
    model.load_state_dict(state)
    return model


def _rms_db(audio: np.ndarray) -> float:
    if audio.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(np.square(audio.astype(np.float32))) + 1e-12))
    return 20.0 * math.log10(max(rms, 1e-6))


def _load_wav_mono(path: Path) -> tuple[np.ndarray, int]:
    """Load a .wav file as mono float32 in [-1, 1]."""
    with wave.open(str(path), "rb") as w:
        n_frames = w.getnframes()
        sample_rate = w.getframerate()
        n_channels = w.getnchannels()
        sampwidth = w.getsampwidth()
        raw = w.readframes(n_frames)
    if sampwidth == 2:
        x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        x = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    elif sampwidth == 1:
        x = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"unsupported sample width: {sampwidth} bytes")
    if n_channels > 1:
        x = x.reshape(-1, n_channels).mean(axis=1)
    return x, sample_rate


class DecoderWorker(QObject):
    """Owns the model + StreamingDecoder + audio capture."""

    # State / progress reporting.
    status_changed = Signal(str)
    error = Signal(str)
    ready_changed = Signal(bool)

    # Live decode.
    transcript_fragment = Signal(str)
    level_db = Signal(float)
    raw_audio = Signal(np.ndarray)

    # File decode.
    file_decoded = Signal(str)

    def __init__(self, *, device: str = "cpu") -> None:
        super().__init__()
        self._device = torch.device(device)
        self._model: RnntModel | None = None
        self._loaded_acoustic: str | None = None
        self._decoder: StreamingDecoder | None = None
        self._capture: AudioCapture | None = None
        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=64)
        self._drain_timer: QTimer | None = None
        self._preset: Preset = PRESETS["live"]
        # Settings overrides (applied on top of preset).
        self._confidence_threshold: float | None = None
        self._digit_threshold: float | None = None
        self._carrier_hz: float = 600.0
        self._running = False

    # ------------------------------------------------------------------ #
    # Lifecycle (called once after moveToThread)
    # ------------------------------------------------------------------ #

    @Slot()
    def initialize(self) -> None:
        """Set up resources that must live on the worker thread."""
        self._drain_timer = QTimer()
        self._drain_timer.setInterval(_DRAIN_INTERVAL_MS)
        self._drain_timer.timeout.connect(self._drain_queue)

    # ------------------------------------------------------------------ #
    # Settings (slot)
    # ------------------------------------------------------------------ #

    @Slot(str)
    def set_preset(self, name: str) -> None:
        if name not in PRESETS:
            self.error.emit(f"unknown preset '{name}'")
            return
        self._preset = PRESETS[name]
        self.status_changed.emit(f"preset → {name}")

    @Slot(float)
    def set_confidence_threshold(self, value: float) -> None:
        self._confidence_threshold = float(value)

    @Slot(float)
    def set_digit_threshold(self, value: float) -> None:
        self._digit_threshold = float(value)

    @Slot(float)
    def set_carrier_hz(self, value: float) -> None:
        self._carrier_hz = float(value)

    # ------------------------------------------------------------------ #
    # Live decode (slots)
    # ------------------------------------------------------------------ #

    @Slot(int)
    def start_live(self, device_index: int) -> None:
        if self._running:
            return
        try:
            self._ensure_model_loaded()
            cfg = self._streaming_config()
            assert self._model is not None
            self._decoder = StreamingDecoder(self._model, cfg, device=self._device)
            self._queue = queue.Queue(maxsize=64)

            def _on_audio(chunk: np.ndarray) -> None:
                try:
                    self._queue.put_nowait(chunk)
                except queue.Full:
                    # Drop the oldest item if we cannot keep up. Audio
                    # latency is preferable to a silent crash.
                    try:
                        self._queue.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        self._queue.put_nowait(chunk)
                    except queue.Full:
                        pass

            self._capture = AudioCapture(
                device_index=device_index,
                target_sample_rate=cfg.sample_rate,
                block_seconds=0.1,
                on_audio=_on_audio,
            )
            self._capture.start()
            assert self._drain_timer is not None
            self._drain_timer.start()
            self._running = True
            self.status_changed.emit(
                f"live • device {device_index} • native "
                f"{self._capture.device_sample_rate} Hz → {cfg.sample_rate} Hz"
            )
        except Exception as e:  # noqa: BLE001 — surface every failure to UI
            self.error.emit(f"start_live failed: {e}")
            self._cleanup_capture()

    @Slot()
    def stop_live(self) -> None:
        if not self._running:
            return
        if self._drain_timer is not None:
            self._drain_timer.stop()
        self._cleanup_capture()
        if self._decoder is not None:
            try:
                tail = self._decoder.flush()
                if tail:
                    self.transcript_fragment.emit(tail)
            finally:
                self._decoder = None
        self._running = False
        self.status_changed.emit("stopped")

    def _cleanup_capture(self) -> None:
        if self._capture is not None:
            try:
                self._capture.stop()
            finally:
                self._capture = None

    @Slot()
    def _drain_queue(self) -> None:
        if not self._running or self._decoder is None:
            return
        chunks: list[np.ndarray] = []
        while True:
            try:
                chunks.append(self._queue.get_nowait())
            except queue.Empty:
                break
        if not chunks:
            return
        audio = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
        # Emit raw audio + level before decoding so the UI updates even
        # if torch is slow.
        self.level_db.emit(_rms_db(audio))
        self.raw_audio.emit(audio)
        try:
            fragments = self._decoder.feed(audio)
        except Exception as e:  # noqa: BLE001
            self.error.emit(f"decode failed: {e}")
            return
        for f in fragments:
            if f:
                self.transcript_fragment.emit(f)

    # ------------------------------------------------------------------ #
    # File decode
    # ------------------------------------------------------------------ #

    @Slot(str)
    def decode_file(self, path: str) -> None:
        p = Path(path)
        if not p.exists():
            self.error.emit(f"file not found: {path}")
            return
        try:
            self._ensure_model_loaded()
            assert self._model is not None
            cfg = self._streaming_config()
            self.status_changed.emit(f"decoding {p.name} …")
            audio, src_rate = _load_wav_mono(p)
            if src_rate != cfg.sample_rate:
                from math import gcd

                from scipy.signal import resample_poly

                g = gcd(src_rate, cfg.sample_rate)
                audio = resample_poly(
                    audio, cfg.sample_rate // g, src_rate // g,
                ).astype(np.float32)
            t0 = time.perf_counter()
            text = decode_offline(self._model, audio, cfg, device=self._device)
            dt = time.perf_counter() - t0
            self.status_changed.emit(
                f"decoded {p.name} in {dt:.1f}s ({len(text)} chars)"
            )
            self.file_decoded.emit(text)
        except Exception as e:  # noqa: BLE001
            self.error.emit(f"decode_file failed: {e}")

    # ------------------------------------------------------------------ #
    # Model loading
    # ------------------------------------------------------------------ #

    def _ensure_model_loaded(self) -> None:
        target = self._preset.acoustic
        if self._model is not None and self._loaded_acoustic == target:
            return
        self.ready_changed.emit(False)
        self.status_changed.emit(f"loading {target} …")
        ckpt_path = resolve_model(target)
        self._model = _load_rnnt(ckpt_path, self._device)
        self._loaded_acoustic = target
        self.ready_changed.emit(True)
        self.status_changed.emit(f"loaded {target} ({ckpt_path.name})")

    def _streaming_config(self) -> StreamingConfig:
        return StreamingConfig(
            confidence_threshold=(
                self._confidence_threshold
                if self._confidence_threshold is not None
                else self._preset.confidence_threshold
            ),
            digit_threshold=(
                self._digit_threshold
                if self._digit_threshold is not None
                else self._preset.digit_threshold
            ),
            carrier_hz=self._carrier_hz,
        )
