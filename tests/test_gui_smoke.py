"""Headless smoke tests for the PySide6 GUI.

These tests must NOT require a display server. They only verify that
modules import cleanly, that the audio-capture device-listing returns
something sensible (possibly empty), and that the pure helpers in
``decoder_worker`` behave correctly.

The full Qt main window is not constructed here — that requires a
``QApplication`` and a display server, which CI runners do not have.
"""

from __future__ import annotations

import math
import wave
from pathlib import Path

import numpy as np
import pytest


def test_gui_modules_import_clean() -> None:
    # Importing must work without a display, without PortAudio devices,
    # and without any model weights on disk.
    import morseformer.gui  # noqa: F401
    from morseformer.gui import (  # noqa: F401
        app,
        audio_capture,
        decoder_worker,
        waveform,
    )
    from morseformer.gui.panels import settings_panel  # noqa: F401
    from morseformer.gui.tabs import file_tab, live_tab  # noqa: F401


def test_list_input_devices_returns_list() -> None:
    from morseformer.gui.audio_capture import list_input_devices

    devices = list_input_devices()
    assert isinstance(devices, list)
    for d in devices:
        assert isinstance(d.index, int)
        assert isinstance(d.name, str)
        assert d.max_channels >= 1
        assert d.default_sample_rate > 0
        assert d.backend in ("sounddevice", "pulse")


def test_pulse_sources_are_picked_up_when_pulse_running() -> None:
    """If pactl + a Pulse socket exist (WSLg, native pulse / pipewire),
    list_input_devices should expose at least one source. We do not
    assert which one — only that the path actually returns something
    when Pulse is reachable."""
    import shutil

    from morseformer.gui.audio_capture import _pulse_available, list_input_devices

    if not (_pulse_available() and shutil.which("pactl")):
        pytest.skip("no PulseAudio available on this host")
    devices = list_input_devices()
    backends = {d.backend for d in devices}
    # If sounddevice already found inputs we may not see pulse here;
    # that's the documented behaviour. Just check we got *something*.
    assert devices, "PulseAudio is reachable but no input devices listed"
    assert backends, "device list missing a backend tag"


def test_rms_db_silent_audio_is_floor() -> None:
    from morseformer.gui.decoder_worker import _rms_db

    db = _rms_db(np.zeros(8000, dtype=np.float32))
    assert db <= -100.0


def test_rms_db_full_scale_is_zero_db() -> None:
    from morseformer.gui.decoder_worker import _rms_db

    db = _rms_db(np.ones(8000, dtype=np.float32))
    assert math.isclose(db, 0.0, abs_tol=0.1)


def test_rms_db_half_amplitude_is_about_minus_six_db() -> None:
    from morseformer.gui.decoder_worker import _rms_db

    db = _rms_db(0.5 * np.ones(8000, dtype=np.float32))
    assert -6.5 <= db <= -5.5


def test_load_wav_mono_round_trip(tmp_path: Path) -> None:
    from morseformer.gui.decoder_worker import _load_wav_mono

    sample_rate = 16000
    n = sample_rate * 1
    samples = (0.5 * np.sin(2 * np.pi * 600.0 * np.arange(n) / sample_rate)).astype(np.float32)
    pcm = (samples * 32767.0).astype(np.int16)
    path = tmp_path / "tone.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())

    audio, sr = _load_wav_mono(path)
    assert sr == sample_rate
    assert audio.shape == (n,)
    assert audio.dtype == np.float32
    # Tolerance allows for int16 quantisation.
    assert np.max(np.abs(audio - samples)) < 1e-3


def test_load_wav_stereo_downmixes_to_mono(tmp_path: Path) -> None:
    from morseformer.gui.decoder_worker import _load_wav_mono

    n = 100
    left = np.full(n, 0.5, dtype=np.float32)
    right = np.full(n, -0.5, dtype=np.float32)
    interleaved = np.empty(2 * n, dtype=np.float32)
    interleaved[0::2] = left
    interleaved[1::2] = right
    pcm = (interleaved * 32767.0).astype(np.int16)
    path = tmp_path / "stereo.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(8000)
        w.writeframes(pcm.tobytes())

    audio, _sr = _load_wav_mono(path)
    assert audio.shape == (n,)
    # Mean of +0.5 / -0.5 = 0, with int16 quantisation noise.
    assert np.max(np.abs(audio)) < 1e-3


def test_decoder_worker_instantiates_without_qapp() -> None:
    # DecoderWorker is a QObject — constructible without a QApplication.
    # Widget instantiation still requires QApplication and is skipped.
    from morseformer.gui.decoder_worker import DecoderWorker

    w = DecoderWorker(device="cpu")
    assert w._preset.name == "live"      # default preset


def test_set_preset_with_unknown_name_emits_error_signal() -> None:
    from morseformer.gui.decoder_worker import DecoderWorker

    w = DecoderWorker(device="cpu")
    captured: list[str] = []
    w.error.connect(captured.append)
    w.set_preset("does-not-exist")
    assert captured
    assert "does-not-exist" in captured[-1]


def test_cli_includes_gui_subcommand() -> None:
    from morseformer.cli import build_parser

    parser = build_parser()
    sub_action = next(
        a for a in parser._actions if a.__class__.__name__ == "_SubParsersAction"
    )
    assert "gui" in sub_action.choices


@pytest.mark.skipif(
    not Path("release/rnnt_phase5_5.pt").exists(),
    reason="no local release checkpoint",
)
def test_decoder_worker_can_load_local_model() -> None:
    """Sanity-check that a release checkpoint loads cleanly through the
    worker's path. Skipped without a local release/ checkpoint."""
    from morseformer.gui.decoder_worker import DecoderWorker

    w = DecoderWorker(device="cpu")
    # Trigger lazy load.
    w._ensure_model_loaded()
    assert w._model is not None


def test_main_window_constructs_offscreen() -> None:
    """Build the full main window with a fake Qt platform. Skipped if the
    platform plugin can't initialise (CI containers without Qt deps)."""
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError:
        pytest.skip("PySide6 not installed")

    from morseformer.gui.app import MainWindow

    _app = QApplication.instance() or QApplication([])
    win = MainWindow()
    try:
        assert win.windowTitle().startswith("morseformer")
        assert [win.tabs.tabText(i) for i in range(win.tabs.count())] == ["Live", "File"]
        # Settings populate from the default preset.
        assert win.settings.preset.currentText() == "live"
        assert win.settings.conf_thr.value() == pytest.approx(0.6)
        assert win.settings.digit_thr.value() == pytest.approx(0.9)
    finally:
        win.close()
        # Don't quit the application — pytest may have other tests using Qt.
