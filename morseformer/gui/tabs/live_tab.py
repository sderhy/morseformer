"""Live-mic decode tab: device picker + transcript + VU + waveform."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from morseformer.gui.audio_capture import (
    InputDevice,
    default_input_device,
    list_input_devices,
)
from morseformer.gui.services.formatting import DisplayOptions
from morseformer.gui.waveform import VuMeter, Waveform
from morseformer.gui.widgets.transcript_view import TranscriptView


class LiveTab(QWidget):
    """UI surface for the live mic flow. Pure widget — the worker is
    owned by the main window and connected via signals/slots."""

    start_requested = Signal(object)   # InputDevice
    stop_requested = Signal()
    refresh_requested = Signal()
    record_toggled = Signal(bool)      # WAV recording on/off

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        options: DisplayOptions | None = None,
    ) -> None:
        super().__init__(parent)
        self._preferred_device: tuple[str, str] | None = None
        self._build_ui(options or DisplayOptions())
        self._populate_devices()

    # ------------------------------------------------------------------ #
    # UI
    # ------------------------------------------------------------------ #

    def _build_ui(self, options: DisplayOptions) -> None:
        layout = QVBoxLayout(self)

        # Row 1: device picker + refresh.
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Input device:"))
        self.device_combo = QComboBox()
        self.device_combo.setMinimumWidth(360)
        row1.addWidget(self.device_combo, 1)
        self.refresh_btn = QPushButton("↻")
        self.refresh_btn.setFixedWidth(32)
        self.refresh_btn.setToolTip("Re-scan audio devices")
        self.refresh_btn.clicked.connect(self._populate_devices)
        row1.addWidget(self.refresh_btn)
        layout.addLayout(row1)

        # Row 2: start / stop / record / clear.
        row2 = QHBoxLayout()
        self.start_btn = QPushButton("● Start")
        self.stop_btn = QPushButton("■ Stop")
        self.stop_btn.setEnabled(False)
        self.record_btn = QPushButton("⦿ Record WAV")
        self.record_btn.setCheckable(True)
        self.record_btn.setToolTip(
            "When armed, the next live session is saved to a .wav file "
            "(the 8 kHz mono stream the model decodes)."
        )
        self.record_btn.toggled.connect(self.record_toggled.emit)
        self.clear_btn = QPushButton("Clear")
        self.start_btn.clicked.connect(self._on_start_clicked)
        self.stop_btn.clicked.connect(self._on_stop_clicked)
        self.clear_btn.clicked.connect(self._clear_transcript)
        row2.addWidget(self.start_btn)
        row2.addWidget(self.stop_btn)
        row2.addWidget(self.record_btn)
        row2.addStretch(1)
        row2.addWidget(self.clear_btn)
        layout.addLayout(row2)

        # VU + waveform.
        self.vu = VuMeter()
        layout.addWidget(self.vu)
        self.wave = Waveform(history_seconds=6.0)
        layout.addWidget(self.wave)

        # Transcript area.
        self.transcript = TranscriptView(
            options=options,
            placeholder=(
                "Pick an input device and press Start. "
                "Decoded text will appear here as it stabilises."
            ),
        )
        layout.addWidget(self.transcript, 1)

    # ------------------------------------------------------------------ #
    # Devices
    # ------------------------------------------------------------------ #

    def set_preferred_device(self, key: tuple[str, str] | None) -> None:
        """Remember the (backend, name) and re-select it in the picker."""
        self._preferred_device = key
        self._populate_devices()

    def _populate_devices(self) -> None:
        self.device_combo.blockSignals(True)
        self.device_combo.clear()
        try:
            devices = list_input_devices()
        except Exception as e:  # noqa: BLE001
            self.device_combo.addItem(f"<no audio backend: {e}>", None)
            self.device_combo.blockSignals(False)
            return
        default = default_input_device()
        default_idx = 0
        for i, d in enumerate(devices):
            self.device_combo.addItem(d.label(), d)
            # Prefer the user's last-used device; fall back to the host default.
            if self._preferred_device is not None:
                if (d.backend, d.name) == self._preferred_device:
                    default_idx = i
            elif default is not None and d.backend == default.backend and d.index == default.index:
                default_idx = i
        if not devices:
            self.device_combo.addItem("<no input devices found>", None)
        self.device_combo.setCurrentIndex(default_idx)
        self.device_combo.blockSignals(False)

    def selected_device(self) -> InputDevice | None:
        data = self.device_combo.currentData()
        return data if isinstance(data, InputDevice) else None

    # ------------------------------------------------------------------ #
    # Slots (called by main window / worker)
    # ------------------------------------------------------------------ #

    def on_transcript_fragment(self, text: str) -> None:
        self.transcript.append_fragment(text)

    def on_level_db(self, db: float) -> None:
        self.vu.set_db(db)

    def on_audio(self, audio) -> None:  # np.ndarray
        self.wave.append(audio)

    def set_display_options(self, options: DisplayOptions) -> None:
        self.transcript.set_options(options)

    def set_running(self, running: bool) -> None:
        self.start_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)
        self.device_combo.setEnabled(not running)
        self.refresh_btn.setEnabled(not running)
        self.record_btn.setEnabled(not running)

    # ------------------------------------------------------------------ #
    # Button handlers
    # ------------------------------------------------------------------ #

    def is_recording_armed(self) -> bool:
        return self.record_btn.isChecked()

    def _on_start_clicked(self) -> None:
        device = self.selected_device()
        if device is None:
            return
        self.start_requested.emit(device)

    def _on_stop_clicked(self) -> None:
        self.stop_requested.emit()

    def _clear_transcript(self) -> None:
        self.transcript.clear_transcript()
