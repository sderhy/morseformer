"""Main window: top settings bar + Live/File tabs + menus + status bar."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QMainWindow,
    QMessageBox,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from morseformer import __version__
from morseformer.gui.decoder_worker import DecoderWorker
from morseformer.gui.panels.settings_panel import SettingsPanel
from morseformer.gui.services import exporter
from morseformer.gui.services.config_store import ConfigStore
from morseformer.gui.services.formatting import DisplayOptions
from morseformer.gui.tabs.file_tab import FileTab
from morseformer.gui.tabs.live_tab import LiveTab


class MainWindow(QMainWindow):
    _start_live_requested = Signal(object)
    _stop_live_requested = Signal()
    _decode_file_requested = Signal(str)
    _set_record_path_requested = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"morseformer · v{__version__}")
        self.resize(960, 640)

        self.config = ConfigStore()
        self.display_options = self.config.get_display_options()

        # Central layout: settings panel + tabs.
        central = QWidget()
        self.setCentralWidget(central)
        v = QVBoxLayout(central)
        v.setContentsMargins(0, 0, 0, 0)

        self.settings = SettingsPanel()
        v.addWidget(self.settings)

        self.tabs = QTabWidget()
        self.live_tab = LiveTab(options=self.display_options)
        self.file_tab = FileTab(options=self.display_options)
        self.tabs.addTab(self.live_tab, "Live")
        self.tabs.addTab(self.file_tab, "File")
        v.addWidget(self.tabs, 1)

        self.live_tab.set_preferred_device(self.config.get_last_device())

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Loading model on first decode …")

        self._build_menu()
        self._build_worker()
        self._wire_signals()

    # ------------------------------------------------------------------ #
    # Menus
    # ------------------------------------------------------------------ #

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        self.save_action = QAction("&Save transcript …", self)
        self.save_action.setShortcut(QKeySequence.StandardKey.Save)
        self.save_action.triggered.connect(self._save_transcript)
        file_menu.addAction(self.save_action)
        file_menu.addSeparator()
        quit_action = QAction("&Quit", self)
        quit_action.setShortcut(QKeySequence.StandardKey.Quit)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        text_menu = self.menuBar().addMenu("&Text")
        o = self.display_options
        self.act_break_tokens = self._toggle_action(
            text_menu, "Line break after = / KN", o.break_tokens,
        )
        self.act_break_after_k = self._toggle_action(
            text_menu, "Line break after standalone K", o.break_after_k,
        )
        self.act_lowercase = self._toggle_action(
            text_menu, "Lower case", o.lowercase,
        )
        for act in (self.act_break_tokens, self.act_break_after_k, self.act_lowercase):
            act.toggled.connect(self._on_display_options_changed)

        help_menu = self.menuBar().addMenu("&Help")
        about_action = QAction("&About morseformer", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _show_about(self) -> None:
        repo = "https://github.com/sderhy/morseformer"
        hf = "https://huggingface.co/sderhy/morseformer"
        box = QMessageBox(self)
        box.setWindowTitle("About morseformer")
        box.setTextFormat(Qt.TextFormat.RichText)
        box.setIconPixmap(self.windowIcon().pixmap(64, 64))
        box.setText(f"<h3>morseformer v{__version__}</h3>")
        box.setInformativeText(
            "<p>Open-source, transformer-based <b>Morse / CW decoder</b> with a "
            "built-in ham-specialised language model. It turns received CW audio "
            "(live from a mic / transceiver, or a WAV file) into readable text, "
            "and links detected callsigns straight to QRZ.com.</p>"
            "<p><b>How it works:</b> an acoustic RNN-T model trained on synthetic "
            "and real on-air CW transcribes the audio; display formatting and "
            "callsign detection run on top.</p>"
            f"<p><b>Project:</b> <a href=\"{repo}\">{repo}</a><br>"
            f"<b>Models:</b> <a href=\"{hf}\">Hugging Face — sderhy/morseformer</a></p>"
            "<p><b>Author:</b> sderhy &lt;sderhy@gmail.com&gt;<br>"
            "<b>License:</b> Apache-2.0</p>"
            "<p><b>Built with:</b> PyTorch · torchaudio · PySide6 (Qt) · NumPy · SciPy. "
            "Thanks to the amateur-radio operators whose on-air recordings made the "
            "real-audio training possible.</p>"
        )
        box.setStandardButtons(QMessageBox.StandardButton.Ok)
        box.exec()

    def _toggle_action(self, menu, label: str, checked: bool) -> QAction:
        act = QAction(label, self, checkable=True)
        act.setChecked(checked)
        menu.addAction(act)
        return act

    def _on_display_options_changed(self) -> None:
        self.display_options = DisplayOptions(
            break_tokens=self.act_break_tokens.isChecked(),
            break_after_k=self.act_break_after_k.isChecked(),
            lowercase=self.act_lowercase.isChecked(),
        )
        self.live_tab.set_display_options(self.display_options)
        self.file_tab.set_display_options(self.display_options)
        self.config.set_display_options(self.display_options)

    def _save_transcript(self) -> None:
        current = self.tabs.currentWidget()
        text = (
            current.display_text()
            if hasattr(current, "display_text")
            else ""
        )
        if not text:
            self.status_bar.showMessage("nothing to save — transcript is empty", 6000)
            return
        default = f"morseformer-{datetime.now():%Y%m%d-%H%M%S}.txt"
        path, _filter = QFileDialog.getSaveFileName(
            self, "Save transcript", default, "Text files (*.txt);;All files (*)",
        )
        if not path:
            return
        try:
            exporter.export_text(text, path)
        except OSError as e:
            QMessageBox.warning(self, "Save failed", str(e))
            return
        self.status_bar.showMessage(f"saved transcript → {Path(path).name}", 6000)

    # ------------------------------------------------------------------ #
    # Worker
    # ------------------------------------------------------------------ #

    def _build_worker(self) -> None:
        self._thread = QThread(self)
        self._thread.setObjectName("decoder-worker")
        self.worker = DecoderWorker(device="cpu")
        self.worker.moveToThread(self._thread)
        self._thread.started.connect(self.worker.initialize)
        self._thread.start()

    def _wire_signals(self) -> None:
        w = self.worker

        # Settings → worker.
        self.settings.preset_changed.connect(w.set_preset)
        self.settings.confidence_threshold_changed.connect(w.set_confidence_threshold)
        self.settings.digit_threshold_changed.connect(w.set_digit_threshold)
        self.settings.carrier_changed.connect(w.set_carrier_hz)
        self.settings.bandwidth_changed.connect(w.set_bandwidth_hz)

        # Live tab ↔ worker. Routed via main-window signals so the QThread
        # call site stays explicit (auto-queued for cross-thread).
        self.live_tab.start_requested.connect(self._on_start_live)
        self.live_tab.stop_requested.connect(self._stop_live_requested.emit)
        self.live_tab.record_toggled.connect(self._on_record_toggled)
        self._start_live_requested.connect(
            w.start_live, type=Qt.ConnectionType.QueuedConnection,
        )
        self._stop_live_requested.connect(
            w.stop_live, type=Qt.ConnectionType.QueuedConnection,
        )
        self._set_record_path_requested.connect(
            w.set_record_path, type=Qt.ConnectionType.QueuedConnection,
        )

        w.transcript_fragment.connect(self.live_tab.on_transcript_fragment)
        w.level_db.connect(self.live_tab.on_level_db)
        w.raw_audio.connect(self.live_tab.on_audio)
        w.recording_saved.connect(self._on_recording_saved)
        w.status_changed.connect(self._on_status)
        w.error.connect(self._on_error)
        w.ready_changed.connect(self._on_ready_changed)

        # File tab ↔ worker.
        self.file_tab.decode_requested.connect(self._decode_file_requested.emit)
        self._decode_file_requested.connect(
            w.decode_file, type=Qt.ConnectionType.QueuedConnection,
        )
        w.file_decoded.connect(self.file_tab.on_file_decoded)
        w.error.connect(lambda _msg: self.file_tab.on_error())

    # ------------------------------------------------------------------ #
    # Live + recording handlers
    # ------------------------------------------------------------------ #

    def _on_start_live(self, device) -> None:
        # Remember the chosen input device for next launch.
        self.config.set_last_device(device.backend, device.name)
        self._start_live_requested.emit(device)

    def _on_record_toggled(self, armed: bool) -> None:
        if not armed:
            self._set_record_path_requested.emit(None)
            return
        default = f"morseformer-{datetime.now():%Y%m%d-%H%M%S}.wav"
        path, _filter = QFileDialog.getSaveFileName(
            self, "Record session to WAV", default, "WAV files (*.wav)",
        )
        if not path:
            # User cancelled → un-arm the toggle.
            self.live_tab.record_btn.setChecked(False)
            return
        self._set_record_path_requested.emit(path)
        self.status_bar.showMessage(f"recording armed → {Path(path).name}", 6000)

    def _on_recording_saved(self, path: str) -> None:
        self.status_bar.showMessage(f"recording saved → {Path(path).name}", 8000)
        self.live_tab.record_btn.setChecked(False)

    # ------------------------------------------------------------------ #
    # Status / lifecycle
    # ------------------------------------------------------------------ #

    def _on_status(self, msg: str) -> None:
        self.status_bar.showMessage(msg, 8000)
        # Approximate live running state from the message — keeps the
        # widget enabled/disabled without an extra signal.
        if msg.startswith("live"):
            self.live_tab.set_running(True)
        elif msg == "stopped":
            self.live_tab.set_running(False)

    def _on_error(self, msg: str) -> None:
        self.status_bar.showMessage(f"[error] {msg}", 12000)
        self.live_tab.set_running(False)

    def _on_ready_changed(self, ready: bool) -> None:
        # Could be used later to gate buttons; today the start button
        # implicitly waits because start_live runs synchronously on the
        # worker.
        if ready:
            self.status_bar.showMessage("model ready", 4000)

    def closeEvent(self, event) -> None:  # noqa: N802, D401
        try:
            self._stop_live_requested.emit()
        except Exception:
            pass
        self.config.sync()
        self._thread.quit()
        self._thread.wait(2000)
        super().closeEvent(event)


def main(argv: list[str] | None = None) -> int:
    app = QApplication(argv if argv is not None else sys.argv)
    app.setApplicationName("morseformer")
    app.setOrganizationName("morseformer")
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
