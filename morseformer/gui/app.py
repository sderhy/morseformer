"""Main window: top settings bar + Live/File tabs + status bar."""

from __future__ import annotations

import sys

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from morseformer import __version__
from morseformer.gui.decoder_worker import DecoderWorker
from morseformer.gui.file_tab import FileTab
from morseformer.gui.live_tab import LiveTab
from morseformer.gui.settings_panel import SettingsPanel


class MainWindow(QMainWindow):
    _start_live_requested = Signal(object)
    _stop_live_requested = Signal()
    _decode_file_requested = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"morseformer · v{__version__}")
        self.resize(960, 640)

        # Central layout: settings panel + tabs.
        central = QWidget()
        self.setCentralWidget(central)
        v = QVBoxLayout(central)
        v.setContentsMargins(0, 0, 0, 0)

        self.settings = SettingsPanel()
        v.addWidget(self.settings)

        self.tabs = QTabWidget()
        self.live_tab = LiveTab()
        self.file_tab = FileTab()
        self.tabs.addTab(self.live_tab, "Live")
        self.tabs.addTab(self.file_tab, "File")
        v.addWidget(self.tabs, 1)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Loading model on first decode …")

        self._build_menu()
        self._build_worker()
        self._wire_signals()

    def _build_menu(self) -> None:
        m = self.menuBar().addMenu("&File")
        quit_action = QAction("&Quit", self)
        quit_action.setShortcut(QKeySequence.StandardKey.Quit)
        quit_action.triggered.connect(self.close)
        m.addAction(quit_action)

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

        # Live tab ↔ worker. Routed via main-window signals so the QThread
        # call site stays explicit (auto-queued for cross-thread).
        self.live_tab.start_requested.connect(self._start_live_requested.emit)
        self.live_tab.stop_requested.connect(self._stop_live_requested.emit)
        self._start_live_requested.connect(
            w.start_live, type=Qt.ConnectionType.QueuedConnection,
        )
        self._stop_live_requested.connect(
            w.stop_live, type=Qt.ConnectionType.QueuedConnection,
        )

        w.transcript_fragment.connect(self.live_tab.on_transcript_fragment)
        w.level_db.connect(self.live_tab.on_level_db)
        w.raw_audio.connect(self.live_tab.on_audio)
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
        self._thread.quit()
        self._thread.wait(2000)
        super().closeEvent(event)


def main(argv: list[str] | None = None) -> int:
    app = QApplication(argv if argv is not None else sys.argv)
    app.setApplicationName("morseformer")
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
