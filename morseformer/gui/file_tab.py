"""Offline WAV decode tab."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtGui import QFont, QGuiApplication
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class FileTab(QWidget):
    decode_requested = Signal(str)  # absolute path

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        row = QHBoxLayout()
        self.open_btn = QPushButton("Open WAV …")
        self.open_btn.clicked.connect(self._on_open)
        self.path_label = QLabel("(no file)")
        self.path_label.setStyleSheet("color: #888;")
        row.addWidget(self.open_btn)
        row.addWidget(self.path_label, 1)
        self.copy_btn = QPushButton("Copy transcript")
        self.copy_btn.setEnabled(False)
        self.copy_btn.clicked.connect(self._on_copy)
        row.addWidget(self.copy_btn)
        layout.addLayout(row)

        self.transcript = QPlainTextEdit()
        self.transcript.setReadOnly(True)
        font = QFont("monospace")
        font.setStyleHint(QFont.StyleHint.Monospace)
        font.setPointSize(13)
        self.transcript.setFont(font)
        self.transcript.setPlaceholderText(
            "Open a .wav file (any sample rate, mono or stereo). "
            "The decoded text appears here when ready."
        )
        layout.addWidget(self.transcript, 1)

    def _on_open(self) -> None:
        path, _filter = QFileDialog.getOpenFileName(
            self, "Open WAV", "", "WAV files (*.wav);;All files (*)",
        )
        if not path:
            return
        self.path_label.setText(Path(path).name)
        self.transcript.setPlainText("")
        self.open_btn.setEnabled(False)
        self.copy_btn.setEnabled(False)
        self.decode_requested.emit(path)

    def _on_copy(self) -> None:
        text = self.transcript.toPlainText()
        if text:
            QGuiApplication.clipboard().setText(text)

    # Worker callbacks.
    def on_file_decoded(self, text: str) -> None:
        self.transcript.setPlainText(text)
        self.open_btn.setEnabled(True)
        self.copy_btn.setEnabled(bool(text))

    def on_error(self) -> None:
        self.open_btn.setEnabled(True)
