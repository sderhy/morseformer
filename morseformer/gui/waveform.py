"""Lightweight VU meter + scrolling waveform widgets (no matplotlib)."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QWidget


class VuMeter(QWidget):
    """Horizontal RMS-dB meter."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._db = -120.0
        self._floor_db = -60.0
        self._ceil_db = 0.0
        self.setMinimumHeight(16)

    @Slot(float)
    def set_db(self, db: float) -> None:
        self._db = max(self._floor_db, min(self._ceil_db, float(db)))
        self.update()

    def paintEvent(self, event) -> None:  # noqa: D401, N802
        p = QPainter(self)
        rect = self.rect()
        p.fillRect(rect, QColor("#1d1d1d"))
        # Map dB to width.
        norm = (self._db - self._floor_db) / (self._ceil_db - self._floor_db)
        norm = max(0.0, min(1.0, norm))
        w = int(rect.width() * norm)
        # Colour green up to -12 dB, yellow to -6 dB, red above.
        if self._db < -12.0:
            colour = QColor("#3ec46d")
        elif self._db < -6.0:
            colour = QColor("#e6c54d")
        else:
            colour = QColor("#e25c5c")
        p.fillRect(0, 0, w, rect.height(), colour)
        # Tick at -12 / -6 dB.
        for tick_db in (-12.0, -6.0):
            x = int(rect.width() * (tick_db - self._floor_db) /
                    (self._ceil_db - self._floor_db))
            p.setPen(QPen(QColor("#555"), 1))
            p.drawLine(x, 0, x, rect.height())


class Waveform(QWidget):
    """Scrolling waveform — fixed-size ring buffer downsampled to widget width."""

    def __init__(self, parent: QWidget | None = None, history_seconds: float = 4.0) -> None:
        super().__init__(parent)
        self._history_seconds = history_seconds
        self._sample_rate = 8000
        self._buf = np.zeros(int(history_seconds * self._sample_rate), dtype=np.float32)
        self.setMinimumHeight(48)

    @Slot(np.ndarray)
    def append(self, audio: np.ndarray) -> None:
        a = np.asarray(audio, dtype=np.float32).ravel()
        if a.size >= self._buf.size:
            self._buf = a[-self._buf.size:].copy()
        else:
            self._buf = np.concatenate([self._buf[a.size:], a])
        self.update()

    def paintEvent(self, event) -> None:  # noqa: D401, N802
        p = QPainter(self)
        rect = self.rect()
        p.fillRect(rect, QColor("#141414"))
        if self._buf.size == 0:
            return
        w = rect.width()
        h = rect.height()
        mid = h // 2
        # Downsample buffer to w columns by min/max bracketing.
        if w <= 0:
            return
        chunk = max(1, self._buf.size // w)
        # Crop to exact multiple of chunk.
        n_full = (self._buf.size // chunk) * chunk
        view = self._buf[-n_full:].reshape(-1, chunk)
        lo = view.min(axis=1)
        hi = view.max(axis=1)
        # Scale.
        scale = mid * 0.95
        p.setPen(QPen(QColor("#5fb1f3"), 1))
        n_cols = lo.size
        for i in range(n_cols):
            x = int(i * w / n_cols)
            y1 = int(mid - hi[i] * scale)
            y2 = int(mid - lo[i] * scale)
            p.drawLine(x, y1, x, y2)
        # Centre line.
        p.setPen(QPen(QColor("#333"), 1, Qt.PenStyle.DashLine))
        p.drawLine(0, mid, w, mid)
