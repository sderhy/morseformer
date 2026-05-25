"""Settings panel exposed in the main window header.

Emits per-knob signals as the user changes things; the worker subscribes
to those signals (cross-thread) so config updates land on the worker
thread before the next decode.
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QWidget,
)

from morseformer.cli.presets import DEFAULT_PRESET, PRESETS


class SettingsPanel(QWidget):
    preset_changed = Signal(str)
    confidence_threshold_changed = Signal(float)
    digit_threshold_changed = Signal(float)
    carrier_changed = Signal(float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._building = True
        self._build_ui()
        self._building = False

    def _build_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)

        layout.addWidget(QLabel("Preset:"))
        self.preset = QComboBox()
        for name, p in PRESETS.items():
            self.preset.addItem(name, p)
            i = self.preset.count() - 1
            self.preset.setItemData(i, p.description, role=3)  # Qt.ToolTipRole
        self.preset.setCurrentText(DEFAULT_PRESET)
        self.preset.currentTextChanged.connect(self._on_preset_changed)
        layout.addWidget(self.preset)

        layout.addSpacing(16)
        layout.addWidget(QLabel("Conf. thr:"))
        self.conf_thr = QDoubleSpinBox()
        self.conf_thr.setRange(0.0, 1.0)
        self.conf_thr.setSingleStep(0.05)
        self.conf_thr.setDecimals(2)
        self.conf_thr.valueChanged.connect(self._emit_conf_thr)
        layout.addWidget(self.conf_thr)

        layout.addWidget(QLabel("Digit thr:"))
        self.digit_thr = QDoubleSpinBox()
        self.digit_thr.setRange(0.0, 1.0)
        self.digit_thr.setSingleStep(0.05)
        self.digit_thr.setDecimals(2)
        self.digit_thr.valueChanged.connect(self._emit_digit_thr)
        layout.addWidget(self.digit_thr)

        layout.addSpacing(16)
        layout.addWidget(QLabel("Carrier (Hz):"))
        self.carrier = QDoubleSpinBox()
        self.carrier.setRange(100.0, 3000.0)
        self.carrier.setSingleStep(50.0)
        self.carrier.setDecimals(0)
        self.carrier.setValue(600.0)
        self.carrier.valueChanged.connect(self._emit_carrier)
        layout.addWidget(self.carrier)

        layout.addStretch(1)

        # Trigger initial population from the default preset.
        self._on_preset_changed(DEFAULT_PRESET)

    def _on_preset_changed(self, name: str) -> None:
        preset = PRESETS.get(name)
        if preset is None:
            return
        self._building = True
        self.conf_thr.setValue(preset.confidence_threshold)
        self.digit_thr.setValue(preset.digit_threshold)
        self._building = False
        # Emit in dependency order so the worker sees the preset first.
        self.preset_changed.emit(name)
        self.confidence_threshold_changed.emit(preset.confidence_threshold)
        self.digit_threshold_changed.emit(preset.digit_threshold)

    def _emit_conf_thr(self, v: float) -> None:
        if self._building:
            return
        self.confidence_threshold_changed.emit(float(v))

    def _emit_digit_thr(self, v: float) -> None:
        if self._building:
            return
        self.digit_threshold_changed.emit(float(v))

    def _emit_carrier(self, v: float) -> None:
        if self._building:
            return
        self.carrier_changed.emit(float(v))
