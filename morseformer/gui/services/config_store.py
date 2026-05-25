"""Persistent GUI preferences, backed by ``QSettings``.

UI-free apart from the Qt dependency. Centralises every persisted key so
that adding a new remembered preference is one method here, not a
QSettings call scattered through a widget. The store survives across runs
in the platform-native location (registry on Windows, plist on macOS,
INI under ``~/.config`` on Linux).
"""

from __future__ import annotations

from PySide6.QtCore import QSettings

from morseformer.gui.services.formatting import DisplayOptions

_ORG = "morseformer"
_APP = "morseformer"

# Keys.
_K_DEVICE_BACKEND = "audio/device_backend"
_K_DEVICE_NAME = "audio/device_name"
_K_FMT_BREAK_TOKENS = "display/break_tokens"
_K_FMT_BREAK_AFTER_K = "display/break_after_k"
_K_FMT_LOWERCASE = "display/lowercase"


class ConfigStore:
    def __init__(self) -> None:
        self._s = QSettings(_ORG, _APP)

    # ---- last-used input device (matched by backend + name) ---------- #

    def get_last_device(self) -> tuple[str, str] | None:
        backend = self._s.value(_K_DEVICE_BACKEND, None)
        name = self._s.value(_K_DEVICE_NAME, None)
        if backend and name:
            return str(backend), str(name)
        return None

    def set_last_device(self, backend: str, name: str) -> None:
        self._s.setValue(_K_DEVICE_BACKEND, backend)
        self._s.setValue(_K_DEVICE_NAME, name)

    # ---- display options --------------------------------------------- #

    def get_display_options(self) -> DisplayOptions:
        defaults = DisplayOptions()
        return DisplayOptions(
            break_tokens=self._as_bool(_K_FMT_BREAK_TOKENS, defaults.break_tokens),
            break_after_k=self._as_bool(_K_FMT_BREAK_AFTER_K, defaults.break_after_k),
            lowercase=self._as_bool(_K_FMT_LOWERCASE, defaults.lowercase),
        )

    def set_display_options(self, opts: DisplayOptions) -> None:
        self._s.setValue(_K_FMT_BREAK_TOKENS, opts.break_tokens)
        self._s.setValue(_K_FMT_BREAK_AFTER_K, opts.break_after_k)
        self._s.setValue(_K_FMT_LOWERCASE, opts.lowercase)

    def sync(self) -> None:
        self._s.sync()

    # ---- helpers ----------------------------------------------------- #

    def _as_bool(self, key: str, default: bool) -> bool:
        v = self._s.value(key, default)
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.lower() in ("1", "true", "yes", "on")
        return bool(v)
