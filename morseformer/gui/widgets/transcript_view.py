"""Read-only transcript view with display formatting + clickable callsigns.

Keeps the *raw* decoder output as the source of truth and re-renders it to
HTML through the current :class:`DisplayOptions` whenever text is appended
or an option changes. Callsigns are detected on the rendered text and
wrapped in QRZ.com links; the licensed core determines the URL and a
DXCC-country tooltip.

Because the raw buffer is preserved, toggling lowercase / line-breaks is
fully reversible and never corrupts the transcript.
"""

from __future__ import annotations

from html import escape

from PySide6.QtGui import QFont
from PySide6.QtWidgets import QTextBrowser, QWidget

from morseformer.gui.services import callsigns
from morseformer.gui.services.formatting import DisplayOptions, apply


class TranscriptView(QTextBrowser):
    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        options: DisplayOptions | None = None,
        placeholder: str = "",
    ) -> None:
        super().__init__(parent)
        self._raw = ""
        self._options = options or DisplayOptions()
        self.setReadOnly(True)
        self.setOpenExternalLinks(True)  # QRZ links open in the browser
        if placeholder:
            self.setPlaceholderText(placeholder)
        font = QFont("monospace")
        font.setStyleHint(QFont.StyleHint.Monospace)
        font.setPointSize(13)
        self.setFont(font)

    # ---- content ----------------------------------------------------- #

    def set_text(self, raw: str) -> None:
        self._raw = raw
        self._render()

    def append_fragment(self, fragment: str) -> None:
        if not fragment:
            return
        self._raw += fragment
        self._render()
        sb = self.verticalScrollBar()
        sb.setValue(sb.maximum())

    def clear_transcript(self) -> None:
        self._raw = ""
        self._render()

    def raw_text(self) -> str:
        return self._raw

    def display_text(self) -> str:
        """The formatted plain text the user sees (for copy / export)."""
        return apply(self._raw, self._options)

    # ---- options ----------------------------------------------------- #

    def set_options(self, options: DisplayOptions) -> None:
        self._options = options
        self._render()

    # ---- rendering --------------------------------------------------- #

    def _render(self) -> None:
        formatted = apply(self._raw, self._options)
        self.setHtml(_to_html(formatted))

    def setPlaceholderText(self, text: str) -> None:  # noqa: N802
        # QTextBrowser inherits placeholder support from QTextEdit.
        super().setPlaceholderText(text)


def _to_html(text: str) -> str:
    """Escape ``text``, linkify callsigns, and turn newlines into <br>."""
    if not text:
        return ""
    matches = callsigns.find_callsigns(text)
    parts: list[str] = []
    cursor = 0
    for m in matches:
        parts.append(_escape_block(text[cursor:m.start]))
        shown = escape(text[m.start:m.end])
        country = m.country
        title = f' title="{escape(country)}"' if country else ""
        parts.append(f'<a href="{escape(m.qrz_url)}"{title}>{shown}</a>')
        cursor = m.end
    parts.append(_escape_block(text[cursor:]))
    body = "".join(parts)
    return f'<div style="white-space: pre-wrap;">{body}</div>'


def _escape_block(text: str) -> str:
    return escape(text).replace("\n", "<br>")
