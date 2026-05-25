"""Unit tests for the UI-free GUI service layer.

These run headless (no QApplication, no display) except for the config
store, which only needs QSettings in a temp location.
"""

from __future__ import annotations

import numpy as np
import pytest

from morseformer.gui.services import callsigns, exporter
from morseformer.gui.services.formatting import DisplayOptions, apply
from morseformer.gui.services.logbook import Logbook
from morseformer.gui.services.recorder import WavRecorder

# ---------------------------------------------------------------------- #
# formatting
# ---------------------------------------------------------------------- #


def test_format_default_breaks_on_tokens() -> None:
    assert apply("A = B KN C", DisplayOptions()) == "A =\nB KN\nC"


def test_format_no_break_tokens_keeps_one_line() -> None:
    out = apply("A = B KN C", DisplayOptions(break_tokens=False))
    assert "\n" not in out


def test_format_break_after_k_and_lowercase() -> None:
    out = apply("CQ K DE", DisplayOptions(break_after_k=True, lowercase=True))
    assert out == "cq k\nde"


def test_format_does_not_break_k_inside_word() -> None:
    # The K in a callsign-like token must not get a line break.
    out = apply("K1ABC", DisplayOptions(break_after_k=True))
    assert out == "K1ABC"


def test_display_options_roundtrip_dict() -> None:
    o = DisplayOptions(break_tokens=False, break_after_k=True, lowercase=True)
    assert DisplayOptions.from_dict(o.as_dict()) == o


# ---------------------------------------------------------------------- #
# callsigns
# ---------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "call", ["F4HYY", "G3ZRJ", "MW0BGL", "2E0ABC", "9A1AA", "HB9XYZ", "IK6ZKD"],
)
def test_detects_valid_callsigns(call: str) -> None:
    text = f"CQ DE {call} K"
    found = [m.call for m in callsigns.find_callsigns(text)]
    assert call in found


@pytest.mark.parametrize("noise", ["599", "5NN", "CQ", "TEST", "73"])
def test_ignores_non_callsigns(noise: str) -> None:
    assert callsigns.find_callsigns(noise) == []


def test_portable_callsign_core_and_url() -> None:
    assert callsigns.qrz_url("DL1ABC/P") == "https://www.qrz.com/db/DL1ABC"
    assert callsigns.qrz_url("F/ON4XX") == "https://www.qrz.com/db/ON4XX"


def test_country_lookup_prefers_longest_prefix() -> None:
    assert callsigns.country_for("F4HYY") == "France"
    assert callsigns.country_for("EA8ABC") == "Canary Islands"  # not "Spain"


def test_match_spans_align_with_source() -> None:
    text = "de f4hyy here"
    (m,) = callsigns.find_callsigns(text)
    assert text[m.start:m.end].upper() == "F4HYY"


# ---------------------------------------------------------------------- #
# transcript HTML rendering (pure helper, no Qt)
# ---------------------------------------------------------------------- #


def test_to_html_linkifies_callsign() -> None:
    from morseformer.gui.widgets.transcript_view import _to_html

    html = _to_html("DE F4HYY")
    assert 'href="https://www.qrz.com/db/F4HYY"' in html
    assert ">F4HYY</a>" in html


def test_to_html_escapes_special_chars() -> None:
    from morseformer.gui.widgets.transcript_view import _to_html

    html = _to_html("A < B & C")
    assert "&lt;" in html and "&amp;" in html
    assert "<B" not in html  # the raw '<' must not survive unescaped


def test_to_html_newlines_become_br() -> None:
    from morseformer.gui.widgets.transcript_view import _to_html

    assert "<br>" in _to_html("A =\nB")


# ---------------------------------------------------------------------- #
# recorder
# ---------------------------------------------------------------------- #


def test_recorder_writes_wav(tmp_path) -> None:
    import wave

    path = tmp_path / "out.wav"
    rec = WavRecorder(path, sample_rate=8000)
    rec.start()
    rec.feed(np.zeros(8000, dtype=np.float32))
    rec.feed(np.ones(4000, dtype=np.float32))  # clipped to 1.0
    saved = rec.stop()
    assert saved == path
    with wave.open(str(path), "rb") as w:
        assert w.getframerate() == 8000
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        assert w.getnframes() == 12000


def test_recorder_discards_empty(tmp_path) -> None:
    path = tmp_path / "empty.wav"
    rec = WavRecorder(path, sample_rate=8000)
    rec.start()
    assert rec.stop() is None
    assert not path.exists()


# ---------------------------------------------------------------------- #
# exporter
# ---------------------------------------------------------------------- #


def test_export_text_with_header(tmp_path) -> None:
    path = tmp_path / "t.txt"
    exporter.export_text("CQ CQ DE F4HYY", path)
    body = path.read_text(encoding="utf-8")
    assert "morseformer" in body
    assert "CQ CQ DE F4HYY" in body


# ---------------------------------------------------------------------- #
# logbook
# ---------------------------------------------------------------------- #


def test_logbook_add_and_list(tmp_path) -> None:
    log = Logbook(tmp_path / "log.sqlite")
    rid = log.add("f4hyy", country="France", excerpt="CQ DE F4HYY")
    assert rid == 1
    assert log.count() == 1
    (entry,) = log.all()
    assert entry.callsign == "F4HYY"  # normalised upper-case
    assert entry.country == "France"
    log.close()


def test_logbook_persists_across_reopen(tmp_path) -> None:
    p = tmp_path / "log.sqlite"
    log = Logbook(p)
    log.add("G3ZRJ")
    log.close()
    reopened = Logbook(p)
    assert reopened.count() == 1
    reopened.close()
