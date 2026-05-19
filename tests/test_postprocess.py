"""Unit tests for ``morseformer.decoding.postprocess.format_output``.

Locks the cosmetic semantics:

* ``BK`` (word-bounded) → ``=``
* ``KN`` and ``=`` each get a trailing newline so a long transcript
  reads as one over / break segment per line.
* Substrings of larger words are not touched.
"""

from __future__ import annotations

from morseformer.decoding.postprocess import format_output


def test_bk_becomes_equals_with_newline() -> None:
    # Trailing newline is trimmed so "TU BK" → "TU ="; a mid-string BK
    # keeps the newline so the next segment starts on a fresh line.
    assert format_output("TU BK") == "TU ="
    assert format_output("TU BK 73") == "TU =\n73"


def test_kn_word_boundary_gets_newline() -> None:
    assert format_output("G6PZ KN") == "G6PZ KN"  # trailing newline trimmed
    assert format_output("G6PZ KN DE F4HYY") == "G6PZ KN\nDE F4HYY"


def test_multiple_breaks_each_get_newline() -> None:
    assert format_output("CQ DE F4HYY KN G6PZ DE F4HYY BK") == (
        "CQ DE F4HYY KN\nG6PZ DE F4HYY ="
    )
    assert format_output("TU = BK = K") == "TU =\n=\n=\nK"


def test_bk_inside_a_word_is_not_substituted() -> None:
    # Word boundary required on both sides — anniversary / numeric calls
    # like K1BK or BKR must stay intact.
    assert format_output("K1BK") == "K1BK"
    assert format_output("BKR DE K1") == "BKR DE K1"


def test_kn_inside_a_word_is_not_broken() -> None:
    # A callsign starting with KN (KN4WG…) must not get a phantom newline.
    assert format_output("KN4WG") == "KN4WG"


def test_trailing_break_does_not_leave_dangling_newline() -> None:
    assert format_output("TU 73 BK") == "TU 73 ="
    assert format_output("PSE QSP KN") == "PSE QSP KN"


def test_empty_input() -> None:
    assert format_output("") == ""


def test_no_break_tokens_passthrough() -> None:
    assert format_output("CQ DE F4HYY K") == "CQ DE F4HYY K"
