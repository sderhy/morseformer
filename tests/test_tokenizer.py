"""Unit tests for the CW character-level tokenizer."""

from __future__ import annotations

from morseformer.core.tokenizer import (
    BLANK_INDEX,
    BLANK_TOKEN,
    INDEX_TO_TOKEN,
    SPACE_INDEX,
    TOKEN_TO_INDEX,
    VOCAB_SIZE,
    ctc_greedy_decode,
    decode,
    encode,
)


def test_vocab_shape() -> None:
    assert VOCAB_SIZE == 46
    assert len(INDEX_TO_TOKEN) == VOCAB_SIZE
    assert len(TOKEN_TO_INDEX) == VOCAB_SIZE


def test_blank_and_space_positions() -> None:
    assert INDEX_TO_TOKEN[BLANK_INDEX] == BLANK_TOKEN
    assert BLANK_INDEX == 0
    assert INDEX_TO_TOKEN[SPACE_INDEX] == " "
    assert SPACE_INDEX == 1


def test_letters_and_digits_contiguous() -> None:
    # Letters A-Z at indices 2..27, digits 0-9 at 28..37.
    for offset, ch in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
        assert INDEX_TO_TOKEN[2 + offset] == ch
    for offset, ch in enumerate("0123456789"):
        assert INDEX_TO_TOKEN[28 + offset] == ch


def test_punctuation_present() -> None:
    for ch in ".,?!/=+-":
        assert ch in TOKEN_TO_INDEX


def test_encode_uppercases() -> None:
    assert encode("abc") == encode("ABC")


def test_encode_decode_roundtrip() -> None:
    text = "CQ CQ DE F6ABC K"
    assert decode(encode(text)) == text


def test_encode_collapses_whitespace() -> None:
    # Multiple spaces, tabs, leading/trailing whitespace all collapse.
    assert encode("  A\t\tB  ") == encode("A B")


def test_encode_drops_out_of_vocab() -> None:
    # '~' and '@' are not in the vocabulary.
    assert decode(encode("A~B@C")) == "ABC"


def test_encode_never_emits_blank() -> None:
    for idx in encode("THE QUICK BROWN FOX 0123456789 .,?!/=+-"):
        assert idx != BLANK_INDEX


def test_ctc_greedy_collapse_repeats() -> None:
    # Raw frame-level argmax with CTC blanks and repeats.
    a = TOKEN_TO_INDEX["A"]
    b = TOKEN_TO_INDEX["B"]
    frames = [BLANK_INDEX, a, a, BLANK_INDEX, a, b, b, BLANK_INDEX, b]
    # Collapse repeats between blanks: A A _ A B B _ B → A A B B
    assert ctc_greedy_decode(frames) == "AABB"


def test_ctc_greedy_all_blank() -> None:
    assert ctc_greedy_decode([BLANK_INDEX] * 10) == ""


def test_decode_ignores_blank() -> None:
    a = TOKEN_TO_INDEX["A"]
    assert decode([BLANK_INDEX, a, BLANK_INDEX, a]) == "AA"


def test_decode_strip_default_removes_edge_spaces() -> None:
    a = TOKEN_TO_INDEX["A"]
    assert decode([SPACE_INDEX, a, SPACE_INDEX]) == "A"


def test_decode_strip_false_preserves_edge_spaces() -> None:
    # Streaming callers need this: an inter-word space at a window
    # boundary must survive concatenation, otherwise consecutive
    # fragments produce "HELLOWORLD" instead of "HELLO WORLD".
    a = TOKEN_TO_INDEX["A"]
    b = TOKEN_TO_INDEX["B"]
    # Leading space (start of a new word in this fragment).
    assert decode([SPACE_INDEX, a, b], strip=False) == " AB"
    # Trailing space (end of a word at the window boundary).
    assert decode([a, b, SPACE_INDEX], strip=False) == "AB "
    # Both — concatenating two such fragments yields exactly one space
    # between the words.
    left = decode([a, b, SPACE_INDEX], strip=False)
    right = decode([a, b], strip=False)
    assert left + right == "AB AB"
