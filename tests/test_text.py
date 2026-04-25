"""Unit tests for the synthetic-text generators."""

from __future__ import annotations

import re
from collections import Counter

import numpy as np
import pytest

from morseformer.core.tokenizer import BLANK_INDEX, TOKEN_TO_INDEX
from morseformer.data import text as text_mod
from morseformer.data.itu_prefixes import ENTRIES
from morseformer.data.text import (
    DEFAULT_MIX,
    PHASE_3_2_MIX,
    TextMix,
    sample_callsign,
    sample_category,
    sample_english_words,
    sample_numeric,
    sample_qcode_abbrev,
    sample_qso_line,
    sample_random_chars,
    sample_text,
)


# --------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "fn",
    [
        sample_callsign,
        sample_qcode_abbrev,
        sample_qso_line,
        sample_numeric,
        sample_english_words,
        sample_random_chars,
        sample_text,
    ],
)
def test_sampler_is_deterministic_under_seed(fn) -> None:
    rng_a = np.random.default_rng(123)
    rng_b = np.random.default_rng(123)
    out_a = [fn(rng_a) for _ in range(50)]
    out_b = [fn(rng_b) for _ in range(50)]
    assert out_a == out_b


def test_different_seeds_give_different_streams() -> None:
    rng_a = np.random.default_rng(1)
    rng_b = np.random.default_rng(2)
    out_a = [sample_text(rng_a) for _ in range(50)]
    out_b = [sample_text(rng_b) for _ in range(50)]
    assert out_a != out_b


# --------------------------------------------------------------------- #
# Vocabulary compliance — every generated char must be a tokenizer token
# --------------------------------------------------------------------- #


def test_all_outputs_stay_in_vocab() -> None:
    rng = np.random.default_rng(7)
    for _ in range(2000):
        t = sample_text(rng)
        for ch in t:
            assert ch in TOKEN_TO_INDEX, f"out-of-vocab char {ch!r} in {t!r}"


def test_no_generator_produces_empty_string() -> None:
    rng = np.random.default_rng(9)
    for _ in range(500):
        for fn in (
            sample_callsign, sample_qcode_abbrev, sample_qso_line,
            sample_numeric, sample_english_words, sample_text,
        ):
            t = fn(rng)
            assert len(t) > 0


# --------------------------------------------------------------------- #
# Token coverage
# --------------------------------------------------------------------- #


def test_every_non_blank_token_is_reachable() -> None:
    rng = np.random.default_rng(0)
    seen: set[str] = set()
    for _ in range(10_000):
        for ch in sample_text(rng):
            seen.add(ch)
    all_non_blank = {tok for tok, idx in TOKEN_TO_INDEX.items() if idx != BLANK_INDEX}
    missing = all_non_blank - seen
    assert not missing, f"tokens never produced in 10k draws: {sorted(missing)}"


# --------------------------------------------------------------------- #
# Callsign structure
# --------------------------------------------------------------------- #

# The structure the callsign generator is *meant* to produce.
# Left part (before any portable slash):
#   - starts with at least one alnum char (the root may contain digits)
#   - optionally one digit (skipped when the root already ends in a digit)
#   - suffix of 1-3 uppercase letters.
# Portable tag: optional, /P /M /MM /QRP /A.
_CALLSIGN_REGEX = re.compile(r"^[A-Z0-9]+[A-Z]{1,3}(/[A-Z]{1,3})?$")
_ALLOWED_PORTABLE = {"P", "M", "MM", "QRP", "A"}


def test_callsign_matches_expected_structure() -> None:
    rng = np.random.default_rng(42)
    for _ in range(1000):
        cs = sample_callsign(rng)
        assert _CALLSIGN_REGEX.match(cs), f"malformed callsign: {cs!r}"
        if "/" in cs:
            assert cs.split("/", 1)[1] in _ALLOWED_PORTABLE, f"bad portable tag: {cs!r}"


def test_portable_tag_frequency_matches_spec() -> None:
    # Spec: ~8% of callsigns carry a portable tag. Allow ±2pt tolerance
    # on 5000 draws.
    rng = np.random.default_rng(11)
    n = 5000
    portable = sum("/" in sample_callsign(rng) for _ in range(n))
    frac = portable / n
    assert 0.06 <= frac <= 0.10, f"portable fraction {frac:.3f} outside [0.06, 0.10]"


def test_hb9_always_has_fixed_digit_9() -> None:
    # Force an HB9 draw by temporarily overriding the entries list.
    from morseformer.data import itu_prefixes as itu
    rng = np.random.default_rng(0)
    hb9 = next(e for e in ENTRIES if e.root == "HB9")
    # Patch the table so every sample_root call returns HB9.
    original_probs = itu._PROBS.copy()
    try:
        new_probs = np.zeros_like(original_probs)
        idx = ENTRIES.index(hb9)
        new_probs[idx] = 1.0
        itu._PROBS = new_probs
        for _ in range(200):
            cs = sample_callsign(rng)
            base = cs.split("/")[0]
            # Must start with "HB9" and not insert another digit.
            assert base.startswith("HB9"), cs
            # Position 3 onward must be only letters (the suffix).
            assert base[3:].isalpha(), cs
    finally:
        itu._PROBS = original_probs


def test_plain_root_gets_random_digit() -> None:
    # When the root is purely alphabetic, the character right after
    # should be a digit 0-9 (the sampled digit).
    from morseformer.data import itu_prefixes as itu
    rng = np.random.default_rng(0)
    k_entry = next(e for e in ENTRIES if e.root == "K")
    original_probs = itu._PROBS.copy()
    try:
        new_probs = np.zeros_like(original_probs)
        new_probs[ENTRIES.index(k_entry)] = 1.0
        itu._PROBS = new_probs
        digits_seen: set[str] = set()
        for _ in range(400):
            cs = sample_callsign(rng)
            base = cs.split("/")[0]
            assert base[0] == "K"
            assert base[1].isdigit(), cs
            digits_seen.add(base[1])
        # With 400 draws we should see most digits at least once.
        assert len(digits_seen) >= 8, f"only saw digits {digits_seen}"
    finally:
        itu._PROBS = original_probs


# --------------------------------------------------------------------- #
# Q-codes & abbreviations
# --------------------------------------------------------------------- #


def test_qcode_output_is_in_table() -> None:
    rng = np.random.default_rng(3)
    table = set(text_mod._QCODES_AND_ABBREVS)
    for _ in range(500):
        assert sample_qcode_abbrev(rng) in table


def test_qcode_core_set_is_reachable() -> None:
    # Five very common items a ham would expect to see.
    rng = np.random.default_rng(0)
    must_appear = {"QRZ", "CQ", "DE", "73", "TU"}
    seen: set[str] = set()
    for _ in range(5000):
        seen.add(sample_qcode_abbrev(rng))
        if must_appear <= seen:
            return
    pytest.fail(f"missing after 5k draws: {must_appear - seen}")


# --------------------------------------------------------------------- #
# QSO grammar
# --------------------------------------------------------------------- #


def test_qso_line_has_no_unfilled_slots() -> None:
    rng = np.random.default_rng(5)
    for _ in range(500):
        line = sample_qso_line(rng)
        assert "{" not in line and "}" not in line, f"unfilled slot: {line!r}"


def test_qso_lines_embed_callsigns() -> None:
    # At least some templates include one or more {cs} slots. A sample
    # of 500 lines should contain many callsign-shaped tokens.
    rng = np.random.default_rng(6)
    call_like = re.compile(r"\b[A-Z0-9]{1,3}\d[A-Z]{1,3}(/[A-Z]+)?\b")
    matches = sum(
        bool(call_like.search(sample_qso_line(rng))) for _ in range(500)
    )
    assert matches > 250, f"too few callsign-bearing lines: {matches}/500"


# --------------------------------------------------------------------- #
# Numeric
# --------------------------------------------------------------------- #


def test_numeric_output_non_empty_and_in_vocab() -> None:
    rng = np.random.default_rng(13)
    for _ in range(500):
        n = sample_numeric(rng)
        assert len(n) > 0
        for ch in n:
            assert ch in TOKEN_TO_INDEX


def test_numeric_covers_several_patterns() -> None:
    # We should see at least "KHZ", "MHZ", "Z" time, "NR", and dates.
    rng = np.random.default_rng(14)
    joined = " ".join(sample_numeric(rng) for _ in range(1000))
    for token in ("KHZ", "MHZ", "NR", "/", "-"):
        assert token in joined, f"{token!r} never appeared"


# --------------------------------------------------------------------- #
# English-word sampler
# --------------------------------------------------------------------- #


def test_english_words_are_space_separated_and_uppercase() -> None:
    rng = np.random.default_rng(15)
    for _ in range(500):
        s = sample_english_words(rng)
        parts = s.split(" ")
        assert 2 <= len(parts) <= 6
        for w in parts:
            assert w.isupper()
            assert w.isalpha()


def test_english_words_sampled_from_corpus() -> None:
    rng = np.random.default_rng(16)
    corpus = set(text_mod._ENGLISH_WORDS)
    for _ in range(200):
        s = sample_english_words(rng)
        for w in s.split(" "):
            assert w in corpus, f"unknown word: {w!r}"


# --------------------------------------------------------------------- #
# Top-level mix
# --------------------------------------------------------------------- #


def test_mix_distribution_respects_weights() -> None:
    rng = np.random.default_rng(17)
    n = 10_000
    counts = Counter(sample_category(rng, DEFAULT_MIX) for _ in range(n))
    expected = {
        "callsign": 0.15, "qcode": 0.20, "qso": 0.35,
        "numeric": 0.15, "words": 0.15,
    }
    for cat, p in expected.items():
        frac = counts[cat] / n
        assert abs(frac - p) < 0.03, f"{cat}: expected {p}, got {frac:.3f}"


def test_custom_mix_is_respected() -> None:
    # Push all the weight onto one category and verify the sampler obeys.
    rng = np.random.default_rng(19)
    mix = TextMix(callsign=1.0, qcode=0.0, qso=0.0, numeric=0.0, words=0.0)
    for _ in range(100):
        assert sample_category(rng, mix) == "callsign"


def test_mix_rejects_negative_weights() -> None:
    with pytest.raises(ValueError):
        TextMix(callsign=-1.0, qcode=1.0, qso=1.0, numeric=1.0, words=1.0).as_array()


def test_mix_rejects_all_zero_weights() -> None:
    with pytest.raises(ValueError):
        TextMix(callsign=0.0, qcode=0.0, qso=0.0, numeric=0.0, words=0.0).as_array()


# --------------------------------------------------------------------- #
# Length distribution — texts should fit a 6 s window at [16, 28] WPM.
# --------------------------------------------------------------------- #


def test_text_length_is_reasonable() -> None:
    rng = np.random.default_rng(23)
    lens = [len(sample_text(rng)) for _ in range(5000)]
    median = float(np.median(lens))
    p95 = float(np.percentile(lens, 95))
    assert 5 <= median <= 20, f"median length {median} outside [5, 20]"
    # Long tail is allowed — the dataset layer will retry if audio > 6 s.
    assert p95 <= 50, f"95th-percentile length {p95} is too long"


# --------------------------------------------------------------------- #
# Random-character sampler (Phase 3.2)
# --------------------------------------------------------------------- #


def test_random_chars_only_in_vocab() -> None:
    rng = np.random.default_rng(31)
    for _ in range(2000):
        s = sample_random_chars(rng)
        assert len(s) > 0
        for ch in s:
            assert ch in TOKEN_TO_INDEX, (
                f"out-of-vocab {ch!r} in random sample {s!r}"
            )


def test_random_chars_covers_letters_digits_and_punct() -> None:
    rng = np.random.default_rng(32)
    seen_letter = seen_digit = seen_punct = False
    punct_set = set(",/?=-+")
    for _ in range(2000):
        for ch in sample_random_chars(rng):
            if ch.isalpha():
                seen_letter = True
            elif ch.isdigit():
                seen_digit = True
            elif ch in punct_set:
                seen_punct = True
        if seen_letter and seen_digit and seen_punct:
            return
    pytest.fail(
        f"after 2000 draws: letter={seen_letter} digit={seen_digit} "
        f"punct={seen_punct}"
    )


def test_random_chars_produces_multi_group_sometimes() -> None:
    """A non-trivial fraction of samples should contain a space
    (multi-group, e.g. cipher-style ABCDE FGHIJ)."""
    rng = np.random.default_rng(33)
    n = 1000
    multi = sum(" " in sample_random_chars(rng) for _ in range(n))
    frac = multi / n
    # Spec is 30 % multi-group; allow a wide tolerance.
    assert 0.15 <= frac <= 0.50, f"multi-group fraction {frac:.3f} outside [0.15, 0.50]"


def test_random_chars_pure_letter_mode_lengths() -> None:
    """All-letter samples are produced by either the letter mode (3-6)
    or by the mixed mode happening to draw 0 digits (4-8). So the
    overall length range is 3-8."""
    rng = np.random.default_rng(34)
    pure_letter_lengths: list[int] = []
    for _ in range(2000):
        s = sample_random_chars(rng)
        if " " not in s and s.isalpha():
            pure_letter_lengths.append(len(s))
    assert len(pure_letter_lengths) > 100
    assert min(pure_letter_lengths) >= 3
    assert max(pure_letter_lengths) <= 8


# --------------------------------------------------------------------- #
# Phase 3.2 mix
# --------------------------------------------------------------------- #


def test_phase_3_2_mix_distribution() -> None:
    rng = np.random.default_rng(41)
    n = 10_000
    counts = Counter(sample_category(rng, PHASE_3_2_MIX) for _ in range(n))
    expected = {
        "callsign": 0.12, "qcode": 0.14, "qso": 0.25,
        "numeric": 0.13, "words": 0.06, "random": 0.30,
    }
    for cat, p in expected.items():
        frac = counts[cat] / n
        assert abs(frac - p) < 0.03, f"{cat}: expected {p}, got {frac:.3f}"


def test_phase_3_2_mix_text_in_vocab() -> None:
    rng = np.random.default_rng(42)
    for _ in range(2000):
        t = sample_text(rng, PHASE_3_2_MIX)
        for ch in t:
            assert ch in TOKEN_TO_INDEX, f"oov {ch!r} in {t!r}"
