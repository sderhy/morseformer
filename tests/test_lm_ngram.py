"""Tests for the char-level n-gram amateur LM (Phase 11 §C)."""

from __future__ import annotations

import tempfile
from pathlib import Path

from morseformer.decoding.lm_ngram import CharNGramLM


def test_score_higher_on_trained_pattern() -> None:
    """The LM should rank a string drawn from the training distribution
    above an unrelated string of the same length."""
    train = ["CQ CQ DE F4HYY", "QRZ DE OM TKS 73", "5NN OM TU 73 GL"] * 20
    lm = CharNGramLM(order=3).fit(train)
    s_amateur = lm.score_per_char("CQ DE F4HYY")
    s_alien = lm.score_per_char("ZZZZZZZZZZZ")
    assert s_amateur > s_alien


def test_split_beats_unsplit_when_split_matches_training() -> None:
    """If training contains 'MY WX IS' (spaced), the LM should prefer
    'MY WX IS' over 'MYWXIS'. This is what the splitter relies on."""
    train = ["MY WX IS SUNNY", "MY WX IS RAINY", "MY WX IS GOOD"] * 50
    lm = CharNGramLM(order=3).fit(train)
    s_split = lm.score_per_char("MY WX IS")
    s_run_on = lm.score_per_char("MYWXIS")
    assert s_split > s_run_on, (
        f"split={s_split:.3f} should beat run-on={s_run_on:.3f}"
    )


def test_unsplit_beats_split_when_unsplit_matches_training() -> None:
    """Inverse case: if training never spaces 'HELLO', the LM should
    prefer the unsplit form. Lets the splitter avoid false positives on
    clean prose."""
    train = ["HELLO WORLD", "SAID HELLO TO", "HELLO HELLO HELLO"] * 50
    lm = CharNGramLM(order=3).fit(train)
    s_unsplit = lm.score_per_char("HELLO")
    s_split = lm.score_per_char("HE L LO")
    assert s_unsplit > s_split


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    train = ["CQ CQ DE F4HYY", "73 73 OM"] * 10
    lm = CharNGramLM(order=3).fit(train)
    score_before = lm.score_per_char("CQ DE F4HYY")
    p = tmp_path / "lm.pkl"
    lm.save(p)
    lm2 = CharNGramLM.load(p)
    score_after = lm2.score_per_char("CQ DE F4HYY")
    assert abs(score_before - score_after) < 1e-9
    assert lm2.vocab_size == lm.vocab_size


def test_empty_text_scores_to_zero() -> None:
    lm = CharNGramLM(order=3).fit(["ABC"])
    assert lm.score("") == 0.0
    assert lm.score_per_char("") == 0.0


def test_unknown_char_does_not_crash() -> None:
    lm = CharNGramLM(order=3).fit(["ABC"])
    s = lm.score("XYZ")
    # Should be finite (penalty applied, not -inf).
    assert s != float("-inf") and not (s != s)  # not NaN
