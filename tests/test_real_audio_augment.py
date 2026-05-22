"""Tests for the word-gap augmentation in ``morseformer.data.real_audio``.

Phase 11 swap: the augmentation used to pick a SPACE position in the
label and linearly interpolate to the audio (Phase 10 — wrong: the
model learned "silence ≠ word gap" because the inserted silence rarely
landed inside actual operator silences). The new version takes the
forced-aligned ``char_starts_s`` and inserts silence in the *true*
inter-word gap.
"""

from __future__ import annotations

import numpy as np

from morseformer.core.tokenizer import SPACE_INDEX, encode
from morseformer.data.real_audio import _augment_word_gap


def _make_audio_two_words(sample_rate: int = 8000) -> tuple[np.ndarray, dict]:
    """Build a fake 6-s clip: 1 s tone, 2 s silence, 1 s tone, 2 s silence."""
    sr = sample_rate
    t = np.arange(sr).astype(np.float32) / sr
    tone = 0.5 * np.sin(2 * np.pi * 600 * t).astype(np.float32)
    silence = np.zeros(2 * sr, dtype=np.float32)
    audio = np.concatenate([tone, silence, tone, silence])
    # Label "A B" → tokens [A, SPACE, B]; SPACE token starts at the
    # beginning of the inter-word silence at t=1.0s.
    label_tokens = encode("A B")
    char_starts_s = [0.0, 1.0, 3.0]
    return audio, {
        "label": "A B",
        "tokens": label_tokens,
        "char_starts_s": char_starts_s,
    }


def test_augment_word_gap_aligned_lands_inside_silence() -> None:
    sr = 8000
    audio, meta = _make_audio_two_words(sr)
    assert meta["tokens"] == [encode("A B")[0], SPACE_INDEX, encode("A B")[2]]

    rng = np.random.default_rng(0)
    out_audio, out_tokens = _augment_word_gap(
        audio, meta["label"],
        sample_rate=sr,
        rng=rng,
        inflation_range=(6.0, 6.0),  # deterministic: exactly 6× = 2.4s
        tokens=meta["tokens"],
        char_starts_s=meta["char_starts_s"],
    )
    expected_gap_samples = int(round(6.0 * 0.4 * sr))  # 19_200
    assert out_audio.shape[0] == audio.shape[0] + expected_gap_samples
    # No target_samples → no truncation → tokens passed through.
    assert out_tokens == meta["tokens"]

    is_silent = np.abs(out_audio) < 1e-6
    runs: list[int] = []
    cur = 0
    for v in is_silent:
        if v:
            cur += 1
        else:
            if cur:
                runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)
    assert max(runs) >= expected_gap_samples


def test_augment_word_gap_truncates_label_when_audio_overflows() -> None:
    """Phase 11b fix: when inflation pushes audio past ``target_samples``,
    the inserted-silence tail eats real CW content from the end of the
    chunk. The returned token list must drop the now-missing trailing
    tokens, not the original full label."""
    sr = 8000
    target = sr * 6  # 48_000

    # Three "words" packed in 6s: tokens at t=0, 2, 4 s. Insert 3s silence
    # at the first inter-word boundary (t≈1s) → audio becomes 9s, then
    # truncated back to 6s. The last word (at t=4s, shifts to t=7s after
    # insertion) falls beyond the 6s horizon and must be dropped.
    audio = np.zeros(target, dtype=np.float32)
    audio[: int(0.5 * sr)] = 0.1
    audio[int(2 * sr) : int(2.5 * sr)] = 0.1
    audio[int(4 * sr) : int(4.5 * sr)] = 0.1
    tokens = [encode("A B C")[0], SPACE_INDEX, encode("A B C")[2],
              SPACE_INDEX, encode("A B C")[4]]
    char_starts_s = [0.0, 1.0, 2.0, 3.0, 4.0]

    rng = np.random.default_rng(0)
    out_audio, out_tokens = _augment_word_gap(
        audio, "A B C",
        sample_rate=sr,
        rng=rng,
        inflation_range=(7.5, 7.5),  # deterministic: 7.5 × 0.4s = 3s
        tokens=tokens,
        char_starts_s=char_starts_s,
        target_samples=target,
    )
    assert out_audio.shape[0] == target
    # First two letters survive (A at t=0 stays, B at t=2 shifts to t=5).
    # The trailing SPACE at original t=3 (shifted to t=6) lands exactly on
    # the boundary and gets dropped (we strip trailing SPACE). C at t=4
    # (shifted to t=7) is past the horizon and dropped.
    assert out_tokens == [tokens[0], SPACE_INDEX, tokens[2]]


def test_augment_word_gap_no_truncation_when_target_not_set() -> None:
    """Backward-compat: without ``target_samples``, the augmentation
    only inserts and returns the longer audio + unchanged tokens."""
    sr = 8000
    audio, meta = _make_audio_two_words(sr)
    rng = np.random.default_rng(0)
    out_audio, out_tokens = _augment_word_gap(
        audio, meta["label"],
        sample_rate=sr,
        rng=rng,
        inflation_range=(1.0, 1.0),
        tokens=meta["tokens"],
        char_starts_s=meta["char_starts_s"],
    )
    assert out_audio.shape[0] > audio.shape[0]
    assert out_tokens == meta["tokens"]


def test_augment_word_gap_linear_fallback_grows_audio() -> None:
    """Without tokens/char_starts_s, the legacy linear-interp path
    still runs, produces a longer audio array, and returns
    ``tokens=None`` so the caller knows to use ``encode(label)``."""
    sr = 8000
    audio = np.zeros(sr * 6, dtype=np.float32)
    audio[: sr * 1] = 0.1  # avoid all-silent
    audio[sr * 2 : sr * 3] = 0.1
    rng = np.random.default_rng(0)
    out_audio, out_tokens = _augment_word_gap(
        audio, "A B",
        sample_rate=sr,
        rng=rng,
        inflation_range=(3.0, 3.0),
    )
    assert out_audio.shape[0] == audio.shape[0] + int(round(3.0 * 0.4 * sr))
    assert out_tokens is None


def test_augment_word_gap_no_space_returns_unchanged() -> None:
    """If the label has no space and no aligned tokens, the audio is
    returned unchanged."""
    sr = 8000
    audio = np.full(sr, 0.1, dtype=np.float32)
    rng = np.random.default_rng(0)
    out_audio, out_tokens = _augment_word_gap(
        audio, "QRZ",
        sample_rate=sr,
        rng=rng,
        inflation_range=(3.0, 3.0),
    )
    assert out_audio.shape == audio.shape
    assert np.array_equal(out_audio, audio)
    assert out_tokens is None


def test_augment_word_gap_aligned_picks_only_space_tokens() -> None:
    """If tokens have no SPACE, the alignment path returns unchanged."""
    sr = 8000
    audio = np.full(sr * 6, 0.1, dtype=np.float32)
    tokens = encode("WORD")  # no space
    char_starts_s = [0.0, 0.5, 1.0, 1.5]
    rng = np.random.default_rng(0)
    out_audio, out_tokens = _augment_word_gap(
        audio, "WORD",
        sample_rate=sr,
        rng=rng,
        inflation_range=(3.0, 3.0),
        tokens=tokens,
        char_starts_s=char_starts_s,
    )
    assert out_audio.shape == audio.shape
    assert out_tokens == tokens
