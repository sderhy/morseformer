"""Evaluation metrics for CW decoding.

Three families:
    * Character Error Rate (CER) — Levenshtein / len(reference)
    * Word Error Rate (WER)      — Levenshtein over whitespace-split words
    * Callsign F1                 — regex-extract valid ITU-style callsigns,
                                    compare as sets, report precision/recall/F1.

All metrics are computed over uppercase-normalised strings; callers do not
need to pre-normalise.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


def _levenshtein(a: list, b: list) -> int:
    """Classical O(len(a)*len(b)) memory-O(min) edit distance."""
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            curr[j] = min(
                prev[j] + 1,           # deletion
                curr[j - 1] + 1,       # insertion
                prev[j - 1] + (0 if ca == cb else 1),  # substitution
            )
        prev = curr
    return prev[-1]


def character_error_rate(reference: str, hypothesis: str) -> float:
    ref = list(reference.upper())
    hyp = list(hypothesis.upper())
    if not ref:
        return 0.0 if not hyp else 1.0
    return _levenshtein(ref, hyp) / len(ref)


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref = reference.upper().split()
    hyp = hypothesis.upper().split()
    if not ref:
        return 0.0 if not hyp else 1.0
    return _levenshtein(ref, hyp) / len(ref)


# ITU-style callsign: 1-2 letters/digits + 1 digit + 1-4 letters.
# This matches the vast majority of ham callsigns; exotic cases (club calls,
# portable operation suffixes like "/P", "/MM") are handled separately.
CALLSIGN_RE = re.compile(r"\b[A-Z0-9]{1,2}[0-9][A-Z]{1,4}\b")


@dataclass(frozen=True)
class CallsignScores:
    precision: float
    recall: float
    f1: float


def callsign_scores(reference: str, hypothesis: str) -> CallsignScores:
    ref_set = set(CALLSIGN_RE.findall(reference.upper()))
    hyp_set = set(CALLSIGN_RE.findall(hypothesis.upper()))
    if not ref_set and not hyp_set:
        return CallsignScores(1.0, 1.0, 1.0)
    if not ref_set:
        return CallsignScores(0.0, 1.0, 0.0)
    if not hyp_set:
        return CallsignScores(1.0, 0.0, 0.0)
    tp = len(ref_set & hyp_set)
    precision = tp / len(hyp_set)
    recall = tp / len(ref_set)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return CallsignScores(precision, recall, f1)
