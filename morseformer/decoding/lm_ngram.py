"""Tiny character n-gram LM for amateur-idiom splitter rescoring.

Phase 11 §C — a focused replacement for the heavyweight neural prose
LM (``lm_phase5_2``) that hurt ragchew because it had learned literary
bigrammes. This LM is trained ONLY on amateur idioms (Q-codes, cut
numbers, callsigns, run-on macros) generated from
:func:`DatasetConfig.phase_9`, so its scoring scale rewards the same
idioms the post-process splitter tries to recover.

Smoothing: stupid backoff (Brants 2007). Not a proper probability —
the relative scores are what matters for rescoring two candidate
segmentations of the same acoustic emission. Stupid backoff is near-KN
at scale, trivial to implement, and saves us the kenlm C++ build
dependency on environments without cmake/gcc.

The serialised model is just a dict of n-gram counts, pickled. Vocab
is implicit (any char appearing in training is in vocab; unseen chars
get a flat penalty at scoring time).

Usage::

    lm = CharNGramLM(order=3)
    lm.fit(["CQ CQ DE F4HYY", "5NN OM TU 73"])
    lm.score("CQ DE F4HYY")           # higher is better
    lm.score_per_char("CQ DE F4HYY")  # length-normalised
    lm.save(Path("checkpoints/lm_amateur_3gram.pkl"))
    lm = CharNGramLM.load(Path("checkpoints/lm_amateur_3gram.pkl"))
"""

from __future__ import annotations

import math
import pickle
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


_BOS = "^"
_EOS = "$"
_UNK_LOG_PROB = math.log(1e-6)


@dataclass
class CharNGramLM:
    """Character n-gram with stupid backoff.

    ``counts[n]`` maps n-gram tuples (``len == n``) to their training
    occurrence count. ``counts[0]`` carries the empty-tuple key with
    the total character count (the unigram denominator).
    """

    order: int = 3
    backoff_alpha: float = 0.4
    counts: list[dict[tuple[str, ...], int]] = field(default_factory=list)
    _total_chars: int = 0
    _vocab: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        if not self.counts:
            self.counts = [defaultdict(int) for _ in range(self.order + 1)]

    @staticmethod
    def _wrap(text: str, order: int) -> str:
        """Prepend ``order - 1`` BOS markers and append a single EOS."""
        return (_BOS * max(order - 1, 1)) + text + _EOS

    def fit(self, texts: Iterable[str]) -> "CharNGramLM":
        """Accumulate n-gram counts from ``texts`` (one string per
        training sentence)."""
        for text in texts:
            if not text:
                continue
            wrapped = self._wrap(text, self.order)
            self._total_chars += len(text) + 1  # exclude BOS, include EOS
            for ch in text:
                self._vocab.add(ch)
            for n in range(1, self.order + 1):
                for i in range(len(wrapped) - n + 1):
                    self.counts[n][tuple(wrapped[i : i + n])] += 1
        return self

    def score(self, text: str) -> float:
        """Return a log-score (sum of per-char stupid-backoff log probs)."""
        if not text:
            return 0.0
        wrapped = self._wrap(text, self.order)
        log_score = 0.0
        # Score each char from position (order-1) onward — the first
        # ``order-1`` slots are BOS padding used as context.
        start = self.order - 1
        for i in range(start, len(wrapped)):
            log_score += self._char_log_prob(wrapped, i)
        return log_score

    def score_per_char(self, text: str) -> float:
        """Length-normalised log-score (mean per-character log prob).

        Use this when comparing segmentations of different lengths, e.g.
        ``"MYWXIS"`` vs ``"MY WX IS"`` — the split version has more
        chars (the spaces) and would lose under raw ``score()``.
        """
        if not text:
            return 0.0
        # Number of "real" emissions scored: len(text) + 1 (the EOS).
        n_scored = (len(text) + 1)
        return self.score(text) / n_scored

    def _char_log_prob(self, wrapped: str, i: int) -> float:
        """Stupid-backoff log prob of ``wrapped[i]`` given preceding
        context ``wrapped[i-(order-1) : i]``."""
        # Start at full order and back off until we get a count.
        alpha_pow = 0  # number of backoff steps so far
        for n in range(self.order, 1, -1):
            ngram = tuple(wrapped[i - n + 1 : i + 1])
            context = ngram[:-1]
            c_full = self.counts[n].get(ngram, 0)
            if c_full > 0:
                c_ctx = self.counts[n - 1].get(context, 0)
                if c_ctx > 0:
                    return (
                        math.log(c_full / c_ctx)
                        + alpha_pow * math.log(self.backoff_alpha)
                    )
            alpha_pow += 1
        # Unigram fallback.
        c_uni = self.counts[1].get((wrapped[i],), 0)
        if c_uni > 0 and self._total_chars > 0:
            return (
                math.log(c_uni / self._total_chars)
                + alpha_pow * math.log(self.backoff_alpha)
            )
        # Truly unseen char — flat penalty.
        return _UNK_LOG_PROB + alpha_pow * math.log(self.backoff_alpha)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump({
                "order": self.order,
                "backoff_alpha": self.backoff_alpha,
                "counts": [dict(c) for c in self.counts],
                "total_chars": self._total_chars,
                "vocab": sorted(self._vocab),
            }, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path) -> "CharNGramLM":
        with path.open("rb") as f:
            payload = pickle.load(f)
        lm = cls(order=payload["order"], backoff_alpha=payload["backoff_alpha"])
        for n, ng_dict in enumerate(payload["counts"]):
            lm.counts[n] = defaultdict(int, ng_dict)
        lm._total_chars = payload["total_chars"]
        lm._vocab = set(payload["vocab"])
        return lm

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)
