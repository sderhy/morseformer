"""Train the amateur-idiom char n-gram LM (Phase 11 §C).

Samples ``--n-samples`` texts from :data:`PHASE_9_MIX` (the same mix
the Phase 11 acoustic curriculum trains on, so the LM and the
acoustic share a distribution over amateur idioms / Q-codes / cut
numbers / callsigns), then fits a stupid-backoff char n-gram and
serialises it.

Run::

    python -m scripts.train_ngram_amateur
    python -m scripts.train_ngram_amateur --n-samples 200000 --order 4
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from morseformer.data.text import PHASE_9_MIX, sample_text
from morseformer.decoding.lm_ngram import CharNGramLM


_DEFAULT_OUT = Path("checkpoints/lm_amateur_3gram.pkl")


def _normalise_for_lm(text: str) -> str:
    """The acoustic emits uppercase chars from the 49-vocab. Match
    that on the LM side so split / unsplit comparisons run in the
    same space."""
    text = text.upper()
    # Collapse runs of whitespace to single space — the splitter
    # operates on space-separated tokens.
    return " ".join(text.split())


def _iter_samples(n_samples: int, seed: int):
    rng = np.random.default_rng(seed)
    for _ in range(n_samples):
        yield _normalise_for_lm(sample_text(rng, PHASE_9_MIX))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--n-samples", type=int, default=100_000)
    p.add_argument("--order", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    p.add_argument(
        "--sanity-corpus", type=Path, default=None,
        help="Optional path to write the first 1000 training samples for "
             "manual inspection.",
    )
    args = p.parse_args(argv)

    print(
        f"[ngram] n_samples={args.n_samples} order={args.order} "
        f"seed={args.seed} → {args.out}"
    )
    t0 = time.time()
    lm = CharNGramLM(order=args.order)

    sanity_buf: list[str] = []

    # Batch in chunks of 5000 so memory stays bounded even at 1M samples.
    n_chunk = 5000
    samples_iter = _iter_samples(args.n_samples, args.seed)
    n_done = 0
    while n_done < args.n_samples:
        batch: list[str] = []
        for _ in range(min(n_chunk, args.n_samples - n_done)):
            batch.append(next(samples_iter))
        lm.fit(batch)
        if args.sanity_corpus and len(sanity_buf) < 1000:
            for s in batch:
                if len(sanity_buf) < 1000:
                    sanity_buf.append(s)
        n_done += len(batch)
        print(f"  fit {n_done:>7d}/{args.n_samples}  elapsed={time.time() - t0:.1f}s")

    n_unigrams = sum(1 for _ in lm.counts[1])
    n_bigrams = sum(1 for _ in lm.counts[2]) if args.order >= 2 else 0
    n_trigrams = sum(1 for _ in lm.counts[3]) if args.order >= 3 else 0
    print(
        f"[ngram] vocab={lm.vocab_size} 1-grams={n_unigrams} "
        f"2-grams={n_bigrams} 3-grams={n_trigrams} "
        f"total_chars={lm._total_chars:,}"
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    lm.save(args.out)
    size_kb = args.out.stat().st_size / 1024
    print(f"[ngram] saved {args.out} ({size_kb:.1f} KB)")

    if args.sanity_corpus:
        args.sanity_corpus.parent.mkdir(parents=True, exist_ok=True)
        args.sanity_corpus.write_text(
            "\n".join(sanity_buf), encoding="utf-8"
        )
        print(f"[ngram] sanity corpus → {args.sanity_corpus}")

    # Quick smoke-test: amateur idioms should outscore alien strings.
    smoke = [
        ("CQ DE F4HYY", "ZZZZQXYWQ"),
        ("5NN OM TU 73", "QABCJWXLKR"),
        ("MY WX IS SUNNY", "MYWXISSUNNY"),
    ]
    print("[ngram] smoke test (higher = better)")
    for good, bad in smoke:
        sg = lm.score_per_char(good)
        sb = lm.score_per_char(bad)
        win = "✓" if sg > sb else "✗"
        print(f"  {win}  '{good}' = {sg:+.3f}   vs   '{bad}' = {sb:+.3f}")

    print(f"[ngram] done in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
