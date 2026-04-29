"""Align FAV22 audio↔text using a sliding-window decode + sequential
fuzzy matching against the reference clair blocks.

Inputs::

    --decoded   data/corpus/fav22_lent_decoded.jsonl  (from decode_fav22.py)
    --blocks    data/corpus/fav22_blocks.jsonl

Output::

    --output    data/corpus/fav22_lent_aligned.jsonl

Each output line::

    {
      "lesson_id": "01-1/2",
      "block_idx": 1,
      "start_s": 42.0,
      "end_s": 86.0,
      "vitesse_signs_per_min": 420,
      "label": "LE PROCESSUS D'ACQUISITION DE L'INFORMATION...",
      "score": 0.78,             # difflib ratio between hyp and label
      "hyp_extract": "LE PROCESSUS DACQUSITION DE LINFOR..."
    }

Algorithm
---------
1. Concatenate the per-chunk hypotheses into one long string. Build a
   parallel ``char_to_time`` array so any character index maps back to
   an audio second (linear interpolation across each chunk's text).
2. Walk the reference *clair* blocks in their JSONL order. For each
   block, slide a window of length ``len(label) * 1.4`` over a search
   region of the global hypothesis starting at the previous block's
   end character. Score each window with
   ``difflib.SequenceMatcher.ratio`` against the label.
3. Pick the window with the best ratio above ``--min-score``. Convert
   the window's character bounds to (start_s, end_s). Advance the
   search start past the matched window.

This is robust to a high CER (~20-30 %) because difflib's LCS-style
ratio matches insertions / deletions / substitutions gracefully.
"""

from __future__ import annotations

import argparse
import difflib
import json
import time
from pathlib import Path


def _load_decoded(path: Path) -> tuple[str, list[float], list[tuple[int, int, float, float]]]:
    """Concatenate per-chunk hypotheses and build a char→time map.

    Returns
    -------
    hyp_text : str
        ``" ".join(rnnt_hyp for each chunk)`` (a single space joins
        adjacent chunks so word boundaries survive the boundary).
    char_to_time : list[float]
        ``len(hyp_text) + 1`` entries; index ``i`` is the audio second
        at character ``hyp_text[i]`` (linearly interpolated within each
        chunk's text segment).
    chunks : list[(char_start, char_end, t_start, t_end)]
        Per-chunk bookkeeping; useful for debugging / sanity.
    """
    hyp_parts: list[str] = []
    chunks: list[tuple[int, int, float, float]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            hyp_parts.append(rec["rnnt_hyp"])
            chunks.append((0, 0, float(rec["start_s"]), float(rec["end_s"])))
    # Build the joined string and update char-bounds in `chunks`.
    out: list[str] = []
    pos = 0
    for i, part in enumerate(hyp_parts):
        if i > 0:
            out.append(" ")
            pos += 1
        cs, _, ts, te = chunks[i]
        chunks[i] = (pos, pos + len(part), ts, te)
        out.append(part)
        pos += len(part)
    hyp_text = "".join(out)

    # char_to_time[i] = time at character i (for i in [0, len(hyp_text)]).
    char_to_time = [0.0] * (len(hyp_text) + 1)
    for cs, ce, ts, te in chunks:
        n = max(1, ce - cs)
        for k in range(cs, ce + 1):
            frac = (k - cs) / n
            char_to_time[k] = ts + frac * (te - ts)
    # Spaces between chunks fall on the "ts" of the next chunk.
    return hyp_text, char_to_time, chunks


def _load_clair_blocks(path: Path) -> list[dict]:
    blocks: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("mode") == "clair":
                blocks.append(rec)
    return blocks


def _best_window(
    hyp_text: str,
    label: str,
    search_start: int,
    search_end: int,
    *,
    win_factor: float = 1.4,
    step: int | None = None,
) -> tuple[int, int, float]:
    """Slide a length-(win_factor * len(label)) window over
    hyp_text[search_start:search_end] and return (best_start, best_end,
    best_ratio)."""
    win = max(int(len(label) * win_factor), 32)
    if step is None:
        step = max(1, len(label) // 12)  # ~12 samples per label length
    sm = difflib.SequenceMatcher(autojunk=False)
    sm.set_seq1(label)
    best = (search_start, min(search_end, search_start + win), 0.0)
    s = search_start
    while s + 32 <= search_end:
        e = min(s + win, search_end)
        sm.set_seq2(hyp_text[s:e])
        r = sm.ratio()
        if r > best[2]:
            best = (s, e, r)
        s += step
    return best


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--decoded", type=Path, required=True)
    p.add_argument("--blocks", type=Path,
                   default=Path("data/corpus/fav22_blocks.jsonl"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--min-score", type=float, default=0.45,
                   help="difflib ratio floor; segments below are dropped.")
    p.add_argument("--max-drift-chars", type=int, default=8000,
                   help="how far past the previous block end to search "
                        "for the next block. Wide enough to skip a "
                        "missed codé block (≈600 chars at 420 sig/min).")
    p.add_argument("--win-factor", type=float, default=1.4)
    p.add_argument("--limit", type=int, default=0,
                   help="stop after N blocks (0 = all). For debugging.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(f"[align] loading decoded hyps {args.decoded}")
    hyp_text, char_to_time, chunks = _load_decoded(args.decoded)
    print(f"[align] hyp: {len(hyp_text):,} chars from {len(chunks):,} chunks "
          f"= {chunks[-1][3]/60:.1f} min audio")
    print(f"[align] loading reference blocks {args.blocks}")
    blocks = _load_clair_blocks(args.blocks)
    if args.limit:
        blocks = blocks[: args.limit]
    print(f"[align] {len(blocks)} clair blocks to align")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    aligned = 0
    skipped = 0
    cursor = 0
    t0 = time.time()
    with args.output.open("w", encoding="utf-8") as f:
        for i, blk in enumerate(blocks):
            label = blk["normalized_text"]
            if not label:
                continue
            search_end = min(len(hyp_text), cursor + args.max_drift_chars)
            cs, ce, score = _best_window(
                hyp_text, label, cursor, search_end,
                win_factor=args.win_factor,
            )
            if score < args.min_score:
                skipped += 1
                # Don't advance the cursor too aggressively on a miss —
                # the next block may still be findable. Slide cursor by
                # ~1 label worth so we make progress.
                cursor = min(search_end, cursor + len(label))
                continue
            start_s = char_to_time[cs]
            end_s = char_to_time[ce]
            f.write(json.dumps({
                "lesson_id": blk["lesson_id"],
                "block_idx": blk["block_idx"],
                "start_s": start_s,
                "end_s": end_s,
                "vitesse_signs_per_min": blk["vitesse_signs_per_min"],
                "label": label,
                "score": score,
                "hyp_extract": hyp_text[cs:ce][:200],
            }, ensure_ascii=False) + "\n")
            aligned += 1
            cursor = ce  # advance past the matched window
            if (i + 1) % 10 == 0:
                print(f"[align] block {i+1}/{len(blocks)} "
                      f"score={score:.2f} t={start_s:.0f}-{end_s:.0f}s "
                      f"cursor={cursor}/{len(hyp_text)}")

    print(f"[align] done in {time.time()-t0:.1f} s. "
          f"aligned={aligned} skipped={skipped} → {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
