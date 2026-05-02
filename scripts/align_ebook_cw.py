"""Align ebook2cw-generated CW audio against its ground-truth text.

The ebook2cw tool deterministically renders a text file as Morse audio
at a fixed WPM. We can therefore obtain a per-chunk training label by:

    1. running the existing model on each ``--chunk-seconds`` window of
       the audio (CTC head, which is sturdier than RNN-T for this kind
       of mostly-clean audio),
    2. concatenating the per-chunk CTC hypotheses into one global
       ``decoded_text``,
    3. running ``difflib.SequenceMatcher`` between ``decoded_text`` and
       the normalised ground-truth ``gt_text`` to obtain a char-level
       alignment, and
    4. for each chunk ``[start_char_decoded, end_char_decoded]``,
       mapping the bounds through the alignment to get the
       corresponding ground-truth slice ``gt_text[start_gt:end_gt]`` —
       which becomes the gold-standard label for that audio segment.

This works even at CER 25-30 % on the decode side, because difflib
finds the matching blocks at much finer granularity than the full
hypothesis — each correct phrase gets its block, and we take the
ground-truth between blocks even when the decoder was wrong.

Output JSONL (one line per chunk):

    {
      "audio_path": "data/real/alice_chapter1.wav",
      "chunk_idx": 12,
      "start_s": 72.0,
      "end_s": 78.0,
      "label": "ALICE WAS BEGINNING TO GET",
      "decoded": "AA IICE WAS BEGINNIT EG TO GET",
      "score": 0.78,
    }

Usage::

    python -m scripts.align_ebook_cw \
        --decoded /tmp/alice_ch1_decoded.jsonl \
        --gt-text "/home/serge/Bureau/wav/Alice-adventures-in-wonderland/Alice-adventures-in-wonderland/Text/chapter1.txt" \
        --audio-path data/real/alice_chapter1.wav \
        --output data/real/alice_chapter1_aligned.jsonl
"""

from __future__ import annotations

import argparse
import difflib
import json
from pathlib import Path

from morseformer.data.text import _normalize_prose


def _build_char_mapping(decoded: str, gt: str) -> list[int]:
    """Return ``mapping[i]`` = position in ``gt`` aligned to ``decoded[i]``.

    Uses ``SequenceMatcher.get_matching_blocks`` to find common
    substrings, then linearly interpolates positions inside gaps. The
    last entry is ``len(gt)`` so callers can use ``mapping[end]`` to get
    the slice end.
    """
    mapping = [0] * (len(decoded) + 1)
    sm = difflib.SequenceMatcher(autojunk=False, a=decoded, b=gt)
    blocks = list(sm.get_matching_blocks())  # ends with (len(a), len(b), 0)
    # Interpolate between successive matching blocks.
    prev_a, prev_b = 0, 0
    for a_start, b_start, size in blocks:
        # Gap [prev_a, a_start) in decoded ↔ [prev_b, b_start) in gt
        n = a_start - prev_a
        m = b_start - prev_b
        for k in range(n):
            mapping[prev_a + k] = (
                prev_b + (k * m // n) if n > 0 else prev_b
            )
        # Matching block: identity mapping
        for k in range(size):
            mapping[a_start + k] = b_start + k
        prev_a = a_start + size
        prev_b = b_start + size
    mapping[len(decoded)] = len(gt)
    return mapping


def _load_decoded(path: Path) -> list[dict]:
    chunks: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--decoded", type=Path, required=True,
                   help="JSONL produced by scripts/decode_fav22.py "
                        "(re-usable for ebook audio).")
    p.add_argument("--gt-text", type=Path, required=True,
                   help="Plain ground-truth text file matching the audio.")
    p.add_argument("--audio-path", type=Path, required=True,
                   help="Path of the wav file (recorded into each output "
                        "row so the dataset loader can locate the audio).")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--language", default="en",
                   choices=("en", "fr", "de", "es"),
                   help="for _normalize_prose()")
    p.add_argument("--use-ctc", action="store_true", default=True,
                   help="use the CTC hypothesis for alignment (default; "
                        "more reliable than RNN-T on ebook2cw audio).")
    p.add_argument("--use-rnnt", dest="use_ctc", action="store_false")
    p.add_argument("--min-label-chars", type=int, default=3,
                   help="drop chunks whose mapped label is shorter than "
                        "this — usually pure-silence boundaries.")
    p.add_argument("--max-label-chars", type=int, default=80,
                   help="drop chunks with abnormally long labels (alignment "
                        "drift). Default 80 = ~13 chars/sec at 20 WPM.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    print(f"[align_ebook] loading decoded {args.decoded}")
    chunks = _load_decoded(args.decoded)
    if not chunks:
        raise SystemExit("[align_ebook] decoded jsonl is empty.")
    field = "ctc_hyp" if args.use_ctc else "rnnt_hyp"

    # Concatenate decoded hypotheses with single-space joiners, and
    # remember the char-range of each chunk so we can map back later.
    decoded_parts: list[str] = []
    chunk_char_ranges: list[tuple[int, int]] = []
    pos = 0
    for i, ch in enumerate(chunks):
        if i > 0:
            decoded_parts.append(" ")
            pos += 1
        text = ch[field]
        chunk_char_ranges.append((pos, pos + len(text)))
        decoded_parts.append(text)
        pos += len(text)
    decoded_text = "".join(decoded_parts)
    print(f"[align_ebook] decoded text: {len(decoded_text):,} chars from {len(chunks)} chunks")

    print(f"[align_ebook] loading + normalising ground-truth {args.gt_text}")
    raw = args.gt_text.read_text(encoding="utf-8")
    gt_text = _normalize_prose(raw, args.language)
    print(f"[align_ebook] gt text:     {len(gt_text):,} chars (normalised, lang={args.language})")

    print(f"[align_ebook] building char-level alignment …")
    mapping = _build_char_mapping(decoded_text, gt_text)

    # Score the alignment globally
    sm = difflib.SequenceMatcher(autojunk=False, a=decoded_text, b=gt_text)
    global_ratio = sm.ratio()
    print(f"[align_ebook] global decode/gt ratio: {global_ratio:.4f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    n_kept = n_dropped = 0
    with args.output.open("w", encoding="utf-8") as f:
        for i, ch in enumerate(chunks):
            cs, ce = chunk_char_ranges[i]
            gt_start = mapping[cs]
            gt_end = mapping[ce]
            label = gt_text[gt_start:gt_end].strip()
            if (
                len(label) < args.min_label_chars
                or len(label) > args.max_label_chars
            ):
                n_dropped += 1
                continue
            # Local alignment quality between the decoded chunk and its
            # mapped GT slice — a sanity score per chunk.
            decoded_chunk = ch[field]
            local_sm = difflib.SequenceMatcher(
                autojunk=False, a=decoded_chunk, b=label
            )
            local_ratio = local_sm.ratio()
            f.write(json.dumps({
                "audio_path": str(args.audio_path),
                "chunk_idx": int(ch["chunk_idx"]),
                "start_s": float(ch["start_s"]),
                "end_s": float(ch["end_s"]),
                "label": label,
                "decoded": decoded_chunk,
                "score": local_ratio,
            }, ensure_ascii=False) + "\n")
            n_kept += 1

    print(f"[align_ebook] kept {n_kept} / {len(chunks)} chunks "
          f"(dropped {n_dropped} short / over-long) → {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
