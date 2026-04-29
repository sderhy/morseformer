"""Parse the FAV22 reference transcript PDF into a structured JSONL.

The FAV22 dataset (training material for the French Morse competition,
courtesy CNMO TSR / Favieres-Vernon) ships as a 43-page PDF whose
content alternates between

    * **codé** blocks — random 5-letter / 5-digit groups, no
      linguistic prior, used for top-speed CW reception drills;
    * **clair** blocks — natural French prose with full diacritics
      (é / è / ê / à / ç / apostrophes), used for context-dependent
      reception drills.

Each block is preceded by a header of the form

    Lundi-Leçon numéro 01-1/2 Vitesse 420 clair
    Dimanche 7/Leçon numéro 2/1 Vitesse 600 codé

The day prefix can be ``<Day>-`` or ``<Day> <N>/`` (latter half of the
PDF). The lesson identifier may contain digits and slashes; the speed
is given in signs / min and the mode is one of ``codé`` / ``clair``.

This script extracts every block, normalises the text into the
49-token tokenizer alphabet (preserving É / À / apostrophe), and
writes the result as a JSONL where each line is::

    {
      "block_idx": 12,
      "day": "Lundi",
      "lesson_id": "01-1/4",
      "vitesse_signs_per_min": 420,
      "mode": "clair",
      "raw_text": "Quand le paquebot géant heurte un iceberg...",
      "normalized_text": "QUAND LE PAQUEBOT GÉANT HEURTE UN ICEBERG..."
    }

The normalised string is exactly what the model is trained on, so it
can be fed straight to ``tokenizer.encode`` for label generation when
we get to the audio↔text alignment stage of Pass B.

Usage::

    python -m scripts.parse_fav22_pdf \
        --pdf /home/serge/Bureau/wav/REF_VAV22-F9TM_Corriges.pdf \
        --output data/corpus/fav22_blocks.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from pypdf import PdfReader

from morseformer.data.text import _normalize_prose


# Match any of the two header formats observed in the PDF:
#   Lundi-Leçon numéro 01-1/4 Vitesse 420 codé
#   Dimanche 7/Leçon numéro 2/1 Vitesse 600 clair
# Allow loose whitespace because PDF extraction occasionally inserts
# soft hyphens and non-breaking spaces around the prefix.
_HEADER_RE = re.compile(
    r"""(?P<day>[A-Za-zéèêàùîôûç]+)            # day name
        (?:\s+(?P<day_n>\d+))?                  # optional " 7"
        [-/]\s*                                 # delimiter '-' or '/'
        Leçon\s+(?:numéro\s+)?                  # 'Leçon numéro' (numéro optional)
        (?P<lesson_id>[\w\-/]+)                 # lesson id (digits, dashes, slashes)
        \s+Vitesse\s+(?P<vitesse>\d+)           # speed in signs per minute
        \s+(?P<mode>cod[éeè]|clair)             # mode (accept variant spellings)
    """,
    re.VERBOSE,
)


def _extract_full_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    return "\n".join(p.extract_text() or "" for p in reader.pages)


def _tidy_codé(text: str) -> str:
    """Lightly clean a codé block: uppercase, collapse runs of
    whitespace, drop characters outside the 49-token vocabulary.

    Codé blocks are 5-letter groups, sometimes with embedded digit
    groups and rare punctuation tokens. They contain neither
    diacritics nor apostrophes by construction, so the FR-aware
    normaliser is overkill — but it gives us the same final alphabet
    so a single ``tokenizer.encode`` works on either mode."""
    return _normalize_prose(text, "fr")


def _tidy_clair(text: str) -> str:
    return _normalize_prose(text, "fr")


def parse_blocks(raw: str) -> list[dict]:
    matches = list(_HEADER_RE.finditer(raw))
    blocks: list[dict] = []
    for i, m in enumerate(matches):
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        body = raw[body_start:body_end].strip()
        mode = m.group("mode")
        # Re-spell variant codé encodings to the canonical form.
        mode_canon = "codé" if mode.startswith("cod") else "clair"
        normalised = (
            _tidy_clair(body) if mode_canon == "clair" else _tidy_codé(body)
        )
        blocks.append(
            {
                "block_idx": i,
                "day": m.group("day"),
                "day_n": m.group("day_n"),
                "lesson_id": m.group("lesson_id"),
                "vitesse_signs_per_min": int(m.group("vitesse")),
                "mode": mode_canon,
                "raw_text": body,
                "normalized_text": normalised,
            }
        )
    return blocks


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--pdf",
        type=Path,
        default=Path("/home/serge/Bureau/wav/REF_VAV22-F9TM_Corriges.pdf"),
        help="Source PDF (default: the FAV22 corrigés on the desktop).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("data/corpus/fav22_blocks.jsonl"),
        help="Destination JSONL.",
    )
    p.add_argument(
        "--summary-only",
        action="store_true",
        help="Print summary statistics without writing the JSONL.",
    )
    return p


def _summarise(blocks: list[dict]) -> str:
    by_mode: dict[str, int] = {}
    by_speed: dict[int, int] = {}
    by_day: dict[str, int] = {}
    total_chars_clair = 0
    total_chars_codé = 0
    new_token_chars = 0
    for b in blocks:
        by_mode[b["mode"]] = by_mode.get(b["mode"], 0) + 1
        by_speed[b["vitesse_signs_per_min"]] = (
            by_speed.get(b["vitesse_signs_per_min"], 0) + 1
        )
        by_day[b["day"]] = by_day.get(b["day"], 0) + 1
        if b["mode"] == "clair":
            total_chars_clair += len(b["normalized_text"])
            new_token_chars += sum(
                b["normalized_text"].count(t) for t in ("É", "À", "'")
            )
        else:
            total_chars_codé += len(b["normalized_text"])
    lines = [
        f"  total blocks: {len(blocks)}",
        f"  by mode: {by_mode}",
        f"  by day: {by_day}",
        f"  by speed (signs/min): "
        f"{dict(sorted(by_speed.items()))}",
        f"  clair char total (normalised): {total_chars_clair}",
        f"  codé char total  (normalised): {total_chars_codé}",
        f"  É / À / apostrophe in clair: {new_token_chars} "
        f"({100 * new_token_chars / max(1, total_chars_clair):.2f} %)",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    raw = _extract_full_text(args.pdf)
    blocks = parse_blocks(raw)
    print(f"[fav22] parsed {len(blocks)} blocks from {args.pdf}")
    print(_summarise(blocks))
    if args.summary_only:
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for b in blocks:
            f.write(json.dumps(b, ensure_ascii=False) + "\n")
    print(f"[fav22] wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
