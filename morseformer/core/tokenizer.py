"""Character-level tokenizer for the CW acoustic model.

A fixed 49-token vocabulary covering everything the synthesiser can
emit plus the CTC blank, including the three French CW characters
(É / À / apostrophe) added in Phase 3.4. Prosigns are not given
dedicated tokens — they are rendered as their printed equivalents
(`<BT>` → `=`, `<AR>` → `+`, etc.) where convenient, so the model
can work in a flat character space without special-case handling.

Layout:

    index  0         →  <blank>  (CTC blank; never emitted by encode())
    index  1         →  ' '      (word boundary)
    index  2–27      →  A–Z
    index 28–37      →  0–9
    index 38–45      →  . , ? ! / = + -
    index 46–48      →  É À '    (Phase 3.4 French CW additions)

Characters outside the vocabulary are silently dropped during encoding.
``encode()`` calls ``str.upper()`` first, so French source text written
in lowercase ("été", "à") is normalised to É / À before tokenisation.
"""

from __future__ import annotations

BLANK_TOKEN = "<blank>"
BLANK_INDEX = 0
SPACE_INDEX = 1

# Phase 3.4 additions appended at the tail so that pretrained checkpoints
# from Phases 2.x / 3.0–3.3 can be extended in-place: the first 46 indices
# keep their meaning and only the new rows in joint.out / pred.embed /
# acoustic.head need random initialisation. See
# scripts/extend_tokenizer_46_to_49.py.
_VOCAB: list[str] = (
    [BLANK_TOKEN, " "]
    + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    + list("0123456789")
    + list(".,?!/=+-")
    + ["É", "À", "'"]
)

# Bidirectional lookup tables.
TOKEN_TO_INDEX: dict[str, int] = {tok: i for i, tok in enumerate(_VOCAB)}
INDEX_TO_TOKEN: list[str] = list(_VOCAB)
VOCAB_SIZE: int = len(_VOCAB)  # 49

# Pre-Phase-3.4 vocabulary size, retained as a constant so that legacy
# checkpoints can be loaded and extended without hard-coding 46 in
# multiple places.
LEGACY_VOCAB_SIZE: int = 46


def encode(text: str) -> list[int]:
    """Encode a text string to a list of token indices.

    Uppercases the input, collapses runs of whitespace to a single space,
    and drops characters that are not in the vocabulary.
    """
    out: list[int] = []
    prev_space = True  # suppress leading space
    for ch in text.upper():
        if ch.isspace():
            if not prev_space:
                out.append(SPACE_INDEX)
                prev_space = True
            continue
        idx = TOKEN_TO_INDEX.get(ch)
        if idx is not None:
            out.append(idx)
            prev_space = False
    # Strip trailing space.
    if out and out[-1] == SPACE_INDEX:
        out.pop()
    return out


def decode(indices: list[int], strip: bool = True) -> str:
    """Decode a list of token indices back to a text string.

    Blanks are ignored; everything else is mapped through `INDEX_TO_TOKEN`.
    Does *not* collapse CTC repeats — use `ctc_greedy_decode` for that.

    ``strip`` defaults to True for backward compatibility (eval / training
    callers want a clean text). Streaming callers must pass ``strip=False``
    so that an inter-word space falling at a window boundary is not
    swallowed and word-collision artefacts ("HELLOWORLD") do not appear
    in concatenated fragments.
    """
    chars: list[str] = []
    for idx in indices:
        if idx == BLANK_INDEX:
            continue
        if 0 <= idx < VOCAB_SIZE:
            chars.append(INDEX_TO_TOKEN[idx])
    out = "".join(chars)
    return out.strip() if strip else out


def ctc_greedy_decode(indices: list[int]) -> str:
    """Collapse CTC-repeated tokens and strip blanks, then decode to text."""
    collapsed: list[int] = []
    prev = -1
    for idx in indices:
        if idx != prev:
            if idx != BLANK_INDEX:
                collapsed.append(idx)
            prev = idx
    return decode(collapsed)
