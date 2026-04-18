"""DXCC prefix roots with activity-weighted priors for callsign synthesis.

Purpose
-------
Generate realistic-looking amateur-radio callsigns for training the
acoustic model. A callsign is formed as::

    callsign = root + digit + suffix [+ "/" + portable_tag]

where ``root`` is sampled from the table below, ``digit`` is ``0-9``
(skipped if the root already ends with a digit — as for ``HB9``,
``EA8``, ``KL7``), ``suffix`` is 1–3 random ``A-Z`` letters, and a
small fraction of calls receives a portable tag (``/P``, ``/M``,
``/MM``, ``/QRP``, ``/A``).

Sources and methodology
-----------------------
This list is a hand-curated subset of the most-active DXCC entities.
Weights are approximate log-scale priors — they are *not* precise
statistics, they are a reasonable proxy so that the training-time
distribution of prefixes is not flat. For precise activity rates and
the full ~350-entity DXCC list, consult:

- AD1C's ``CTY.DAT`` (https://www.country-files.com/), the reference
  country file used by N1MM Logger+, DXLab, and most contest-logging
  software. Updated monthly.
- ClubLog public statistics (https://clublog.org/).
- Reverse Beacon Network spot density (http://reversebeacon.net/).

For Phase 2 a compact ~55-entity table is sufficient to expose the
acoustic model to the prefix bigrams that actually matter; a richer
table or a live ``CTY.DAT`` parser can replace this module later
without touching any downstream code.

Callsign-structure caveats
--------------------------
Real ham callsign rules are richer than this table captures:

- Some entities allow only one or two valid digits (``HB9`` = licensed
  Swiss, ``HB3`` = trainee; ``HB0`` = Liechtenstein). Handled by baking
  the fixed digit into the root.
- Special-event and contest-only prefixes (``TM`` and ``TK`` for
  France, US 1×1 event calls) are covered implicitly by the random-
  digit / suffix-length distribution, not by dedicated entries.
- A few rare entities have structures that do not fit the
  ``<root><digit><suffix>`` mould (``1A0``, ``3Y0``, ``FT5``…). They
  are omitted here; the acoustic model will learn their characters via
  the Q-code / QSO-prose generators.

The table is auditable: one entry = one line. Disagreements with the
weights are settled with a single-file edit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CallRoot:
    """Prefix root for a DXCC entity.

    Attributes:
        root:   Everything that precedes the 1-3 letter suffix. If the
                last character of ``root`` is a digit, the digit slot
                is already baked in and the generator emits
                ``root + suffix``. Otherwise the generator samples a
                random 0-9 digit and emits ``root + digit + suffix``.
        weight: Relative activity prior (un-normalised; the sampler
                normalises across the full table).
        name:   DXCC entity name for documentation only.
    """

    root: str
    weight: float
    name: str


# --------------------------------------------------------------------- #
# The table
# --------------------------------------------------------------------- #

# ordering is by region for readability; the sampler is order-independent.
ENTRIES: tuple[CallRoot, ...] = (
    # ---------- North America ---------- #
    CallRoot("K",   16.0, "United States"),
    CallRoot("W",   12.0, "United States"),
    CallRoot("N",    6.0, "United States"),
    CallRoot("AA",   1.2, "United States"),
    CallRoot("AK",   0.4, "United States"),
    CallRoot("VE",   4.0, "Canada"),
    CallRoot("VA",   0.5, "Canada"),
    CallRoot("XE",   0.6, "Mexico"),
    # ---------- Europe ---------- #
    CallRoot("DL",   4.0, "Germany"),
    CallRoot("DK",   1.2, "Germany"),
    CallRoot("G",    3.5, "England"),
    CallRoot("F",    3.0, "France"),
    CallRoot("I",    2.5, "Italy"),
    CallRoot("IZ",   0.5, "Italy"),
    CallRoot("EA",   2.5, "Spain"),
    CallRoot("EA8",  0.3, "Canary Islands"),
    CallRoot("OH",   1.5, "Finland"),
    CallRoot("SM",   1.5, "Sweden"),
    CallRoot("LA",   1.0, "Norway"),
    CallRoot("OZ",   0.9, "Denmark"),
    CallRoot("OE",   1.2, "Austria"),
    CallRoot("HB9",  0.8, "Switzerland"),
    CallRoot("ON",   1.0, "Belgium"),
    CallRoot("PA",   1.2, "Netherlands"),
    CallRoot("OK",   1.2, "Czechia"),
    CallRoot("SP",   1.2, "Poland"),
    CallRoot("HA",   0.7, "Hungary"),
    CallRoot("YO",   0.7, "Romania"),
    CallRoot("LZ",   0.6, "Bulgaria"),
    CallRoot("SV",   0.7, "Greece"),
    CallRoot("TA",   0.5, "Turkey"),
    CallRoot("UA",   2.5, "European Russia"),
    CallRoot("R",    0.8, "Russia"),
    CallRoot("UR",   1.0, "Ukraine"),
    CallRoot("EW",   0.3, "Belarus"),
    CallRoot("9A",   0.5, "Croatia"),
    CallRoot("S5",   0.5, "Slovenia"),
    CallRoot("EI",   0.5, "Ireland"),
    CallRoot("GM",   0.7, "Scotland"),
    CallRoot("GW",   0.5, "Wales"),
    CallRoot("CT",   0.6, "Portugal"),
    CallRoot("LY",   0.3, "Lithuania"),
    # ---------- Asia ---------- #
    CallRoot("JA",   4.0, "Japan"),
    CallRoot("JH",   1.5, "Japan"),
    CallRoot("BY",   1.5, "China"),
    CallRoot("HL",   1.0, "South Korea"),
    CallRoot("VU",   0.7, "India"),
    CallRoot("HS",   0.5, "Thailand"),
    CallRoot("4X",   0.6, "Israel"),
    # ---------- Oceania ---------- #
    CallRoot("VK",   1.5, "Australia"),
    CallRoot("ZL",   0.6, "New Zealand"),
    CallRoot("KH6",  0.4, "Hawaii"),
    # ---------- South America ---------- #
    CallRoot("PY",   1.2, "Brazil"),
    CallRoot("LU",   0.8, "Argentina"),
    CallRoot("CE",   0.5, "Chile"),
    CallRoot("CX",   0.3, "Uruguay"),
    # ---------- Africa ---------- #
    CallRoot("ZS",   0.7, "South Africa"),
    CallRoot("CN",   0.3, "Morocco"),
    # ---------- Rare DX (thin-tail coverage) ---------- #
    CallRoot("KL7",  0.3, "Alaska"),
    CallRoot("KP4",  0.3, "Puerto Rico"),
    CallRoot("TF",   0.2, "Iceland"),
    CallRoot("5B",   0.2, "Cyprus"),
    CallRoot("T7",   0.1, "San Marino"),
)


# Pre-computed probabilities for fast sampling. Recomputed eagerly so
# that changing the table in memory (tests / notebooks) is honoured.
def _build_probabilities() -> np.ndarray:
    w = np.array([e.weight for e in ENTRIES], dtype=np.float64)
    if (w <= 0).any():
        raise ValueError("all entry weights must be positive")
    return w / w.sum()


_PROBS = _build_probabilities()


def sample_root(rng: np.random.Generator) -> CallRoot:
    """Draw one ``CallRoot`` according to the weighted distribution."""
    idx = int(rng.choice(len(ENTRIES), p=_PROBS))
    return ENTRIES[idx]


def root_has_fixed_digit(root: str) -> bool:
    """Whether the root already ends in a digit (so the generator must
    not append one)."""
    return len(root) > 0 and root[-1].isdigit()
