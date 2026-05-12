"""ITU/DXCC callsign-shape prior for Phase 7 beam-search rescoring.

A score for "does this string parse as an amateur-radio callsign?" In a
beam-search decoder this is added to a hypothesis's running log-prob
whenever a word boundary is reached, biasing the beam toward word
hypotheses that *look* like callsigns — without touching the acoustic
score elsewhere.

The structural test is the canonical amateur format::

    <root> [digit] <suffix> [/portable]

with::

    root    : 1 letter or 1 letter-letter or 1 letter-digit
              (digit is allowed only as the second char)
    digit   : 0-9  (omitted if the root already ends in a digit)
    suffix  : 1-4 letters
    portable: /P /M /MM /A /AM /QRP  (optional, ignored for matching)

Examples that match::

    DL5WW   F4HYY   N6ZZ   G4LIOW   HB9DNX   9A2BC   EA8DAQ   KL7AB
    DL5WW/P  HB9DNX/QRP

Examples that do not match::

    HELLO   CQ   599   ABC   K   K1   K1ABCD2  (suffix > 4)
    73     QRZ  KN

The scorer returns a positive log-prob bonus when the parse succeeds.
Hits where the root is one of the DXCC roots curated for training in
:mod:`morseformer.data.itu_prefixes` get the full bonus; structurally
valid hits with an unknown root get half. Everything else gets zero —
this is purely additive, so non-callsign words are not penalised.

This module is intentionally a self-contained helper so the rescoring
logic can be unit-tested without spinning up a beam search.
"""

from __future__ import annotations

import re

from morseformer.data.itu_prefixes import ENTRIES


# Set of known DXCC roots taken from the synthesis table. The synthesis
# table is the smallest possible subset that the training-time mix
# actually exposes the model to — biasing the prior to those same roots
# is consistent with what the acoustic head has been calibrated against.
KNOWN_ROOTS: frozenset[str] = frozenset(e.root for e in ENTRIES)


# Strict amateur-callsign structural regex.
#
# Group 1 (root) is greedy and includes the digit slot if present.
# Group 2 (suffix) is 1-4 letters. Group 3 (portable) is optional.
#
# Root cases:
#   * One letter: K, W, N, G, F, I, R   (mostly USA/UK/IT/RU/FR)
#   * Two letters: DL, DK, JA, JH, ON, ...
#   * Letter-digit: 9A, 4X, 5B, 7J, T7
#   * Letter-letter-digit: HB9, EA8, KH6, KL7, KP4
#   * Digit-letter: 3V, 9H, 4U (less common but valid)
#
# After the root, a single digit unless the root already ended in one.
# The pattern below uses a non-greedy alternation that lets the matcher
# try the longest root first.
_CALLSIGN_RE = re.compile(
    r"""^
    (                                         # 1: root + digit slot
        (?: [A-Z]{1,2} \d        )            #    DL5, G4, K1, HB9
      | (?:  \d  [A-Z]   \d      )            #    9A2, 4X1, 5B6
      | (?: [A-Z]{2}  \d{1,2}    )            #    KP4, KH6, KL7, EA8
      | (?: [A-Z]    \d          )            #    K1, F4 etc (already covered)
    )
    ( [A-Z]{1,4} )                            # 2: suffix
    (?: / ( P|M|MM|A|AM|QRP ) )?              # 3: optional portable
    $""",
    re.VERBOSE,
)


def _try_parse(word: str) -> tuple[str, str] | None:
    """Parse ``word`` as a callsign. Return (root, suffix) on success.

    The root *includes* the digit slot (e.g. ``DL5`` for ``DL5WW``).
    """
    m = _CALLSIGN_RE.match(word)
    if m is None:
        return None
    return m.group(1), m.group(2)


def is_callsign_shape(word: str) -> bool:
    """True iff ``word`` parses as the amateur-callsign structural form.

    Does not consult the DXCC table; only checks the shape.
    """
    return _CALLSIGN_RE.match(word) is not None


def _root_is_known(root_with_digit: str) -> bool:
    """Strip a trailing digit (if any) and check against KNOWN_ROOTS.

    The DXCC table mixes both *root-only* entries (``DL``, ``G``) and
    *root-with-fixed-digit* entries (``HB9``, ``KL7``, ``EA8``). The
    parsed group 1 always *includes* the digit slot. We try the
    canonical form first (the digit-bearing root, e.g. ``HB9``), then
    fall back to dropping the trailing digit (e.g. ``DL5`` → ``DL``).
    """
    if root_with_digit in KNOWN_ROOTS:
        return True
    if root_with_digit[-1].isdigit() and root_with_digit[:-1] in KNOWN_ROOTS:
        return True
    return False


def score_callsign(
    word: str,
    *,
    weight: float = 1.0,
    unknown_root_fraction: float = 0.5,
) -> float:
    """Return a non-negative log-prob bonus for ``word``.

    The bonus is::

        weight                        if word is a valid callsign shape AND
                                      its root is in the DXCC table
        weight * unknown_root_fraction
                                      if word is a valid callsign shape but
                                      the root is unknown to us
        0                             otherwise

    Args:
        word: Candidate, uppercase ASCII. Trailing portable tag is
            optional and ignored. Empty string returns 0.
        weight: Maximum bonus, i.e. the ``λ_prior`` knob the beam
            search will tune.
        unknown_root_fraction: Multiplier for the bonus when the root
            does not appear in the DXCC table. Set to 0.0 to require
            an exact root match.
    """
    if not word:
        return 0.0
    parsed = _try_parse(word)
    if parsed is None:
        return 0.0
    root, _suffix = parsed
    if _root_is_known(root):
        return weight
    return weight * unknown_root_fraction
