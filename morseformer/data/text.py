"""Synthetic text generators for the morseformer training pipeline.

Five sources are mixed to produce the training-time text distribution:

    sample_callsign       — realistic amateur callsigns drawn from the
                            ITU prefix table, with occasional portable
                            suffixes (/P, /M, /MM, /QRP, /A).
    sample_qcode_abbrev   — Q-codes (QRM, QSB, QTH, …) and the most
                            common CW operating abbreviations.
    sample_qso_line       — short, realistic QSO fragments filled from
                            parametric templates. Callsigns are embedded
                            directly in the fragment (as they would be
                            in real CW traffic), in addition to being
                            sampled standalone.
    sample_numeric        — serial numbers, RST reports, frequencies,
                            dates, times, and other digit-heavy material.
    sample_english_words  — 2–6 short common English words, to expose
                            the acoustic model to natural bigrams beyond
                            the ham-specific lexicon.

Every generator takes a single ``np.random.Generator`` so seeding the
RNG fully determines the output. The top-level ``sample_text`` picks a
category according to ``DEFAULT_MIX`` and delegates.

All outputs use only characters that the 49-token tokenizer accepts
(``A-Z 0-9 space . , ? ! / = + - É À '``). Out-of-vocab characters
are silently dropped by both the tokenizer and the synthesiser, but
we strive to produce clean in-vocab text anyway for clarity and
reproducibility. The Phase 3.4 additions (É / À / apostrophe) are
preserved by ``_normalize_prose`` so French prose retains its
diacritics end-to-end.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from morseformer.data.itu_prefixes import root_has_fixed_digit, sample_root


# --------------------------------------------------------------------- #
# Callsigns
# --------------------------------------------------------------------- #

# Suffix length distribution (1-by-1 calls are rare special events,
# 1-by-2 and 1-by-3 are the common US / EU formats).
_SUFFIX_LEN_PROBS = np.array([0.05, 0.40, 0.55])

# Probability of a portable tag after the base call, and the tag mix.
_PORTABLE_PROB = 0.08
_PORTABLE_TAGS = ("P", "M", "MM", "QRP", "A")
_PORTABLE_PROBS = np.array([0.45, 0.35, 0.05, 0.10, 0.05])


def _random_letters(rng: np.random.Generator, n: int) -> str:
    return "".join(chr(ord("A") + int(i)) for i in rng.integers(0, 26, size=n))


def sample_callsign(rng: np.random.Generator) -> str:
    """Sample one callsign as ``root + [digit] + suffix [+ /portable]``.

    - ``root`` is drawn from the ITU prefix table with activity weights.
    - ``digit`` is skipped if the root already ends in a digit.
    - ``suffix`` is 1–3 random letters (distribution ≈ real 1×N calls).
    - 8 % of calls receive a portable tag.
    """
    entry = sample_root(rng)
    root = entry.root
    digit = "" if root_has_fixed_digit(root) else str(int(rng.integers(0, 10)))
    suffix_len = int(rng.choice(3, p=_SUFFIX_LEN_PROBS)) + 1
    suffix = _random_letters(rng, suffix_len)
    base = f"{root}{digit}{suffix}"
    if rng.random() < _PORTABLE_PROB:
        tag = _PORTABLE_TAGS[int(rng.choice(len(_PORTABLE_TAGS), p=_PORTABLE_PROBS))]
        return f"{base}/{tag}"
    return base


# --------------------------------------------------------------------- #
# Q-codes & CW abbreviations
# --------------------------------------------------------------------- #

# Standard ITU Q-codes that ham operators actually send, plus the most
# common CW operating abbreviations. Drawn uniformly at random.
_QCODES_AND_ABBREVS: tuple[str, ...] = (
    # ---- Q-codes ----
    "QRA", "QRG", "QRH", "QRK", "QRL", "QRM", "QRN", "QRO", "QRP", "QRQ",
    "QRS", "QRT", "QRU", "QRV", "QRX", "QRZ", "QSB", "QSK", "QSL", "QSO",
    "QSY", "QTH", "QTR",
    # ---- Courtesy / closings ----
    "TU", "FB", "73", "88", "SK", "AR", "BK", "KN", "CL",
    # ---- Formal prefixes ----
    "CQ", "DE", "K", "ES",
    # ---- People / addressing ----
    "OM", "YL", "XYL", "HR", "UR",
    # ---- Signal reports ----
    "RST", "599", "579", "569", "559", "449", "339",
    # ---- Conditions / gear ----
    "WX", "ANT", "RIG", "PWR", "DX", "QRO", "QRP",
    # ---- Affirmatives / negatives ----
    "R", "OK", "NO", "WKD", "CFM", "NIL", "AGN",
    # ---- Greetings ----
    "GE", "GA", "GM", "GN", "GL", "CUL", "HI",
    # ---- Filler / connectives ----
    "PSE", "TNX", "TKS", "FER", "NW", "NR", "ABT", "WID", "BURO", "DIR",
    "CPY", "HW",
)


def sample_qcode_abbrev(rng: np.random.Generator) -> str:
    return str(_QCODES_AND_ABBREVS[int(rng.integers(0, len(_QCODES_AND_ABBREVS)))])


# --------------------------------------------------------------------- #
# QSO templates (slot-filling grammar)
# --------------------------------------------------------------------- #

# Short QSO fragments — the kind of thing you'd hear on a single over.
# Full multi-over QSOs are reconstructed downstream; here we just need
# realistic on-air phrases that fit inside a few seconds of audio.
_QSO_TEMPLATES: tuple[str, ...] = (
    "CQ CQ DE {cs} {cs} K",
    "CQ CQ CQ DE {cs} K",
    "CQ DX DE {cs} K",
    "CQ TEST DE {cs} TEST",
    "QRZ DE {cs} K",
    "QRZ QRZ DE {cs}",
    "{cs} DE {cs} K",
    "{cs} DE {cs} KN",
    "{cs} DE {cs} GE OM",
    "{cs} DE {cs} GM OM UR {rst}",
    "{cs} DE {cs} UR RST {rst} {rst}",
    "{cs} UR {rst} IN {qth}",
    "NAME {name} QTH {qth}",
    "MY NAME {name} ES QTH {qth}",
    "{cs} TNX FER CALL UR {rst}",
    "{cs} TU FB QSO 73",
    "TU {cs} 73 SK",
    "TU 73 ES GL DE {cs} SK",
    "{cs} DE {cs} PSE QSL VIA BURO",
    "{cs} QSL VIA {cs} TNX",
    "RIG IC {rig} PWR {pwr} W",
    "ANT {ant} HR",
    "WX {wx} HR TEMP {temp}",
    "{cs} DE {cs} = {serial} = {zone}",
    "{serial} {serial} {zone}",
    "{cs} 599 {serial}",
    "UR {rst} HW?",
    "HW CPY? BK",
    "AGN AGN PSE",
    "QRZ DE {cs} QRZ",
    "CQ CONTEST DE {cs}",
    "{cs} {rst} {zone}",
    "DE {cs} UR {rst} OP {name}",
    "FB OM TU 73",
    "{cs} PSE K",
    # End-of-message / end-of-QSO prosigns (+ = AR, SK = VA).
    "{cs} DE {cs} + TU",
    "{cs} DE {cs} 73 +",
    "TU QSO DE {cs} SK +",
    # Comma- and bang-flavoured fragments — rare in CW but they exist
    # and the tokenizer supports them, so expose them occasionally.
    "OP {name}, RIG IC {rig}",
    "NAME {name}, AGE {zone}",
    "FB OM! TU 73",
    "GL! ES 73",
    "TNX OM! CUL",
)


# Slot fillers. Kept short and CW-typical so that full lines remain
# tractable for a 6 s audio window at moderate WPM.
_NAMES: tuple[str, ...] = (
    "JOHN", "PAUL", "MIKE", "BOB", "JIM", "BILL", "FRANK", "TOM", "DAVE",
    "TED", "RON", "JACK", "JEFF", "JOE", "JERRY", "MARK", "MATT", "NICK",
    "PHIL", "RICK", "STEVE", "TIM", "TONY", "WILL", "ED", "AL", "KEN",
    "KEVIN", "LARRY", "DAN", "PETE", "ANDY", "ROB", "ERIC", "JAMES",
    "CHARLES", "SERGE", "PIERRE", "JEAN", "ALAIN", "MARC", "LUC", "YVES",
    "ANTONIO", "LUIGI", "CARLO", "MARIO", "FABIO", "PAOLO",
    "HANS", "DIETER", "WOLF", "KLAUS", "OLAF", "PETER", "STEFAN",
    "YUKI", "TARO", "KEN", "HIRO", "AKIRA",
    "IVAN", "OLEG", "SERGEY", "MIKHAIL", "VASSILI",
    "MARIA", "ANNA", "SARA", "LINDA", "JANE", "LAURA",
)

_QTHS: tuple[str, ...] = (
    "NICE", "PARIS", "LYON", "NANTES", "MARSEILLE",
    "ROME", "MILAN", "TORINO", "NAPLES",
    "BERLIN", "MUNICH", "HAMBURG", "FRANKFURT",
    "LONDON", "BIRMINGHAM", "MANCHESTER", "LEEDS",
    "MADRID", "BARCELONA", "SEVILLA",
    "VIENNA", "LINZ", "GRAZ",
    "BERN", "ZURICH", "GENEVA",
    "AMSTERDAM", "ROTTERDAM", "UTRECHT",
    "BRUSSELS", "ANTWERP", "LIEGE",
    "OSLO", "BERGEN", "TRONDHEIM",
    "STOCKHOLM", "MALMO", "GOTHENBURG",
    "HELSINKI", "TAMPERE", "TURKU",
    "COPENHAGEN", "AARHUS", "ODENSE",
    "WARSAW", "KRAKOW", "GDANSK",
    "PRAGUE", "BRNO", "OSTRAVA",
    "BUDAPEST", "DEBRECEN",
    "ATHENS", "THESSALONIKI",
    "SOFIA", "BUCHAREST",
    "MOSCOW", "SPB", "KAZAN",
    "KIEV", "LVIV", "ODESSA",
    "TOKYO", "OSAKA", "KYOTO", "SAPPORO",
    "BEIJING", "SHANGHAI", "HONGKONG",
    "SEOUL", "BUSAN",
    "DELHI", "MUMBAI",
    "BANGKOK",
    "SYDNEY", "MELBOURNE", "PERTH", "BRISBANE",
    "AUCKLAND", "WELLINGTON",
    "NYC", "BOSTON", "CHICAGO", "DENVER", "SEATTLE", "DALLAS", "ATLANTA",
    "LA", "SF", "PORTLAND", "PHOENIX", "MIAMI", "HOUSTON",
    "TORONTO", "MONTREAL", "VANCOUVER", "OTTAWA",
    "SAO PAULO", "RIO", "BRASILIA",
    "BUENOS AIRES", "SANTIAGO",
)

_RSTS: tuple[str, ...] = (
    "599", "579", "569", "559", "449", "339", "229", "119",
    "589", "578", "568", "478", "468",
)

_RIG_MODELS: tuple[str, ...] = (
    "7300", "7610", "705", "718", "9700", "756", "910", "991",
    "891", "857", "817", "847", "590", "5000", "7100", "3000",
    "2000", "1000",
)

_POWERS: tuple[str, ...] = (
    "5", "10", "25", "50", "75", "100", "200", "400", "500", "1000",
)

_ANTENNAS: tuple[str, ...] = (
    "DIPOLE", "VERTICAL", "YAGI", "LOOP", "BEAM", "GP",
    "HEXBEAM", "WINDOM", "MAGLOOP", "ENDFED", "LONGWIRE",
    "G5RV", "OCF", "FAN DIPOLE", "INV V",
)

_WX: tuple[str, ...] = (
    "SUNNY", "CLOUDY", "RAINY", "WINDY", "CLEAR", "FOGGY",
    "SNOWY", "COLD", "WARM", "MILD", "HOT", "FAIR", "COOL",
)


def _fill_slot(slot: str, rng: np.random.Generator) -> str:
    if slot == "cs":
        return sample_callsign(rng)
    if slot == "rst":
        return _RSTS[int(rng.integers(0, len(_RSTS)))]
    if slot == "name":
        return _NAMES[int(rng.integers(0, len(_NAMES)))]
    if slot == "qth":
        return _QTHS[int(rng.integers(0, len(_QTHS)))]
    if slot == "rig":
        return _RIG_MODELS[int(rng.integers(0, len(_RIG_MODELS)))]
    if slot == "pwr":
        return _POWERS[int(rng.integers(0, len(_POWERS)))]
    if slot == "ant":
        return _ANTENNAS[int(rng.integers(0, len(_ANTENNAS)))]
    if slot == "wx":
        return _WX[int(rng.integers(0, len(_WX)))]
    if slot == "temp":
        # -10 to 40 °C, CW-typical.
        t = int(rng.integers(-10, 41))
        return f"{t}" if t >= 0 else f"-{abs(t)}"
    if slot == "serial":
        return f"{int(rng.integers(1, 1000)):03d}"
    if slot == "zone":
        return str(int(rng.integers(1, 41)))
    raise KeyError(f"unknown slot: {slot!r}")


def _render_template(tpl: str, rng: np.random.Generator) -> str:
    """Manual slot substitution — ``str.format`` does not allow drawing
    a fresh random value for each occurrence of the same slot."""
    out: list[str] = []
    i = 0
    n = len(tpl)
    while i < n:
        c = tpl[i]
        if c == "{":
            j = tpl.index("}", i + 1)
            out.append(_fill_slot(tpl[i + 1 : j], rng))
            i = j + 1
        else:
            out.append(c)
            i += 1
    return "".join(out)


def sample_qso_line(rng: np.random.Generator) -> str:
    tpl = _QSO_TEMPLATES[int(rng.integers(0, len(_QSO_TEMPLATES)))]
    return _render_template(tpl, rng)


# --------------------------------------------------------------------- #
# Numeric / punctuation material
# --------------------------------------------------------------------- #


def _numeric_serial(rng: np.random.Generator) -> str:
    return f"{int(rng.integers(1, 10_000))}"


def _numeric_rst_pair(rng: np.random.Generator) -> str:
    a = _RSTS[int(rng.integers(0, len(_RSTS)))]
    b = _RSTS[int(rng.integers(0, len(_RSTS)))]
    return f"{a} {b}"


_CW_BANDS: tuple[tuple[int, int], ...] = (
    (1810, 1838), (3500, 3570), (7000, 7040), (10100, 10140),
    (14000, 14070), (18068, 18100), (21000, 21070),
    (24890, 24915), (28000, 28070),
)


def _numeric_frequency(rng: np.random.Generator) -> str:
    lo, hi = _CW_BANDS[int(rng.integers(0, len(_CW_BANDS)))]
    khz = int(rng.integers(lo, hi + 1))
    return f"{khz} KHZ"


def _numeric_frequency_mhz(rng: np.random.Generator) -> str:
    # Decimal-MHZ notation, e.g. 14.040 MHZ — less frequent than kHz in
    # CW logs but common enough that the model should see the decimal.
    lo, hi = _CW_BANDS[int(rng.integers(0, len(_CW_BANDS)))]
    khz = int(rng.integers(lo, hi + 1))
    return f"{khz // 1000}.{khz % 1000:03d} MHZ"


def _numeric_time_z(rng: np.random.Generator) -> str:
    hh = int(rng.integers(0, 24))
    mm = int(rng.integers(0, 60))
    return f"{hh:02d}{mm:02d}Z"


def _numeric_date(rng: np.random.Generator) -> str:
    dd = int(rng.integers(1, 29))
    mm = int(rng.integers(1, 13))
    yy = int(rng.integers(20, 35))
    return f"{dd:02d}/{mm:02d}/{yy:02d}"


def _numeric_qso_nr(rng: np.random.Generator) -> str:
    return f"NR {int(rng.integers(1, 500)):03d}"


def _numeric_dash_serial(rng: np.random.Generator) -> str:
    a = int(rng.integers(1, 1000))
    b = int(rng.integers(1, 1000))
    return f"{a:03d}-{b:03d}"


_NUMERIC_SAMPLERS = (
    _numeric_serial,
    _numeric_rst_pair,
    _numeric_frequency,
    _numeric_frequency_mhz,
    _numeric_time_z,
    _numeric_date,
    _numeric_qso_nr,
    _numeric_dash_serial,
)


def sample_numeric(rng: np.random.Generator) -> str:
    sampler = _NUMERIC_SAMPLERS[int(rng.integers(0, len(_NUMERIC_SAMPLERS)))]
    return sampler(rng)


# --------------------------------------------------------------------- #
# English word sequences
# --------------------------------------------------------------------- #

# ~300 common English words, 2–8 letters, A-Z only. Frequency-ordered at
# the head, then ham-flavoured additions and deliberate rare-letter
# coverage (Q, X, Z, J) at the tail. Not a contest corpus — just enough
# to expose the acoustic model to natural letter bigrams outside the
# ham-specific lexicon.
_ENGLISH_WORDS: tuple[str, ...] = tuple(
    """
    THE AND FOR ARE BUT NOT YOU ALL CAN HAS HAD HER WAS ONE OUR OUT
    DAY GET HIM HIS HOW MAN NEW NOW OLD SEE TWO WAY WHO BOY DID ITS
    LET PUT SAY SHE TOO USE THAT WITH HAVE THIS WILL YOUR FROM THEY
    KNOW WANT BEEN GOOD MUCH SOME TIME VERY WHEN COME HERE JUST LIKE
    LONG MAKE MANY OVER SUCH TAKE THAN THEM WELL WERE YEAR WORD DOWN
    EACH HELP HIGH LAST MADE MOST NEED NEXT ONLY PART PLAY SAID STAY
    STOP TELL WORK KEEP LEFT MORE NAME OPEN POOR REST ROAD ROOM SEEM
    SHOW SIDE SLOW SOON STEP SURE TOLD TURN USED WAIT WALK WARM WENT
    WOULD ABOUT OTHER WHICH THEIR THERE THESE FIRST AFTER THOSE STILL
    NEVER WHERE EVERY THINK EIGHT THREE UNDER SHOULD COULD RIGHT UNTIL
    YOUNG WHILE BEING GOING AGAIN ASKED GREAT FOUND GIVEN HOUSE KNOW
    MIGHT MONEY NIGHT PLACE POINT ROUND SHALL SINCE SMALL SOUND STORY
    TABLE TODAY WATCH WATER WORLD WORTH WRITE YEARS BEFORE LITTLE
    AROUND CHANGE COMMON COURSE DURING ENOUGH FAMILY FOLLOW FRIEND
    LISTEN MATTER MOMENT MOTHER NUMBER PERSON PLEASE PRETTY REALLY
    SECOND SIMPLE SISTER SUMMER WINTER ANOTHER AGAINST ALREADY BETWEEN
    COUNTRY EVENING GENERAL HUNDRED LONGER MORNING PRESENT SEVERAL
    STUDENT WITHOUT WRITING
    RADIO SIGNAL STATION BAND NOISE TONE KEY CODE MORSE VOICE BEACON
    RELAY FILTER POWER TOWER DIPOLE LOOP BEAM TEST TALK FRIEND CONTACT
    RECEIVE SEND COPY CLEAR STRONG WEAK ATOM LINE CURVE WAVE RANGE
    FIELD NORTH SOUTH EAST WEST SPRING AUTUMN
    FOX TAX MIX FIX SIX BOX MAX WAX
    QUICK QUIET QUITE QUIZ QUEEN
    ZERO ZONE SIZE JAZZ PRIZE MAZE HAZE
    JAR JOY JUDGE JUMP JULY JOIN JOKE
    HELLO HI
    """.split()
)


def sample_english_words(rng: np.random.Generator) -> str:
    n = int(rng.integers(2, 7))  # 2..6 words
    idx = rng.integers(0, len(_ENGLISH_WORDS), size=n)
    return " ".join(_ENGLISH_WORDS[int(i)] for i in idx)


# --------------------------------------------------------------------- #
# Random character clumps (no linguistic prior)
# --------------------------------------------------------------------- #

# Punctuation tokens / prosigns sampled into the random stream. These
# are all in the 49-token vocabulary (space and SK are space + S + K).
_RANDOM_PUNCT: tuple[str, ...] = (",", "/", "?", "=", "-", "+", " SK")


def _random_digits(rng: np.random.Generator, n: int) -> str:
    return "".join(str(int(d)) for d in rng.integers(0, 10, size=n))


def sample_random_chars(rng: np.random.Generator) -> str:
    """Random short character clumps to break linguistic priors.

    The acoustic model trained only on English / Q-codes / callsigns
    learns linguistic priors that bias decoding toward plausible-looking
    text on noise (the "letter-soup hallucination" failure mode observed
    in the 2026-04-25 London↔French QSO). This sampler emits sequences
    with no linguistic structure so the model is forced to rely on the
    pure acoustic→character mapping.

    Four modes (uniformly weighted):

        * letters: 3-6 random A-Z
        * digits:  2-5 random 0-9
        * mixed:   4-8 chars, 70 % letters / 30 % digits
        * with_punct: a base sequence + one inserted punctuation /
          prosign chosen from ``_RANDOM_PUNCT``

    Sometimes a sample is multi-group (e.g. ``ABCDE FGHIJ``) to mimic
    cipher-style five-letter groups that occur in real CW practice
    transmissions.
    """
    # Multi-group with 30 % probability — exposes the model to multiple
    # short clumps separated by inter-word gaps in one utterance.
    n_groups = 1 if rng.random() < 0.7 else int(rng.integers(2, 4))
    groups: list[str] = []
    for _ in range(n_groups):
        mode = ("letters", "digits", "mixed", "with_punct")[
            int(rng.integers(0, 4))
        ]
        if mode == "letters":
            length = int(rng.integers(3, 7))           # 3..6
            chunk = _random_letters(rng, length)
        elif mode == "digits":
            length = int(rng.integers(2, 6))           # 2..5
            chunk = _random_digits(rng, length)
        elif mode == "mixed":
            length = int(rng.integers(4, 9))           # 4..8
            chars: list[str] = []
            for _ in range(length):
                if rng.random() < 0.30:
                    chars.append(str(int(rng.integers(0, 10))))
                else:
                    chars.append(chr(ord("A") + int(rng.integers(0, 26))))
            chunk = "".join(chars)
        else:  # with_punct
            base_len = int(rng.integers(3, 6))
            base_chars: list[str] = []
            for _ in range(base_len):
                if rng.random() < 0.30:
                    base_chars.append(str(int(rng.integers(0, 10))))
                else:
                    base_chars.append(chr(ord("A") + int(rng.integers(0, 26))))
            punct = _RANDOM_PUNCT[int(rng.integers(0, len(_RANDOM_PUNCT)))]
            # Append, prepend, or insert at a random position.
            place = int(rng.integers(0, 3))
            if place == 0:
                chunk = "".join(base_chars) + punct
            elif place == 1:
                chunk = punct + "".join(base_chars) if not punct.startswith(" ") else "".join(base_chars) + punct
            else:
                pos = int(rng.integers(1, max(2, len(base_chars))))
                chunk = "".join(base_chars[:pos]) + punct + "".join(base_chars[pos:])
        groups.append(chunk)
    return " ".join(groups)


# --------------------------------------------------------------------- #
# Phase 4.0 — pure-acoustic random chars
# --------------------------------------------------------------------- #
#
# Phase 4.0 retraining drops all prose / Q-codes / callsigns from the
# acoustic curriculum and relies on this sampler exclusively. Rationale:
# the post-3.5 series (3.6/3.7/3.8) showed that any prose redistribution
# trades one sub-domain for another (catastrophic forgetting under fixed
# capacity). The Phase 4.0 architectural decision splits responsibilities
# — acoustic = char-level only, language structure = LM at decode time
# (Phase 5.0 fusion). The sampler must therefore expose the model to a
# distribution with **no linguistic prior whatsoever**, while still
# preserving the 49-token vocabulary acquired in Phase 3.5 (including
# É / À / apostrophe at 92–100 % per-token precision).
#
# Four modes (planned mix 50/25/15/10):
#
#   * mode_a (50 %): 5–25 chars, mix letters + digits, no punct
#   * mode_b (25 %): 5–15 chars + 1–2 punctuations dispersed; ~10 % of
#                    mode_b samples force-include one accent token
#                    (É / À / ') so these tokens see ~2.5 % of training
#                    samples — well above their natural-ASCII rate of
#                    0 % and enough to defend against catastrophic
#                    forgetting under 30k bootstrap steps.
#   * mode_c (15 %): 3–8 chars all-digit (RST / dates / freqs out of
#                    context). Concentrates training on the digit
#                    tokens, which are sparse in the 3.x prose corpora.
#   * mode_d (10 %): cipher-style 5-letter groups (5+5 or 5+5+5).
#                    Mimics the canonical contest / training-tape
#                    formats and exercises inter-group spaces.

# Phase 4.0 punctuation pool. Excludes the multi-char " SK" prosign that
# the 3.x sampler used: in a no-prior curriculum we want each token to
# fire on its own, not glued into a prosign.
_RANDOM_PUNCT_PHASE4: tuple[str, ...] = (".", ",", "?", "!", "/", "=", "+", "-")
_ACCENT_TOKENS: tuple[str, ...] = ("É", "À", "'")
_PHASE_4_MODE_PROBS: tuple[float, ...] = (0.50, 0.25, 0.15, 0.10)
# Probability that a mode_b sample is force-rewritten to contain at
# least one accent token. With mode_b at 25 % of the mix, this gives
# accents ≈ 2.5 % of training characters at the sample level.
_PHASE_4_ACCENT_PROB: float = 0.10


def _phase4_letters_or_digits(rng: np.random.Generator, length: int,
                               digit_prob: float = 0.30) -> list[str]:
    out: list[str] = []
    for _ in range(length):
        if rng.random() < digit_prob:
            out.append(str(int(rng.integers(0, 10))))
        else:
            out.append(chr(ord("A") + int(rng.integers(0, 26))))
    return out


def sample_random_chars_phase4(
    rng: np.random.Generator,
    max_chars: int | None = None,
) -> str:
    """Phase 4.0 random-char sampler — 4 modes (a/b/c/d) with accent boost.

    See module-level comment above for motivation and mode definitions.
    Returned text always passes ``encode()`` cleanly and never contains
    the SK prosign (single tokens only).

    ``max_chars`` (when provided) clips every per-mode length range to
    ``[1, max_chars]`` so the sampler can be driven from a wpm-derived
    budget at synthesis time. mode_d (5+5 cipher groups) requires at
    least 11 characters; if the cap is below that, the call re-rolls
    to one of the other three modes. With no cap (the default), the
    sampler returns its full distribution — useful for unit tests and
    interactive inspection.
    """
    if max_chars is not None and max_chars < 1:
        # Pathological budget — emit a single letter so the caller still
        # gets a non-empty token sequence.
        return chr(ord("A") + int(rng.integers(0, 26)))

    # Re-roll the mode if the chosen one would require more chars than
    # the cap allows. Bound the loop to avoid pathological infinite
    # re-roll on tiny caps (only mode_d is gated; mode_a/b/c always
    # have valid sub-distributions for max_chars >= 1).
    for _ in range(8):
        mode_idx = int(rng.choice(4, p=_PHASE_4_MODE_PROBS))
        if mode_idx == 3 and max_chars is not None and max_chars < 11:
            continue
        break
    else:
        mode_idx = 0  # safe fallback

    if mode_idx == 0:  # mode_a
        hi = 26 if max_chars is None else min(26, max_chars + 1)
        lo = min(5, hi - 1) if hi > 1 else 1
        length = int(rng.integers(lo, hi))
        return "".join(_phase4_letters_or_digits(rng, length))

    if mode_idx == 1:  # mode_b
        # Reserve 1 slot for punct (and another for accent if it fires)
        # so the final string still respects ``max_chars``.
        accent_will_fire = rng.random() < _PHASE_4_ACCENT_PROB
        n_punct = int(rng.integers(1, 3))                  # 1 or 2
        reserve = n_punct + (1 if accent_will_fire else 0)
        if max_chars is None:
            base_hi = 16                                    # 5..15
            base_lo = 5
        else:
            base_hi = max(2, max_chars - reserve + 1)       # exclusive
            base_lo = min(3, base_hi - 1)
        base_len = int(rng.integers(base_lo, base_hi))
        chars = _phase4_letters_or_digits(rng, base_len)
        for _ in range(n_punct):
            punct = _RANDOM_PUNCT_PHASE4[
                int(rng.integers(0, len(_RANDOM_PUNCT_PHASE4)))
            ]
            pos = int(rng.integers(0, len(chars) + 1))
            chars.insert(pos, punct)
        if accent_will_fire:
            accent = _ACCENT_TOKENS[
                int(rng.integers(0, len(_ACCENT_TOKENS)))
            ]
            pos = int(rng.integers(0, len(chars) + 1))
            chars.insert(pos, accent)
        return "".join(chars)

    if mode_idx == 2:  # mode_c — all digits
        hi = 9 if max_chars is None else min(9, max_chars + 1)
        lo = min(3, hi - 1) if hi > 1 else 1
        length = int(rng.integers(lo, hi))
        return _random_digits(rng, length)

    # mode_d — 5-char groups, 2 or 3 of them. Already gated above so
    # max_chars >= 11 if we reach here.
    if max_chars is None or max_chars >= 17:
        n_groups = int(rng.integers(2, 4))                  # 2 or 3
    else:
        n_groups = 2                                        # exactly 11 chars
    groups = ["".join(_phase4_letters_or_digits(rng, 5))
              for _ in range(n_groups)]
    return " ".join(groups)


# --------------------------------------------------------------------- #
# Multilingual prose (Phase 3.3)
# --------------------------------------------------------------------- #

# German umlauts have no NFKD decomposition that produces ASCII letters,
# so we must transliterate them explicitly *before* the NFKD pass.
_DE_TRANSLIT = str.maketrans({
    "ä": "AE", "ö": "OE", "ü": "UE", "ß": "SS",
    "Ä": "AE", "Ö": "OE", "Ü": "UE",
})

# Apostrophe-class characters that all collapse to the canonical ASCII
# apostrophe (kept in the Phase 3.4 vocabulary). Curly quotes, prime,
# backtick, acute as quote — every variant the corpus produces.
_APOSTROPHE_CHARS = "'’‘ʼ`´"

# Wider quote-mark class that becomes whitespace (no semantic CW
# equivalent — opening/closing quotes don't survive into Morse).
_QUOTE_CHARS = "\"“”«»‹›„‟"

# Allowed output characters (matches the 49-token tokenizer:
# A-Z 0-9 punct + Phase 3.4 É À '). Lowercase é/à are pre-uppercased
# in ``_normalize_prose`` before this set is consulted.
_PROSE_ALLOWED = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,?!/=+-ÉÀ'")

# Cap how much text we keep per language to avoid loading 8 MB into
# memory when only a fraction is sampled per epoch. 500 KB per language
# is more than enough variety for prose-mix at 10–15 %.
_PROSE_CAP_BYTES = 500_000

_PROSE_HEADER_RE = re.compile(r"^=== LANG=([A-Za-z]+) ID=\d+", re.MULTILINE)

_PROSE_PATH = Path(__file__).resolve().parents[2] / "data" / "corpus" / "prose.txt"

_PROSE_CACHE: dict[str, str] | None = None


def _normalize_prose(text: str, lang: str) -> str:
    """Map a unicode prose chunk to the 49-token vocabulary.

    Phase 3.4 update: É / À / apostrophe are preserved end-to-end so
    French prose retains the diacritics that the CW tokenizer can now
    represent. Other accented characters (è / ê / ç / ñ / ü …) still
    fall back to their ASCII base letter via NFKD.

    Steps:
        1. DE umlauts → ASCII digraphs (no NFKD path produces letters).
        2. Apostrophe-class glyphs → ASCII ``'`` (kept).
        3. Wider quote-mark class → space.
        4. Em/en dashes → ``-``.
        5. Uppercase the whole string. ``é`` → ``É`` and ``à`` → ``À``;
           both are in ``_PROSE_ALLOWED`` so they pass through unchanged.
        6. For any character still outside ``_PROSE_ALLOWED``, run a
           per-character NFKD pass and strip combining marks. This keeps
           the pre-3.4 behaviour for unsupported diacritics
           (``ê`` → ``E``, ``ç`` → ``C``, ``ñ`` → ``N``).
        7. Collapse whitespace.
    """
    if lang == "de":
        text = text.translate(_DE_TRANSLIT)
    for q in _APOSTROPHE_CHARS:
        text = text.replace(q, "'")
    for q in _QUOTE_CHARS:
        text = text.replace(q, " ")
    for d in ("—", "–", "‒", "―"):
        text = text.replace(d, "-")
    text = text.upper()
    out: list[str] = []
    for c in text:
        if c in _PROSE_ALLOWED:
            out.append(c)
            continue
        if c.isspace():
            out.append(" ")
            continue
        # Per-char NFKD fallback for diacritics we don't tokenise.
        nf = unicodedata.normalize("NFKD", c)
        nf = "".join(x for x in nf if not unicodedata.combining(x))
        for nc in nf:
            if nc in _PROSE_ALLOWED:
                out.append(nc)
            elif nc.isspace():
                out.append(" ")
            # else: drop silently
    return re.sub(r"\s+", " ", "".join(out)).strip()


def _load_prose() -> dict[str, str]:
    """Lazy-load and normalize ``data/corpus/prose.txt``.

    Returns ``{lang: normalized_concatenated_text}``. Empty dict if the
    file is missing (CI / fresh checkout without data) — callers should
    fall back gracefully.
    """
    global _PROSE_CACHE
    if _PROSE_CACHE is not None:
        return _PROSE_CACHE
    if not _PROSE_PATH.exists():
        _PROSE_CACHE = {}
        return _PROSE_CACHE
    raw = _PROSE_PATH.read_text(encoding="utf-8")
    headers = list(_PROSE_HEADER_RE.finditer(raw))
    by_lang: dict[str, list[str]] = {}
    for i, m in enumerate(headers):
        lang = m.group(1).lower()
        body_start = raw.find("\n", m.end()) + 1
        body_end = headers[i + 1].start() if i + 1 < len(headers) else len(raw)
        body = raw[body_start:body_end]
        by_lang.setdefault(lang, []).append(body)
    out: dict[str, str] = {}
    for lang, parts in by_lang.items():
        normed = _normalize_prose("\n".join(parts), lang)
        if len(normed) > _PROSE_CAP_BYTES:
            normed = normed[:_PROSE_CAP_BYTES]
        if len(normed) >= 200:
            out[lang] = normed
    _PROSE_CACHE = out
    return out


def _snap_to_word_boundary(text: str, start: int, end: int, slack: int = 12) -> tuple[int, int]:
    """Adjust ``[start, end)`` to nearest word boundaries within ``slack`` chars."""
    n = len(text)
    if 0 < start < n and text[start - 1] != " ":
        nxt = text.find(" ", start, min(start + slack, n))
        if nxt != -1:
            start = nxt + 1
    if 0 < end < n and text[end] != " ":
        prv = text.rfind(" ", max(end - slack, start), end)
        if prv != -1:
            end = prv
    return start, end


def _sample_prose_from_lang(
    rng: np.random.Generator,
    lang: str,
    min_chars: int,
    max_chars: int,
) -> str:
    prose = _load_prose()
    text = prose.get(lang, "")
    if not text:
        return sample_english_words(rng)
    n = len(text)
    target = int(rng.integers(min_chars, max_chars + 1))
    if n <= target:
        return text.strip()
    start = int(rng.integers(0, n - target + 1))
    end = start + target
    start, end = _snap_to_word_boundary(text, start, end)
    fragment = text[start:end].strip()
    return fragment if fragment else text[start : start + target].strip()


def sample_prose(
    rng: np.random.Generator,
    min_chars: int = 5,
    max_chars: int = 22,
) -> str:
    """Sample a prose fragment uniformly across available languages.

    Default length window matches the CW-duration budget at the bottom
    of the WPM range (~9 chars at 16 WPM in 6 s); the dataset's
    ``_sample_fitting_text`` retry loop will discard any over-budget
    fragments, so keeping the cap conservative avoids wasting samples.

    Falls back to ``sample_english_words`` if the corpus file is missing
    (e.g. CI without ``data/corpus/prose.txt``) — never raises.
    """
    prose = _load_prose()
    if not prose:
        return sample_english_words(rng)
    langs = sorted(prose.keys())
    lang = langs[int(rng.integers(0, len(langs)))]
    return _sample_prose_from_lang(rng, lang, min_chars, max_chars)


def sample_prose_fr(
    rng: np.random.Generator,
    min_chars: int = 5,
    max_chars: int = 22,
) -> str:
    """Sample a French-only prose fragment.

    Phase 3.4 introduces the É / À / apostrophe tokens. The natural
    density of these characters in French prose is ~3 % per character,
    so dedicating a slice of the text mix exclusively to FR gives the
    model enough gradient on the new vocab rows during a short
    fine-tune. Falls back to multilingual ``sample_prose`` (which itself
    falls back to English words) if no FR text is loaded — keeps tests
    that ship without the corpus from breaking.
    """
    return _sample_prose_from_lang(rng, "fr", min_chars, max_chars)


# --------------------------------------------------------------------- #
# FAV22 corpus + French adversarial sampler (Phase 3.6)
# --------------------------------------------------------------------- #

_FAV22_PATH = (
    Path(__file__).resolve().parents[2]
    / "data" / "corpus" / "fav22_blocks.jsonl"
)

_FAV22_CACHE: str | None = None

# Patterns where the Phase 3.5 model emits false positives:
#   * ``W + vowel`` (in-word or across an inter-word gap) — `WA` morse is
#     ``.--.-`` which is exactly À when run together.
#   * ``QU + vowel`` — short inter-element gap on the U after the Q
#     produces patterns the model maps to É.
# Pre-compiled at import; matches both inside words and around word
# boundaries (e.g. ``WAS A`` or ``L'EAU`` for QU/U+vowel).
_ADVERSARIAL_FR_PATTERNS = (
    re.compile(r"W[ \-]?[AEIOUYÀÉ]"),
    re.compile(r"QU[AEIOUYÀÉ]"),
)


def _load_fav22_clair() -> str:
    """Lazy-load the clair FAV22 blocks as a single normalised stream.

    Only ``mode=clair`` blocks are kept (codé blocks are random 5-letter
    groups already covered by ``sample_random_chars``). Blocks are
    joined with double spaces so position-based extraction never
    straddles two unrelated lessons.
    """
    global _FAV22_CACHE
    if _FAV22_CACHE is not None:
        return _FAV22_CACHE
    if not _FAV22_PATH.exists():
        _FAV22_CACHE = ""
        return _FAV22_CACHE
    chunks: list[str] = []
    with _FAV22_PATH.open(encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("mode") != "clair":
                continue
            text = rec.get("normalized_text", "").strip()
            if text:
                chunks.append(text)
    _FAV22_CACHE = "  ".join(chunks)
    return _FAV22_CACHE


def _adversarial_corpus_text() -> str:
    """Concatenated FR-adversarial corpus: FR prose + FAV22 clair."""
    fr = _load_prose().get("fr", "")
    fav = _load_fav22_clair()
    if fr and fav:
        return fr + "  " + fav
    return fr or fav


_ADVERSARIAL_FR_POSITIONS_CACHE: list[int] | None = None


def _adversarial_fr_positions() -> tuple[str, list[int]]:
    """Return the FR-adversarial corpus and a list of match start
    positions for the target patterns. Cached after first call."""
    global _ADVERSARIAL_FR_POSITIONS_CACHE
    text = _adversarial_corpus_text()
    if _ADVERSARIAL_FR_POSITIONS_CACHE is not None:
        return text, _ADVERSARIAL_FR_POSITIONS_CACHE
    positions: list[int] = []
    if text:
        for pat in _ADVERSARIAL_FR_PATTERNS:
            for m in pat.finditer(text):
                positions.append(m.start())
        positions.sort()
    _ADVERSARIAL_FR_POSITIONS_CACHE = positions
    return text, positions


def sample_french_adversarial(
    rng: np.random.Generator,
    min_chars: int = 6,
    max_chars: int = 24,
) -> str:
    """Hybrid sampler that exposes the model to ``W + vowel`` and
    ``QU + vowel`` patterns at the rate it sees them on real French
    traffic — orders of magnitude above their natural prose density.

    Two modes (50/50):

    * **synthetic / positional**: pick a random ``W·vowel`` or
      ``QU·vowel`` match in the FR-adversarial corpus (FR prose + FAV22
      clair) and extract a window of 6-24 chars centred on it. The
      window is snapped to word boundaries so the fragment reads
      naturally.
    * **corpus**: draw a longer fragment of FR prose + FAV22 clair
      *without* requiring a target bigram, so the model still sees
      authentic FR distribution around the adversarial samples and does
      not over-fit to the specific patterns.

    Falls back to :func:`sample_prose_fr` if the corpus is unavailable
    or has no matches (CI without ``data/`` dir).
    """
    text, positions = _adversarial_fr_positions()
    if not text or not positions:
        return sample_prose_fr(rng, min_chars=min_chars, max_chars=max_chars)
    n = len(text)
    if rng.random() < 0.5:
        # Positional / adversarial draw.
        pos = positions[int(rng.integers(0, len(positions)))]
        target = int(rng.integers(min_chars, max_chars + 1))
        half = target // 2
        start = max(0, pos - half)
        end = min(n, start + target)
        start, end = _snap_to_word_boundary(text, start, end)
        fragment = text[start:end].strip()
        return (
            fragment
            if fragment
            else text[start : start + target].strip()
        )
    # Corpus draw — uniform window, no positional constraint.
    target = int(rng.integers(min_chars, max_chars + 1))
    if n <= target:
        return text.strip()
    start = int(rng.integers(0, n - target + 1))
    end = start + target
    start, end = _snap_to_word_boundary(text, start, end)
    fragment = text[start:end].strip()
    return fragment if fragment else text[start : start + target].strip()


# --------------------------------------------------------------------- #
# Top-level mix
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class TextMix:
    """Category weights for the top-level text sampler. Weights do not
    need to sum to 1.0; they are normalised on use.

    The ``random`` category emits sequences with no linguistic structure
    (random A-Z / 0-9 / punctuation clumps, see :func:`sample_random_chars`).
    Phase 3.2 introduces this category at significant weight to break the
    linguistic priors that cause "letter-soup" hallucination on weak
    signal — a model that has only ever seen English text learns to
    output plausible-looking English on noise. Random clumps force the
    pure acoustic→character mapping.

    Defaults to 0.0 to preserve pre-Phase-3.2 behaviour for older configs.
    """
    callsign: float
    qcode: float
    qso: float
    numeric: float
    words: float
    random: float = 0.0
    prose: float = 0.0
    prose_fr: float = 0.0
    adversarial_fr: float = 0.0
    # Phase 4.0 — the new pure-acoustic random-char distribution
    # (`sample_random_chars_phase4`). Distinct from `random` so existing
    # 3.x presets stay byte-for-byte equivalent. Phase 4.0 sets this to
    # 1.0 and zeroes everything else.
    random_phase4: float = 0.0

    def is_random_phase4_only(self) -> bool:
        """True iff ``random_phase4`` is the sole non-zero category.

        Used by the synthesis pipeline to dispatch into the wpm-aware
        fast path that avoids the Q-code fallback (which would
        contaminate the no-prior curriculum).
        """
        return (
            self.random_phase4 > 0
            and self.callsign == 0 and self.qcode == 0 and self.qso == 0
            and self.numeric == 0 and self.words == 0 and self.random == 0
            and self.prose == 0 and self.prose_fr == 0
            and self.adversarial_fr == 0
        )

    def as_array(self) -> np.ndarray:
        w = np.array(
            [
                self.callsign, self.qcode, self.qso,
                self.numeric, self.words, self.random,
                self.prose, self.prose_fr, self.adversarial_fr,
                self.random_phase4,
            ],
            dtype=np.float64,
        )
        if (w < 0).any():
            raise ValueError("text-mix weights must be non-negative")
        total = w.sum()
        if total <= 0:
            raise ValueError("text-mix weights must sum to > 0")
        return w / total


DEFAULT_MIX = TextMix(
    callsign=0.15,
    qcode=0.20,
    qso=0.35,
    numeric=0.15,
    words=0.15,
)


PHASE_3_2_MIX = TextMix(
    callsign=0.12,
    qcode=0.14,
    qso=0.25,
    numeric=0.13,
    words=0.06,
    random=0.30,
)


# Phase 3.3: add multilingual prose (FR/DE/ES/EN) to fight the English-prior
# bias observed on real French QSOs at v0.2.0 (e.g. "TOM" inside "AUTOMNE").
# Prose budget is taken mostly from `random` and `words` — the random
# category is preserved at meaningful weight so the anti-hallucination
# benefit from Phase 3.2 is not lost.
PHASE_3_3_MIX = TextMix(
    callsign=0.12,
    qcode=0.14,
    qso=0.25,
    numeric=0.13,
    words=0.04,
    random=0.20,
    prose=0.12,
)


# Phase 3.4: tokenizer extended to 49 (É / À / apostrophe). The new tokens
# only appear in French prose (and incidentally in apostrophised English),
# so we double the prose budget vs Phase 3.3 and split it as 1/3
# multilingual + 2/3 French-only (``prose_fr``). At ~3 % accent density in
# the FR corpus this gives the model ≈ 0.5 % of training characters per
# new token — enough to converge fresh-init head/embedding rows in a
# Phase-3.3-length fine-tune without starving the other categories.
PHASE_3_4_MIX = TextMix(
    callsign=0.10,
    qcode=0.12,
    qso=0.20,
    numeric=0.12,
    words=0.04,
    random=0.18,
    prose=0.08,
    prose_fr=0.16,
)


# Phase 3.6: shave 6 % off the existing FR / random / words slices to
# fund a 6 % adversarial-FR stream. The stream concentrates training
# weight on the exact patterns where Phase 3.5 still false-positives in
# live (W + vowel → À, QU + vowel → É) without changing the overall
# language balance — we simply trade a fraction of generic FR prose for
# adversarial FR prose drawn from the same corpus.
PHASE_3_6_MIX = TextMix(
    callsign=0.10,
    qcode=0.12,
    qso=0.20,
    numeric=0.12,
    words=0.03,
    random=0.16,
    prose=0.08,
    prose_fr=0.13,
    adversarial_fr=0.06,
)


# Phase 4.0: 100 % random_phase4. No prose, no callsigns, no Q-codes, no
# linguistic structure. The acoustic model is retrained as a pure
# character recognizer; language structure is restored at decode time
# via beam search + LM fusion (Phase 5.0). All weights other than
# `random_phase4` are zero — the architectural pivot is the entire
# point of the phase.
PHASE_4_0_MIX = TextMix(
    callsign=0.0,
    qcode=0.0,
    qso=0.0,
    numeric=0.0,
    words=0.0,
    random_phase4=1.0,
)


_CATEGORIES = (
    "callsign", "qcode", "qso", "numeric", "words",
    "random", "prose", "prose_fr", "adversarial_fr",
    "random_phase4",
)
_SAMPLERS = {
    "callsign": sample_callsign,
    "qcode": sample_qcode_abbrev,
    "qso": sample_qso_line,
    "numeric": sample_numeric,
    "words": sample_english_words,
    "random": sample_random_chars,
    "prose": sample_prose,
    "prose_fr": sample_prose_fr,
    "adversarial_fr": sample_french_adversarial,
    "random_phase4": sample_random_chars_phase4,
}


def sample_category(rng: np.random.Generator, mix: TextMix = DEFAULT_MIX) -> str:
    """Pick which source to draw from, under ``mix``."""
    probs = mix.as_array()
    idx = int(rng.choice(len(_CATEGORIES), p=probs))
    return _CATEGORIES[idx]


def sample_text(rng: np.random.Generator, mix: TextMix = DEFAULT_MIX) -> str:
    """Draw one training text example under the configured mix."""
    category = sample_category(rng, mix)
    return _SAMPLERS[category](rng)
