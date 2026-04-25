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

All outputs use only characters that the 46-token tokenizer accepts
(``A-Z 0-9 space . , ? ! / = + -``). Out-of-vocab characters are
silently dropped by both the tokenizer and the synthesiser, but we
strive to produce clean in-vocab text anyway for clarity and
reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass

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
# are all in the 46-token vocabulary (space and SK are space + S + K).
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

    def as_array(self) -> np.ndarray:
        w = np.array(
            [
                self.callsign, self.qcode, self.qso,
                self.numeric, self.words, self.random,
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


_CATEGORIES = ("callsign", "qcode", "qso", "numeric", "words", "random")
_SAMPLERS = {
    "callsign": sample_callsign,
    "qcode": sample_qcode_abbrev,
    "qso": sample_qso_line,
    "numeric": sample_numeric,
    "words": sample_english_words,
    "random": sample_random_chars,
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
