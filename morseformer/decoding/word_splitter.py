"""Dictionary-based word splitter for run-on RNN-T output.

Real CW operators routinely collapse the inter-word gap so the
acoustic emits things like ``DROMCHRIS`` (= ``DR OM CHRIS``) or
``MYWXIS`` (= ``MY WX IS``). The acoustic model has no way to insert
a word break that has no acoustic evidence — that is a semantic
problem, not a timing one. We solve it post-hoc by re-segmenting each
output token against an embedded amateur + English vocabulary.

Pipeline (``apply``):

  1. Structural pre-pass — detach ``DE`` from a glued callsign,
     normalise ``+`` / ``=`` (BT prosign), and isolate end-of-message
     ``K`` / ``KN`` / ``SK``. Mostly thanks to user-provided regex
     heuristics.
  2. Per-token DP segmentation — for every whitespace-separated token
     that does NOT look like a callsign and is NOT already in the
     dictionary, try to split it into a sequence of dictionary words
     using shortest-path DP. Accept only segmentations that cover
     ``min_coverage`` of the token with ≥ ``min_words`` dictionary
     pieces.
  3. Whitespace cleanup.

The dictionary is curated for amateur radio: the high-confidence
short idioms (``DR``, ``OM``, ``ES``, ``UR``, ``WX``, ``TKS``, ``TNX``,
``QRM`` …) plus equipment / antenna terms plus a small core of the
top English words. Adding a word is cheap; removing one risks letting
a run-on slip through, so the bar to remove is "this word collides
with a callsign and causes real damage on bench".

Designed to be applied AFTER :func:`format_output` — the splitter
expects ``=`` and ``\\n`` already in place.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# --------------------------------------------------------------------------- #
# Dictionary
# --------------------------------------------------------------------------- #
#
# Words are stored in a single set; segmentation scores are derived from
# length (longer matches preferred) and a small priority bump for amateur
# idioms. Keep alphabetical / grouped for the maintenance cost.

_AMATEUR_HIGH: tuple[str, ...] = (
    # Personal + status
    "DR", "OM", "YL", "OP", "MY", "UR", "MR", "MS", "DE",
    # Reports / civility
    "ES", "FB", "HR", "P", "GD", "OK", "RR", "TU", "VY", "NW",
    "TKS", "TNX", "FER", "PSE", "AGN", "ABT", "BTU", "CUL",
    "WID", "WD", "RPRT", "RPT", "GA", "GE", "GM", "GN", "GLD",
    "BCNU", "BURO", "INFO", "INR", "GUD", "MNI", "EVE", "NITE",
    "HW", "CPY", "HI", "OL", "OLT", "CONDX", "CONDITIONS",
    # Q-codes (most common)
    "QRA", "QRM", "QSB", "QRN", "QRP", "QRT", "QRX", "QRZ", "QSL",
    "QSO", "QSP", "QSY", "QTH", "QTR", "QRL",
    # Operating
    "RIG", "ANT", "PWR", "WX", "RST", "TEMP", "TEMPERATURE",
    "NAME", "AGE", "BAND", "FREQ", "MODE", "CW",
    "MHZ", "KHZ", "W", "KW", "WATTS", "M",
    # Prosigns / canonicals
    "K", "KN", "SK", "AR", "AS", "KA", "BK", "BT",
    "CQ", "TEST", "CONTEST", "DX",
    # Reports / numbers (cut + plain)
    "5NN", "599", "73", "88", "44",
    # Hardware brands / models — keep multi-token strings split here
    "FT", "IC", "K3", "K3S", "K4", "KX", "KX2", "KX3", "K2",
    "TS", "FTDX", "ACOM", "ALPHA", "SPE", "ELECRAFT", "YAESU",
    "ICOM", "KENWOOD", "ARRL",
    # Antenna shorthand heard a lot on-air
    "LW", "LWA", "DP", "EWE", "INVV",
    # Antenna shapes
    "DIPOLE", "VERTICAL", "YAGI", "BEAM", "LOOP", "GP",
    "HEXBEAM", "WINDOM", "MAGLOOP", "ENDFED", "LONGWIRE",
    "G5RV", "OCF", "OCFD", "EFHW", "INV", "VEE", "ZEPP",
    "QUAD", "DELTA", "MOXON", "WHIP", "MOBILE", "BEVERAGE",
    "MULTIBAND", "TRAP", "MAST", "TOWER",
    # Common amateur conversation
    "AT", "ON", "IN", "OF", "FOR", "WITH", "FROM", "AND", "OR",
    "THE", "IS", "WAS", "BE", "ARE",
    "TODAY", "HERE", "GOOD", "MORNING", "EVENING", "NIGHT",
    "WEATHER", "WIND", "NORTH", "SOUTH", "EAST", "WEST",
    "RAIN", "SUN", "CLEAR", "CLOUDY", "WINDY", "RAINY",
    "SUNNY", "COLD", "WARM", "MILD", "HOT", "FAIR", "COOL",
    "SNOW", "SNOWY", "FOGGY", "WET", "DRY",
    # Sloppy short combos heard on-air
    "OLDIE", "PHIL", "JOHN", "STEVE", "TOM", "BOB", "JIM",
    "JOE", "MIKE", "DAVE", "JACK", "JAC", "CHRIS",
    "SUM",  # "sum rain" = some rain
    "MMNT",  # moment
    "MIN",  # minute(s)
    "CB",   # CB = citizen band (rare in amateur but sometimes used)
)


# Smaller English fill-list — enough to catch ragchew prose without
# becoming a magnet for callsign collisions. Sourced from common-
# usage lists, hand-edited to drop one- and two-letter forms that
# overlap heavily with prosigns or cut-numbers.
_ENGLISH_COMMON: tuple[str, ...] = (
    "A", "ABOUT", "ABOVE", "ACROSS", "AFTER", "AGAIN", "AGAINST",
    "ALL", "ALMOST", "ALONE", "ALONG", "ALREADY", "ALSO", "ALTHOUGH",
    "ALWAYS", "AMONG", "ANOTHER", "ANY", "ANYONE", "ANYTHING",
    "AROUND", "BACK", "BAD", "BECAUSE", "BECOME", "BEEN", "BEFORE",
    "BEGAN", "BEGIN", "BEHIND", "BEING", "BELOW", "BEST", "BETTER",
    "BETWEEN", "BIG", "BLACK", "BLUE", "BOOK", "BOTH", "BRING",
    "BUT", "BUY", "CALL", "CAME", "CAN", "CANNOT", "CAR", "CARE",
    "CHILD", "CHILDREN", "CITY", "CLOSE", "COMING", "COULD",
    "COUNTRY", "DAY", "DAYS", "DID", "DIFFERENT", "DO", "DOES",
    "DOG", "DOING", "DONE", "DOOR", "DOWN", "DURING", "EACH",
    "EARLY", "EITHER", "END", "ENOUGH", "EVEN", "EVER", "EVERY",
    "EVERYONE", "EVERYTHING", "EYE", "EYES", "FACE", "FACT",
    "FAR", "FATHER", "FEEL", "FEW", "FIELD", "FIND", "FINE", "FIRST",
    "FIVE", "FOLLOWING", "FOOD", "FOOT", "FOUND", "FOUR",
    "FRIEND", "FULL", "GAVE", "GET", "GIVE", "GIVEN", "GO",
    "GOES", "GOING", "GONE", "GOT", "GREAT", "GREEN", "GROUND",
    "GROUP", "HAD", "HALF", "HAND", "HANDS", "HAPPEN", "HARD",
    "HAS", "HAVE", "HE", "HEAR", "HEARD", "HELP", "HER", "HERE",
    "HIGH", "HIM", "HIMSELF", "HIS", "HOME", "HOPE", "HOUR",
    "HOUSE", "HOW", "HUMAN", "HUNDRED", "I", "IDEA", "IF", "IM",
    "IMPORTANT", "INCLUDING", "INSIDE", "INTO", "IT", "ITS",
    "JUST", "KEEP", "KIND", "KNEW", "KNOW", "KNOWN", "LAND",
    "LARGE", "LAST", "LATE", "LATER", "LEAD", "LEAST", "LEAVE",
    "LEFT", "LESS", "LET", "LETTER", "LIFE", "LIGHT", "LIKE",
    "LINE", "LITTLE", "LIVE", "LIVED", "LOCAL", "LONG", "LOOK",
    "LOOKED", "LOOKING", "LOOKS", "LOT", "LOVE", "LOW", "MADE",
    "MAKE", "MAKING", "MAN", "MANY", "MAY", "MAYBE", "ME", "MEAN",
    "MEANS", "MEMBER", "MEN", "MIGHT", "MIND", "MINE", "MINUTES",
    "MISS", "MOMENT", "MONEY", "MONTH", "MONTHS", "MORE", "MOST",
    "MOTHER", "MOVE", "MUCH", "MUST", "MYSELF", "NEAR", "NEED",
    "NEVER", "NEW", "NEXT", "NIGHT", "NO", "NOT", "NOTHING",
    "NOW", "NUMBER", "ODD", "OFF", "OFFICE", "OFTEN", "OH",
    "OLD", "ONCE", "ONE", "ONLY", "OPEN", "OPENED", "ORDER",
    "OTHER", "OTHERS", "OUR", "OURSELVES", "OUT", "OUTSIDE",
    "OVER", "OWN", "PAGE", "PART", "PASSED", "PAST", "PEOPLE",
    "PERHAPS", "PERSON", "PIECE", "PLACE", "PLAY", "POINT",
    "POSSIBLE", "POWER", "PROBABLY", "PROBLEM", "PROBLEMS",
    "PUT", "QUESTION", "QUITE", "RATHER", "READ", "READY", "REAL",
    "RECEIVED", "REMEMBER", "RIGHT", "ROOM", "RUN", "SAID", "SAME",
    "SAW", "SAY", "SAYING", "SCHOOL", "SECOND", "SEE", "SEEM",
    "SEEMED", "SEEN", "SET", "SEVERAL", "SHE", "SHORT", "SHOULD",
    "SHOW", "SHOWED", "SHOWN", "SIDE", "SINCE", "SMALL", "SO",
    "SOCIAL", "SOME", "SOMEONE", "SOMETHING", "SOMETIMES",
    "SOON", "SORT", "SOUND", "SOURCE", "STATE", "STATES", "STILL",
    "STOP", "STORY", "SUCH", "SURE", "TAKE", "TAKEN", "TALK",
    "TALKING", "TELL", "TEN", "THAN", "THANK", "THANKS", "THAT",
    "THEIR", "THEM", "THEN", "THERE", "THESE", "THEY", "THINK",
    "THIRD", "THIS", "THOSE", "THOUGH", "THOUGHT", "THREE", "THROUGH",
    "TILL", "TIME", "TIMES", "TODAY", "TOGETHER", "TOLD", "TOO",
    "TOOK", "TOP", "TOWARD", "TRY", "TRYING", "TWO", "UNDER",
    "UNTIL", "UP", "UPON", "US", "USE", "USED", "USING", "USUAL",
    "USUALLY", "VERY", "WALK", "WALKED", "WANT", "WANTED", "WAR",
    "WAS", "WATER", "WAY", "WAYS", "WE", "WEEK", "WELL", "WENT",
    "WERE", "WHAT", "WHEN", "WHERE", "WHETHER", "WHICH", "WHILE",
    "WHITE", "WHO", "WHOLE", "WHOSE", "WHY", "WILL", "WIND",
    "WINDOW", "WISH", "WITHIN", "WITHOUT", "WOMAN", "WOMEN", "WORD",
    "WORK", "WORLD", "WOULD", "WRITE", "YEAR", "YEARS", "YES",
    "YOU", "YOUNG", "YOUR", "YOURSELF",
)


# Build the runtime dictionary
DICT: frozenset[str] = frozenset(set(_AMATEUR_HIGH) | set(_ENGLISH_COMMON))


# --------------------------------------------------------------------------- #
# Callsign detection
# --------------------------------------------------------------------------- #
#
# ITU callsign rough shape: 1-2 letters / digits prefix + 1 digit + 1-4 letter
# suffix. Includes /P /M /MM /QRP suffixes. Avoid splitting these.
_CALLSIGN_RE = re.compile(
    r"^[A-Z0-9]{1,3}[0-9][A-Z0-9]{1,4}(/[A-Z0-9]{1,4})?$"
)


def is_callsign(token: str) -> bool:
    """Returns True if ``token`` looks like an amateur callsign."""
    if not (3 <= len(token) <= 12):
        return False
    if not _CALLSIGN_RE.match(token):
        return False
    # Must have at least one digit (the number) AND at least one letter
    # (callsigns are not pure-digit).
    return any(c.isdigit() for c in token) and any(c.isalpha() for c in token)


# --------------------------------------------------------------------------- #
# Structural pre-pass (user-suggested regex)
# --------------------------------------------------------------------------- #
#
# Splits the obvious structural glue without consulting the dictionary.

# DE glued to its callsign suffix (DEF4HYY → DE F4HYY) OR its
# preceding letter/digit (YDE → Y DE). Both directions seen in the
# 2026-05-19 audit on g3ses + g6pz.
_DE_SUFFIX_RE = re.compile(r"\bDE([A-Z0-9]{3,})\b")
_DE_PREFIX_RE = re.compile(r"\b([A-Z0-9]{1,3})DE\b")
_BT_PROSIGN_RE = re.compile(r"\s*[+=]\s*")
# End-of-transmission markers, including EE (sometimes sent in
# place of K by tired operators).
_END_TX_RE = re.compile(r"\s+(K|KN|SK|EE)(?=\s|$)")
# Punctuation aeration — separate ? , . + from an immediately
# following letter / digit so the splitter sees clean tokens.
# ``/`` is deliberately excluded — it is the standard portable
# suffix separator (F4HYY/P, MM0XYZ/M) where we MUST NOT add a
# space.
_PUNCT_AERATE_RE = re.compile(r"([?,.+])([A-Z0-9])")
# Spaced-callsign reconstruction. Real CW transcripts sometimes look
# like ``F 4 H Y Y`` (fully spaced) or ``F4 H Y Y`` (prefix glued to
# its digit) because the model emits per-character spaces when the
# inter-character gap is unusually long. ``\s*`` between prefix and
# digit allows the partially-glued form; ``\s+`` between the digit
# and each suffix letter requires at least one separator there so
# already-correct callsigns like ``F4HYY`` are not re-touched.
_SPACED_CALLSIGN_RE = re.compile(
    r"\b([A-Z]{1,2})\s*(\d)((?:\s+[A-Z]){1,4})(?:\s*(/[A-Z0-9]{1,4}))?\b"
)
_WS_RE = re.compile(r"[ \t]{2,}")


def _join_spaced_callsign(match: re.Match) -> str:
    prefix = match.group(1)
    digit = match.group(2)
    suffix_raw = match.group(3)
    portable = match.group(4) or ""
    suffix = "".join(suffix_raw.split())
    return f"{prefix}{digit}{suffix}{portable}"


def structural_normalise(text: str) -> str:
    """Apply the structural-clean regex pre-pass.

    Order matters: spaced-callsign reconstruction runs before
    punctuation aeration so a callsign followed by ``?`` is glued
    first and then aerated; DE-detachment runs before BT-prosign
    normalisation so ``F4HYYDE`` does not eat the following ``=``.
    """
    text = text.strip()
    # 1. Spaced-callsign reconstruction (F 4 H Y Y → F4HYY).
    text = _SPACED_CALLSIGN_RE.sub(_join_spaced_callsign, text)
    # 2. DE detached from its callsign neighbour, both directions.
    text = _DE_SUFFIX_RE.sub(r"DE \1", text)
    text = _DE_PREFIX_RE.sub(r"\1 DE", text)
    # 3. Aerate punctuation glued to a letter/digit.
    text = _PUNCT_AERATE_RE.sub(r"\1 \2", text)
    # 4. Normalise BT prosign with surrounding whitespace + line break.
    text = _BT_PROSIGN_RE.sub(" = \n", text)
    # 5. Isolate end-of-transmission markers — keep leading space and
    # add a blank line after. The trailing context (\s|$) lets a
    # message-final marker still get the treatment.
    text = _END_TX_RE.sub(r" \1 \n\n", text)
    # 6. Collapse runs of horizontal whitespace (newlines preserved).
    text = _WS_RE.sub(" ", text)
    # 7. Strip leading + trailing whitespace from each line — leading
    # spaces appear when the structural pre-pass injects a newline
    # right before a space-separated continuation.
    text = "\n".join(line.strip() for line in text.splitlines())
    return text


# --------------------------------------------------------------------------- #
# DP segmentation
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SplitterConfig:
    """Knobs for :func:`split_token`.

    Attributes:
        min_word_chars:  reject candidate sub-words shorter than this
                         (lone letters in mid-segment are usually noise,
                         not amateur idiom).
        min_coverage:    only accept a segmentation if the dict words
                         cover ≥ this fraction of the original token.
        min_words:       require at least this many dict words in the
                         segmentation (a single dict word matched out
                         of a longer token is treated as no match).
        min_token_to_split:
                         skip tokens shorter than this. A 4-letter
                         common word like ``DEAR`` would otherwise
                         greedily split into ``DE`` + ``AR`` (both
                         amateur idioms in the dict) and silently
                         introduce a space error on clean prose.
                         Default 5 protects 4-letter words while
                         still allowing the canonical 6+ run-ons
                         (``MYWXIS``, ``DROMCHRIS``, ``ESANTISLW``).
        max_token_chars: ignore very long tokens (likely garbage or
                         pre-segmented prose). Caps the DP cost.
        length_bonus:    score bonus per char of matched dict word.
                         Longer matches beat shorter ones, all else
                         equal.
    """
    min_word_chars: int = 2
    min_coverage: float = 0.90
    min_words: int = 2
    min_token_to_split: int = 6
    max_token_chars: int = 30
    length_bonus: float = 1.0


_DEFAULT_CFG = SplitterConfig()


def split_token(token: str, cfg: SplitterConfig = _DEFAULT_CFG) -> list[str]:
    """Segment ``token`` against ``DICT`` via greedy longest-match.

    Walks the token left-to-right; at each position, accepts the
    longest dictionary word that starts there (≥ ``cfg.min_word_chars``
    chars). Characters that do not start a dictionary match accumulate
    into an "unknown" chunk that is emitted as one piece between
    dictionary words. Unknown chunks are kept in the output so the
    user still sees what the acoustic emitted.

    Returns ``[token]`` unchanged when:
      * ``token`` is empty, already a dictionary word, or a callsign;
      * the token is shorter than ``min_word_chars × min_words`` or
        longer than ``max_token_chars`` (DP cost cap);
      * the segmentation produces fewer than ``cfg.min_words``
        dictionary words, OR the dictionary words cover less than
        ``cfg.min_coverage`` of the total length.
    """
    if not token:
        return [token]
    if token in DICT:
        return [token]
    if is_callsign(token):
        return [token]
    n = len(token)
    if n < cfg.min_token_to_split or n > cfg.max_token_chars:
        return [token]
    out: list[str] = []
    unknown: list[str] = []
    # Track only the pieces that were explicitly matched by the
    # ≥ ``min_word_chars`` search so an accidental 1-char unknown
    # chunk that happens to equal a dict entry (e.g. "P" → ``DICT``)
    # does not count toward the coverage threshold.
    matched_lens: list[int] = []
    i = 0
    # Cap longest-match search at 12 chars — covers the longest
    # entries in DICT comfortably (MULTIBAND, ELECTRAFT, …).
    max_word_chars = 12
    while i < n:
        match: str | None = None
        for end in range(min(n, i + max_word_chars), i + cfg.min_word_chars - 1, -1):
            piece = token[i:end]
            if piece in DICT:
                match = piece
                break
        if match is not None:
            if unknown:
                out.append("".join(unknown))
                unknown = []
            out.append(match)
            matched_lens.append(len(match))
            i += len(match)
        else:
            unknown.append(token[i])
            i += 1
    if unknown:
        out.append("".join(unknown))
    if len(matched_lens) < cfg.min_words:
        return [token]
    if sum(matched_lens) / n < cfg.min_coverage:
        return [token]
    return out


# --------------------------------------------------------------------------- #
# Top-level entry point
# --------------------------------------------------------------------------- #


def _split_line(line: str, cfg: SplitterConfig) -> str:
    """Split each whitespace-separated token in ``line`` and re-join."""
    out: list[str] = []
    for token in line.split():
        out.extend(split_token(token, cfg))
    return " ".join(out)


def apply(text: str, cfg: SplitterConfig = _DEFAULT_CFG) -> str:
    """Full pipeline: structural pre-pass + per-token DP segmentation.

    Newlines from the structural pre-pass are preserved; segmentation
    operates line-by-line so amateur prosign breaks stay intact.
    """
    text = structural_normalise(text)
    return "\n".join(_split_line(line, cfg) for line in text.splitlines())
