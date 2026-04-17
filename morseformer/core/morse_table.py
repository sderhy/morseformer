"""International Morse code table, prosigns, and PARIS timing helpers.

The PARIS standard defines 1 WPM as one transmission of the word "PARIS" per
minute. "PARIS" is exactly 50 dot-units long including trailing inter-word gap,
so at N WPM one dot-unit is 1.2 / N seconds.
"""

from __future__ import annotations

# Single-character codes: A-Z, 0-9, common punctuation.
MORSE_TABLE: dict[str, str] = {
    "A": ".-",      "B": "-...",    "C": "-.-.",    "D": "-..",
    "E": ".",       "F": "..-.",    "G": "--.",     "H": "....",
    "I": "..",      "J": ".---",    "K": "-.-",     "L": ".-..",
    "M": "--",      "N": "-.",      "O": "---",     "P": ".--.",
    "Q": "--.-",    "R": ".-.",     "S": "...",     "T": "-",
    "U": "..-",     "V": "...-",    "W": ".--",     "X": "-..-",
    "Y": "-.--",    "Z": "--..",
    "0": "-----",   "1": ".----",   "2": "..---",   "3": "...--",
    "4": "....-",   "5": ".....",   "6": "-....",   "7": "--...",
    "8": "---..",   "9": "----.",
    ".": ".-.-.-",  ",": "--..--",  "?": "..--..",  "'": ".----.",
    "!": "-.-.--",  "/": "-..-.",   "(": "-.--.",   ")": "-.--.-",
    "&": ".-...",   ":": "---...",  ";": "-.-.-.",  "=": "-...-",
    "+": ".-.-.",   "-": "-....-",  "_": "..--.-",  '"': ".-..-.",
    "$": "...-..-", "@": ".--.-.",
}

# Prosigns: digraphs sent without inter-letter gap. Notation uses angle brackets,
# e.g. <BT> = break text / paragraph separator (also rendered as "=").
PROSIGNS: dict[str, str] = {
    "<BT>": "-...-",    # break text / new paragraph (= in printed copy)
    "<AR>": ".-.-.",    # end of message (+ in printed copy)
    "<SK>": "...-.-",   # end of contact
    "<KN>": "-.--.",    # go only / named station continue
    "<AS>": ".-...",    # wait / stand by
    "<HH>": "........", # error
    "<SN>": "...-.",    # understood
    "<CT>": "-.-.-",    # start of transmission (copy)
}

# Reverse lookup for decoding. Single-char codes take precedence over prosigns
# when the dit/dah sequence is ambiguous (prosign-only codes have no single-char
# equivalent by construction, so there is no real collision — but e.g. "-...-"
# is both <BT> and "=", which is intended: they are printed identically).
INVERSE_TABLE: dict[str, str] = {v: k for k, v in MORSE_TABLE.items()}
INVERSE_PROSIGNS: dict[str, str] = {v: k for k, v in PROSIGNS.items()}


def unit_seconds(wpm: float) -> float:
    """Duration of one Morse dot-unit at the given speed (PARIS standard)."""
    if wpm <= 0:
        raise ValueError(f"WPM must be positive, got {wpm}")
    return 1.2 / wpm


def encode(text: str) -> list[str]:
    """Encode a text string to a list of per-character Morse codes.

    Unknown characters are silently dropped. Case is ignored. The returned
    list preserves words (consecutive runs of non-space chars); word
    boundaries are represented by the empty string `""` between words.
    """
    out: list[str] = []
    for word_i, word in enumerate(text.upper().split()):
        if word_i > 0:
            out.append("")
        for ch in word:
            code = MORSE_TABLE.get(ch)
            if code is not None:
                out.append(code)
    return out


def decode_code(code: str) -> str:
    """Decode a single dit/dah string to its character, or '' if unknown."""
    if code in INVERSE_TABLE:
        return INVERSE_TABLE[code]
    if code in INVERSE_PROSIGNS:
        return INVERSE_PROSIGNS[code]
    return ""
