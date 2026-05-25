"""Display-time text formatting for the GUI transcript.

This is a thin, UI-free service: it owns *how decoded text is shown*, not
how it is decoded. The transcript widgets keep the raw decoder output and
re-render it through :func:`apply` whenever the user toggles an option in
the Text menu, so changing a preference never loses or corrupts the
underlying transcript.

The heavy lifting (prosign substitution, break-token newlines) is reused
from :func:`morseformer.decoding.postprocess.format_output`; this module
only carries the user-facing option set.
"""

from __future__ import annotations

from dataclasses import dataclass

from morseformer.decoding.postprocess import format_output


@dataclass(frozen=True)
class DisplayOptions:
    """User-selectable display transforms, persisted via the config store.

    Defaults match the decoder's historical display behaviour (break after
    ``=`` / ``KN``, upper case) so a fresh install looks like before.
    """

    break_tokens: bool = True   # newline after = / KN
    break_after_k: bool = False  # newline after a standalone K
    lowercase: bool = False

    def as_dict(self) -> dict[str, bool]:
        return {
            "break_tokens": self.break_tokens,
            "break_after_k": self.break_after_k,
            "lowercase": self.lowercase,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> DisplayOptions:
        def _b(key: str, default: bool) -> bool:
            v = data.get(key, default)
            if isinstance(v, str):
                return v.lower() in ("1", "true", "yes", "on")
            return bool(v)

        return cls(
            break_tokens=_b("break_tokens", True),
            break_after_k=_b("break_after_k", False),
            lowercase=_b("lowercase", False),
        )


def apply(text: str, opts: DisplayOptions) -> str:
    """Render ``text`` for display under the given options."""
    return format_output(
        text,
        break_tokens=opts.break_tokens,
        break_after_k=opts.break_after_k,
        lowercase=opts.lowercase,
    )
