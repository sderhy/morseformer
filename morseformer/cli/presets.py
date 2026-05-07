"""Decode presets — a small named table of (model, thresholds, fusion) tuples.

A preset bundles the four user-facing knobs that matter at inference:

- ``acoustic``        — name in the model registry (see ``registry.py``).
- ``confidence_threshold`` — gate on the acoustic softmax (decode_live's
  ``--confidence-threshold``).
- ``digit_threshold`` — stricter gate applied only to digit tokens 0-9.
- ``lm`` / ``fusion_weight`` — optional shallow-fusion LM. ``lm=None``
  disables fusion. The ``live`` subcommand drops the LM regardless
  because streaming fusion is currently broken (see
  ``project_streaming_fusion_failed`` in memory).

Preset choice rationale:

- ``live`` (default): the v0.5.3 streaming defaults. Threshold 0.6 +
  digit-threshold 0.90 cure the 5.1/5.2 live FP modes.
- ``prose``: same acoustic + LM fusion λ=0.7 for offline file decoding
  on prose audio (Alice gain −11.4 % CER on n=120).
- ``contest``: relaxed thresholds for fast contest-style exchanges
  where missing a character is worse than emitting one wrongly.
- ``conservative``: tightened thresholds for very noisy bands where
  silence is preferable to a guess.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Preset:
    name: str
    description: str
    acoustic: str
    confidence_threshold: float
    digit_threshold: float
    lm: str | None
    fusion_weight: float


PRESETS: dict[str, Preset] = {
    "live": Preset(
        name="live",
        description="v0.5.3 streaming defaults. Best for IC-7300 + receiver "
                    "real-time decode.",
        acoustic="rnnt_phase5_7",
        confidence_threshold=0.6,
        digit_threshold=0.90,
        lm=None,
        fusion_weight=0.0,
    ),
    "prose": Preset(
        name="prose",
        description="Offline file decode with LM shallow fusion (λ=0.7). "
                    "Best for poetry / ragchew / book reading from a "
                    "recorded .wav.",
        acoustic="rnnt_phase5_7",
        confidence_threshold=0.6,
        digit_threshold=0.90,
        lm="lm_phase5_2",
        fusion_weight=0.7,
    ),
    "contest": Preset(
        name="contest",
        description="Looser thresholds for fast contest exchanges where "
                    "missing characters costs more than rare false positives.",
        acoustic="rnnt_phase5_7",
        confidence_threshold=0.5,
        digit_threshold=0.80,
        lm=None,
        fusion_weight=0.0,
    ),
    "conservative": Preset(
        name="conservative",
        description="Tightened thresholds for very noisy bands. Prefers "
                    "silence to a guess.",
        acoustic="rnnt_phase5_7",
        confidence_threshold=0.75,
        digit_threshold=0.95,
        lm=None,
        fusion_weight=0.0,
    ),
}

DEFAULT_PRESET = "live"


def get_preset(name: str) -> Preset:
    if name not in PRESETS:
        raise SystemExit(
            f"[morseformer] unknown preset '{name}'. "
            f"Known: {', '.join(PRESETS)}."
        )
    return PRESETS[name]
