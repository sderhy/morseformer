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

- ``live`` (default): v0.5.3 streaming defaults on top of the
  **v0.6.4 acoustic** (rnnt_phase11b — forced-alignment-aware real-
  audio retrain from phase5_5/best, -34 % real-OTA mean CER vs 5.5).
  Threshold 0.6 + digit-threshold 0.90 cure the 5.1/5.2 live FP modes.
- ``prose``: same acoustic + dictionary splitter + amateur char
  n-gram LM rescoring (lm_amateur_3gram, Phase 11 §C) for offline
  file decoding. The neural prose LM (lm_phase5_2) was dropped at
  v0.6.3 because it hurt amateur jargon on literary prose.
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
    # When True, run the dictionary-based word splitter
    # (``morseformer.decoding.word_splitter``) on the decoder output
    # to re-segment amateur run-on words (DROMCHRIS → DR OM CHRIS).
    # Off by default; on for the ``prose`` preset where offline
    # readability is the primary goal.
    post_segment: bool = False


PRESETS: dict[str, Preset] = {
    "live": Preset(
        name="live",
        description="v0.5.3 streaming defaults. Best for IC-7300 + receiver "
                    "real-time decode.",
        acoustic="rnnt_phase11b",
        confidence_threshold=0.6,
        digit_threshold=0.90,
        lm=None,
        fusion_weight=0.0,
    ),
    "prose": Preset(
        name="prose",
        description="Offline file decode with the dictionary-based word "
                    "splitter on the output (re-segments run-on amateur "
                    "words + restructures by transmission boundaries). "
                    "Best for ragchew / file decode where readability "
                    "matters. LM fusion is now off by default — it was "
                    "trained on literary prose and consistently hurts "
                    "amateur jargon (see project_streaming_fusion_failed "
                    "+ post-2026-05-21 live test on g3ses C7).",
        acoustic="rnnt_phase11b",
        confidence_threshold=0.6,
        digit_threshold=0.90,
        lm=None,
        fusion_weight=0.0,
        post_segment=True,
    ),
    "contest": Preset(
        name="contest",
        description="Looser thresholds for fast contest exchanges where "
                    "missing characters costs more than rare false positives.",
        acoustic="rnnt_phase11b",
        confidence_threshold=0.5,
        digit_threshold=0.80,
        lm=None,
        fusion_weight=0.0,
    ),
    "conservative": Preset(
        name="conservative",
        description="Tightened thresholds for very noisy bands. Prefers "
                    "silence to a guess.",
        acoustic="rnnt_phase11b",
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
