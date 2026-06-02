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
- ``contest``: relaxed thresholds *and* a narrow 100 Hz front-end
  band-pass (vs the 200 Hz default) for dense contest passbands
  with multiple adjacent CW signals. Narrowing was validated on a
  CQ WPX CW recording set on 2026-05-31..06-02 (see
  ``reports/wpx_diagnosis_2026_05_31/``): ``cwcwwDA1A`` goes from
  inexploitable to a readable QSO. A factorial bandwidth×threshold
  bench (2026-06-02) showed 100 Hz captures essentially all the
  contest benefit (DA1A QSO recovered, ``cwcww5`` CQ de-hallucinated)
  while bw=60 over-narrows: it costs +2.0 CER / +14.5 WER on the
  real-OTA ``websdr-fav22`` clip and even re-hallucinates ``cwcww5``
  (CMT). At 100 Hz the clean LCWO mean CER is unchanged (2.56) and
  mean WER actually improves (8.89→8.77). A tighter 60 Hz remains
  available via ``--bandwidth 60`` (CLI) or the BW field (GUI) for very
  dense pile-ups, at the cost of some accuracy on isolated stations.
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
    # Front-end band-pass width in Hz around the carrier. 200 Hz is the
    # historical default (matches StreamingConfig). 100 Hz is enabled for
    # the contest preset to reject adjacent stations in dense WPX-style
    # passbands without over-narrowing (60 Hz regressed real-OTA clips);
    # see project_wpx_diagnosis_2026_05_31 in memory.
    bandwidth_hz: float = 200.0


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
                    "missing characters costs more than rare false positives, "
                    "and a narrow 100 Hz front-end band-pass to reject "
                    "adjacent stations in dense WPX-style passbands "
                    "(see reports/wpx_diagnosis_2026_05_31/).",
        acoustic="rnnt_phase11b",
        confidence_threshold=0.5,
        digit_threshold=0.80,
        lm=None,
        fusion_weight=0.0,
        bandwidth_hz=100.0,
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
