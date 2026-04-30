"""PyTorch IterableDataset that streams synthetic training examples.

Each item is a dict:

    features:  torch.float32, shape ``[T, F]``  — front-end features
               at ``FrontendConfig.frame_rate`` (500 fps default, F=1).
    tokens:    torch.int64,   shape ``[L]``    — target token indices
               (no blank, no padding).
    n_frames:  int                              — T (constant per batch).
    n_tokens:  int                              — L (varies across items).

Because the audio is rendered at a fixed duration, every ``features``
tensor has the same shape and batches stack directly. ``tokens`` are
padded in the ``collate`` helper below to the batch-max length with the
CTC blank index — ignored by ``ctc_loss`` via ``target_lengths``.

Phase 2.0 config defaults: clean audio, ideal operator timing, WPM
uniform in [16, 28], 6.0 s at 8 kHz. See
``memory/project_phase2_decisions.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import IterableDataset

import math

from morse_synth.channel import ChannelConfig
from morse_synth.core import render
from morse_synth.keying import KeyingConfig
from morse_synth.operator import OperatorConfig
from morseformer.core.morse_table import MORSE_TABLE, unit_seconds
from morseformer.core.tokenizer import BLANK_INDEX, encode
from morseformer.data.text import (
    DEFAULT_MIX,
    PHASE_3_2_MIX,
    PHASE_3_3_MIX,
    PHASE_3_4_MIX,
    PHASE_3_6_MIX,
    PHASE_4_0_MIX,
    TextMix,
    sample_random_chars_phase4,
    sample_text,
)
from morseformer.features import FrontendConfig, extract_features


def estimate_cw_duration_s(text: str, wpm: float) -> float:
    """Upper-bound estimate of the audio duration of ``text`` at ``wpm``.

    Counts PARIS-standard units for every known character: each dit is 1
    unit, each dah is 3 units, inter-element gaps are 1 unit, inter-
    character gaps are 3 units, and inter-word gaps are 7 units. Adds a
    50 ms tail for the keying ramp-down.

    Slightly over-estimates real audio duration, which is fine for
    pre-filtering overly long texts before rendering.
    """
    u = unit_seconds(wpm)
    total_units = 0.0
    words = text.upper().split()
    for w_idx, word in enumerate(words):
        if w_idx > 0:
            total_units += 7.0  # inter-word gap
        first_char = True
        for ch in word:
            code = MORSE_TABLE.get(ch)
            if code is None:
                continue
            if not first_char:
                total_units += 3.0  # inter-character gap
            first_char = False
            # Elements plus inter-element gaps.
            total_units += sum(1.0 if e == "." else 3.0 for e in code)
            total_units += max(0, len(code) - 1)
    return total_units * u + 0.05


def _random_phase4_max_chars(wpm: float, budget_s: float) -> int:
    """Largest N such that an N-char random_phase4 sequence is likely
    to fit in ``budget_s`` of audio at ``wpm`` keying speed.

    Uses the PARIS-standard average of ~10 dot-units per character
    (including the inter-character gap), with an 11-dot conservative
    factor to leave headroom for operator-jitter slip in 4.0b/4.0c
    and for the heavier punctuation tokens in mode_b (',' / '.' /
    '?' / '!' all clock in around 9–12 dot-units). The 50 ms keying
    tail matches :func:`estimate_cw_duration_s`. Returns at least 1
    so the sampler always has work to do — the caller relies on this
    invariant to skip its retry loop entirely.
    """
    u = unit_seconds(wpm)
    available_dots = max(0.0, (budget_s - 0.05) / u)
    return max(1, int(available_dots // 11.0))


def _FALLBACK_SHORT_TEXTS() -> tuple[str, ...]:
    # Always-very-short material used when sampling keeps returning texts
    # that exceed the target duration — e.g. at 16 WPM where prose is
    # hard to fit in six seconds. Guaranteed to fit ≥ 6 s at WPM ≥ 12.
    return ("K", "TU", "73", "CQ", "QRZ", "DE", "R", "SK", "BK")


@dataclass
class DatasetConfig:
    """Hyperparameters for the synthetic CW training stream.

    Attributes:
        target_duration_s: Fixed utterance duration in seconds.
        sample_rate:       Audio sample rate in Hz. Must be a multiple of
                           ``frontend.frame_rate``.
        freq_hz:           CW carrier frequency in Hz.
        wpm_range:         Inclusive (low, high) range for uniform WPM
                           sampling.
        text_mix:          Category weights for the text sampler.
        frontend:          DSP front-end config.
        keying:            Keying-edge shape config.

        channel_probability: Probability of applying the HF channel at
                           all. 0.0 = every sample is clean (Phase 2.0
                           curriculum). 1.0 = every sample sees the
                           channel. Fractional values mix clean and
                           channel-affected samples.
        snr_db_range:      Uniform SNR range (inclusive) in dB when the
                           channel is applied. Ignored when
                           ``channel_probability == 0``.
        rx_filter_bw:      Receiver-side bandpass filter bandwidth (Hz).
                           ``None`` disables the filter. Only applied
                           when the channel is applied.
        operator_element_jitter_range: Uniform range for per-sample
                           ``OperatorConfig.element_jitter`` (stddev of
                           per-element length noise, in dot-units).
                           Default (0, 0) = perfectly mechanical timing.
        operator_gap_jitter_range:     Same, for gap lengths.

        seed:              Base RNG seed. Combined with the worker id
                           so multi-worker DataLoaders get disjoint streams.
        max_text_retries:  If the synthesised audio is longer than the
                           target duration, re-draw the text up to this
                           many times. After the final retry the audio
                           is truncated (tokens are kept intact; CTC is
                           robust to mild length mismatch).
    """

    target_duration_s: float = 6.0
    sample_rate: int = 8000
    freq_hz: float = 600.0
    wpm_range: tuple[float, float] = (16.0, 28.0)
    text_mix: TextMix = field(default_factory=lambda: DEFAULT_MIX)
    frontend: FrontendConfig = field(default_factory=FrontendConfig)
    keying: KeyingConfig = field(
        default_factory=lambda: KeyingConfig(shape="raised_cosine", rise_ms=5.0)
    )

    # --- Phase 2.1 extensions: defaults leave Phase 2.0 behaviour intact ---
    channel_probability: float = 0.0
    snr_db_range: tuple[float, float] = (0.0, 30.0)
    rx_filter_bw: float | None = None
    operator_element_jitter_range: tuple[float, float] = (0.0, 0.0)
    operator_gap_jitter_range: tuple[float, float] = (0.0, 0.0)

    # --- Phase 3.1 extensions: richer HF channel, all disabled by default ---
    #
    # These knobs turn on capabilities that already exist in
    # :class:`morse_synth.channel.ChannelConfig` (QSB, QRN, carrier drift)
    # plus two genuinely new ones (per-sample carrier-frequency jitter
    # and QRM co-channel interference). The Phase 3.1 preset wires them
    # into realistic defaults; the plain ``DatasetConfig`` inherits the
    # Phase 2.0 silence so older tests keep producing identical audio.

    # Per-sample tone-frequency jitter (Hz). Sample ``freq = freq_hz + U(lo, hi)``
    # for every rendered utterance. Models the user's imperfect zero-beat
    # on the rig. (lo == hi) disables jitter — both zero = Phase 2.1 behaviour.
    freq_offset_range_hz: tuple[float, float] = (0.0, 0.0)

    # Per-sample QSB (slow amplitude fading) rate and depth, sampled
    # uniformly from the ranges. Depth in dB; 20 dB ≈ fades down to 1/10.
    qsb_rate_range_hz: tuple[float, float] = (0.0, 0.0)
    qsb_depth_range_db: tuple[float, float] = (0.0, 0.0)

    # Per-sample QRN (atmospheric impulse) Poisson rate. Peak amplitude
    # and decay are fixed to the ChannelConfig defaults.
    qrn_rate_range_per_sec: tuple[float, float] = (0.0, 0.0)

    # Per-sample carrier-drift stddev, in Hz per second. Each utterance
    # gets an independent random walk.
    carrier_drift_sigma_range_hz_per_s: tuple[float, float] = (0.0, 0.0)

    # Probability of emitting a *pure-noise* sample (AWGN only, no CW),
    # labelled with the empty string. Kills the "hallucinate on silence"
    # failure mode observed on real-air audio.
    empty_sample_probability: float = 0.0

    # Phase 3.6 — post-emission silence curriculum. With this
    # probability, draw a *very short* text, render it normally, and
    # let the standard pad-or-truncate fill the rest of the buffer with
    # silence. Label = real tokens (non-empty), so the model is forced
    # to learn "after the last symbol → emit nothing on the trailing
    # silence". Targets the residual sentence-boundary É hallucination
    # observed in the post-Phase-3.5 live test (e.g. ``PLEURE.É``).
    post_emission_silence_probability: float = 0.0
    # Length window for the short text (in characters, before render).
    post_emission_silence_text_chars: tuple[int, int] = (1, 5)

    # QRM: co-channel interference. With probability ``qrm_probability``,
    # render a second independent CW utterance at ``freq_hz + U(offset)``
    # and mix it into the primary at ``U(qrm_rel_db_range)`` relative power.
    # 0.0 probability disables the branch entirely.
    qrm_probability: float = 0.0
    qrm_offset_range_hz: tuple[float, float] = (-300.0, 300.0)
    qrm_rel_db_range: tuple[float, float] = (-20.0, -8.0)

    # Phase 4.0 — quiet-zone padding before / after the rendered CW.
    # When the CW is positioned at t=0 of the buffer (3.x behaviour) the
    # model never sees "silence-then-CW" in training and is prone to
    # hallucinating phantom characters at chunk boundaries in streaming
    # decode (the pseudo-`4` on silence observed live at v0.4.0).
    #
    # With these ranges set, every sample inserts a random pre-CW
    # silence ``U(pre_quiet_zone_range_s)`` and reserves a minimum
    # post-CW silence ``post_quiet_zone_min_s`` (the rest of the buffer
    # after the CW gets the trailing silence "for free" via padding).
    # The whole buffer — including the quiet zones — is then run through
    # the channel pipeline, so the quiet zones inherit AWGN, QSB, QRN
    # and (with non-zero ``qrm_probability``) co-channel CW. Label is
    # the CW tokens only; the model must learn that the channel
    # artefacts in the quiet zone are not characters. This is the
    # interpretation-(C) "noisy quiet zone" of the Phase 4.0 plan.
    #
    # Defaults keep zero-length zones so the 3.x presets are unchanged.
    pre_quiet_zone_range_s: tuple[float, float] = (0.0, 0.0)
    post_quiet_zone_min_s: float = 0.0

    seed: int = 0
    max_text_retries: int = 5

    @property
    def target_samples(self) -> int:
        return int(round(self.target_duration_s * self.sample_rate))

    @classmethod
    def phase_2_0(cls, **overrides) -> "DatasetConfig":
        """Explicit Phase 2.0 preset (clean + ideal timing). Equivalent
        to the bare default constructor — provided for symmetry with
        :meth:`phase_2_1` and for readability at call sites."""
        return cls(**overrides)

    @classmethod
    def phase_2_1(cls, **overrides) -> "DatasetConfig":
        """Moderate-noise curriculum: channel on every sample, SNR
        uniform in [0, 30] dB, mild operator jitter, RX bandpass at
        500 Hz. These are the defaults used for the Phase 2.1 training
        pass that produces the benchmark number against the
        rule-based baseline.
        """
        base = dict(
            channel_probability=1.0,
            snr_db_range=(0.0, 30.0),
            rx_filter_bw=500.0,
            operator_element_jitter_range=(0.0, 0.05),
            operator_gap_jitter_range=(0.0, 0.10),
        )
        base.update(overrides)
        return cls(**base)

    @classmethod
    def phase_2_2(cls, **overrides) -> "DatasetConfig":
        """Phase 2.1 curriculum with widened operator-jitter ranges.

        The Phase 2.1 model exposed a jitter-OOD gap at low SNR: the
        benchmark generator (eval.datasets.generate_noisy) uses
        element_jitter=0.08 and gap_jitter=0.15, roughly 2× our
        Phase 2.1 training means. Phase 2.2 trains on a jitter
        distribution that spans the benchmark operator profile
        (element ∈ [0, 0.12], gap ∈ [0, 0.20]) so the model
        generalises to realistic hand-keyed timing.
        """
        base = dict(
            channel_probability=1.0,
            snr_db_range=(0.0, 30.0),
            rx_filter_bw=500.0,
            operator_element_jitter_range=(0.0, 0.12),
            operator_gap_jitter_range=(0.0, 0.20),
        )
        base.update(overrides)
        return cls(**base)

    @classmethod
    def phase_3_2(cls, **overrides) -> "DatasetConfig":
        """Phase 3.2 anti-hallucination curriculum.

        Builds on the Phase 3.1 realistic-HF channel and adds two
        targeted distributions to cure the failure modes seen on the
        2026-04-25 London↔French live test (``project_live_observations_phase3_1.md``):

        * **30 % random-character samples** (letters / digits / mixed /
          with-punct, see :func:`morseformer.data.text.sample_random_chars`).
          Breaks the linguistic priors that make the model fall back to
          plausible-English letter-soup on weak signal.
        * **20 % empty-audio samples** (up from Phase 3.1's 5 %), with
          three sub-modes (pure AWGN, AWGN + QRN bursts, distant weak
          CW labelled empty). Teaches "no decodable signal → emit
          nothing" across the realistic distribution of "no decodable
          signal".

        Channel impairments are kept identical to Phase 3.1 — the gain
        comes entirely from the text and label distributions.
        """
        base = dict(
            channel_probability=1.0,
            snr_db_range=(0.0, 30.0),
            rx_filter_bw=500.0,
            operator_element_jitter_range=(0.0, 0.08),
            operator_gap_jitter_range=(0.0, 0.15),
            freq_offset_range_hz=(-50.0, 50.0),
            qsb_rate_range_hz=(0.05, 1.0),
            qsb_depth_range_db=(0.0, 15.0),
            qrn_rate_range_per_sec=(0.0, 1.0),
            carrier_drift_sigma_range_hz_per_s=(0.0, 1.0),
            empty_sample_probability=0.20,
            qrm_probability=0.25,
            qrm_offset_range_hz=(-300.0, 300.0),
            qrm_rel_db_range=(-18.0, -8.0),
            text_mix=PHASE_3_2_MIX,
        )
        base.update(overrides)
        return cls(**base)

    @classmethod
    def phase_3_6(cls, **overrides) -> "DatasetConfig":
        """Phase 3.6 adversarial-FR + post-emission-silence curriculum.

        Builds on Phase 3.5 (wider operator-jitter, full Phase 3.4
        French mix) and adds two targeted distributions to close the
        last three live failure modes observed at the end of the
        Phase 3.5 evaluation:

        * 6 % ``adversarial_fr`` text — fragments of FR prose +
          FAV22-clair concentrated on ``W + vowel`` and ``QU + vowel``
          patterns where the Phase 3.5 model still false-positives É /
          À (`WAS A AN` → `ÀS A AN`, `QUAND` → `QÉND`).
        * 10 % ``post_emission_silence`` samples — short text padded
          with trailing silence, label = real tokens, so the model
          learns that the silence after the last symbol must not
          produce a hallucinated follow-up token (e.g. spurious É at
          sentence end).

        Channel and jitter knobs are kept identical to Phase 3.5; the
        gain comes from the new text and label distributions only.
        Bootstrap target: ``checkpoints/phase3_5/best_rnnt.pt``.
        """
        base = dict(
            channel_probability=1.0,
            snr_db_range=(0.0, 30.0),
            rx_filter_bw=500.0,
            operator_element_jitter_range=(0.0, 0.15),
            operator_gap_jitter_range=(0.0, 0.25),
            freq_offset_range_hz=(-50.0, 50.0),
            qsb_rate_range_hz=(0.05, 1.0),
            qsb_depth_range_db=(0.0, 15.0),
            qrn_rate_range_per_sec=(0.0, 1.0),
            carrier_drift_sigma_range_hz_per_s=(0.0, 1.0),
            empty_sample_probability=0.20,
            post_emission_silence_probability=0.10,
            qrm_probability=0.25,
            qrm_offset_range_hz=(-300.0, 300.0),
            qrm_rel_db_range=(-18.0, -8.0),
            text_mix=PHASE_3_6_MIX,
        )
        base.update(overrides)
        return cls(**base)

    @classmethod
    def _phase_4_0_base(cls, **overrides) -> "DatasetConfig":
        """Common knobs for the Phase 4.0 sub-curricula.

        Phase 4.0 retrains the acoustic model from the Phase 3.5
        checkpoint as a pure character recognizer:

        * 100 % ``random_phase4`` text (no prose, no callsigns, no
          Q-codes — see :data:`PHASE_4_0_MIX`).
        * Quiet-zone padding: every sample inserts a random pre-CW
          silence in [0.5, 2.0] s and reserves ≥ 0.3 s of post-CW
          silence. Together with the channel pipeline applied to the
          full buffer, this teaches the model "decode only when you
          see ≥ 2 coherent CW characters" — kills the
          pseudo-character-on-silence failure mode observed live in
          v0.4.0 streaming.
        * No empty-sample / post-emission-silence branches — those 3.x
          curriculum tricks are subsumed by the global quiet-zone
          design.
        * No QRM in 4.0a/4.0b (added back in 4.0c via the standard
          ``qrm_probability`` knob).
        * WPM range widened to [14, 32] (vs 3.x's [16, 28]).

        Sub-phases differ only in jitter and channel: clean (4.0a) →
        + jitter (4.0b) → + full HF channel (4.0c). The 4.0a/b are
        short stabilisation runs (~10k steps each); 4.0c is the
        long-running phase (~100–150k steps from 3.5 best), since
        the architectural pivot's whole point is massive char-level
        training.

        Window / WPM / quiet-zone calibration (validated 2026-04-30):

        * ``target_duration_s = 4.0`` (down from the 3.x default of 6.0).
          Char-level output doesn't need the word-length context that
          motivated 6 s — shrinking the window cuts memory ~33 % per
          sample, which buys back ~33 % more iterations at the same
          wall-clock budget. The streaming-inference pipeline still
          uses its 6 s decode window: the encoder is a Conformer, so
          training and inference window sizes are independent.
        * ``wpm_range = (18, 32)``: the 14 WPM floor of the original
          plan was incompatible with the 4 s budget after quiet-zone
          reservation (only 2-char samples would fit). 18 WPM is
          still slow CW in practical terms and lets the wpm-aware
          sampler emit 3+ chars even at the floor.
        * ``pre_quiet_zone_range_s = (0.3, 1.0)``,
          ``post_quiet_zone_min_s = 0.3``: half the original (0.5, 2.0)
          reservation. Preserves the streaming-inertia training value
          (model still sees 0.3-1.0 s of channel artefacts before any
          CW) while leaving 2.43 s budget for the CW itself —
          enough for 3-6 chars across the WPM range.
        """
        base = dict(
            target_duration_s=4.0,
            wpm_range=(18.0, 32.0),
            text_mix=PHASE_4_0_MIX,
            pre_quiet_zone_range_s=(0.3, 1.0),
            post_quiet_zone_min_s=0.3,
            empty_sample_probability=0.0,
            post_emission_silence_probability=0.0,
        )
        base.update(overrides)
        return cls(**base)

    @classmethod
    def phase_4_0_a(cls, **overrides) -> "DatasetConfig":
        """Phase 4.0a — clean curriculum, no jitter, no channel.

        Bootstrap target: ``checkpoints/phase3_5/best_rnnt.pt``. The
        clean prefix lets the random-char distribution and quiet-zone
        positioning stabilise on top of the Phase 3.5 weights before
        we re-introduce HF impairments.
        """
        return cls._phase_4_0_base(**overrides)

    @classmethod
    def phase_4_0_b(cls, **overrides) -> "DatasetConfig":
        """Phase 4.0b — 4.0a + Phase-3.5-style operator jitter.

        Same channel-off setting as 4.0a; only operator timing jitter
        is added. Bootstrap target: ``checkpoints/phase4_0_a/best_rnnt.pt``.
        """
        base = dict(
            operator_element_jitter_range=(0.0, 0.15),
            operator_gap_jitter_range=(0.0, 0.25),
        )
        base.update(overrides)
        return cls._phase_4_0_base(**base)

    @classmethod
    def phase_4_0_c(cls, **overrides) -> "DatasetConfig":
        """Phase 4.0c — full HF channel (AWGN + QSB + QRN + drift + QRM).

        Identical channel envelope to Phase 3.5/3.6; the difference is
        the underlying text mix (random-only) and quiet-zone padding,
        so the model learns the same impairments without any
        linguistic prior. Bootstrap target:
        ``checkpoints/phase4_0_b/best_rnnt.pt``.
        """
        base = dict(
            operator_element_jitter_range=(0.0, 0.15),
            operator_gap_jitter_range=(0.0, 0.25),
            channel_probability=1.0,
            snr_db_range=(0.0, 30.0),
            rx_filter_bw=500.0,
            freq_offset_range_hz=(-50.0, 50.0),
            qsb_rate_range_hz=(0.05, 1.0),
            qsb_depth_range_db=(0.0, 15.0),
            qrn_rate_range_per_sec=(0.0, 1.0),
            carrier_drift_sigma_range_hz_per_s=(0.0, 1.0),
            qrm_probability=0.25,
            qrm_offset_range_hz=(-300.0, 300.0),
            qrm_rel_db_range=(-18.0, -8.0),
        )
        base.update(overrides)
        return cls._phase_4_0_base(**base)

    @classmethod
    def phase_3_5(cls, **overrides) -> "DatasetConfig":
        """Phase 3.5 wider-jitter curriculum.

        Identical to Phase 3.4 except for the operator-jitter ranges,
        which are widened from ``(0, 0.08)`` element / ``(0, 0.15)``
        gap to ``(0, 0.15)`` element / ``(0, 0.25)`` gap. The 2026-04-28
        live test on real morning keying surfaced systematic
        false-positive emissions of É / À at sentence boundaries and
        on tight ``W + vowel`` patterns: the user's hand-keyed timing
        sat at the upper edge of the Phase 3.4 jitter envelope, so
        characters like ``F + space + E`` or ``LD`` produced acoustic
        signals that fell into the basin of attraction of the new
        tokens (``É`` = ``..-..`` shares its prefix with ``F`` =
        ``..-.``). Widening the training jitter teaches the model to
        require *clearer* acoustic evidence for the new tokens.

        Bootstrap target: ``checkpoints/phase3_4/last.pt`` — the
        49-vocab tokenizer is already in place, so no checkpoint
        extension is needed.
        """
        base = dict(
            channel_probability=1.0,
            snr_db_range=(0.0, 30.0),
            rx_filter_bw=500.0,
            operator_element_jitter_range=(0.0, 0.15),
            operator_gap_jitter_range=(0.0, 0.25),
            freq_offset_range_hz=(-50.0, 50.0),
            qsb_rate_range_hz=(0.05, 1.0),
            qsb_depth_range_db=(0.0, 15.0),
            qrn_rate_range_per_sec=(0.0, 1.0),
            carrier_drift_sigma_range_hz_per_s=(0.0, 1.0),
            empty_sample_probability=0.20,
            qrm_probability=0.25,
            qrm_offset_range_hz=(-300.0, 300.0),
            qrm_rel_db_range=(-18.0, -8.0),
            text_mix=PHASE_3_4_MIX,
        )
        base.update(overrides)
        return cls(**base)

    @classmethod
    def phase_3_4(cls, **overrides) -> "DatasetConfig":
        """Phase 3.4 French-CW curriculum.

        Identical channel and label distributions to Phases 3.2 / 3.3 —
        only the text mix changes. The tokenizer is extended from 46 to
        49 tokens (É / À / apostrophe added) and ``PHASE_3_4_MIX``
        biases the text sampler toward French-only prose so the freshly
        initialised vocab rows for the new tokens see enough gradient
        during a short fine-tune.

        Motivation: Phase 3.3 closed the multilingual prose gap on real
        French QSOs by exposing the model to natural FR bigrams, but
        diacritics were stripped to ASCII (``ÉTÉ`` → ``ETE``). Phase 3.4
        preserves the diacritics end-to-end so the model can transcribe
        French CW the way it is actually written and sent on air.
        """
        base = dict(
            channel_probability=1.0,
            snr_db_range=(0.0, 30.0),
            rx_filter_bw=500.0,
            operator_element_jitter_range=(0.0, 0.08),
            operator_gap_jitter_range=(0.0, 0.15),
            freq_offset_range_hz=(-50.0, 50.0),
            qsb_rate_range_hz=(0.05, 1.0),
            qsb_depth_range_db=(0.0, 15.0),
            qrn_rate_range_per_sec=(0.0, 1.0),
            carrier_drift_sigma_range_hz_per_s=(0.0, 1.0),
            empty_sample_probability=0.20,
            qrm_probability=0.25,
            qrm_offset_range_hz=(-300.0, 300.0),
            qrm_rel_db_range=(-18.0, -8.0),
            text_mix=PHASE_3_4_MIX,
        )
        base.update(overrides)
        return cls(**base)

    @classmethod
    def phase_3_3(cls, **overrides) -> "DatasetConfig":
        """Phase 3.3 multilingual prose curriculum.

        Identical channel and label distributions to Phase 3.2 — the
        only change is ``text_mix=PHASE_3_3_MIX``, which adds a 12 %
        slice of multilingual prose (FR/DE/ES/EN, normalised to ASCII)
        sampled from ``data/corpus/prose.txt``.

        Motivation: the v0.2.0 live test on a real French QSO surfaced
        an English-prior bias (e.g. ``TOM`` hallucinated inside
        ``AUTOMNE``). Exposing the acoustic model to natural letter
        bigrams in the user's actual language closes that gap.

        The 30 % random-clump weight from Phase 3.2 is reduced to 20 %
        to make room for prose; the anti-hallucination benefit is
        preserved at meaningful weight while the linguistic prior is
        broadened across four languages instead of one.
        """
        base = dict(
            channel_probability=1.0,
            snr_db_range=(0.0, 30.0),
            rx_filter_bw=500.0,
            operator_element_jitter_range=(0.0, 0.08),
            operator_gap_jitter_range=(0.0, 0.15),
            freq_offset_range_hz=(-50.0, 50.0),
            qsb_rate_range_hz=(0.05, 1.0),
            qsb_depth_range_db=(0.0, 15.0),
            qrn_rate_range_per_sec=(0.0, 1.0),
            carrier_drift_sigma_range_hz_per_s=(0.0, 1.0),
            empty_sample_probability=0.20,
            qrm_probability=0.25,
            qrm_offset_range_hz=(-300.0, 300.0),
            qrm_rel_db_range=(-18.0, -8.0),
            text_mix=PHASE_3_3_MIX,
        )
        base.update(overrides)
        return cls(**base)

    @classmethod
    def phase_3_1(cls, **overrides) -> "DatasetConfig":
        """Phase 3.1 realistic-HF curriculum.

        Turns on every channel impairment at realistic magnitudes:

            * Carrier-frequency jitter ±50 Hz (user's imperfect tune)
            * QSB (0.05 – 1 Hz fading, 0 – 15 dB depth)
            * QRN (0 – 1 impulse / sec at −6 dB peak)
            * Carrier drift (0 – 1 Hz / s)
            * Wider operator jitter (element ≤ 0.08, gap ≤ 0.15) so the
              model sees the benchmark operator profile
            * 5 % empty-audio samples (pure AWGN, empty label) to kill
              the silence-hallucination failure mode
            * 25 % QRM (second CW signal at ±50 – 300 Hz offset,
              −18 to −8 dB relative)

        The SNR range is held at (0, 30) dB — same as Phase 2.1 — to
        preserve a direct comparison with the Phase 3.0 benchmark. All
        gains in this curriculum come from channel realism, not from
        moving the SNR operating point.
        """
        base = dict(
            channel_probability=1.0,
            snr_db_range=(0.0, 30.0),
            rx_filter_bw=500.0,
            operator_element_jitter_range=(0.0, 0.08),
            operator_gap_jitter_range=(0.0, 0.15),
            freq_offset_range_hz=(-50.0, 50.0),
            qsb_rate_range_hz=(0.05, 1.0),
            qsb_depth_range_db=(0.0, 15.0),
            qrn_rate_range_per_sec=(0.0, 1.0),
            carrier_drift_sigma_range_hz_per_s=(0.0, 1.0),
            empty_sample_probability=0.05,
            qrm_probability=0.25,
            qrm_offset_range_hz=(-300.0, 300.0),
            qrm_rel_db_range=(-18.0, -8.0),
        )
        base.update(overrides)
        return cls(**base)


def _pad_or_truncate(audio: np.ndarray, target: int) -> np.ndarray:
    """Trim to `target` samples or zero-pad at the end."""
    if audio.size == target:
        return audio
    if audio.size > target:
        return audio[:target]
    out = np.zeros(target, dtype=audio.dtype)
    out[: audio.size] = audio
    return out


def _worker_seed(base_seed: int, worker_id: int) -> int:
    """Combine base seed and worker id deterministically. Distinct worker
    ids yield well-separated streams; same (base, id) always replays."""
    return int((np.uint64(base_seed) * np.uint64(1_000_003)
                + np.uint64(worker_id) * np.uint64(2_654_435_761)) & 0x7FFFFFFF)


class SyntheticCWDataset(IterableDataset):
    """Infinite stream of synthetic CW utterances for acoustic training."""

    def __init__(self, cfg: DatasetConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or DatasetConfig()
        if self.cfg.sample_rate % self.cfg.frontend.frame_rate != 0:
            raise ValueError(
                f"sample_rate={self.cfg.sample_rate} is not a multiple of "
                f"frame_rate={self.cfg.frontend.frame_rate}"
            )

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = 0 if worker_info is None else int(worker_info.id)
        rng = np.random.default_rng(_worker_seed(self.cfg.seed, worker_id))
        while True:
            yield self._generate_one(rng)

    def _sample_operator(self, rng: np.random.Generator, wpm: float) -> OperatorConfig:
        lo_e, hi_e = self.cfg.operator_element_jitter_range
        lo_g, hi_g = self.cfg.operator_gap_jitter_range
        element_jitter = 0.0 if hi_e <= lo_e else float(rng.uniform(lo_e, hi_e))
        gap_jitter = 0.0 if hi_g <= lo_g else float(rng.uniform(lo_g, hi_g))
        # Derive a per-sample operator seed so that timing is reproducible
        # given the parent RNG state.
        op_seed = int(rng.integers(0, 2**31 - 1))
        return OperatorConfig(
            wpm=wpm,
            element_jitter=element_jitter,
            gap_jitter=gap_jitter,
            seed=op_seed,
        )

    @staticmethod
    def _uniform_or_zero(
        rng: np.random.Generator, rng_range: tuple[float, float]
    ) -> float:
        lo, hi = rng_range
        return float(rng.uniform(lo, hi)) if hi > lo else 0.0

    def _sample_channel(
        self, rng: np.random.Generator, snr_override: float | None = None,
    ) -> ChannelConfig | None:
        """Sample a :class:`ChannelConfig` for one utterance.

        Phase 2.1 knobs (AWGN SNR + RX filter) are joined by the
        Phase 3.1 extensions (QSB / QRN / carrier drift) when their
        ranges are non-trivial. Per-sample randomisation keeps the
        training distribution wide without complicating the caller.

        ``snr_override`` forces a specific SNR (used by empty-sample
        rendering so the AWGN level is set explicitly).
        """
        cfg = self.cfg
        if cfg.channel_probability <= 0.0 and snr_override is None:
            return None
        if (
            snr_override is None
            and cfg.channel_probability < 1.0
            and rng.random() >= cfg.channel_probability
        ):
            return None

        if snr_override is not None:
            snr_db = float(snr_override)
        else:
            snr_db = self._uniform_or_zero(rng, cfg.snr_db_range)

        return ChannelConfig(
            snr_db=snr_db,
            rx_filter_bw=cfg.rx_filter_bw,
            rx_filter_centre=cfg.freq_hz,
            qsb_rate_hz=self._uniform_or_zero(rng, cfg.qsb_rate_range_hz),
            qsb_depth_db=self._uniform_or_zero(rng, cfg.qsb_depth_range_db),
            qrn_rate_per_sec=self._uniform_or_zero(
                rng, cfg.qrn_rate_range_per_sec
            ),
            carrier_drift_hz_per_s=self._uniform_or_zero(
                rng, cfg.carrier_drift_sigma_range_hz_per_s
            ),
            seed=int(rng.integers(0, 2**31 - 1)),
        )

    def _sample_fitting_text(self, rng: np.random.Generator, wpm: float) -> str:
        """Draw a text whose estimated rendered duration fits the target.

        Uses ``estimate_cw_duration_s`` as a cheap pre-filter so we never
        render a clip that will have to be truncated. On pathological
        cases (e.g. every sampled text is too long at very slow WPM),
        falls back to a hardcoded short Q-code so the label always
        matches the audio.

        Phase 4.0 fast path: when the mix is ``random_phase4``-only the
        method computes a wpm-derived ``max_chars`` from the budget and
        calls the random-char sampler directly. This avoids the Q-code
        fallback (which would inject linguistic prior into the no-prior
        curriculum) and the wasted retries on long sequences that
        cannot fit at low WPM in a 6 s window with quiet zones.
        """
        cfg = self.cfg
        # Reserve room for the Phase 4.0 quiet zones (zero by default in
        # 3.x presets, so this is a no-op there): the worst-case pre-CW
        # silence is the upper bound of pre_quiet_zone_range_s, and the
        # post-CW silence has a guaranteed minimum.
        pre_max = max(cfg.pre_quiet_zone_range_s[1], 0.0)
        post_min = max(cfg.post_quiet_zone_min_s, 0.0)
        # Leave a 10% margin on top of what's left so that jitter /
        # keying tail can't push us past the budget.
        budget = (cfg.target_duration_s - pre_max - post_min) * 0.9

        if cfg.text_mix.is_random_phase4_only():
            return sample_random_chars_phase4(
                rng, max_chars=_random_phase4_max_chars(wpm, budget)
            )

        for _ in range(cfg.max_text_retries):
            text = sample_text(rng, cfg.text_mix)
            if estimate_cw_duration_s(text, wpm) <= budget:
                return text
        # Fallback.
        short = _FALLBACK_SHORT_TEXTS()
        return short[int(rng.integers(0, len(short)))]

    def _sample_carrier_freq(self, rng: np.random.Generator) -> float:
        """Return the per-sample tone frequency, jittered around
        ``cfg.freq_hz`` by ``cfg.freq_offset_range_hz``. Models the
        user's imperfect zero-beat on the rig."""
        return self.cfg.freq_hz + self._uniform_or_zero(
            rng, self.cfg.freq_offset_range_hz
        )

    def _maybe_qrm_audio(
        self,
        rng: np.random.Generator,
        target_samples: int,
    ) -> np.ndarray | None:
        """Optionally render a secondary CW signal at an offset
        frequency to simulate adjacent-channel interference. Returns
        ``None`` if QRM is not drawn this iteration."""
        cfg = self.cfg
        if cfg.qrm_probability <= 0.0 or rng.random() >= cfg.qrm_probability:
            return None
        qrm_wpm = float(rng.uniform(*cfg.wpm_range))
        qrm_text = self._sample_fitting_text(rng, qrm_wpm)
        qrm_op = self._sample_operator(rng, qrm_wpm)
        qrm_offset = self._uniform_or_zero(rng, cfg.qrm_offset_range_hz)
        qrm_rel_db = self._uniform_or_zero(rng, cfg.qrm_rel_db_range)
        qrm_audio = render(
            qrm_text,
            operator=qrm_op,
            keying=cfg.keying,
            channel=None,     # QRM is mixed *before* the receiver's channel
            freq=cfg.freq_hz + qrm_offset,
            sample_rate=cfg.sample_rate,
        )
        qrm_audio = _pad_or_truncate(qrm_audio, target_samples)
        return qrm_audio.astype(np.float32) * (10.0 ** (qrm_rel_db / 20.0))

    def _post_emission_silence_features(
        self, rng: np.random.Generator
    ) -> tuple[np.ndarray, list[int]]:
        """Render a *short* utterance and let the trailing buffer fill
        with silence. Label is the real (non-empty) token sequence.

        This branch teaches the model that the silence following the
        last emitted symbol must produce *no* output — addressing the
        residual end-of-phrase token hallucination (e.g. spurious É at
        sentence end) seen on the post-Phase-3.5 live test.
        """
        cfg = self.cfg
        target_samples = cfg.target_samples
        lo_chars, hi_chars = cfg.post_emission_silence_text_chars

        text: str | None = None
        for _ in range(8):
            candidate = sample_text(rng, cfg.text_mix)
            if lo_chars <= len(candidate) <= hi_chars:
                text = candidate
                break
        if text is None:
            short = _FALLBACK_SHORT_TEXTS()
            text = short[int(rng.integers(0, len(short)))]

        wpm = float(rng.uniform(*cfg.wpm_range))
        operator = self._sample_operator(rng, wpm)
        freq = self._sample_carrier_freq(rng)
        clean = render(
            text,
            operator=operator,
            keying=cfg.keying,
            channel=None,
            freq=freq,
            sample_rate=cfg.sample_rate,
        )
        clean = _pad_or_truncate(
            clean.astype(np.float32), target_samples
        )
        channel = self._sample_channel(rng)
        if channel is not None:
            from morse_synth.channel import apply_channel
            audio = apply_channel(clean, cfg.sample_rate, channel).astype(
                np.float32
            )
        else:
            audio = clean
        audio = _pad_or_truncate(audio, target_samples)
        tokens = encode(text)
        return audio, list(tokens)

    def _empty_sample_features(
        self, rng: np.random.Generator
    ) -> tuple[np.ndarray, list[int]]:
        """Render an audio clip paired with an empty label.

        Three sub-modes, drawn uniformly. Each teaches a distinct
        "no decodable signal" distribution that the model must reduce
        to the empty hypothesis:

        1. **Pure AWGN** — quiet band, no signal at all.
        2. **AWGN + QRN bursts** — atmospheric clicks, still no carrier.
        3. **Distant weak CW (SNR −35 to −25 dB)** — a real CW signal
           so faint it must be ignored, paired with an empty label. This
           is the Phase 3.2 mode that addresses the live-test failure
           where the model emits letter-soup on weak / out-of-passband
           signals it should not be decoding.
        """
        cfg = self.cfg
        target_samples = cfg.target_samples
        noise_rms = 0.1
        mode = int(rng.integers(0, 3))

        if mode == 2:
            # Distant weak CW: render at the usual carrier and drown it.
            wpm = float(rng.uniform(*cfg.wpm_range))
            text = self._sample_fitting_text(rng, wpm)
            operator = self._sample_operator(rng, wpm)
            clean = render(
                text,
                operator=operator,
                keying=cfg.keying,
                channel=None,
                freq=cfg.freq_hz,
                sample_rate=cfg.sample_rate,
            )
            clean = _pad_or_truncate(
                clean.astype(np.float32), target_samples
            )
            from morse_synth.channel import apply_channel
            ch = ChannelConfig(
                snr_db=float(rng.uniform(-35.0, -25.0)),
                rx_filter_bw=cfg.rx_filter_bw,
                rx_filter_centre=cfg.freq_hz,
                seed=int(rng.integers(0, 2**31 - 1)),
            )
            audio = apply_channel(clean, cfg.sample_rate, ch).astype(
                np.float32
            )
            return audio, []

        # Modes 0 and 1: start from white noise.
        audio = rng.normal(0.0, noise_rms, size=target_samples).astype(
            np.float32
        )
        if mode == 1:
            # Atmospheric impulse clicks on top of the noise floor.
            from morse_synth.channel import _add_qrn
            audio = _add_qrn(
                audio,
                cfg.sample_rate,
                rate=float(rng.uniform(0.5, 3.0)),
                amp_db=-3.0,
                decay_ms=1.0,
                rng=rng,
            ).astype(np.float32)
        # Always apply the RX filter so empty samples are spectrally
        # comparable to the regular distribution after the receiver stage.
        if cfg.rx_filter_bw is not None and cfg.rx_filter_bw > 0:
            from morse_synth.channel import _apply_rx_filter
            audio = _apply_rx_filter(
                audio, cfg.sample_rate, cfg.rx_filter_bw, cfg.freq_hz
            ).astype(np.float32)
        return audio, []

    def _generate_one(self, rng: np.random.Generator) -> dict:
        cfg = self.cfg

        # Empty-sample branch (Phase 3.1): short-circuit rendering and
        # emit pure noise labelled with an empty token sequence so the
        # model learns that "no CW energy → no emission".
        if (
            cfg.empty_sample_probability > 0.0
            and rng.random() < cfg.empty_sample_probability
        ):
            audio, tokens_list = self._empty_sample_features(rng)
            features = extract_features(audio, cfg.sample_rate, cfg.frontend)
            return {
                "features": torch.from_numpy(features),
                "tokens": torch.tensor(tokens_list, dtype=torch.int64),
                "n_frames": int(features.shape[0]),
                "n_tokens": 0,
            }

        # Post-emission silence branch (Phase 3.6): short text + long
        # trailing silence, label = real tokens. Forces the model to
        # learn "after the last symbol → silence, not a hallucinated
        # follow-up token".
        if (
            cfg.post_emission_silence_probability > 0.0
            and rng.random() < cfg.post_emission_silence_probability
        ):
            audio, tokens_list = self._post_emission_silence_features(rng)
            features = extract_features(audio, cfg.sample_rate, cfg.frontend)
            return {
                "features": torch.from_numpy(features),
                "tokens": torch.tensor(tokens_list, dtype=torch.int64),
                "n_frames": int(features.shape[0]),
                "n_tokens": int(len(tokens_list)),
            }

        wpm = float(rng.uniform(*cfg.wpm_range))
        text = self._sample_fitting_text(rng, wpm)
        operator = self._sample_operator(rng, wpm)
        freq = self._sample_carrier_freq(rng)

        # Optional QRM: render the second signal *before* the primary's
        # channel so both see the same receiver filter and AWGN.
        qrm_audio = self._maybe_qrm_audio(rng, cfg.target_samples)

        # Primary signal is rendered clean here so we can mix QRM in
        # before the channel stage. The channel (AWGN + QSB + QRN +
        # carrier drift + RX filter) is then applied to the combined
        # audio in one pass — this matches the physical receiver chain.
        clean_audio = render(
            text,
            operator=operator,
            keying=cfg.keying,
            channel=None,
            freq=freq,
            sample_rate=cfg.sample_rate,
        ).astype(np.float32)
        # Phase 4.0 quiet-zone prepend: shift the CW into the buffer by a
        # random pre-CW silence. The post-CW silence is created
        # automatically by ``_pad_or_truncate`` — its length is whatever
        # remains of the buffer; ``_sample_fitting_text`` already
        # reserves ``post_quiet_zone_min_s`` from the text budget so the
        # tail is never shorter than the configured minimum.
        pre_zone_s = self._uniform_or_zero(rng, cfg.pre_quiet_zone_range_s)
        if pre_zone_s > 0.0:
            pre_samples = int(round(pre_zone_s * cfg.sample_rate))
            clean_audio = np.concatenate(
                [np.zeros(pre_samples, dtype=np.float32), clean_audio]
            )
        clean_audio = _pad_or_truncate(clean_audio, cfg.target_samples)
        if qrm_audio is not None:
            clean_audio = clean_audio + qrm_audio

        channel = self._sample_channel(rng)
        if channel is not None:
            from morse_synth.channel import apply_channel
            audio = apply_channel(clean_audio, cfg.sample_rate, channel)
        else:
            audio = clean_audio

        audio = _pad_or_truncate(audio, cfg.target_samples)
        features = extract_features(audio, cfg.sample_rate, cfg.frontend)
        tokens = encode(text)

        return {
            "features": torch.from_numpy(features),
            "tokens": torch.tensor(tokens, dtype=torch.int64),
            "n_frames": int(features.shape[0]),
            "n_tokens": int(len(tokens)),
        }


def collate(batch: list[dict]) -> dict:
    """Stack features, pad tokens to batch-max length with BLANK_INDEX.

    The CTC loss ignores positions beyond ``target_lengths``, so the
    specific pad value does not affect the loss — using the blank
    index is conventional but arbitrary.
    """
    if not batch:
        raise ValueError("cannot collate an empty batch")

    features = torch.stack([b["features"] for b in batch], dim=0)   # [B, T, F]
    n_frames = torch.tensor([b["n_frames"] for b in batch], dtype=torch.int64)
    n_tokens = torch.tensor([b["n_tokens"] for b in batch], dtype=torch.int64)

    max_l = int(n_tokens.max().item())
    if max_l == 0:
        tokens = torch.zeros((len(batch), 0), dtype=torch.int64)
    else:
        tokens = torch.full((len(batch), max_l), BLANK_INDEX, dtype=torch.int64)
        for i, b in enumerate(batch):
            n = b["n_tokens"]
            if n > 0:
                tokens[i, :n] = b["tokens"]

    return {
        "features": features,
        "tokens": tokens,
        "n_frames": n_frames,
        "n_tokens": n_tokens,
    }
