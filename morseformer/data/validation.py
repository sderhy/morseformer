"""Deterministic clean-audio validation set for Phase 2 training.

The training stream (``synthetic.SyntheticCWDataset``) samples WPM
uniformly in ``[16, 28]``; for validation we instead grid the WPM over
a fixed set of bins so that every run reports CER at the same operating
points. Everything else (text mix, channel = none, keying = raised
cosine, ideal operator timing) matches training exactly.

The set is small enough (~200 samples, ~40 MB of float32 features) to
keep in RAM and re-use across many validation passes per epoch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

from morse_synth.channel import ChannelConfig, apply_channel
from morse_synth.core import render
from morse_synth.keying import KeyingConfig
from morse_synth.operator import OperatorConfig
from morseformer.core.tokenizer import encode
from morseformer.data.synthetic import (
    DatasetConfig,
    _pad_or_truncate,
    _random_phase4_max_chars,
)
from morseformer.data.text import (
    DEFAULT_MIX,
    TextMix,
    sample_random_chars_phase4,
    sample_text,
)
from morseformer.features import FrontendConfig, extract_features


@dataclass
class ValidationConfig:
    """Hyperparameters for the validation-set builder.

    Defaults are chosen so that a bare ``ValidationConfig()`` keeps the
    historical Phase-2/3 validation distribution byte-for-byte stable:
    ideal operator timing for ``build_clean_validation`` /
    ``build_snr_ladder_validation`` and the hardcoded Phase-3.1 channel
    envelope for ``build_realistic_ladder_validation``. Use
    :meth:`matching` to inherit the *actual* training distribution from
    a :class:`DatasetConfig` — that is what closes the test/train
    visibility gap documented in ``project_phase4_0b_result``.
    """

    n_per_wpm: int = 40
    wpm_bins: tuple[float, ...] = (16.0, 20.0, 22.0, 25.0, 28.0)
    target_duration_s: float = 6.0
    sample_rate: int = 8000
    freq_hz: float = 600.0
    text_mix: TextMix = field(default_factory=lambda: DEFAULT_MIX)
    frontend: FrontendConfig = field(default_factory=FrontendConfig)
    keying: KeyingConfig = field(
        default_factory=lambda: KeyingConfig(shape="raised_cosine", rise_ms=5.0)
    )
    seed: int = 20_260_418
    max_text_retries: int = 5

    # Phase 4.0 — match the quiet-zone padding from the training
    # ``DatasetConfig``. Defaults stay zero so 3.x validation sets are
    # bit-identical to before. ``ValidationConfig.matching()``
    # propagates the training values automatically.
    pre_quiet_zone_range_s: tuple[float, float] = (0.0, 0.0)
    post_quiet_zone_min_s: float = 0.0

    # Operator timing envelope — mirrors ``DatasetConfig``. Defaults are
    # the "ideal mechanical" values so historical
    # ``build_clean_validation`` / ``build_snr_ladder_validation`` output
    # is unchanged when the user does not call :meth:`matching`.
    operator_element_jitter_range: tuple[float, float] = (0.0, 0.0)
    operator_gap_jitter_range: tuple[float, float] = (0.0, 0.0)
    operator_dash_dot_ratio_range: tuple[float, float] = (3.0, 3.0)
    operator_gap_inflation_range: tuple[float, float] = (1.0, 1.0)
    operator_word_gap_inflation_range: tuple[float, float] = (1.0, 1.0)
    operator_run_on_pairs: tuple[tuple[str, str, float], ...] = ()

    # Realistic-channel envelope — mirrors ``DatasetConfig``. Defaults
    # match the numbers that ``_render_one_realistic`` hardcoded before
    # this extension, so a bare ``ValidationConfig()`` still produces
    # the Phase 3.1 ladder used by every checkpoint up to v0.6.3.
    freq_offset_range_hz: tuple[float, float] = (-50.0, 50.0)
    qsb_rate_range_hz: tuple[float, float] = (0.05, 1.0)
    qsb_depth_range_db: tuple[float, float] = (0.0, 15.0)
    qrn_rate_range_per_sec: tuple[float, float] = (0.0, 1.0)
    carrier_drift_sigma_range_hz_per_s: tuple[float, float] = (0.0, 1.0)
    qrm_probability: float = 0.25
    qrm_offset_range_hz: tuple[float, float] = (-300.0, 300.0)
    qrm_rel_db_range: tuple[float, float] = (-18.0, -8.0)

    # Empty-sample / post-emission-silence branches — propagated from
    # ``DatasetConfig`` for visibility, even though
    # ``build_noise_only_validation`` does not consume the probabilities
    # (it forces the modes explicitly). ``empty_sample_pseudo_morse_enabled``
    # *does* gate a 4th mode in that builder so the Phase 5.6 sub-
    # distribution is testable.
    empty_sample_probability: float = 0.0
    empty_sample_pseudo_morse_enabled: bool = False
    post_emission_silence_probability: float = 0.0
    post_emission_silence_text_chars: tuple[int, int] = (1, 5)

    @property
    def target_samples(self) -> int:
        return int(round(self.target_duration_s * self.sample_rate))

    @property
    def total_size(self) -> int:
        return self.n_per_wpm * len(self.wpm_bins)

    @classmethod
    def matching(cls, ds_cfg: DatasetConfig, **overrides) -> ValidationConfig:
        """Build a ``ValidationConfig`` whose audio pipeline matches
        ``ds_cfg`` exactly. Use this so the val set sees the same
        synth / front-end / operator / channel settings as the training
        stream — fixes the long-standing gap where the val measured
        "ideal CW" while training was jittered/noisy (see
        ``project_phase4_0b_result``).
        """
        defaults = dict(
            target_duration_s=ds_cfg.target_duration_s,
            sample_rate=ds_cfg.sample_rate,
            freq_hz=ds_cfg.freq_hz,
            text_mix=ds_cfg.text_mix,
            frontend=ds_cfg.frontend,
            keying=ds_cfg.keying,
            max_text_retries=ds_cfg.max_text_retries,
            pre_quiet_zone_range_s=ds_cfg.pre_quiet_zone_range_s,
            post_quiet_zone_min_s=ds_cfg.post_quiet_zone_min_s,
            operator_element_jitter_range=ds_cfg.operator_element_jitter_range,
            operator_gap_jitter_range=ds_cfg.operator_gap_jitter_range,
            operator_dash_dot_ratio_range=ds_cfg.operator_dash_dot_ratio_range,
            operator_gap_inflation_range=ds_cfg.operator_gap_inflation_range,
            operator_word_gap_inflation_range=ds_cfg.operator_word_gap_inflation_range,
            operator_run_on_pairs=ds_cfg.operator_run_on_pairs,
            freq_offset_range_hz=ds_cfg.freq_offset_range_hz,
            qsb_rate_range_hz=ds_cfg.qsb_rate_range_hz,
            qsb_depth_range_db=ds_cfg.qsb_depth_range_db,
            qrn_rate_range_per_sec=ds_cfg.qrn_rate_range_per_sec,
            carrier_drift_sigma_range_hz_per_s=ds_cfg.carrier_drift_sigma_range_hz_per_s,
            qrm_probability=ds_cfg.qrm_probability,
            qrm_offset_range_hz=ds_cfg.qrm_offset_range_hz,
            qrm_rel_db_range=ds_cfg.qrm_rel_db_range,
            empty_sample_probability=ds_cfg.empty_sample_probability,
            empty_sample_pseudo_morse_enabled=ds_cfg.empty_sample_pseudo_morse_enabled,
            post_emission_silence_probability=ds_cfg.post_emission_silence_probability,
            post_emission_silence_text_chars=ds_cfg.post_emission_silence_text_chars,
        )
        defaults.update(overrides)
        return cls(**defaults)


@dataclass
class ValidationSample:
    """One fully-rendered validation item, held in memory."""

    features: torch.Tensor     # [T, 1] float32
    tokens: torch.Tensor       # [L] int64
    text: str                  # original text (for CER reporting)
    wpm: float                 # the forced WPM this sample was rendered at
    snr_db: float              # math.inf for clean samples
    n_frames: int
    n_tokens: int

    def as_batch_item(self) -> dict:
        """Return the same dict layout as ``SyntheticCWDataset`` yields,
        so the existing ``collate`` helper works unchanged."""
        return {
            "features": self.features,
            "tokens": self.tokens,
            "n_frames": self.n_frames,
            "n_tokens": self.n_tokens,
        }


def _uniform_or_zero(
    rng: np.random.Generator, rng_range: tuple[float, float]
) -> float:
    """Mirror of :meth:`SyntheticCWDataset._uniform_or_zero` so val and
    train sample channel knobs through the exact same convention."""
    lo, hi = rng_range
    return float(rng.uniform(lo, hi)) if hi > lo else 0.0


def _sample_operator(
    rng: np.random.Generator, cfg: ValidationConfig, wpm: float
) -> OperatorConfig:
    """Per-sample :class:`OperatorConfig`, mirroring
    :meth:`SyntheticCWDataset._sample_operator`. Lets the validation
    audio carry the same operator-timing envelope as training when
    :meth:`ValidationConfig.matching` was used; falls through to ideal
    mechanical timing when the ranges are degenerate (legacy
    behaviour for bare ``ValidationConfig()``).
    """
    lo_e, hi_e = cfg.operator_element_jitter_range
    lo_g, hi_g = cfg.operator_gap_jitter_range
    element_jitter = 0.0 if hi_e <= lo_e else float(rng.uniform(lo_e, hi_e))
    gap_jitter = 0.0 if hi_g <= lo_g else float(rng.uniform(lo_g, hi_g))
    lo_r, hi_r = cfg.operator_dash_dot_ratio_range
    dash_dot_ratio = 3.0 if hi_r <= lo_r else float(rng.uniform(lo_r, hi_r))
    lo_i, hi_i = cfg.operator_gap_inflation_range
    gap_inflation = 1.0 if hi_i <= lo_i else float(rng.uniform(lo_i, hi_i))
    lo_w, hi_w = cfg.operator_word_gap_inflation_range
    word_gap_inflation = (
        1.0 if hi_w <= lo_w else float(rng.uniform(lo_w, hi_w))
    )
    op_seed = int(rng.integers(0, 2**31 - 1))
    return OperatorConfig(
        wpm=wpm,
        element_jitter=element_jitter,
        gap_jitter=gap_jitter,
        dash_dot_ratio=dash_dot_ratio,
        gap_inflation=gap_inflation,
        word_gap_inflation=word_gap_inflation,
        run_on_pairs=cfg.operator_run_on_pairs,
        seed=op_seed,
    )


def _render_one(
    text: str,
    wpm: float,
    cfg: ValidationConfig,
    snr_db: float,
    rx_filter_bw: float | None,
    channel_seed: int,
    rng: np.random.Generator,
    pre_silence_samples: int = 0,
) -> np.ndarray:
    """Render one validation sample with optional quiet-zone prepend.

    The CW is rendered first as a clean waveform; we then prepend
    ``pre_silence_samples`` of zeros and pad / truncate to the
    target window. Any AWGN + RX-filter channel is applied last,
    over the full buffer (silence + CW + tail), so the pre-CW silence
    inherits the same channel artefacts as in training — this is the
    "noisy quiet zone" the training pipeline produces.

    ``rng`` drives per-sample operator timing — when ``cfg`` came from
    :meth:`ValidationConfig.matching`, the rendered audio carries the
    same jitter / dash-dot-ratio / gap-inflation envelope as the
    training stream.
    """
    operator = _sample_operator(rng, cfg, wpm)
    clean = render(
        text,
        operator=operator,
        keying=cfg.keying,
        channel=None,
        freq=cfg.freq_hz,
        sample_rate=cfg.sample_rate,
    ).astype(np.float32)
    if pre_silence_samples > 0:
        clean = np.concatenate(
            [np.zeros(pre_silence_samples, dtype=np.float32), clean]
        )
    clean = _pad_or_truncate(clean, cfg.target_samples)

    if math.isfinite(snr_db) or rx_filter_bw is not None:
        channel = ChannelConfig(
            snr_db=snr_db,
            rx_filter_bw=rx_filter_bw,
            rx_filter_centre=cfg.freq_hz,
            seed=channel_seed,
        )
        return apply_channel(clean, cfg.sample_rate, channel)
    return clean


def _one_sample(
    rng: np.random.Generator,
    cfg: ValidationConfig,
    wpm: float,
    snr_db: float,
    rx_filter_bw: float | None,
    *,
    realistic: bool = False,
) -> ValidationSample:
    """Render one validation sample.

    If ``realistic=True``, the channel also carries QSB, QRN, carrier
    drift, a per-sample tone-frequency offset (±50 Hz) and a 25 %
    chance of QRM — matching the Phase 3.1 training distribution.
    All channel magnitudes are drawn deterministically from ``rng``,
    so the sample is reproducible given the caller's seed.
    """
    # Use duration-estimate pre-filtering so that labels always match
    # the rendered audio — same policy as the training stream.
    from morseformer.data.synthetic import (
        _FALLBACK_SHORT_TEXTS,
        estimate_cw_duration_s,
    )

    # Reserve room for the quiet zones (zero-effect in 3.x configs).
    pre_max = max(cfg.pre_quiet_zone_range_s[1], 0.0)
    post_min = max(cfg.post_quiet_zone_min_s, 0.0)
    budget = (cfg.target_duration_s - pre_max - post_min) * 0.9

    if cfg.text_mix.is_random_phase4_only():
        # Phase 4.0 fast path — mirror the training-side dispatch:
        # call the random-char sampler directly with a wpm-derived
        # max_chars so we never fall back to Q-codes (which would
        # contaminate the no-prior eval the same way they would
        # contaminate training).
        text = sample_random_chars_phase4(
            rng, max_chars=_random_phase4_max_chars(wpm, budget)
        )
    else:
        text = ""
        for _ in range(cfg.max_text_retries):
            candidate = sample_text(rng, cfg.text_mix)
            if estimate_cw_duration_s(candidate, wpm) <= budget:
                text = candidate
                break
        if not text:
            fallbacks = _FALLBACK_SHORT_TEXTS()
            text = fallbacks[int(rng.integers(0, len(fallbacks)))]

    # Per-sample quiet-zone prepend (deterministic via rng so the val
    # set stays reproducible across runs).
    pre_lo, pre_hi = cfg.pre_quiet_zone_range_s
    if pre_hi > pre_lo:
        pre_silence_s = float(rng.uniform(pre_lo, pre_hi))
    else:
        pre_silence_s = 0.0
    pre_silence_samples = int(round(pre_silence_s * cfg.sample_rate))

    channel_seed = int(rng.integers(0, 2**31 - 1))
    if realistic:
        audio = _render_one_realistic(
            text, wpm, cfg, snr_db, rx_filter_bw, channel_seed, rng,
            pre_silence_samples=pre_silence_samples,
        )
    else:
        audio = _render_one(
            text, wpm, cfg, snr_db, rx_filter_bw, channel_seed, rng,
            pre_silence_samples=pre_silence_samples,
        )
    audio = _pad_or_truncate(audio, cfg.target_samples)
    features = extract_features(audio, cfg.sample_rate, cfg.frontend)
    tokens = encode(text)

    return ValidationSample(
        features=torch.from_numpy(features),
        tokens=torch.tensor(tokens, dtype=torch.int64),
        text=text,
        wpm=wpm,
        snr_db=snr_db,
        n_frames=int(features.shape[0]),
        n_tokens=len(tokens),
    )


def _render_one_realistic(
    text: str,
    wpm: float,
    cfg: ValidationConfig,
    snr_db: float,
    rx_filter_bw: float | None,
    channel_seed: int,
    rng: np.random.Generator,
    pre_silence_samples: int = 0,
) -> np.ndarray:
    """Render a sample with the full realistic channel stack.

    Mirrors the augmentations used by the training dataset (freq jitter,
    QSB, QRN, carrier drift, optional QRM) using ranges drawn from
    ``cfg`` so the val distribution tracks the training distribution
    when ``cfg`` came from :meth:`ValidationConfig.matching`. Defaults
    on ``ValidationConfig`` preserve the Phase 3.1 numbers this function
    historically hardcoded.

    Operator timing is sampled per-utterance from ``cfg`` (same
    convention as :func:`_render_one`).

    ``pre_silence_samples`` (Phase 4.0) prepends a silent quiet zone
    to the primary CW; QRM and channel are applied to the full buffer
    so the quiet zone inherits the same channel artefacts as training.
    """
    operator = _sample_operator(rng, cfg, wpm)
    freq = cfg.freq_hz + _uniform_or_zero(rng, cfg.freq_offset_range_hz)
    qsb_rate = _uniform_or_zero(rng, cfg.qsb_rate_range_hz)
    qsb_depth = _uniform_or_zero(rng, cfg.qsb_depth_range_db)
    qrn_rate = _uniform_or_zero(rng, cfg.qrn_rate_range_per_sec)
    drift = _uniform_or_zero(rng, cfg.carrier_drift_sigma_range_hz_per_s)

    clean = render(
        text,
        operator=operator,
        keying=cfg.keying,
        channel=None,
        freq=freq,
        sample_rate=cfg.sample_rate,
    ).astype(np.float32)
    if pre_silence_samples > 0:
        clean = np.concatenate(
            [np.zeros(pre_silence_samples, dtype=np.float32), clean]
        )
    clean = _pad_or_truncate(clean, cfg.target_samples)

    if cfg.qrm_probability > 0.0 and rng.random() < cfg.qrm_probability:
        qrm_wpm = float(rng.uniform(cfg.wpm_bins[0], cfg.wpm_bins[-1]))
        from morseformer.data.synthetic import (
            _FALLBACK_SHORT_TEXTS,
        )
        fallbacks = _FALLBACK_SHORT_TEXTS()
        qrm_text = fallbacks[int(rng.integers(0, len(fallbacks)))]
        qrm_freq = cfg.freq_hz + _uniform_or_zero(rng, cfg.qrm_offset_range_hz)
        qrm_rel_db = _uniform_or_zero(rng, cfg.qrm_rel_db_range)
        qrm_operator = _sample_operator(rng, cfg, qrm_wpm)
        qrm = render(
            qrm_text,
            operator=qrm_operator,
            keying=cfg.keying,
            channel=None,
            freq=qrm_freq,
            sample_rate=cfg.sample_rate,
        )
        qrm = _pad_or_truncate(qrm.astype(np.float32), cfg.target_samples)
        clean = clean + qrm * (10.0 ** (qrm_rel_db / 20.0))

    channel = ChannelConfig(
        snr_db=snr_db,
        rx_filter_bw=rx_filter_bw,
        rx_filter_centre=cfg.freq_hz,
        qsb_rate_hz=qsb_rate,
        qsb_depth_db=qsb_depth,
        qrn_rate_per_sec=qrn_rate,
        carrier_drift_hz_per_s=drift,
        seed=channel_seed,
    )
    return apply_channel(clean, cfg.sample_rate, channel)


def build_clean_validation(
    cfg: ValidationConfig | None = None,
) -> list[ValidationSample]:
    """Build a deterministic, in-memory validation set.

    Structure: ``n_per_wpm`` samples per WPM bin, text drawn under the
    same mix as training, rendered clean (no channel, ideal timing).
    The RNG is derived from ``cfg.seed`` and the WPM bin index, so
    individual bins are reproducible independently and the total set is
    bit-for-bit identical across runs with the same seed.
    """
    cfg = cfg or ValidationConfig()
    samples: list[ValidationSample] = []
    for bin_idx, wpm in enumerate(cfg.wpm_bins):
        rng = np.random.default_rng(cfg.seed + bin_idx * 10_007)
        for _ in range(cfg.n_per_wpm):
            samples.append(
                _one_sample(rng, cfg, wpm,
                            snr_db=math.inf, rx_filter_bw=None)
            )
    return samples


def build_snr_ladder_validation(
    snrs_db: tuple[float, ...] = (20.0, 10.0, 5.0, 0.0, -5.0, -10.0),
    *,
    cfg: ValidationConfig | None = None,
    rx_filter_bw: float | None = 500.0,
    n_per_cell: int | None = None,
) -> list[ValidationSample]:
    """Build a deterministic (WPM × SNR)-laddered validation set.

    Every (wpm_bin, snr) cell gets the same number of samples — by
    default ``cfg.n_per_wpm`` — so the total size is
    ``len(wpm_bins) * len(snrs_db) * n_per_cell``. Intended for
    tracking the Phase 2.1 benchmark (CER vs. SNR) during training.

    The default SNR list (+20 down to −10 dB) matches the existing
    rule-based baseline ladder in ``eval.snr_ladder`` so the numbers
    are directly comparable.
    """
    cfg = cfg or ValidationConfig()
    n = n_per_cell if n_per_cell is not None else cfg.n_per_wpm
    samples: list[ValidationSample] = []

    for wpm_idx, wpm in enumerate(cfg.wpm_bins):
        for snr_idx, snr in enumerate(snrs_db):
            rng_seed = cfg.seed + wpm_idx * 10_007 + snr_idx * 33_091
            rng = np.random.default_rng(rng_seed)
            for _ in range(n):
                samples.append(
                    _one_sample(rng, cfg, wpm,
                                snr_db=snr, rx_filter_bw=rx_filter_bw,
                                realistic=False)
                )
    return samples


def build_noise_only_validation(
    *,
    cfg: ValidationConfig | None = None,
    rx_filter_bw: float | None = 500.0,
    n_per_mode: int = 50,
) -> list[ValidationSample]:
    """Build a "no decodable signal" validation set with empty labels.

    The bench mirrors the empty-audio sampler used by
    :class:`SyntheticCWDataset`:

        1. Pure AWGN — quiet band, no signal.
        2. AWGN + QRN bursts — atmospheric clicks, no signal.
        3. Distant weak CW (SNR -35 to -25 dB) — real CW so faint it
           must be ignored.
        4. Pseudo-Morse pulse rhythm (only when
           ``cfg.empty_sample_pseudo_morse_enabled`` is True) — clean
           dots/dashes at the carrier with malformed inter-pulse gaps,
           labelled empty. Mirrors the Phase 5.6 sub-mode designed to
           kill the v0.5.1 digit-hallucination on rhythmic on-band
           noise.

    Every sample has ``n_tokens == 0`` and ``text == ""``. The intended
    metric is the mean number of characters emitted per sample —
    target ≈ 0. This is the false-positive rate that the
    anti-hallucination curriculum is designed to drive down.
    """
    cfg = cfg or ValidationConfig()
    samples: list[ValidationSample] = []
    target_samples = cfg.target_samples
    noise_rms = 0.1

    # Mode 0 — pure AWGN (+ optional RX filter).
    for i in range(n_per_mode):
        rng = np.random.default_rng(cfg.seed + 401_059 + i)
        audio = rng.normal(0.0, noise_rms, size=target_samples).astype(np.float32)
        if rx_filter_bw is not None and rx_filter_bw > 0:
            from morse_synth.channel import _apply_rx_filter
            audio = _apply_rx_filter(
                audio, cfg.sample_rate, rx_filter_bw, cfg.freq_hz
            ).astype(np.float32)
        samples.append(_empty_sample_from_audio(audio, cfg))

    # Mode 1 — AWGN + QRN bursts (+ optional RX filter).
    for i in range(n_per_mode):
        rng = np.random.default_rng(cfg.seed + 401_063 + i)
        audio = rng.normal(0.0, noise_rms, size=target_samples).astype(np.float32)
        from morse_synth.channel import _add_qrn
        audio = _add_qrn(
            audio,
            cfg.sample_rate,
            rate=float(rng.uniform(0.5, 3.0)),
            amp_db=-3.0,
            decay_ms=1.0,
            rng=rng,
        ).astype(np.float32)
        if rx_filter_bw is not None and rx_filter_bw > 0:
            from morse_synth.channel import _apply_rx_filter
            audio = _apply_rx_filter(
                audio, cfg.sample_rate, rx_filter_bw, cfg.freq_hz
            ).astype(np.float32)
        samples.append(_empty_sample_from_audio(audio, cfg))

    # Mode 2 — distant weak CW (real signal, SNR -35 to -25, label empty).
    from morse_synth.channel import apply_channel
    from morseformer.data.synthetic import _FALLBACK_SHORT_TEXTS
    for i in range(n_per_mode):
        rng = np.random.default_rng(cfg.seed + 401_069 + i)
        wpm = float(rng.choice(cfg.wpm_bins))
        text = sample_text(rng, cfg.text_mix)
        if not text:
            text = _FALLBACK_SHORT_TEXTS()[0]
        clean = render(
            text,
            operator=OperatorConfig(wpm=wpm),
            keying=cfg.keying,
            channel=None,
            freq=cfg.freq_hz,
            sample_rate=cfg.sample_rate,
        )
        clean = _pad_or_truncate(clean.astype(np.float32), target_samples)
        ch = ChannelConfig(
            snr_db=float(rng.uniform(-35.0, -25.0)),
            rx_filter_bw=rx_filter_bw,
            rx_filter_centre=cfg.freq_hz,
            seed=int(rng.integers(0, 2**31 - 1)),
        )
        audio = apply_channel(clean, cfg.sample_rate, ch).astype(np.float32)
        samples.append(_empty_sample_from_audio(audio, cfg))

    # Mode 3 — pseudo-Morse pulse rhythm (Phase 5.6). Only emitted when
    # the training distribution actually enables this branch, so the bench
    # stays bit-identical to its 3-mode form for legacy presets.
    if cfg.empty_sample_pseudo_morse_enabled:
        from morse_synth.channel import apply_channel as _apply_channel
        from morse_synth.keying import render_events
        wpm_lo = float(cfg.wpm_bins[0])
        wpm_hi = float(cfg.wpm_bins[-1])
        for i in range(n_per_mode):
            rng = np.random.default_rng(cfg.seed + 401_077 + i)
            wpm = float(rng.uniform(wpm_lo, wpm_hi))
            unit_s = 1.2 / wpm
            n_pulses = int(rng.integers(4, 26))
            events: list[tuple[bool, float]] = []
            for _ in range(n_pulses):
                on_units = (1.0 if rng.random() < 0.5 else 3.0) * float(
                    rng.uniform(0.85, 1.15)
                )
                off_units = float(rng.uniform(0.5, 8.0))
                events.append((True, on_units * unit_s))
                events.append((False, off_units * unit_s))
            clean = render_events(
                events,
                keying=cfg.keying,
                freq=cfg.freq_hz,
                sample_rate=cfg.sample_rate,
            ).astype(np.float32)
            clean = _pad_or_truncate(clean, target_samples)
            ch = ChannelConfig(
                snr_db=float(rng.uniform(5.0, 25.0)),
                rx_filter_bw=rx_filter_bw,
                rx_filter_centre=cfg.freq_hz,
                seed=int(rng.integers(0, 2**31 - 1)),
            )
            audio = _apply_channel(clean, cfg.sample_rate, ch).astype(
                np.float32
            )
            samples.append(_empty_sample_from_audio(audio, cfg))

    return samples


def _empty_sample_from_audio(
    audio: np.ndarray, cfg: ValidationConfig
) -> ValidationSample:
    """Wrap a noise-only waveform as a :class:`ValidationSample` with an
    empty label."""
    features = extract_features(audio, cfg.sample_rate, cfg.frontend)
    return ValidationSample(
        features=torch.from_numpy(features),
        tokens=torch.zeros(0, dtype=torch.int64),
        text="",
        wpm=0.0,
        snr_db=math.inf,           # distinguishes from any real-SNR sample
        n_frames=int(features.shape[0]),
        n_tokens=0,
    )


def build_realistic_ladder_validation(
    snrs_db: tuple[float, ...] = (20.0, 10.0, 5.0, 0.0, -5.0, -10.0),
    *,
    cfg: ValidationConfig | None = None,
    rx_filter_bw: float | None = 500.0,
    n_per_cell: int | None = None,
) -> list[ValidationSample]:
    """Like :func:`build_snr_ladder_validation` but the channel at each
    SNR also carries the full Phase 3.1 augmentation stack — carrier
    frequency offset, QSB, QRN, carrier drift, and a 25 % chance of a
    secondary co-channel signal (QRM). This bench is the **target
    metric** for Phase 3.1: the gain here is what the fine-tune is
    optimising for. The AWGN-only ladder from
    :func:`build_snr_ladder_validation` stays as the Phase 3.0
    regression guard.
    """
    cfg = cfg or ValidationConfig()
    n = n_per_cell if n_per_cell is not None else cfg.n_per_wpm
    samples: list[ValidationSample] = []
    for wpm_idx, wpm in enumerate(cfg.wpm_bins):
        for snr_idx, snr in enumerate(snrs_db):
            # A different seed offset from the AWGN ladder so the two
            # benches use independent random draws — otherwise the
            # realistic bench would simply be the AWGN bench with
            # channel noise added on top of the same text.
            rng_seed = cfg.seed + wpm_idx * 10_007 + snr_idx * 33_091 + 77_003
            rng = np.random.default_rng(rng_seed)
            for _ in range(n):
                samples.append(
                    _one_sample(rng, cfg, wpm,
                                snr_db=snr, rx_filter_bw=rx_filter_bw,
                                realistic=True)
                )
    return samples
