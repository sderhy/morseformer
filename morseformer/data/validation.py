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

from dataclasses import dataclass, field

import math

import numpy as np
import torch

from morse_synth.channel import ChannelConfig
from morse_synth.core import render
from morse_synth.keying import KeyingConfig
from morse_synth.operator import OperatorConfig
from morseformer.core.tokenizer import encode
from morseformer.data.synthetic import DatasetConfig, _pad_or_truncate
from morseformer.data.text import DEFAULT_MIX, TextMix, sample_text
from morseformer.features import FrontendConfig, extract_features


@dataclass
class ValidationConfig:
    """Hyperparameters for the validation-set builder."""

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

    @property
    def target_samples(self) -> int:
        return int(round(self.target_duration_s * self.sample_rate))

    @property
    def total_size(self) -> int:
        return self.n_per_wpm * len(self.wpm_bins)

    @classmethod
    def matching(cls, ds_cfg: DatasetConfig, **overrides) -> "ValidationConfig":
        """Build a ``ValidationConfig`` whose audio pipeline matches
        ``ds_cfg`` exactly. Use this so the val set sees the same
        synth / front-end settings as the training stream."""
        defaults = dict(
            target_duration_s=ds_cfg.target_duration_s,
            sample_rate=ds_cfg.sample_rate,
            freq_hz=ds_cfg.freq_hz,
            text_mix=ds_cfg.text_mix,
            frontend=ds_cfg.frontend,
            keying=ds_cfg.keying,
            max_text_retries=ds_cfg.max_text_retries,
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


def _render_one(
    text: str,
    wpm: float,
    cfg: ValidationConfig,
    snr_db: float,
    rx_filter_bw: float | None,
    channel_seed: int,
) -> np.ndarray:
    channel: ChannelConfig | None = None
    if math.isfinite(snr_db) or rx_filter_bw is not None:
        channel = ChannelConfig(
            snr_db=snr_db,
            rx_filter_bw=rx_filter_bw,
            rx_filter_centre=cfg.freq_hz,
            seed=channel_seed,
        )
    return render(
        text,
        operator=OperatorConfig(wpm=wpm),
        keying=cfg.keying,
        channel=channel,
        freq=cfg.freq_hz,
        sample_rate=cfg.sample_rate,
    )


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

    budget = cfg.target_duration_s * 0.9
    text = ""
    for _ in range(cfg.max_text_retries):
        candidate = sample_text(rng, cfg.text_mix)
        if estimate_cw_duration_s(candidate, wpm) <= budget:
            text = candidate
            break
    if not text:
        fallbacks = _FALLBACK_SHORT_TEXTS()
        text = fallbacks[int(rng.integers(0, len(fallbacks)))]

    channel_seed = int(rng.integers(0, 2**31 - 1))
    if realistic:
        audio = _render_one_realistic(
            text, wpm, cfg, snr_db, rx_filter_bw, channel_seed, rng
        )
    else:
        audio = _render_one(text, wpm, cfg, snr_db, rx_filter_bw, channel_seed)
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
) -> np.ndarray:
    """Render a sample with the full Phase 3.1 channel stack.

    Mirrors the augmentations used by the Phase 3.1 training dataset
    (freq jitter, QSB, QRN, carrier drift, and a 25 % chance of QRM)
    but with deterministic sampling driven by the provided ``rng`` so
    the validation set is reproducible across runs.
    """
    from morse_synth.channel import apply_channel

    freq = cfg.freq_hz + float(rng.uniform(-50.0, 50.0))
    qsb_rate = float(rng.uniform(0.05, 1.0))
    qsb_depth = float(rng.uniform(0.0, 15.0))
    qrn_rate = float(rng.uniform(0.0, 1.0))
    drift = float(rng.uniform(0.0, 1.0))

    clean = render(
        text,
        operator=OperatorConfig(wpm=wpm),
        keying=cfg.keying,
        channel=None,
        freq=freq,
        sample_rate=cfg.sample_rate,
    )
    clean = _pad_or_truncate(clean.astype(np.float32), cfg.target_samples)

    if rng.random() < 0.25:
        qrm_wpm = float(rng.uniform(cfg.wpm_bins[0], cfg.wpm_bins[-1]))
        from morseformer.data.synthetic import (
            _FALLBACK_SHORT_TEXTS,
        )
        fallbacks = _FALLBACK_SHORT_TEXTS()
        qrm_text = fallbacks[int(rng.integers(0, len(fallbacks)))]
        qrm_freq = cfg.freq_hz + float(rng.uniform(-300.0, 300.0))
        qrm_rel_db = float(rng.uniform(-18.0, -8.0))
        qrm = render(
            qrm_text,
            operator=OperatorConfig(wpm=qrm_wpm),
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

    The bench mirrors the 3-mode empty-audio sampler used by
    :class:`SyntheticCWDataset` in Phase 3.2:

        1. Pure AWGN — quiet band, no signal.
        2. AWGN + QRN bursts — atmospheric clicks, no signal.
        3. Distant weak CW (SNR -35 to -25 dB) — real CW so faint it
           must be ignored.

    Every sample has ``n_tokens == 0`` and ``text == ""``. The intended
    metric is the mean number of characters emitted per sample —
    target ≈ 0. This is the false-positive rate that Phase 3.2's
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
