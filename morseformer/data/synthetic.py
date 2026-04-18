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
from morseformer.core.tokenizer import BLANK_INDEX, encode
from morseformer.data.text import DEFAULT_MIX, TextMix, sample_text
from morseformer.features import FrontendConfig, extract_features


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

    def _sample_channel(
        self, rng: np.random.Generator
    ) -> ChannelConfig | None:
        cfg = self.cfg
        if cfg.channel_probability <= 0.0:
            return None
        if cfg.channel_probability < 1.0 and rng.random() >= cfg.channel_probability:
            return None
        lo, hi = cfg.snr_db_range
        snr_db = float(rng.uniform(lo, hi)) if hi > lo else float(lo)
        ch_seed = int(rng.integers(0, 2**31 - 1))
        return ChannelConfig(
            snr_db=snr_db,
            rx_filter_bw=cfg.rx_filter_bw,
            rx_filter_centre=cfg.freq_hz,
            seed=ch_seed,
        )

    def _generate_one(self, rng: np.random.Generator) -> dict:
        cfg = self.cfg
        wpm = float(rng.uniform(*cfg.wpm_range))

        text = ""
        audio: np.ndarray | None = None
        for _ in range(cfg.max_text_retries):
            text = sample_text(rng, cfg.text_mix)
            operator = self._sample_operator(rng, wpm)
            channel = self._sample_channel(rng)
            audio = render(
                text,
                operator=operator,
                keying=cfg.keying,
                channel=channel,
                freq=cfg.freq_hz,
                sample_rate=cfg.sample_rate,
            )
            if audio.size <= cfg.target_samples:
                break
        assert audio is not None

        audio = _pad_or_truncate(audio, cfg.target_samples)
        features = extract_features(audio, cfg.sample_rate, cfg.frontend)  # [T, 1]
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
