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

import numpy as np
import torch

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
    text: str, wpm: float, cfg: ValidationConfig
) -> np.ndarray:
    return render(
        text,
        operator=OperatorConfig(wpm=wpm),
        keying=cfg.keying,
        channel=None,
        freq=cfg.freq_hz,
        sample_rate=cfg.sample_rate,
    )


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
            # Retry on over-long text, same policy as the training stream.
            audio: np.ndarray | None = None
            text = ""
            for _ in range(cfg.max_text_retries):
                text = sample_text(rng, cfg.text_mix)
                audio = _render_one(text, wpm, cfg)
                if audio.size <= cfg.target_samples:
                    break
            assert audio is not None

            audio = _pad_or_truncate(audio, cfg.target_samples)
            features = extract_features(audio, cfg.sample_rate, cfg.frontend)
            tokens = encode(text)

            samples.append(
                ValidationSample(
                    features=torch.from_numpy(features),
                    tokens=torch.tensor(tokens, dtype=torch.int64),
                    text=text,
                    wpm=wpm,
                    n_frames=int(features.shape[0]),
                    n_tokens=len(tokens),
                )
            )
    return samples
