"""Real-audio CW dataset (Phase 3.7).

Streams ``(audio_chunk, label)`` pairs from a JSONL produced by
``scripts/align_ebook_cw.py``. The JSONL maps timestamp ranges in a
master wav file to ground-truth text labels obtained by char-level
alignment of the model's own decode against the source ebook.

Each record on the JSONL has the schema::

    {
      "audio_path": "data/real/alice_chapter1.wav",
      "chunk_idx": 12,
      "start_s": 72.0,
      "end_s": 78.0,
      "label": "ALICE WAS BEGINNING TO GET",
      "decoded": "AA IICE WAS BEGINNIT EG TO GET",
      "score": 0.78
    }

The loader keeps each referenced wav file in memory (typical ebook2cw
audio is < 100 MB at 8 kHz mono int16) so per-sample slicing is just a
numpy view, not a disk seek.

Mixing with the synthetic stream: pair this with
:class:`SyntheticCWDataset` via :class:`MixedCWDataset`, which draws
each item from the real-audio source with a configurable probability.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile
from torch.utils.data import IterableDataset

from morseformer.core.tokenizer import encode
from morseformer.data.synthetic import _pad_or_truncate, _worker_seed
from morseformer.features import FrontendConfig, extract_features


@dataclass
class RealAudioConfig:
    """Hyperparameters for :class:`RealAudioCWDataset`.

    Attributes:
        jsonl_path:        Path to the aligned JSONL file.
        target_duration_s: Audio window length, must match the synthetic
                           stream so batches stack.
        sample_rate:       Audio sample rate. The wav files referenced
                           by ``audio_path`` are expected to already be
                           at this rate (the converter step in
                           ``scripts/align_ebook_cw.py`` ensures it).
        freq_hz:           Carrier frequency for the front-end.
        frontend:          DSP front-end config.
        score_threshold:   Drop records below this alignment score.
                           Default 0.5 keeps almost everything; raise
                           to 0.7+ for a stricter dataset.
        seed:              Base RNG seed (combined with worker id).

        word_gap_augment_prob:
            Phase 10 — probability of inserting an inflated silence at
            one random word boundary of the chunk's audio before
            yielding. Real operators do not pause 6× between words, so
            the unaugmented stream collapses the model's
            ``operator_word_gap_inflation`` learning from the synthetic
            curriculum (cf the +7.9 pp word_gap_inflation_6× regression
            measured on Phase 8). 0.0 disables augmentation (legacy
            behaviour).
        word_gap_augment_inflation_range:
            Range for the per-augmentation inflation factor. Each
            augmented sample picks ``inflation = U(lo, hi)`` and
            inserts ``inflation × 0.4 s`` (= one nominal 20-WPM word
            gap) of silence at the chosen boundary, then truncates the
            chunk back to ``target_samples``.
    """

    jsonl_path: Path
    target_duration_s: float = 6.0
    sample_rate: int = 8000
    freq_hz: float = 600.0
    frontend: FrontendConfig = field(default_factory=FrontendConfig)
    score_threshold: float = 0.5
    seed: int = 0

    word_gap_augment_prob: float = 0.0
    word_gap_augment_inflation_range: tuple[float, float] = (1.5, 5.0)

    @property
    def target_samples(self) -> int:
        return int(round(self.target_duration_s * self.sample_rate))


def _load_wav_to_float32(path: Path, target_sr: int) -> np.ndarray:
    """Read a wav file and return a float32 array normalised to ``[-1, 1]``.

    Refuses to silently resample — the JSONL is built against a fixed
    sample-rate-converted dataset, so any mismatch is a setup error.
    """
    sr, audio = wavfile.read(str(path))
    if sr != target_sr:
        raise RuntimeError(
            f"{path}: expected sample_rate {target_sr}, got {sr}"
        )
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    if np.issubdtype(audio.dtype, np.integer):
        audio = audio / float(np.iinfo(audio.dtype).max)
    max_abs = float(np.max(np.abs(audio)))
    if max_abs > 1.5:
        audio = audio / max_abs
    return audio


class RealAudioCWDataset(IterableDataset):
    """Infinite stream of real-audio CW samples from an aligned JSONL.

    Records are filtered on construction by ``score_threshold``, then
    drawn uniformly at random for each yielded item. Each ``__iter__``
    creates an independent RNG seeded by ``(cfg.seed, worker_id)`` so
    multi-worker DataLoaders see disjoint streams.
    """

    def __init__(self, cfg: RealAudioConfig) -> None:
        super().__init__()
        self.cfg = cfg
        with cfg.jsonl_path.open(encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]
        records = [
            r for r in records if float(r.get("score", 1.0)) >= cfg.score_threshold
        ]
        if not records:
            raise ValueError(
                f"No records in {cfg.jsonl_path} after applying "
                f"score_threshold={cfg.score_threshold}."
            )
        self.records = records

        # Pre-load each referenced wav file once. Audio files are small
        # (< 100 MB each at 8 kHz mono float32), so caching them in
        # memory avoids per-sample disk I/O.
        self._audio_cache: dict[str, np.ndarray] = {}
        for r in records:
            path = r["audio_path"]
            if path not in self._audio_cache:
                self._audio_cache[path] = _load_wav_to_float32(
                    Path(path), cfg.sample_rate
                )

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = 0 if worker_info is None else int(worker_info.id)
        rng = np.random.default_rng(_worker_seed(self.cfg.seed, worker_id))
        n = len(self.records)
        target_samples = self.cfg.target_samples
        while True:
            i = int(rng.integers(0, n))
            rec = self.records[i]
            audio_full = self._audio_cache[rec["audio_path"]]
            start = int(round(float(rec["start_s"]) * self.cfg.sample_rate))
            end = int(round(float(rec["end_s"]) * self.cfg.sample_rate))
            audio = audio_full[start:end].astype(np.float32, copy=True)
            label = rec["label"]
            if (
                self.cfg.word_gap_augment_prob > 0.0
                and rng.random() < self.cfg.word_gap_augment_prob
            ):
                audio = _augment_word_gap(
                    audio, label,
                    sample_rate=self.cfg.sample_rate,
                    rng=rng,
                    inflation_range=self.cfg.word_gap_augment_inflation_range,
                )
            audio = _pad_or_truncate(audio, target_samples)
            features = extract_features(audio, self.cfg.sample_rate, self.cfg.frontend)
            tokens = encode(label)
            yield {
                "features": torch.from_numpy(features),
                "tokens": torch.tensor(tokens, dtype=torch.int64),
                "n_frames": int(features.shape[0]),
                "n_tokens": int(len(tokens)),
            }


def _augment_word_gap(
    audio: np.ndarray,
    label: str,
    *,
    sample_rate: int,
    rng: np.random.Generator,
    inflation_range: tuple[float, float],
    nominal_gap_s: float = 0.4,
) -> np.ndarray:
    """Insert an inflated silent gap at one random word boundary of
    ``audio``. Returns a new array; caller is responsible for
    pad/truncate to the target window.

    The word boundary is picked from the literal SPACE positions of
    ``label``; the audio time corresponding to that boundary is
    estimated by linear interpolation (uniform-speed assumption — the
    aligned chunks are short enough, ~6 s, that this is a reasonable
    approximation). The inflation factor is sampled per call from
    ``inflation_range`` and multiplied by ``nominal_gap_s`` (one
    word-gap-equivalent at ~20 WPM by default).

    Phase 10 motivation: the un-augmented real-audio stream has
    word gaps that hover around 1× (normal operator timing), while the
    synthetic stream uses ``operator_word_gap_inflation_range`` up to
    8×. The two distributions are contradictory and the model
    collapses toward the narrow end (cf the +7.9 pp regression on
    ``word_gap_inflation_6x`` measured for Phase 8 / 8a / 9). With
    this augmentation, the real stream also exposes the inflated
    end so the model retains the Phase-5.5 word-gap robustness while
    still benefiting from real-operator acoustic cues.
    """
    if " " not in label or len(audio) == 0:
        return audio
    boundaries = [i for i, c in enumerate(label) if c == " "]
    if not boundaries:
        return audio
    bi = boundaries[int(rng.integers(0, len(boundaries)))]
    audio_pos = int(round(bi / len(label) * len(audio)))
    audio_pos = max(0, min(audio_pos, len(audio)))
    inflation = float(rng.uniform(*inflation_range))
    gap_samples = int(round(inflation * nominal_gap_s * sample_rate))
    silence = np.zeros(gap_samples, dtype=audio.dtype)
    return np.concatenate([audio[:audio_pos], silence, audio[audio_pos:]])


class MixedCWDataset(IterableDataset):
    """Round-robin mix of two infinite IterableDatasets.

    Each yielded item comes from the real-audio dataset with probability
    ``real_probability``, otherwise from the synthetic dataset. Using
    a per-iter RNG keyed by worker id keeps the mix deterministic but
    distinct across workers.
    """

    def __init__(
        self,
        synthetic: IterableDataset,
        real_audio: IterableDataset,
        real_probability: float = 0.2,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if not 0.0 <= real_probability <= 1.0:
            raise ValueError(
                f"real_probability must be in [0, 1], got {real_probability}"
            )
        self.synthetic = synthetic
        self.real_audio = real_audio
        self.real_probability = real_probability
        self.seed = seed

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = 0 if worker_info is None else int(worker_info.id)
        rng = np.random.default_rng(_worker_seed(self.seed, worker_id))
        synth_it = iter(self.synthetic)
        real_it = iter(self.real_audio)
        while True:
            if rng.random() < self.real_probability:
                yield next(real_it)
            else:
                yield next(synth_it)
