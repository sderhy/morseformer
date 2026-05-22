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

from morseformer.core.tokenizer import SPACE_INDEX, encode
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
            tokens_override: list[int] | None = None
            if (
                self.cfg.word_gap_augment_prob > 0.0
                and rng.random() < self.cfg.word_gap_augment_prob
            ):
                audio, tokens_override = _augment_word_gap(
                    audio, label,
                    sample_rate=self.cfg.sample_rate,
                    rng=rng,
                    inflation_range=self.cfg.word_gap_augment_inflation_range,
                    tokens=rec.get("tokens"),
                    char_starts_s=rec.get("char_starts_s"),
                    target_samples=target_samples,
                )
            audio = _pad_or_truncate(audio, target_samples)
            features = extract_features(audio, self.cfg.sample_rate, self.cfg.frontend)
            tokens = tokens_override if tokens_override is not None else encode(label)
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
    tokens: list[int] | None = None,
    char_starts_s: list[float] | None = None,
    target_samples: int | None = None,
) -> tuple[np.ndarray, list[int] | None]:
    """Insert an inflated silent gap at one random word boundary of
    ``audio``. Returns ``(augmented_audio, tokens_to_use)`` where
    ``tokens_to_use`` reflects any label trimming triggered by audio
    truncation.

    When ``tokens`` and ``char_starts_s`` are provided (forced-aligned
    JSONLs from ``scripts/force_align_real_qso.py``), the insertion
    point is chosen *inside the true inter-word silence*: midway
    between the SPACE token's onset and the next character's onset.

    When ``target_samples`` is provided AND alignment data is
    available, audio that exceeds the target after insertion is
    truncated AND the tokens whose audio start falls beyond the
    truncation point are dropped from the returned token list. This
    fix (Phase 11b) closes the silence_fp regression observed in
    Phase 11 — the previous version returned the longer audio and
    relied on the dataset's ``_pad_or_truncate`` to silently cut the
    tail, leaving the label referring to content that no longer
    existed in the audio. The model would learn "sometimes silence
    contains content" and hallucinate on noise-only inputs.

    Without alignment data, the linear-interp fallback runs (legacy
    Phase 10 path) and returns ``tokens=None`` so the caller knows to
    use ``encode(label)`` directly. The fallback retains the
    truncation bug — only the alignment-aware path is safe.

    The inflation factor is sampled per call from ``inflation_range``
    and multiplied by ``nominal_gap_s`` (one word-gap-equivalent at
    ~20 WPM by default).
    """
    if len(audio) == 0:
        return audio, tokens
    inflation = float(rng.uniform(*inflation_range))
    gap_samples = int(round(inflation * nominal_gap_s * sample_rate))
    if gap_samples <= 0:
        return audio, tokens
    silence = np.zeros(gap_samples, dtype=audio.dtype)

    # Phase 11 path: choose a boundary using true forced-alignment.
    if tokens is not None and char_starts_s is not None:
        space_idxs = [i for i, tok in enumerate(tokens) if tok == SPACE_INDEX]
        if not space_idxs:
            return audio, tokens
        i = int(rng.integers(0, len(space_idxs)))
        sp = space_idxs[i]
        space_t = float(char_starts_s[sp])
        # Insert in the middle of the actual silence: between the
        # SPACE token's onset and the next character's onset.
        if sp + 1 < len(char_starts_s):
            next_t = float(char_starts_s[sp + 1])
            pos_s = 0.5 * (space_t + next_t)
        else:
            pos_s = space_t
        audio_pos = int(round(pos_s * sample_rate))
        audio_pos = max(0, min(audio_pos, len(audio)))
        new_audio = np.concatenate([audio[:audio_pos], silence, audio[audio_pos:]])

        if target_samples is not None and new_audio.shape[0] > target_samples:
            # Drop tokens whose post-insertion start time falls past
            # the truncation horizon. char_starts_s is in *original*
            # audio time; tokens after audio_pos shift by gap_samples.
            insertion_t = audio_pos / sample_rate
            gap_s = gap_samples / sample_rate
            target_t = target_samples / sample_rate
            kept: list[int] = []
            for tok, t0 in zip(tokens, char_starts_s):
                t_shifted = t0 + gap_s if t0 >= insertion_t else t0
                if t_shifted < target_t:
                    kept.append(tok)
                else:
                    break  # char_starts_s is monotonic
            # Trim trailing SPACE — it would point to truncated audio.
            while kept and kept[-1] == SPACE_INDEX:
                kept.pop()
            new_audio = new_audio[:target_samples]
            return new_audio, kept

        return new_audio, tokens

    # Phase 10 fallback: linear interpolation from label SPACE
    # positions. Returns tokens=None so the caller knows to encode
    # ``label`` directly. The truncation bug remains in this path —
    # use the alignment-aware path for any new datasets.
    if " " not in label:
        return audio, None
    boundaries = [i for i, c in enumerate(label) if c == " "]
    if not boundaries:
        return audio, None
    bi = boundaries[int(rng.integers(0, len(boundaries)))]
    audio_pos = int(round(bi / len(label) * len(audio)))
    audio_pos = max(0, min(audio_pos, len(audio)))
    new_audio = np.concatenate([audio[:audio_pos], silence, audio[audio_pos:]])
    return new_audio, None


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
