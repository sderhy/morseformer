"""Sliding-window streaming RNN-T decoder.

The training distribution for the acoustic model is fixed at 6 s clips,
so any real-time decoder must keep its inputs at that length (longer
clips collapse the RNN-T greedy decoder; see scripts/decode_audio.py).
The naïve approach — non-overlapping 6 s chunks — is what
:mod:`scripts.decode_live` v0 does, and live-testing the user's IC-7300
on 2026-04-24 surfaced two big costs of that design:

* **First-character stutter at chunk boundaries.** RNN-T greedy emits
  the first character several times at ``t=0`` before the prediction
  LSTM stabilises, producing artefacts like ``CCCCQ`` and ``DDDDE``.
* **Word-boundary cuts.** A non-overlapping cut in the middle of a
  word breaks ``F4HYY`` into ``F4HY`` + ``Y``, ``LONDON`` into ``LON`` +
  ``DON`` etc.

This module addresses both with a sliding window of the same
``window_seconds`` length and a shorter ``hop_seconds`` advance. For
each window we keep only the emissions whose absolute audio timestamps
fall in the *central* zone — the part of the window that is far from
both edges. Adjacent central zones tile the audio without gaps and
without overlap, so we never re-emit a token and we never see a chunk
boundary in the committed text.

Latency is ``window/2 + hop/2`` worst-case (4 s with the defaults), down
from 6 s in v0.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from morseformer.core.tokenizer import decode
from morseformer.features import FrontendConfig, extract_features
from morseformer.models.rnnt import RnntModel


# Conformer subsampling factor (two stride-2 conv layers).
_ENC_SUBSAMPLE: int = 4


@dataclass
class StreamingConfig:
    """Configuration for :class:`StreamingDecoder`.

    Attributes:
        window_seconds: Decoder input length. Must equal the training
            clip length (6.0 s) — the model is not robust to other lengths.
        hop_seconds: How often we re-decode. Smaller hop → lower latency
            and more compute. The central zone we commit per window has
            width ``hop_seconds`` (so the audio is exactly tiled by
            those zones).
        sample_rate: Rate at which audio is delivered to ``feed()``. Must
            be a multiple of ``frame_rate``.
        frame_rate: Front-end output rate. Default 500 Hz matches
            training.
        carrier_hz: Carrier frequency the front-end demodulates around.
        bandwidth_hz: Front-end BPF width.
        max_emit_per_frame: Cap on RNN-T emissions per encoder frame
            (matches training default).
        confidence_threshold: Minimum softmax probability of the predicted
            non-blank token; below this the frame is treated as a blank
            and we advance. ``0.0`` disables gating. Raising to 0.3 - 0.5
            on a model trained without enough silence/noise data
            suppresses the "letter soup" hallucination on weak signal at
            inference cost only.
    """

    window_seconds: float = 6.0
    hop_seconds: float = 2.0
    sample_rate: int = 8000
    frame_rate: int = 500
    carrier_hz: float = 600.0
    bandwidth_hz: float = 200.0
    max_emit_per_frame: int = 5
    confidence_threshold: float = 0.0


class StreamingDecoder:
    """Emit text incrementally from a continuous audio feed.

    Push audio with :meth:`feed`; it returns any newly-committed text
    fragments as they become stable. Call :meth:`flush` at end-of-stream
    to commit the trailing audio.

    Pure logic, no I/O — the live and offline harnesses both use this
    class.
    """

    def __init__(
        self,
        model: RnntModel,
        cfg: StreamingConfig,
        device: torch.device | str = "cpu",
    ) -> None:
        if cfg.sample_rate % cfg.frame_rate != 0:
            raise ValueError(
                f"sample_rate={cfg.sample_rate} must be a multiple of "
                f"frame_rate={cfg.frame_rate}"
            )
        if cfg.hop_seconds <= 0 or cfg.hop_seconds > cfg.window_seconds:
            raise ValueError(
                f"hop_seconds must be in (0, window_seconds]; "
                f"got hop={cfg.hop_seconds} window={cfg.window_seconds}"
            )

        self.model = model
        self.cfg = cfg
        self.device = torch.device(device)

        self._fcfg = FrontendConfig(
            tone_freq=cfg.carrier_hz,
            bandwidth=cfg.bandwidth_hz,
            frame_rate=cfg.frame_rate,
        )

        # Window/hop in samples (integer, derived once).
        self._window_samples = int(round(cfg.window_seconds * cfg.sample_rate))
        self._hop_samples = int(round(cfg.hop_seconds * cfg.sample_rate))
        # Each encoder frame corresponds to this many input samples.
        # frame_rate × subsample / sample_rate = frames per sample, so the
        # inverse is samples per encoder frame.
        self._enc_samples_per_frame = (
            cfg.sample_rate * _ENC_SUBSAMPLE // cfg.frame_rate
        )

        # State.
        self._buffer = np.zeros(0, dtype=np.float32)
        self._buffer_origin_samples = 0   # absolute index of buffer[0]
        self._total_samples = 0           # total samples ever fed
        self._chunk_idx = 0               # number of windows decoded
        self._committed_until_samples = 0  # high-water mark of commit zones

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def feed(self, audio: np.ndarray) -> list[str]:
        """Push more audio. Return the list of newly-committed text
        fragments (one per window decoded during this call).
        """
        if audio.size == 0:
            return []
        a = np.asarray(audio, dtype=np.float32)
        self._buffer = np.concatenate([self._buffer, a])
        self._total_samples += a.size
        return self._drain()

    def flush(self) -> str:
        """Commit the remaining audio. Returns the final un-committed
        text fragment (may be empty).
        """
        if self._committed_until_samples >= self._total_samples:
            return ""
        if self._buffer.size == 0:
            return ""
        return self._decode_and_commit(
            window_start_samples=self._buffer_origin_samples,
            window_audio=self._buffer,
            is_first=(self._chunk_idx == 0),
            is_final=True,
        )

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _drain(self) -> list[str]:
        """Decode every full window we now have data for."""
        out: list[str] = []
        win = self._window_samples
        hop = self._hop_samples
        while True:
            i = self._chunk_idx
            window_start = i * hop
            window_end = window_start + win
            if self._total_samples < window_end:
                break

            # Slice window from buffer (absolute → buffer-relative).
            buf_lo = window_start - self._buffer_origin_samples
            buf_hi = window_end - self._buffer_origin_samples
            window_audio = self._buffer[buf_lo:buf_hi]

            text = self._decode_and_commit(
                window_start_samples=window_start,
                window_audio=window_audio,
                is_first=(i == 0),
                is_final=False,
            )
            if text:
                out.append(text)
            self._chunk_idx += 1

            # Drop samples we'll never need again: the next window starts
            # at (i+1) * hop.
            new_origin = (i + 1) * hop
            drop = new_origin - self._buffer_origin_samples
            if drop > 0:
                self._buffer = self._buffer[drop:]
                self._buffer_origin_samples = new_origin
        return out

    def _decode_and_commit(
        self,
        *,
        window_start_samples: int,
        window_audio: np.ndarray,
        is_first: bool,
        is_final: bool,
    ) -> str:
        """Decode one window and return the text falling in its commit
        zone. Updates ``_committed_until_samples``."""
        if window_audio.size == 0:
            return ""

        feats = extract_features(window_audio, self.cfg.sample_rate, self._fcfg)
        if feats.shape[0] == 0:
            return ""

        x = torch.from_numpy(feats).unsqueeze(0).to(self.device)
        lengths = torch.tensor(
            [feats.shape[0]], dtype=torch.long, device=self.device
        )
        with torch.no_grad():
            aligned = self.model.greedy_rnnt_decode_aligned(
                x,
                lengths,
                max_emit_per_frame=self.cfg.max_emit_per_frame,
                confidence_threshold=self.cfg.confidence_threshold,
            )[0]

        commit_lo, commit_hi = self._commit_zone_samples(
            window_start_samples=window_start_samples,
            window_audio_size=window_audio.size,
            is_first=is_first,
            is_final=is_final,
        )

        # Filter emissions by absolute timestamp.
        kept_tokens: list[int] = []
        for tok, frame_idx in aligned:
            abs_sample = (
                window_start_samples + frame_idx * self._enc_samples_per_frame
            )
            if abs_sample < commit_lo or abs_sample >= commit_hi:
                continue
            # Defensive dedup: never re-emit a token with a timestamp at
            # or before the previous commit cutoff. Adjacent central
            # zones do not overlap by construction, but encoder frame
            # rounding can put a token a few ms before the boundary.
            if abs_sample < self._committed_until_samples:
                continue
            kept_tokens.append(tok)

        self._committed_until_samples = commit_hi
        return decode(kept_tokens)

    def _commit_zone_samples(
        self,
        *,
        window_start_samples: int,
        window_audio_size: int,
        is_first: bool,
        is_final: bool,
    ) -> tuple[int, int]:
        """Compute the [lo, hi) absolute-sample range of the commit zone
        for the current window.

        For interior windows the zone is centred at the window centre
        with width ``hop_seconds``. The first window also covers
        everything to its left (no earlier window will), and the final
        window also covers everything to its right (no later window
        will).
        """
        hop = self._hop_samples
        win = self._window_samples
        # Centre of the window in absolute samples.
        centre = window_start_samples + win // 2
        # Standard interior zone: [centre - hop/2, centre + hop/2).
        lo = centre - hop // 2
        hi = centre + (hop - hop // 2)

        if is_first:
            lo = 0
        if is_final:
            # Cover everything from previous commit cutoff to end of audio.
            hi = window_start_samples + window_audio_size
            if not is_first:
                lo = self._committed_until_samples
        # Clamp lo to the previous commit cutoff to avoid any backslide
        # from rounding when window_seconds / hop_seconds is non-integer.
        if lo < self._committed_until_samples:
            lo = self._committed_until_samples
        if hi < lo:
            hi = lo
        return lo, hi
