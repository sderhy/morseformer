"""Tests for the sliding-window streaming RNN-T decoder."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from morseformer.decoding.streaming import (  # noqa: E402
    StreamingConfig,
    StreamingDecoder,
)
from morseformer.models.acoustic import AcousticConfig  # noqa: E402
from morseformer.models.rnnt import RnntConfig, RnntModel  # noqa: E402


def _tiny_rnnt() -> RnntModel:
    cfg = RnntConfig(
        encoder=AcousticConfig(
            d_model=32, n_heads=4, n_layers=2,
            ff_expansion=2, conv_kernel=7, dropout=0.0,
        ),
        d_pred=32,
        pred_lstm_layers=1,
        d_joint=32,
        dropout=0.0,
    )
    return RnntModel(cfg).eval()


def _default_cfg(**overrides) -> StreamingConfig:
    base = dict(
        window_seconds=6.0,
        hop_seconds=2.0,
        sample_rate=8000,
        frame_rate=500,
        carrier_hz=600.0,
        bandwidth_hz=200.0,
    )
    base.update(overrides)
    return StreamingConfig(**base)


# --------------------------------------------------------------------- #
# Commit-zone arithmetic
# --------------------------------------------------------------------- #


def test_commit_zones_tile_audio_without_gaps() -> None:
    """Adjacent central zones must abut exactly (no gap, no overlap).

    With window=6 s and hop=2 s, chunk i covers absolute samples
    [i*hop + 2 s, i*hop + 4 s). Verify there is no gap between
    consecutive zones.
    """
    sd = StreamingDecoder(_tiny_rnnt(), _default_cfg(), device="cpu")
    hop_s = sd._hop_samples
    zones: list[tuple[int, int]] = []
    for i in range(5):
        lo, hi = sd._commit_zone_samples(
            window_start_samples=i * hop_s,
            window_audio_size=sd._window_samples,
            is_first=(i == 0),
            is_final=False,
        )
        zones.append((lo, hi))
        # Mimic what _decode_and_commit does so subsequent zones see the
        # updated cutoff.
        sd._committed_until_samples = hi
    # First zone starts at 0 and reaches the centre of chunk-0's central zone
    # (i.e. 4 s).
    assert zones[0][0] == 0
    # Subsequent zones must start exactly where the previous ended.
    for prev, nxt in zip(zones, zones[1:]):
        assert prev[1] == nxt[0]
    # Each non-first zone must have width = hop in samples.
    for lo, hi in zones[1:]:
        assert hi - lo == hop_s


def test_first_chunk_zone_starts_at_zero() -> None:
    sd = StreamingDecoder(_tiny_rnnt(), _default_cfg(), device="cpu")
    lo, hi = sd._commit_zone_samples(
        window_start_samples=0,
        window_audio_size=sd._window_samples,
        is_first=True,
        is_final=False,
    )
    assert lo == 0
    # Right edge of first zone = centre + hop/2 = window/2 + hop/2 = 4 s.
    expected_hi = sd._window_samples // 2 + sd._hop_samples // 2
    assert hi == expected_hi


def test_final_zone_extends_to_end_of_audio() -> None:
    sd = StreamingDecoder(_tiny_rnnt(), _default_cfg(), device="cpu")
    sd._committed_until_samples = 4 * sd.cfg.sample_rate  # already committed [0, 4)
    lo, hi = sd._commit_zone_samples(
        window_start_samples=2 * sd.cfg.sample_rate,
        window_audio_size=sd._window_samples,
        is_first=False,
        is_final=True,
    )
    assert lo == sd._committed_until_samples
    assert hi == 2 * sd.cfg.sample_rate + sd._window_samples


def test_short_final_window_zone_is_clamped() -> None:
    """If flush is called with less than window_seconds buffered, the
    zone must still be valid."""
    sd = StreamingDecoder(_tiny_rnnt(), _default_cfg(), device="cpu")
    sd._committed_until_samples = 3 * sd.cfg.sample_rate
    short = sd.cfg.sample_rate  # 1 s of audio
    lo, hi = sd._commit_zone_samples(
        window_start_samples=2 * sd.cfg.sample_rate,
        window_audio_size=short,
        is_first=False,
        is_final=True,
    )
    assert lo == 3 * sd.cfg.sample_rate
    assert hi == 2 * sd.cfg.sample_rate + short  # = 3 s
    assert hi >= lo


# --------------------------------------------------------------------- #
# End-to-end shape / flow
# --------------------------------------------------------------------- #


def test_feed_returns_no_text_until_first_window_full() -> None:
    sd = StreamingDecoder(_tiny_rnnt(), _default_cfg(), device="cpu")
    rng = np.random.default_rng(0)
    # Feed less than window_seconds.
    audio = rng.standard_normal(int(0.5 * sd.cfg.sample_rate)).astype(np.float32)
    out = sd.feed(audio)
    assert out == []
    out = sd.feed(audio)
    assert out == []  # still < 6 s
    assert sd._chunk_idx == 0


def test_feed_decodes_one_chunk_per_hop_after_first_window() -> None:
    sd = StreamingDecoder(_tiny_rnnt(), _default_cfg(), device="cpu")
    rng = np.random.default_rng(0)
    # Push exactly window_seconds: should decode chunk 0.
    audio = rng.standard_normal(sd._window_samples).astype(np.float32)
    out = sd.feed(audio)
    assert sd._chunk_idx == 1
    # Push another hop_seconds: should decode chunk 1.
    audio = rng.standard_normal(sd._hop_samples).astype(np.float32)
    out = sd.feed(audio)
    assert sd._chunk_idx == 2


def test_buffer_does_not_grow_unbounded() -> None:
    """As we feed many hops, the buffer should stay ≈ window_seconds."""
    sd = StreamingDecoder(_tiny_rnnt(), _default_cfg(), device="cpu")
    rng = np.random.default_rng(1)
    full = rng.standard_normal(sd._window_samples).astype(np.float32)
    sd.feed(full)
    for _ in range(10):
        sd.feed(rng.standard_normal(sd._hop_samples).astype(np.float32))
    # After 10 hops post-first-window, buffer should be ≤ window samples
    # (approximately equal).
    assert sd._buffer.size <= sd._window_samples


def test_flush_after_partial_audio_returns_string() -> None:
    sd = StreamingDecoder(_tiny_rnnt(), _default_cfg(), device="cpu")
    rng = np.random.default_rng(2)
    # Feed half a window.
    sd.feed(rng.standard_normal(sd._window_samples // 2).astype(np.float32))
    out = sd.flush()
    # No error, returns a (possibly empty) string.
    assert isinstance(out, str)


def test_flush_is_noop_after_full_decode() -> None:
    sd = StreamingDecoder(_tiny_rnnt(), _default_cfg(), device="cpu")
    rng = np.random.default_rng(3)
    full = rng.standard_normal(sd._window_samples).astype(np.float32)
    sd.feed(full)
    # We've decoded chunk 0 with central zone [0, 4 s); some audio (4-6 s)
    # is still uncommitted, so flush should commit it.
    tail = sd.flush()
    assert isinstance(tail, str)
    # Now everything is committed.
    second_flush = sd.flush()
    assert second_flush == ""


def test_invalid_config_rejected() -> None:
    with pytest.raises(ValueError):
        StreamingDecoder(_tiny_rnnt(), _default_cfg(hop_seconds=0), device="cpu")
    with pytest.raises(ValueError):
        StreamingDecoder(_tiny_rnnt(), _default_cfg(hop_seconds=10), device="cpu")
    with pytest.raises(ValueError):
        StreamingDecoder(_tiny_rnnt(), _default_cfg(sample_rate=8001), device="cpu")
