"""Inference-time decoding helpers (streaming, batch).

The :mod:`morseformer.decoding.streaming` module provides
:class:`StreamingDecoder` — a sliding-window wrapper around an RNN-T
checkpoint that emits text incrementally from a continuous audio feed.
"""

from morseformer.decoding.postprocess import StreamFormatter, format_output
from morseformer.decoding.streaming import (
    StreamingConfig,
    StreamingDecoder,
)

__all__ = [
    "StreamingConfig",
    "StreamingDecoder",
    "StreamFormatter",
    "format_output",
]
