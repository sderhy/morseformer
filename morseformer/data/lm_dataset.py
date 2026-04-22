"""Streaming text dataset for the Phase-4 language model.

The LM training loop consumes fixed-length token windows drawn from the
same synthetic text distribution as the acoustic dataset. Each window
is built by concatenating successive :func:`sample_text` outputs
separated by ``EOS_INDEX`` (which doubles as BOS for the next doc),
until the requested context length is reached.

The dataset is an :class:`IterableDataset` with no notion of epoch —
synthetic text is effectively infinite. Each worker holds its own
``np.random.Generator`` seeded deterministically from the worker id
and the config seed, so a fresh-seeded run is fully reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from morseformer.core.tokenizer import encode
from morseformer.data.text import DEFAULT_MIX, TextMix, sample_text
from morseformer.models.lm import EOS_INDEX


@dataclass
class LmDatasetConfig:
    """Hyperparameters for the streaming LM dataset.

    ``context_length`` is the fixed window length of every sample
    yielded (the ``(input, target)`` pair are both of this length).
    ``mix`` controls which text categories are drawn. ``seed`` fixes
    the top-level RNG; each DataLoader worker further salts it.
    """

    context_length: int = 256
    mix: TextMix = field(default_factory=lambda: DEFAULT_MIX)
    seed: int = 0


class LmStreamDataset(IterableDataset):
    """Infinite stream of ``(input, target)`` pairs for causal LM training.

    Each yielded item is a dict:

        ``{"input": LongTensor[T], "target": LongTensor[T]}``

    where ``target[i] = input[i + 1]`` and the final target position is
    the EOS-continuation token (i.e. the first token of the next chunk,
    which is why we draw one extra token internally).
    """

    def __init__(self, cfg: LmDatasetConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or LmDatasetConfig()

    def _token_stream(self, rng: np.random.Generator) -> Iterator[int]:
        """Yield an endless stream of token ids, EOS-separated."""
        while True:
            text = sample_text(rng, self.cfg.mix)
            for idx in encode(text):
                yield idx
            yield EOS_INDEX

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        # Salt the seed per worker so parallel loaders don't collide.
        rng = np.random.default_rng(self.cfg.seed + worker_id * 10007 + 1)
        stream = self._token_stream(rng)

        T = self.cfg.context_length
        # We need T + 1 tokens to build a (input[:T], target[:T]) pair.
        buf: list[int] = []
        for tok in stream:
            buf.append(tok)
            if len(buf) >= T + 1:
                window = buf[: T + 1]
                buf = buf[T:]  # keep last token as seed for next window
                t = torch.tensor(window, dtype=torch.long)
                yield {"input": t[:-1], "target": t[1:]}
