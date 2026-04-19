"""Top-level acoustic model for morseformer.

Architecture: `[B, T, F]` features  →  ConvSubsampling (4×)  →  stack of
`ConformerBlock`s  →  linear head  →  per-frame log-softmax over the
tokenizer vocabulary (including the CTC blank at index 0).

The model returns *both* the log-probabilities and the sub-sampled
frame lengths, since CTC loss requires the effective post-encoder
length for each item in the batch.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from morseformer.core.tokenizer import BLANK_INDEX, VOCAB_SIZE
from morseformer.models.conformer import (
    ConformerBlock,
    ConvSubsampling,
    init_parameters,
)


@dataclass
class AcousticConfig:
    """Hyperparameters for the acoustic model.

    Defaults target a mid-sized model (~5M params) suitable for Phase 2
    training on a single CPU/GPU. Larger variants will be explored once
    the baseline pipeline is validated.
    """

    input_dim: int = 1           # front-end feature channels (F=1 in Phase 2)
    d_model: int = 144           # Conformer-S width
    n_heads: int = 4
    n_layers: int = 8
    ff_expansion: int = 4
    conv_kernel: int = 31        # depthwise-conv receptive field (post-subsample)
    dropout: float = 0.1
    vocab_size: int = VOCAB_SIZE
    blank_index: int = BLANK_INDEX


class AcousticModel(nn.Module):
    def __init__(self, cfg: AcousticConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or AcousticConfig()
        c = self.cfg

        self.subsample = ConvSubsampling(c.input_dim, c.d_model)
        self.input_dropout = nn.Dropout(c.dropout)
        self.blocks = nn.ModuleList(
            [
                ConformerBlock(
                    d_model=c.d_model,
                    n_heads=c.n_heads,
                    ff_expansion=c.ff_expansion,
                    conv_kernel=c.conv_kernel,
                    dropout=c.dropout,
                )
                for _ in range(c.n_layers)
            ]
        )
        self.head = nn.Linear(c.d_model, c.vocab_size)

        init_parameters(self)

    def encode(
        self,
        features: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run the sub-sampling + Conformer stack, returning the
        pre-head encoder representation.

        This is the shared trunk used both by the CTC head (forward())
        and by the RNN-T joint decoder (see :mod:`morseformer.models.rnnt`).

        Returns:
            enc_out:     `[B, T', d_model]` float32 encoder features.
            lengths_out: `[B]` post-subsample valid lengths, or ``None``
                         when ``lengths`` was ``None``.
        """
        x = self.subsample(features)  # [B, T', D]
        x = self.input_dropout(x)

        padding_mask: torch.Tensor | None = None
        lengths_out: torch.Tensor | None = None
        if lengths is not None:
            lengths_out = ConvSubsampling.subsampled_lengths(lengths)
            t_out = x.size(1)
            idx = torch.arange(t_out, device=x.device).unsqueeze(0)
            padding_mask = idx >= lengths_out.unsqueeze(1)

        for block in self.blocks:
            x = block(x, padding_mask=padding_mask)
        return x, lengths_out

    def forward(
        self,
        features: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Encode frame-rate features to per-frame token log-probs.

        Args:
            features: `[B, T, F]` float32.
            lengths:  `[B]` int64 of valid input frames, or `None` for
                      full-length batches (no padding).

        Returns:
            log_probs:   `[B, T', V]` log-softmax, where `T' ≈ T / 4`.
            lengths_out: `[B]` post-subsample valid lengths, or `None`
                         when `lengths` was `None`.
        """
        enc_out, lengths_out = self.encode(features, lengths)
        logits = self.head(enc_out)
        return torch.log_softmax(logits, dim=-1), lengths_out

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
