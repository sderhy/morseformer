"""Compact Transformer acoustic model with a CTC head.

Input:  frame-rate features `[B, T, F]` float32, as produced by
        `morseformer.features.extract_features` (F = 1 in Phase 2.1).
Output: per-frame token log-probabilities `[B, T, V]` where V is the
        tokenizer vocab size (46, including the CTC blank at index 0).

The model is deliberately small — a few hundred-K parameters at most.
CW is a narrow, structured signal; the acoustic problem does not need
a speech-sized network. Real capacity will go into the downstream LM.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from morseformer.core.tokenizer import BLANK_INDEX, VOCAB_SIZE


@dataclass
class AcousticConfig:
    """Hyperparameters for the acoustic model."""

    input_dim: int = 1           # front-end feature channels
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    dim_feedforward: int = 256
    dropout: float = 0.1
    vocab_size: int = VOCAB_SIZE
    max_len: int = 4096          # upper bound on frame sequence length
    blank_index: int = BLANK_INDEX


class SinusoidalPositionalEncoding(nn.Module):
    """Classic sin/cos positional encoding; no learned params."""

    def __init__(self, d_model: int, max_len: int) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, d_model]
        return x + self.pe[:, : x.size(1)]


class AcousticModel(nn.Module):
    """Feature → Transformer encoder → CTC log-probs."""

    def __init__(self, cfg: AcousticConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or AcousticConfig()
        c = self.cfg

        self.input_proj = nn.Linear(c.input_dim, c.d_model)
        self.pos_enc = SinusoidalPositionalEncoding(c.d_model, c.max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=c.d_model,
            nhead=c.n_heads,
            dim_feedforward=c.dim_feedforward,
            dropout=c.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=c.n_layers)
        self.head = nn.Linear(c.d_model, c.vocab_size)

    def forward(
        self,
        features: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute per-frame CTC log-probabilities.

        Args:
            features: `[B, T, F]` float32.
            lengths:  `[B]` int64 of valid frame counts. If provided, padded
                      positions are masked out of attention.

        Returns:
            log_probs: `[B, T, V]` float32, log-softmax over the vocab axis.
        """
        x = self.input_proj(features)
        x = self.pos_enc(x)

        key_padding_mask: torch.Tensor | None = None
        if lengths is not None:
            # True where position is padding.
            t = x.size(1)
            idx = torch.arange(t, device=x.device).unsqueeze(0)
            key_padding_mask = idx >= lengths.unsqueeze(1)

        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        logits = self.head(x)
        return torch.log_softmax(logits, dim=-1)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
