"""Conformer encoder blocks for the morseformer acoustic model.

References
----------
- Gulati et al., 2020. *Conformer: Convolution-augmented Transformer for
  Speech Recognition.* https://arxiv.org/abs/2005.08100
- Su et al., 2021. *RoFormer: Enhanced Transformer with Rotary Position
  Embedding.* https://arxiv.org/abs/2104.09864
- Kim et al., 2022. *Squeezeformer: An Efficient Transformer for
  Automatic Speech Recognition.* https://arxiv.org/abs/2206.00888
  (motivation for LayerNorm-over-BatchNorm in the conv module).

Departures from the original Conformer paper
--------------------------------------------
1. **RoPE** in self-attention instead of Transformer-XL-style relative
   bias. Parameter-free, handles arbitrary sequence lengths, and matches
   or exceeds relative-bias on ASR benchmarks — while being dramatically
   simpler to implement and reason about.
2. **Pre-norm throughout**, including inside the convolution module.
   More stable gradients when stacking many blocks with modern
   optimisers (AdamW + warmup + cosine).
3. **LayerNorm instead of BatchNorm** in the conv module. BN's mean/var
   get polluted by padded positions and break streaming inference;
   modern ASR architectures (Squeezeformer, Zipformer) have moved away
   from BN for exactly this reason.
4. **4× time sub-sampling** at the input (two stride-2 Conv1d). At the
   500 fps front-end we use, this yields 125 fps at the encoder — still
   five frames per dit at 60 WPM while keeping self-attention
   quadratic-cost tractable on multi-second utterances.
"""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


# --------------------------------------------------------------------- #
# Rotary position embedding (RoPE)
# --------------------------------------------------------------------- #


class RotaryEmbedding(nn.Module):
    """Half-split RoPE, as used in LLaMA and most modern transformers.

    Precomputes cos/sin tables lazily. The same table is reused across
    heads and across the batch.
    """

    def __init__(self, head_dim: int, base: float = 10000.0) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")
        self.head_dim = head_dim
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cos: torch.Tensor | None = None
        self._sin: torch.Tensor | None = None

    def _maybe_build_cache(
        self, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> None:
        cached = self._cos
        if (
            cached is not None
            and cached.size(0) >= seq_len
            and cached.device == device
            and cached.dtype == dtype
        ):
            return
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq.to(device))  # [T, head_dim/2]
        self._cos = freqs.cos().to(dtype)
        self._sin = freqs.sin().to(dtype)

    def forward(
        self, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._maybe_build_cache(seq_len, device, dtype)
        assert self._cos is not None and self._sin is not None
        return self._cos[:seq_len], self._sin[:seq_len]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Apply RoPE to `x` of shape `[..., T, head_dim]`.

    `cos` and `sin` are `[T, head_dim/2]`. They are duplicated along the
    last dim to match the half-split rotation scheme.
    """
    cos_full = torch.cat((cos, cos), dim=-1)  # [T, head_dim]
    sin_full = torch.cat((sin, sin), dim=-1)
    # Broadcast over any leading batch / head dims.
    while cos_full.dim() < x.dim():
        cos_full = cos_full.unsqueeze(0)
        sin_full = sin_full.unsqueeze(0)
    return x * cos_full + _rotate_half(x) * sin_full


# --------------------------------------------------------------------- #
# Multi-head self-attention with RoPE
# --------------------------------------------------------------------- #


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout
        self.rope = RotaryEmbedding(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, t, _ = x.shape
        qkv = self.qkv(x).reshape(b, t, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each [B, T, H, D_h]
        q = q.transpose(1, 2)  # [B, H, T, D_h]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        cos, sin = self.rope(t, x.device, x.dtype)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        attn_mask: torch.Tensor | None = None
        if key_padding_mask is not None:
            # SDPA bool convention: True = attend, False = mask out.
            # Our key_padding_mask is True where padding; invert.
            attn_mask = (~key_padding_mask)[:, None, None, :]  # [B, 1, 1, T]

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).contiguous().reshape(b, t, self.d_model)
        return self.out(out)


# --------------------------------------------------------------------- #
# Macaron feed-forward module
# --------------------------------------------------------------------- #


class FeedForwardModule(nn.Module):
    """Macaron half-step FFN: pre-norm → Linear → Swish → Linear → drop.

    Residual is added with weight 0.5 (two such modules sandwich the
    attention + convolution in a Conformer block).
    """

    def __init__(self, d_model: int, expansion: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ff1 = nn.Linear(d_model, d_model * expansion, bias=False)
        self.act = nn.SiLU()
        self.drop1 = nn.Dropout(dropout)
        self.ff2 = nn.Linear(d_model * expansion, d_model, bias=False)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm(x)
        y = self.ff1(y)
        y = self.act(y)
        y = self.drop1(y)
        y = self.ff2(y)
        y = self.drop2(y)
        return x + 0.5 * y


# --------------------------------------------------------------------- #
# Convolution module
# --------------------------------------------------------------------- #


class ConvolutionModule(nn.Module):
    """LN → PW-conv (×2) → GLU → DW-conv → LN → Swish → PW-conv → drop.

    Padded frames are zeroed before the depthwise conv so that the
    local-context filter cannot leak pad values into valid positions.
    """

    def __init__(
        self, d_model: int, kernel_size: int = 31, dropout: float = 0.0
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd to keep T constant")
        self.norm = nn.LayerNorm(d_model)
        self.pw_conv_1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        self.dw_conv = nn.Conv1d(
            d_model, d_model,
            kernel_size=kernel_size,
            groups=d_model,
            padding=kernel_size // 2,
        )
        self.post_norm = nn.LayerNorm(d_model)  # over channel dim
        self.act = nn.SiLU()
        self.pw_conv_2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.drop = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        residual = x
        y = self.norm(x).transpose(1, 2)  # [B, D, T]
        y = self.pw_conv_1(y)
        y = self.glu(y)
        if padding_mask is not None:
            y = y.masked_fill(padding_mask[:, None, :], 0.0)
        y = self.dw_conv(y)
        y = y.transpose(1, 2)  # [B, T, D]
        y = self.post_norm(y)
        y = self.act(y)
        y = y.transpose(1, 2)  # [B, D, T]
        y = self.pw_conv_2(y)
        y = y.transpose(1, 2)  # [B, T, D]
        y = self.drop(y)
        return residual + y


# --------------------------------------------------------------------- #
# Conformer block
# --------------------------------------------------------------------- #


class ConformerBlock(nn.Module):
    """FF (half) → MHSA → Conv → FF (half) → LN."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ff_expansion: int = 4,
        conv_kernel: int = 31,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.ff1 = FeedForwardModule(d_model, ff_expansion, dropout)
        self.mhsa_norm = nn.LayerNorm(d_model)
        self.mhsa = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.mhsa_drop = nn.Dropout(dropout)
        self.conv = ConvolutionModule(d_model, conv_kernel, dropout)
        self.ff2 = FeedForwardModule(d_model, ff_expansion, dropout)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self, x: torch.Tensor, padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = self.ff1(x)
        y = self.mhsa_norm(x)
        y = self.mhsa(y, key_padding_mask=padding_mask)
        x = x + self.mhsa_drop(y)
        x = self.conv(x, padding_mask=padding_mask)
        x = self.ff2(x)
        return self.final_norm(x)


# --------------------------------------------------------------------- #
# Convolutional sub-sampling (4× time reduction)
# --------------------------------------------------------------------- #


class ConvSubsampling(nn.Module):
    """Two stacked stride-2 Conv1d layers → 4× time reduction.

    Maps `[B, T, F_in]` → `[B, ceil(ceil(T/2)/2), d_model]`.
    """

    def __init__(self, in_channels: int, d_model: int) -> None:
        super().__init__()
        hidden = d_model // 2
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv1d(hidden, d_model, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.transpose(1, 2)  # [B, F_in, T]
        y = self.conv(y)
        return y.transpose(1, 2)  # [B, T', d_model]

    @staticmethod
    def subsampled_lengths(lengths: torch.Tensor) -> torch.Tensor:
        """Length transform matching the conv stack above (k=3, s=2, p=1).

        For each stride-2 layer: L_out = (L + 1) // 2.
        """
        lengths = (lengths + 1) // 2
        lengths = (lengths + 1) // 2
        return lengths


def init_parameters(module: nn.Module) -> None:
    """Xavier-uniform for Linear / Conv1d weights, zero bias, default LN."""
    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
