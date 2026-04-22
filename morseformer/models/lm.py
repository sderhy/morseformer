"""Character-level language model for morseformer (Phase 4).

A small decoder-only Transformer trained on the synthetic ham-radio text
distribution from :mod:`morseformer.data.text`. Its job in the full
pipeline is to provide a text prior for shallow fusion with the acoustic
model's RNN-T / CTC head, especially in the weak-signal regime where
the acoustic posterior is noisy.

Design choices (2025-standard small-LM recipe)
---------------------------------------------
- **RMSNorm**, pre-norm. LLaMA / Phi / Gemma default — slightly better
  conditioned than LayerNorm and one cheap multiplication instead of a
  mean+var pair.
- **SwiGLU** FFN with ``2/3 × 4d`` hidden so param budget matches the
  vanilla 4× MLP. Standard modern replacement for GELU-MLP.
- **RoPE** position encoding (re-using :class:`RotaryEmbedding` from the
  acoustic conformer for a single RoPE implementation across the code-
  base).
- **Tied input / output embeddings** — saves ~12 k params at d=256 and
  is known to help perplexity on tiny vocabularies.
- **No biases anywhere** (LLaMA convention): fewer params, no
  accuracy loss on any benchmark we care about.
- Causal self-attention via :func:`F.scaled_dot_product_attention` with
  ``is_causal=True``. PyTorch routes to FlashAttention on CUDA.

Vocabulary
----------
We reuse the 46-token acoustic tokenizer verbatim so shallow fusion is
a trivial per-step log-prob add. In LM context we repurpose the CTC
blank (index 0) as the sequence boundary / end-of-document marker —
the acoustic model never emits index 0, the LM never sees a blank in
the acoustic sense, so the overloaded slot is harmless.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from morseformer.core.tokenizer import BLANK_INDEX, VOCAB_SIZE
from morseformer.models.conformer import RotaryEmbedding, apply_rope


# Re-export the overloaded semantic — blank index doubles as EOS for the LM.
EOS_INDEX = BLANK_INDEX


# --------------------------------------------------------------------- #
# Normalisation + FFN
# --------------------------------------------------------------------- #


class RMSNorm(nn.Module):
    """Root-mean-square layer norm.

    ``y = x / sqrt(mean(x²) + eps) * gamma`` — no mean subtraction, no
    bias. Matches LLaMA's implementation.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        xf = x.float()
        rms = xf.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (xf * rms).to(dtype) * self.weight


class SwiGLU(nn.Module):
    """SwiGLU FFN: ``out = W_out(SiLU(W_gate(x)) * W_up(x))``.

    The hidden dimension is set to ``round(2/3 * 4 * d_model)``, which
    keeps the parameter count equivalent to a vanilla ``4d`` GELU MLP
    while giving the gated architecture.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        hidden = int(round(2 / 3 * 4 * d_model))
        # Round up to multiple of 8 for kernel alignment.
        hidden = ((hidden + 7) // 8) * 8
        self.w_gate = nn.Linear(d_model, hidden, bias=False)
        self.w_up = nn.Linear(d_model, hidden, bias=False)
        self.w_out = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_out(F.silu(self.w_gate(x)) * self.w_up(x))


# --------------------------------------------------------------------- #
# Causal self-attention
# --------------------------------------------------------------------- #


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with RoPE.

    Uses :func:`F.scaled_dot_product_attention` with ``is_causal=True``
    so PyTorch can pick the FlashAttention kernel on CUDA.
    """

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        qkv = self.qkv(x).reshape(b, t, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)            # [B, T, H, D_h]
        q = q.transpose(1, 2)                   # [B, H, T, D_h]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        cos, sin = self.rope(t, x.device, x.dtype)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        out = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).contiguous().reshape(b, t, self.d_model)
        return self.out(out)


# --------------------------------------------------------------------- #
# Transformer block
# --------------------------------------------------------------------- #


class GptBlock(nn.Module):
    """Pre-norm block: ``x = x + Attn(Norm(x)); x = x + FFN(Norm(x))``."""

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ff_norm = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.attn(self.attn_norm(x)))
        x = x + self.drop(self.ffn(self.ff_norm(x)))
        return x


# --------------------------------------------------------------------- #
# Top-level LM
# --------------------------------------------------------------------- #


@dataclass
class LmConfig:
    vocab_size: int = VOCAB_SIZE
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 6
    dropout: float = 0.1
    eos_index: int = EOS_INDEX
    # Initialisation scale for residual-path weights (GPT-2 recipe).
    # Final layer weights are rescaled by ``1 / sqrt(2 * n_layers)`` to
    # keep activation variance stable through deep residual stacks.
    init_std: float = 0.02


class GptLM(nn.Module):
    """Small decoder-only Transformer language model."""

    def __init__(self, cfg: LmConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or LmConfig()
        c = self.cfg

        self.embed = nn.Embedding(c.vocab_size, c.d_model)
        self.drop = nn.Dropout(c.dropout)
        self.blocks = nn.ModuleList(
            [GptBlock(c.d_model, c.n_heads, c.dropout) for _ in range(c.n_layers)]
        )
        self.final_norm = RMSNorm(c.d_model)
        # Output head tied to the input embedding — we expose it as
        # ``self.head`` for clarity but share the weight tensor.
        self.head = nn.Linear(c.d_model, c.vocab_size, bias=False)
        self.head.weight = self.embed.weight

        self._init_parameters()

    # ------------------------------------------------------------- #
    # Initialisation
    # ------------------------------------------------------------- #
    def _init_parameters(self) -> None:
        std = self.cfg.init_std
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=std)
        # GPT-2 residual rescaling: scale output projections by
        # 1 / sqrt(2 * n_layers) so residual variance stays bounded.
        scale = (2 * self.cfg.n_layers) ** -0.5
        for block in self.blocks:
            block.attn.out.weight.data.mul_(scale)
            block.ffn.w_out.weight.data.mul_(scale)

    # ------------------------------------------------------------- #
    # Forward
    # ------------------------------------------------------------- #
    def forward(
        self,
        tokens: torch.Tensor,
        targets: torch.Tensor | None = None,
        ignore_index: int = -100,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run the LM.

        Args:
            tokens:  ``[B, T]`` int64 input token ids.
            targets: optional ``[B, T]`` int64 target ids. If given, a
                     cross-entropy loss is returned alongside the logits.
            ignore_index: target ids equal to this value contribute zero
                          loss (used for padding positions).

        Returns:
            ``(logits [B, T, V], loss or None)``.
        """
        x = self.embed(tokens)
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        logits = self.head(x)
        loss: torch.Tensor | None = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=ignore_index,
            )
        return logits, loss

    def num_parameters(self, non_embedding: bool = False) -> int:
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if non_embedding:
            total -= self.embed.weight.numel()
        return total

    # ------------------------------------------------------------- #
    # Sampling (for qualitative eval and sanity checks)
    # ------------------------------------------------------------- #
    @torch.no_grad()
    def generate(
        self,
        prompt: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        stop_on_eos: bool = True,
    ) -> torch.Tensor:
        """Autoregressive sampling. ``prompt`` is ``[B, T_prompt]``."""
        self.eval()
        tokens = prompt
        for _ in range(max_new_tokens):
            logits, _ = self.forward(tokens)
            next_logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                v, _ = torch.topk(next_logits, top_k)
                threshold = v[:, -1, None]
                next_logits = torch.where(
                    next_logits < threshold,
                    torch.full_like(next_logits, float("-inf")),
                    next_logits,
                )
            probs = F.softmax(next_logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)    # [B, 1]
            tokens = torch.cat((tokens, next_tok), dim=1)
            if stop_on_eos and (next_tok == self.cfg.eos_index).all():
                break
        return tokens
