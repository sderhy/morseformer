"""Unit tests for the Conformer components (conformer.py)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from morseformer.models.conformer import (  # noqa: E402
    ConformerBlock,
    ConvolutionModule,
    ConvSubsampling,
    FeedForwardModule,
    MultiHeadSelfAttention,
    RotaryEmbedding,
    apply_rope,
)


# --------------------------------------------------------------------- #
# RoPE
# --------------------------------------------------------------------- #


def test_rope_preserves_norm() -> None:
    rope = RotaryEmbedding(head_dim=16)
    x = torch.randn(2, 4, 10, 16)  # [B, H, T, D_h]
    cos, sin = rope(10, x.device, x.dtype)
    y = apply_rope(x, cos, sin)
    torch.testing.assert_close(x.norm(dim=-1), y.norm(dim=-1), atol=1e-5, rtol=1e-5)


def test_rope_relative_position_invariance() -> None:
    # The core RoPE property: <RoPE(q, m), RoPE(k, n)> depends only on m-n.
    rope = RotaryEmbedding(head_dim=16)
    q = torch.randn(1, 1, 1, 16)
    k = torch.randn(1, 1, 1, 16)
    cos, sin = rope(20, q.device, q.dtype)

    def score(m: int, n: int) -> float:
        qm = apply_rope(q, cos[m : m + 1], sin[m : m + 1])
        kn = apply_rope(k, cos[n : n + 1], sin[n : n + 1])
        return float((qm * kn).sum())

    # Same relative offset → same score.
    s_02 = score(0, 2)
    s_35 = score(3, 5)
    s_1012 = score(10, 12)
    assert abs(s_02 - s_35) < 1e-5
    assert abs(s_35 - s_1012) < 1e-5


def test_rope_requires_even_head_dim() -> None:
    with pytest.raises(ValueError):
        RotaryEmbedding(head_dim=15)


# --------------------------------------------------------------------- #
# Multi-head self-attention with RoPE
# --------------------------------------------------------------------- #


def test_mhsa_output_shape() -> None:
    mhsa = MultiHeadSelfAttention(d_model=32, n_heads=4).eval()
    x = torch.randn(2, 12, 32)
    with torch.no_grad():
        y = mhsa(x)
    assert y.shape == x.shape
    assert y.dtype == torch.float32


def test_mhsa_padding_mask_changes_result() -> None:
    torch.manual_seed(0)
    mhsa = MultiHeadSelfAttention(d_model=32, n_heads=4).eval()
    x = torch.randn(1, 20, 32)
    pad = torch.zeros(1, 20, dtype=torch.bool)
    pad[:, 10:] = True  # last 10 positions are padding
    with torch.no_grad():
        y_masked = mhsa(x, key_padding_mask=pad)
        y_plain = mhsa(x)
    # The outputs over the valid prefix must differ: padded keys are
    # visible without the mask but excluded with it.
    assert not torch.allclose(y_masked[:, :10], y_plain[:, :10], atol=1e-6)


def test_mhsa_ignores_padded_keys() -> None:
    # Attention over the valid prefix should not depend on what is in the
    # padded tail when the padding mask is supplied.
    torch.manual_seed(1)
    mhsa = MultiHeadSelfAttention(d_model=32, n_heads=4).eval()
    valid = 15
    t = 30
    x_a = torch.randn(1, t, 32)
    x_b = x_a.clone()
    x_b[:, valid:] = torch.randn(1, t - valid, 32) * 10.0
    pad = torch.zeros(1, t, dtype=torch.bool)
    pad[:, valid:] = True
    with torch.no_grad():
        y_a = mhsa(x_a, key_padding_mask=pad)
        y_b = mhsa(x_b, key_padding_mask=pad)
    torch.testing.assert_close(y_a[:, :valid], y_b[:, :valid], atol=1e-5, rtol=1e-5)


# --------------------------------------------------------------------- #
# FFN, ConvModule, Block
# --------------------------------------------------------------------- #


def test_ffn_shape_and_residual_scale() -> None:
    ff = FeedForwardModule(d_model=16, expansion=4, dropout=0.0).eval()
    x = torch.randn(2, 5, 16)
    with torch.no_grad():
        y = ff(x)
    assert y.shape == x.shape


def test_conv_module_preserves_T_and_D() -> None:
    conv = ConvolutionModule(d_model=32, kernel_size=7).eval()
    x = torch.randn(2, 20, 32)
    with torch.no_grad():
        y = conv(x)
    assert y.shape == x.shape


def test_conv_module_rejects_even_kernel() -> None:
    with pytest.raises(ValueError):
        ConvolutionModule(d_model=16, kernel_size=8)


def test_conformer_block_shape() -> None:
    block = ConformerBlock(d_model=32, n_heads=4, conv_kernel=7).eval()
    x = torch.randn(2, 24, 32)
    with torch.no_grad():
        y = block(x)
    assert y.shape == x.shape


def test_conformer_block_respects_padding() -> None:
    torch.manual_seed(2)
    block = ConformerBlock(d_model=32, n_heads=4, conv_kernel=7).eval()
    t = 40
    valid = 25
    x_a = torch.randn(1, t, 32)
    x_b = x_a.clone()
    x_b[:, valid:] = torch.randn(1, t - valid, 32) * 5.0
    pad = torch.zeros(1, t, dtype=torch.bool)
    pad[:, valid:] = True
    with torch.no_grad():
        y_a = block(x_a, padding_mask=pad)[:, :valid]
        y_b = block(x_b, padding_mask=pad)[:, :valid]
    torch.testing.assert_close(y_a, y_b, atol=1e-5, rtol=1e-5)


# --------------------------------------------------------------------- #
# Convolutional sub-sampling
# --------------------------------------------------------------------- #


def test_subsampling_output_shape() -> None:
    sub = ConvSubsampling(in_channels=1, d_model=16).eval()
    x = torch.randn(2, 100, 1)
    with torch.no_grad():
        y = sub(x)
    # Two stride-2 steps: 100 → 50 → 25.
    assert y.shape == (2, 25, 16)


@pytest.mark.parametrize("t_in,t_out", [(100, 25), (99, 25), (17, 5), (1, 1), (2, 1)])
def test_subsampled_lengths_formula(t_in: int, t_out: int) -> None:
    lengths = torch.tensor([t_in], dtype=torch.long)
    assert int(ConvSubsampling.subsampled_lengths(lengths).item()) == t_out


def test_subsampled_lengths_matches_conv_output() -> None:
    # For each T the formula reports, the conv stack must actually yield
    # that many output frames.
    sub = ConvSubsampling(in_channels=1, d_model=16).eval()
    for t_in in [1, 2, 4, 5, 8, 16, 17, 33, 100, 101]:
        x = torch.randn(1, t_in, 1)
        with torch.no_grad():
            y = sub(x)
        expected = int(
            ConvSubsampling.subsampled_lengths(torch.tensor([t_in])).item()
        )
        assert y.size(1) == expected, f"T_in={t_in}: conv={y.size(1)} formula={expected}"
