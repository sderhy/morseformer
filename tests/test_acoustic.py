"""Unit tests for the acoustic Transformer model."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from morseformer.core.tokenizer import VOCAB_SIZE
from morseformer.models.acoustic import AcousticConfig, AcousticModel


def test_forward_shape_and_dtype() -> None:
    cfg = AcousticConfig(d_model=32, n_heads=2, n_layers=2, dim_feedforward=64)
    model = AcousticModel(cfg).eval()
    b, t, f = 2, 64, cfg.input_dim
    x = torch.randn(b, t, f)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (b, t, VOCAB_SIZE)
    assert out.dtype == torch.float32


def test_output_is_log_softmax() -> None:
    cfg = AcousticConfig(d_model=32, n_heads=2, n_layers=2, dim_feedforward=64)
    model = AcousticModel(cfg).eval()
    x = torch.randn(1, 16, cfg.input_dim)
    with torch.no_grad():
        out = model(x)
    probs = out.exp().sum(dim=-1)
    torch.testing.assert_close(probs, torch.ones_like(probs), atol=1e-5, rtol=1e-5)


def test_padding_mask_changes_nothing_for_full_batch() -> None:
    cfg = AcousticConfig(d_model=32, n_heads=2, n_layers=2, dim_feedforward=64)
    model = AcousticModel(cfg).eval()
    b, t = 2, 20
    x = torch.randn(b, t, cfg.input_dim)
    lengths = torch.tensor([t, t], dtype=torch.long)
    with torch.no_grad():
        out_masked = model(x, lengths=lengths)
        out_plain = model(x)
    torch.testing.assert_close(out_masked, out_plain, atol=1e-5, rtol=1e-5)


def test_padding_mask_ignores_padded_frames() -> None:
    # When a batch contains two sequences with different real lengths, the
    # output over the valid portion of the shorter one should not depend on
    # the padded tail.
    cfg = AcousticConfig(d_model=32, n_heads=2, n_layers=2, dim_feedforward=64)
    model = AcousticModel(cfg).eval()
    t = 40
    valid = 25
    x = torch.randn(1, t, cfg.input_dim)
    x_padded_a = x.clone()
    x_padded_b = x.clone()
    x_padded_a[:, valid:] = 0.0
    x_padded_b[:, valid:] = torch.randn(1, t - valid, cfg.input_dim) * 5.0
    lengths = torch.tensor([valid], dtype=torch.long)
    with torch.no_grad():
        out_a = model(x_padded_a, lengths=lengths)[:, :valid]
        out_b = model(x_padded_b, lengths=lengths)[:, :valid]
    torch.testing.assert_close(out_a, out_b, atol=1e-5, rtol=1e-5)


def test_ctc_loss_runs() -> None:
    cfg = AcousticConfig(d_model=32, n_heads=2, n_layers=2, dim_feedforward=64)
    model = AcousticModel(cfg).train()
    b, t = 2, 80
    x = torch.randn(b, t, cfg.input_dim)
    log_probs = model(x).transpose(0, 1)  # CTC wants [T, B, V]
    targets = torch.tensor([2, 3, 4, 5, 6, 7], dtype=torch.long)  # flattened
    input_lengths = torch.tensor([t, t], dtype=torch.long)
    target_lengths = torch.tensor([3, 3], dtype=torch.long)
    loss = torch.nn.functional.ctc_loss(
        log_probs, targets, input_lengths, target_lengths,
        blank=cfg.blank_index, zero_infinity=True,
    )
    assert torch.isfinite(loss)
    loss.backward()
    # Any gradient on the head proves end-to-end connectivity.
    assert model.head.weight.grad is not None
    assert torch.any(model.head.weight.grad != 0)


def test_parameter_count_is_modest() -> None:
    cfg = AcousticConfig()  # default hyperparameters
    model = AcousticModel(cfg)
    n = model.num_parameters()
    # ~150k–800k params for the default 4-layer / d=128 / ff=256 config.
    assert 50_000 < n < 2_000_000
