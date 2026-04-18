"""Unit tests for the acoustic Transformer model."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from morseformer.core.tokenizer import VOCAB_SIZE  # noqa: E402
from morseformer.models.acoustic import AcousticConfig, AcousticModel  # noqa: E402


def _small_cfg(**overrides) -> AcousticConfig:
    """Tiny config used for fast unit tests (far smaller than default)."""
    base = dict(d_model=32, n_heads=4, n_layers=2, ff_expansion=2, conv_kernel=7)
    base.update(overrides)
    return AcousticConfig(**base)


def test_forward_shape_and_dtype() -> None:
    cfg = _small_cfg()
    model = AcousticModel(cfg).eval()
    b, t, f = 2, 80, cfg.input_dim
    x = torch.randn(b, t, f)
    with torch.no_grad():
        log_probs, lengths_out = model(x)
    # Two stride-2 sub-sampling steps: 80 → 40 → 20.
    assert log_probs.shape == (b, 20, VOCAB_SIZE)
    assert log_probs.dtype == torch.float32
    assert lengths_out is None


def test_output_is_log_softmax() -> None:
    cfg = _small_cfg()
    model = AcousticModel(cfg).eval()
    x = torch.randn(1, 64, cfg.input_dim)
    with torch.no_grad():
        log_probs, _ = model(x)
    probs_sum = log_probs.exp().sum(dim=-1)
    torch.testing.assert_close(
        probs_sum, torch.ones_like(probs_sum), atol=1e-5, rtol=1e-5
    )


def test_lengths_out_is_subsampled() -> None:
    cfg = _small_cfg()
    model = AcousticModel(cfg).eval()
    x = torch.randn(2, 80, cfg.input_dim)
    lengths = torch.tensor([80, 60], dtype=torch.long)
    with torch.no_grad():
        _, lengths_out = model(x, lengths=lengths)
    assert lengths_out is not None
    # 4× sub-sampling: ceil(ceil(L/2)/2).
    assert lengths_out.tolist() == [20, 15]


def test_padding_is_respected() -> None:
    # Changing the padded tail of one sequence must not change the
    # output over its valid prefix.
    torch.manual_seed(0)
    cfg = _small_cfg()
    model = AcousticModel(cfg).eval()
    t, valid = 80, 52
    x_a = torch.randn(1, t, cfg.input_dim)
    x_b = x_a.clone()
    x_b[:, valid:] = torch.randn(1, t - valid, cfg.input_dim) * 5.0
    lengths = torch.tensor([valid], dtype=torch.long)
    with torch.no_grad():
        out_a, _ = model(x_a, lengths=lengths)
        out_b, _ = model(x_b, lengths=lengths)
    valid_out = int(((valid + 1) // 2 + 1) // 2)
    torch.testing.assert_close(
        out_a[:, :valid_out], out_b[:, :valid_out], atol=1e-5, rtol=1e-5
    )


def test_ctc_loss_runs_end_to_end() -> None:
    cfg = _small_cfg()
    model = AcousticModel(cfg).train()
    b, t = 2, 200
    x = torch.randn(b, t, cfg.input_dim)
    lengths_in = torch.tensor([t, t], dtype=torch.long)
    log_probs, lengths_out = model(x, lengths=lengths_in)
    assert lengths_out is not None
    # CTC expects [T, B, V].
    log_probs = log_probs.transpose(0, 1)
    targets = torch.tensor([2, 3, 4, 5, 6, 7], dtype=torch.long)
    target_lengths = torch.tensor([3, 3], dtype=torch.long)
    loss = torch.nn.functional.ctc_loss(
        log_probs, targets, lengths_out, target_lengths,
        blank=cfg.blank_index, zero_infinity=True,
    )
    assert torch.isfinite(loss)
    loss.backward()
    # Any gradient on the head proves end-to-end connectivity.
    assert model.head.weight.grad is not None
    assert torch.any(model.head.weight.grad != 0)


def test_default_config_parameter_count_is_modest() -> None:
    # Default config is the real Phase-2 model; assert it is in a sensible
    # range (~1–15 M params).
    model = AcousticModel(AcousticConfig())
    n = model.num_parameters()
    assert 1_000_000 < n < 15_000_000, f"default model has {n} params"
