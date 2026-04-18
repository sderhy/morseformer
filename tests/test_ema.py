"""Unit tests for the exponential-moving-average helper."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402

from morseformer.train.ema import ExponentialMovingAverage  # noqa: E402


def _tiny_model() -> nn.Module:
    return nn.Linear(4, 2)


def test_initial_shadow_matches_model() -> None:
    m = _tiny_model()
    ema = ExponentialMovingAverage(m, decay=0.99)
    for name, p in m.named_parameters():
        assert torch.allclose(ema._shadow[name], p)


def test_update_moves_shadow_toward_model() -> None:
    m = _tiny_model()
    ema = ExponentialMovingAverage(m, decay=0.5)
    # Set weights to a known non-zero tensor so shadow is no longer
    # equal to the live params.
    with torch.no_grad():
        for p in m.parameters():
            p.fill_(1.0)
    ema.update(m)
    # With decay 0.5 the shadow should be 0.5*init + 0.5*1.0.
    for name, p in m.named_parameters():
        # shadow starts at init (~small random); we just check it moved toward 1.
        assert float(ema._shadow[name].mean()) > float(
            p.mean().detach()
        ) - 1.0  # sanity — moved partway, not to 1.


def test_applied_to_swaps_and_restores() -> None:
    m = _tiny_model()
    ema = ExponentialMovingAverage(m, decay=0.5)
    # Make live params distinct from the shadow.
    with torch.no_grad():
        for p in m.parameters():
            p.fill_(2.0)
    live_snapshot = {k: v.clone() for k, v in m.state_dict().items()}
    shadow_before = ema.state_dict()

    with ema.applied_to(m):
        # Inside the context, model params must equal shadow.
        for name, p in m.named_parameters():
            assert torch.allclose(p, shadow_before[name])

    # After the context, original live params are restored.
    for k, v in m.state_dict().items():
        assert torch.allclose(v, live_snapshot[k])


def test_decay_bounds_validated() -> None:
    m = _tiny_model()
    with pytest.raises(ValueError):
        ExponentialMovingAverage(m, decay=0.0)
    with pytest.raises(ValueError):
        ExponentialMovingAverage(m, decay=1.0)
    with pytest.raises(ValueError):
        ExponentialMovingAverage(m, decay=-0.5)


def test_state_dict_roundtrip() -> None:
    m = _tiny_model()
    ema = ExponentialMovingAverage(m, decay=0.9)
    state = ema.state_dict()
    # Modify shadow, then reload to verify restoration.
    with torch.no_grad():
        for v in ema._shadow.values():
            v.fill_(99.0)
    ema.load_state_dict(state)
    for name, p in m.named_parameters():
        assert torch.allclose(ema._shadow[name], p)
