"""Exponential moving average of model parameters.

Evaluating with EMA'd weights smooths out late-stage optimizer noise
and typically buys a small but reliable quality bump on ASR-style
tasks. Usage::

    ema = ExponentialMovingAverage(model, decay=0.9999)

    for step in range(...):
        # ... forward, backward, optimizer.step() ...
        ema.update(model)

    # For validation, swap in the EMA weights, restore when done:
    with ema.applied_to(model):
        cer = evaluate(model, val_set)
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch
from torch import nn


class ExponentialMovingAverage:
    """Keeps a shadow copy of every float parameter.

    Buffers (batch-norm running stats, etc.) are *not* averaged — they
    track the live model. The acoustic model uses LayerNorm only, so
    this simplification is free.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError("decay must be in (0, 1)")
        self.decay = decay
        self._shadow: dict[str, torch.Tensor] = {}
        for name, p in model.named_parameters():
            if p.requires_grad and p.dtype.is_floating_point:
                self._shadow[name] = p.detach().clone()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, p in model.named_parameters():
            if name in self._shadow:
                self._shadow[name].mul_(self.decay).add_(
                    p.detach(), alpha=1.0 - self.decay
                )

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {k: v.clone() for k, v in self._shadow.items()}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        for k, v in state.items():
            if k in self._shadow:
                self._shadow[k].copy_(v)

    @contextmanager
    def applied_to(self, model: nn.Module) -> Iterator[None]:
        """Temporarily swap EMA weights into ``model``; restore on exit."""
        backup: dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for name, p in model.named_parameters():
                if name in self._shadow:
                    backup[name] = p.detach().clone()
                    p.data.copy_(self._shadow[name])
        try:
            yield
        finally:
            with torch.no_grad():
                for name, p in model.named_parameters():
                    if name in backup:
                        p.data.copy_(backup[name])
