"""Learning-rate schedule: linear warmup then cosine decay to a floor.

The schedule returns a *multiplier* over the optimizer's base LR. Wire
it into a ``torch.optim.lr_scheduler.LambdaLR`` so that the underlying
optimizer keeps reporting its base LR cleanly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class WarmupCosineSchedule:
    """Linear warmup over ``warmup_steps``, then half-cosine decay from
    the peak multiplier down to ``min_lr_ratio`` at ``total_steps``.
    After ``total_steps`` the multiplier stays clamped at the floor.

    Attributes:
        warmup_steps: Steps of linear warmup from 0 → 1.
        total_steps: Step at which the cosine reaches the floor. Must be
                     strictly greater than ``warmup_steps``.
        min_lr_ratio: Final multiplier at ``total_steps`` (and after).
                      Keeping a non-zero floor avoids degenerate updates
                      at the end of training.
    """

    warmup_steps: int
    total_steps: int
    min_lr_ratio: float = 0.1

    def __post_init__(self) -> None:
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0")
        if self.total_steps <= self.warmup_steps:
            raise ValueError("total_steps must be > warmup_steps")
        if not 0.0 <= self.min_lr_ratio <= 1.0:
            raise ValueError("min_lr_ratio must be in [0, 1]")

    def __call__(self, step: int) -> float:
        if step < self.warmup_steps:
            if self.warmup_steps == 0:
                return 1.0
            return step / self.warmup_steps
        if step >= self.total_steps:
            return self.min_lr_ratio
        progress = (step - self.warmup_steps) / (
            self.total_steps - self.warmup_steps
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine
