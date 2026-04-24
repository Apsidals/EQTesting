"""Regression losses for magnitude prediction."""

from __future__ import annotations

from torch import nn


def huber_loss(delta: float = 1.0) -> nn.Module:
    """Huber = quadratic near 0, linear beyond delta. Robust to label noise
    from mixed ml/mw scales without the gradient-explosion risk of pure MAE."""
    return nn.HuberLoss(delta=delta)


def mae_loss() -> nn.Module:
    return nn.L1Loss()
