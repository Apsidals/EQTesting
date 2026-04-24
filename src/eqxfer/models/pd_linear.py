"""Rung 1: Pd-only log-linear regression.

    M_pred = intercept + slope * log10(Pd)

Fit via weighted least squares. Weights equalize magnitude bins up to
20× (CLAUDE.md class-imbalance rule)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

LOG10_FLOOR = 1e-20


def bin_balanced_weights(
    magnitudes: np.ndarray,
    bin_width: float = 1.0,
    cap: float = 20.0,
) -> np.ndarray:
    """Per-sample weights that equalize 1.0-wide magnitude bins, capped at
    `cap`× the most-populated bin."""
    bins = np.floor(magnitudes / bin_width).astype(int)
    unique, counts = np.unique(bins, return_counts=True)
    count_of = dict(zip(unique.tolist(), counts.tolist()))
    max_count = counts.max()
    weights = np.array([max_count / count_of[int(b)] for b in bins], dtype=np.float64)
    return np.minimum(weights, cap)


@dataclass(frozen=True)
class PdLinear:
    intercept: float
    slope: float

    @classmethod
    def fit(
        cls,
        pd_values: np.ndarray,
        magnitudes: np.ndarray,
        weights: np.ndarray | None = None,
    ) -> "PdLinear":
        if len(pd_values) != len(magnitudes):
            raise ValueError("pd_values and magnitudes length mismatch")
        log_pd = np.log10(np.maximum(pd_values, LOG10_FLOOR))
        x = np.column_stack([np.ones_like(log_pd), log_pd])
        y = np.asarray(magnitudes, dtype=np.float64)
        if weights is None:
            coef, *_ = np.linalg.lstsq(x, y, rcond=None)
        else:
            sqrt_w = np.sqrt(np.asarray(weights, dtype=np.float64))
            coef, *_ = np.linalg.lstsq(x * sqrt_w[:, None], y * sqrt_w, rcond=None)
        return cls(intercept=float(coef[0]), slope=float(coef[1]))

    def predict(self, pd_values: np.ndarray) -> np.ndarray:
        log_pd = np.log10(np.maximum(pd_values, LOG10_FLOOR))
        return self.intercept + self.slope * log_pd

    def to_dict(self) -> dict[str, float]:
        return {"intercept": self.intercept, "slope": self.slope}
