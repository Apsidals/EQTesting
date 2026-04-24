"""Embedding alignment diagnostics — is the Universal Physics Encoder's
output distribution actually shared across regions, or has it memorized
California?

Metrics:
- MMD (maximum mean discrepancy) with an RBF kernel between source and
  target embedding clouds. Low MMD = overlapping distributions = good
  transfer evidence.
- Silhouette score of region labels. Lower = regions are mixed in
  embedding space = good.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import silhouette_score


def _pairwise_sq_dists(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_sq = np.sum(a**2, axis=1, keepdims=True)
    b_sq = np.sum(b**2, axis=1, keepdims=True)
    cross = a @ b.T
    return a_sq + b_sq.T - 2.0 * cross


def rbf_mmd2(
    source: np.ndarray,
    target: np.ndarray,
    bandwidth: float | None = None,
) -> float:
    """Biased MMD² estimate with an RBF kernel.

    bandwidth=None uses the median heuristic on the combined cloud, which is
    the standard default for two-sample MMD."""
    if source.ndim != 2 or target.ndim != 2:
        raise ValueError("source/target must be 2D (N, D)")
    if source.shape[1] != target.shape[1]:
        raise ValueError(f"dim mismatch: {source.shape[1]} vs {target.shape[1]}")

    if bandwidth is None:
        combined = np.vstack([source, target])
        d = _pairwise_sq_dists(combined, combined)
        # median of non-zero entries (exclude diagonal)
        off_diag = d[~np.eye(d.shape[0], dtype=bool)]
        med = float(np.median(off_diag))
        bandwidth = np.sqrt(med / 2.0) if med > 0 else 1.0

    gamma = 1.0 / (2.0 * bandwidth**2)
    k_ss = np.exp(-gamma * _pairwise_sq_dists(source, source))
    k_tt = np.exp(-gamma * _pairwise_sq_dists(target, target))
    k_st = np.exp(-gamma * _pairwise_sq_dists(source, target))
    return float(k_ss.mean() + k_tt.mean() - 2.0 * k_st.mean())


def region_silhouette(embeddings: np.ndarray, region_labels: np.ndarray) -> float:
    """sklearn silhouette score of region labels. Returns nan if only one
    region or fewer than 2 samples per region."""
    unique = np.unique(region_labels)
    if len(unique) < 2:
        return float("nan")
    counts = np.array([np.sum(region_labels == u) for u in unique])
    if counts.min() < 2:
        return float("nan")
    return float(silhouette_score(embeddings, region_labels))
