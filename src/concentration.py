"""Shapley concentration metrics per §6.6."""

import numpy as np


def gini(values: np.ndarray) -> float:
    """Gini coefficient of non-negative values."""
    v = np.sort(values)
    n = len(v)
    if n == 0 or v.sum() == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float((2 * (index * v).sum() / (n * v.sum())) - (n + 1) / n)


def top_k_share(phi: np.ndarray, frac: float = 0.5) -> int:
    """Smallest k such that the top-k positive Shapley values hold ≥ frac of positive mass."""
    pos = np.sort(phi[phi > 0])[::-1]
    if len(pos) == 0:
        return 0
    total = pos.sum()
    cumsum = np.cumsum(pos)
    k = int(np.searchsorted(cumsum, total * frac) + 1)
    return min(k, len(pos))


def concentration_metrics(phi: np.ndarray) -> dict:
    """Compute all concentration metrics for a Shapley value vector."""
    pos_phi = phi[phi > 0]
    neg_phi = phi[phi < 0]

    return {
        "gini_positive": gini(pos_phi) if len(pos_phi) > 0 else 0.0,
        "top_k_50": top_k_share(phi, 0.5),
        "top_k_80": top_k_share(phi, 0.8),
        "top_k_90": top_k_share(phi, 0.9),
        "n_positive": int(len(pos_phi)),
        "n_negative": int(len(neg_phi)),
        "n_zero": int((phi == 0).sum()),
        "positive_mass": float(pos_phi.sum()),
        "negative_mass": float(neg_phi.sum()),
        "max_phi": float(phi.max()) if len(phi) > 0 else 0.0,
        "min_phi": float(phi.min()) if len(phi) > 0 else 0.0,
    }
