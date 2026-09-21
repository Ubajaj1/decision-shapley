"""Bootstrap flip probability estimation per §6.5."""

import numpy as np
from scipy import stats


def flip_probability(d: np.ndarray, B: int = 10_000, seed: int = 42) -> dict:
    """Resample items with replacement B times; measure how often the decision flips.

    Args:
        d: per-item differential array (d_i ∈ {-1, 0, +1}), length n_items.
        B: number of bootstrap replicates.
        seed: RNG seed.

    Returns dict with flip_prob, flip_count, tie_count, B, and Wilson CI.
    """
    rng = np.random.default_rng(seed)
    n = len(d)
    original_margin = d.sum()
    assert original_margin > 0, "Original decision must have m1 winning (margin > 0)"

    flips = 0
    ties = 0

    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        resampled_margin = d[idx].sum()
        if resampled_margin < 0:
            flips += 1
        elif resampled_margin == 0:
            ties += 1

    flip_prob = flips / B
    tie_prob = ties / B
    flip_or_tie_prob = (flips + ties) / B

    ci_lo, ci_hi = _wilson_interval(flips + ties, B)

    return {
        "flip_prob": flip_prob,
        "tie_prob": tie_prob,
        "flip_or_tie_prob": flip_or_tie_prob,
        "flip_count": flips,
        "tie_count": ties,
        "B": B,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "original_margin": int(original_margin),
        "seed": seed,
    }


def flip_probability_vectorized(d: np.ndarray, B: int = 10_000, seed: int = 42) -> dict:
    """Vectorized bootstrap — faster for large item counts."""
    rng = np.random.default_rng(seed)
    n = len(d)
    original_margin = d.sum()
    assert original_margin > 0

    indices = rng.integers(0, n, size=(B, n))
    resampled_margins = d[indices].sum(axis=1)

    flips = int((resampled_margins < 0).sum())
    ties = int((resampled_margins == 0).sum())

    ci_lo, ci_hi = _wilson_interval(flips + ties, B)

    return {
        "flip_prob": flips / B,
        "tie_prob": ties / B,
        "flip_or_tie_prob": (flips + ties) / B,
        "flip_count": flips,
        "tie_count": ties,
        "B": B,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "original_margin": int(original_margin),
        "seed": seed,
    }


def _wilson_interval(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 1.0)
    z = stats.norm.ppf(1 - alpha / 2)
    p_hat = k / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    spread = z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, center - spread), min(1.0, center + spread))
