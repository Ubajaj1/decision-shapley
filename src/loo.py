"""Leave-one-out flip analysis per §6.5."""

import numpy as np


def loo_flips(d: np.ndarray) -> dict:
    """Compute LOO flips: items whose removal changes the decision.

    Removal of item i changes the margin from M to M - d_i.
    A flip (or tie) occurs when M - d_i <= 0, i.e. d_i >= M.
    Since d_i ∈ {-1, 0, +1}, this is only possible when M = 1 and d_i = 1.

    Returns dict with flip item indices, counts, and analysis.
    """
    margin = int(d.sum())
    n = len(d)
    assert margin > 0, "Original decision must have m1 winning"

    flip_to_tie_indices = []
    flip_to_loss_indices = []

    for i in range(n):
        new_margin = margin - int(d[i])
        if new_margin == 0:
            flip_to_tie_indices.append(i)
        elif new_margin < 0:
            flip_to_loss_indices.append(i)

    return {
        "margin": margin,
        "n_items": n,
        "n_loo_tie": len(flip_to_tie_indices),
        "n_loo_loss": len(flip_to_loss_indices),
        "n_loo_any": len(flip_to_tie_indices) + len(flip_to_loss_indices),
        "loo_tie_indices": np.array(flip_to_tie_indices, dtype=int),
        "loo_loss_indices": np.array(flip_to_loss_indices, dtype=int),
        "loo_vulnerable": margin <= 1,
    }


def decisive_item_count(d: np.ndarray) -> int:
    """Smallest set of items whose removal flips the decision.

    Greedy: remove d_i = +1 items first (each reduces margin by 1).
    Need to remove enough +1 items to bring margin to 0.
    """
    margin = int(d.sum())
    n_plus = int((d == 1).sum())
    if margin <= 0:
        return 0
    return min(margin, n_plus)
