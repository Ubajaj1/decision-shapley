"""Identify top-1 vs runner-up models and compute per-item differentials."""

import numpy as np


def model_scores(R: np.ndarray) -> np.ndarray:
    """Mean accuracy per model (row-wise mean of binary matrix)."""
    return R.mean(axis=1)


def top_two(R: np.ndarray, model_ids: list[str], rank1: int = 1) -> dict:
    """Identify a contest between adjacent-ranked models.

    Args:
        rank1: which rank to start from (1-based). rank1=1 gives #1 vs #2,
               rank1=3 gives #3 vs #4, etc.  Skips ties to find the first
               strict winner at or after the requested rank.

    Returns dict with keys: m1_idx, m2_idx, m1_id, m2_id, m1_score, m2_score,
    d (per-item differential), margin (sum of d), rank1, rank2.
    """
    scores = model_scores(R)
    ranking = np.argsort(-scores)

    start = max(0, rank1 - 1)
    for i in range(start, len(ranking) - 1):
        m1_idx = int(ranking[i])
        m2_idx = int(ranking[i + 1])

        if scores[m1_idx] == scores[m2_idx]:
            continue

        a = R[m1_idx].astype(np.int8)
        b = R[m2_idx].astype(np.int8)
        d = a - b

        return {
            "m1_idx": m1_idx,
            "m2_idx": m2_idx,
            "m1_id": model_ids[m1_idx],
            "m2_id": model_ids[m2_idx],
            "m1_score": float(scores[m1_idx]),
            "m2_score": float(scores[m2_idx]),
            "d": d,
            "margin": int(d.sum()),
            "n_plus": int((d == 1).sum()),
            "n_minus": int((d == -1).sum()),
            "n_zero": int((d == 0).sum()),
            "rank1": i + 1,
            "rank2": i + 2,
            "n_tied_at_top": i,
        }

    raise ValueError(f"No valid contest found at or after rank {rank1}.")


def value_function(d_subset: np.ndarray) -> float:
    """Binary flip-indicator value function v(S) per §6.2."""
    s = d_subset.sum()
    if s > 0:
        return 1.0
    elif s == 0:
        return 0.5
    else:
        return 0.0


def value_function_empty() -> float:
    """v({}) = 0.5 (tie by convention)."""
    return 0.5
