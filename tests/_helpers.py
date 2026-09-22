"""Shared test helpers: synthetic 2PL data and a generic exact Shapley."""

from itertools import permutations
from math import factorial

import numpy as np


def make_2pl_matrix(n_models, item_specs, seed=0):
    """Binary response matrix from a 2PL model.

    item_specs: list of (a, b) discrimination/difficulty pairs.
    Returns (R, theta).
    """
    rng = np.random.default_rng(seed)
    theta = rng.normal(0.0, 1.0, n_models)
    R = np.zeros((n_models, len(item_specs)), dtype=np.float64)
    for j, (a, b) in enumerate(item_specs):
        p = 1.0 / (1.0 + np.exp(-a * (theta - b)))
        R[:, j] = (rng.random(n_models) < p).astype(np.float64)
    return R, theta


def find_qualifying_pair(R, sign=1, min_same_sign=4):
    """Return a deterministic strict-score model pair with enough active items.

    Calibration unit tests exercise same-sign filtering rather than leaderboard
    tie-breaking. Selecting the top adjacent pair made those tests depend on the
    sort order of equal-scoring synthetic models, which varies across NumPy
    versions. This helper chooses the first qualifying pair by model index.
    """
    R = np.asarray(R)
    scores = R.mean(axis=1)
    for m1_idx in range(R.shape[0]):
        for m2_idx in range(R.shape[0]):
            if scores[m1_idx] <= scores[m2_idx]:
                continue
            d = R[m1_idx] - R[m2_idx]
            if int((d == sign).sum()) >= min_same_sign:
                return {
                    "m1_idx": m1_idx,
                    "m2_idx": m2_idx,
                    "n_plus": int((d == 1).sum()),
                    "n_minus": int((d == -1).sum()),
                }
    raise AssertionError("Synthetic matrix contains no qualifying model pair.")


def exact_shapley(n, value_fn):
    """Exhaustive-permutation exact Shapley for an arbitrary value function.

    value_fn takes a tuple/list of player indices (a coalition) and returns its
    value; value_fn(()) is the empty-coalition value. Returns phi of length n.
    """
    phi = np.zeros(n, dtype=np.float64)
    for perm in permutations(range(n)):
        S, prev = [], value_fn(())
        for i in perm:
            S.append(i)
            cur = value_fn(tuple(S))
            phi[i] += cur - prev
            prev = cur
    return phi / factorial(n)
