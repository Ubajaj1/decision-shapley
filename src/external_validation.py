"""Model-free external validation measurement layer (Step 4).

Pre-registered in notes/information-vf-preregistration.md (Amendment 1 + calibration
update). The primary external measure is near-ability-twin pivotality, computed without
any value function:

  - abilities theta_m = logit(mean score)                       [VF-free]
  - twin pairs = k = ceil(sqrt(n_models)) real model pairs nearest the contest pair
    (theta_m1, theta_m2) in ability space
  - item i is pivotal for a pair iff removing it flips whether p beats q;
    pivotality_i = fraction of the k pairs for which i is pivotal.

simulate_2pl provides the parametric ceiling's data generator.
"""

import math

import numpy as np


# --------------------------------------------------------------------------- #
# Abilities
# --------------------------------------------------------------------------- #

def model_abilities(R):
    """theta_m = logit of each model's mean score across items. Value-function free."""
    R = np.asarray(R, dtype=float)
    p = np.clip(R.mean(axis=1), 0.001, 0.999)
    return np.log(p / (1.0 - p))


# --------------------------------------------------------------------------- #
# Twin neighborhood
# --------------------------------------------------------------------------- #

def default_twin_k(n_models):
    """Pre-registered neighborhood size: ceil(sqrt(n_models)). No free bandwidth."""
    return math.ceil(math.sqrt(n_models))


def nearest_twin_pairs(theta, m1_idx, m2_idx, k):
    """The k ordered model pairs (p, q) nearest the contest pair in ability space.

    Distance is squared Euclidean from (theta[m1_idx], theta[m2_idx]). Self-pairs
    (p == q) are excluded. The contest pair (m1_idx, m2_idx) is distance 0 and so
    sorts first. Returns a list of k (p, q) tuples sorted by ascending distance.
    """
    theta = np.asarray(theta, dtype=float)
    n = theta.shape[0]
    dp = (theta - theta[m1_idx]) ** 2          # cost of p standing in for m1
    dq = (theta - theta[m2_idx]) ** 2          # cost of q standing in for m2

    # Large n: the full n x n distance matrix is O(n^2) memory (360 MB at n=6700) and is
    # rebuilt per contest. The k nearest pair-sums dp[p]+dq[q] are guaranteed to come
    # from the m smallest dp and m smallest dq (m = 2k+buffer), so we only score an
    # m x m grid -> O(n + k^2). Below the threshold we keep the exact full-matrix path so
    # the small-input tie ordering is unchanged.
    if n * n > 4_000_000 and n > 2 * k + 4:
        m = min(n, 2 * k + 4)
        pi = np.argpartition(dp, m - 1)[:m]
        qi = np.argpartition(dq, m - 1)[:m]
        sums = dp[pi][:, None] + dq[qi][None, :]
        sums[pi[:, None] == qi[None, :]] = np.inf      # forbid self-pairs
        order = np.argsort(sums.ravel(), kind="stable")[:k]
        return [(int(pi[idx // m]), int(qi[idx % m])) for idx in order]

    dist = dp[:, None] + dq[None, :]
    np.fill_diagonal(dist, np.inf)             # forbid degenerate self-pairs
    order = np.argsort(dist.ravel(), kind="stable")[:k]
    return [(int(idx // n), int(idx % n)) for idx in order]


# --------------------------------------------------------------------------- #
# Item pivotality
# --------------------------------------------------------------------------- #

def item_pivotality(R, twin_pairs):
    """Fraction of twin pairs for which removing each item flips who wins.

    For pair (p, q): d = R[p] - R[q], margin M = sum(d), convention "p wins" = M > 0.
    Item i is pivotal iff dropping it (M - d[i]) changes the sign decision.
    """
    R = np.asarray(R, dtype=float)
    n_items = R.shape[1]
    piv = np.zeros(n_items)
    if not twin_pairs:
        return piv
    for p, q in twin_pairs:
        d = R[p] - R[q]
        M = d.sum()
        wins = M > 0
        flips = ((M - d) > 0) != wins
        piv += flips
    return piv / len(twin_pairs)


# --------------------------------------------------------------------------- #
# End-to-end + simulator
# --------------------------------------------------------------------------- #

def external_pivotality(R, m1_idx, m2_idx, k=None):
    """Run the full VF-free pipeline for one contest; k defaults to ceil(sqrt(n))."""
    R = np.asarray(R, dtype=float)
    theta = model_abilities(R)
    if k is None:
        k = default_twin_k(R.shape[0])
    pairs = nearest_twin_pairs(theta, m1_idx, m2_idx, k)
    piv = item_pivotality(R, pairs)
    return {"pivotality": piv, "twin_pairs": pairs, "k": k, "theta": theta}


def simulate_2pl(a, b, theta, rng):
    """Bernoulli(sigmoid(a_i (theta_m - b_i))) response matrix for the 2PL ceiling."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    theta = np.asarray(theta, dtype=float)
    p = 1.0 / (1.0 + np.exp(-a * (theta[:, None] - b)))
    return (rng.random(p.shape) < p).astype(float)
