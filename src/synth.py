"""Synthetic contest generator with KNOWN deciding structure (revision Step #1).

Purpose
-------
The reviewer's central objection is that the confidence / information value
functions are *engineered to avoid degeneracy, not validated against ground
truth*.  To answer it we need synthetic contests in which the true, graded
decisiveness of each item is known by construction and is defined WITHOUT
reference to any value function -- otherwise the validation is circular.

Design (non-circular)
---------------------
1. A 2PL/testlet IRT generative model produces a full binary response matrix
   ``R`` (models x items).  Conditional on ability, items are independent
   EXCEPT that items sharing a *testlet group* also share a per-model latent
   effect -> they become correlated ("redundant", they ask the same thing).
   Redundancy strength ``lam`` and discrimination ``a`` are both controllable,
   so we can probe each lever separately.

2. Ground truth = **population pivotality**: holding the item parameters and
   the two focal models fixed, we regenerate the responses many times and
   measure, per item, the fraction of resamples in which *removing that item
   alone* flips the m1-vs-m2 decision.  This is a value-function-free, graded
   measure of "how often is this item the lone pivot".  Redundant decisive
   items are individually LESS pivotal (their testlet partner covers them),
   independent decisive items MORE so -- exactly the gradation a correlation-
   aware attribution should recover and a correlation-blind one (flip-Shapley,
   Banzhaf, LOO, raw d) cannot.

The single-sample estimators (confidence / information / flip Shapley, LOO,
Banzhaf, raw d) each see ONE realized ``R`` and are scored by how well their
per-item ranking matches this population pivotality (see experiments/e18).

This module is the data-generating half; src/baselines.py holds the estimators
and experiments/e18_attribution_validation.py is the driver.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ContestParams:
    """Parameters of a synthetic contest's data-generating process.

    Attributes:
        theta:   (M,) latent abilities of all models.
        a:        (N,) item discriminations (true generative slopes).
        b:        (N,) item difficulties.
        group:    (N,) testlet-group id per item; -1 means "no group" (only the
                  shared-effect items use a non-negative id). Items in the same
                  group share a per-model latent effect (redundancy).
        lam:      (N,) loading on the shared group effect; 0 for ungrouped items.
        decisive: (K,) indices of the items designed to be contest-relevant
                  (difficulty near the focal pair's ability midpoint). The
                  validation is computed over the decisive set.
        m1, m2:   indices of the two focal models (m1 the intended winner).
    """

    theta: np.ndarray
    a: np.ndarray
    b: np.ndarray
    group: np.ndarray
    lam: np.ndarray
    decisive: np.ndarray
    m1: int
    m2: int
    meta: dict = field(default_factory=dict)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def sample_responses(p: ContestParams, rng: np.random.Generator) -> np.ndarray:
    """Draw one binary response matrix R (M x N) from the testlet 2PL model.

    logit P(R[m,i]=1) = a_i (theta_m - b_i) + lam_i * u[m, group_i]
    with u[m,g] ~ N(0,1) shared across items in group g for model m.
    """
    M = p.theta.shape[0]
    N = p.a.shape[0]
    n_groups = int(p.group.max()) + 1 if p.group.size and p.group.max() >= 0 else 0

    # Shared per-(model, group) latent effects.
    if n_groups > 0:
        u = rng.standard_normal((M, n_groups))
        shared = np.zeros((M, N), dtype=np.float64)
        grouped = p.group >= 0
        shared[:, grouped] = u[:, p.group[grouped]] * p.lam[grouped]
    else:
        shared = 0.0

    linpred = p.a[None, :] * (p.theta[:, None] - p.b[None, :]) + shared
    probs = _sigmoid(linpred)
    return (rng.random((M, N)) < probs).astype(np.int8)


def make_contest(
    n_models: int = 60,
    n_decisive: int = 8,
    n_neutral: int = 40,
    redundant_pairs: int = 2,
    lam: float = 1.5,
    disc_low: float = 0.4,
    disc_high: float = 1.6,
    ability_gap: float = 1.1,
    seed: int = 0,
) -> ContestParams:
    """Build a synthetic contest whose decisive items have known structure.

    The decisive items sit at difficulty == focal-pair midpoint so they
    actively separate m1 from m2. ``redundant_pairs`` of them are placed in
    shared testlet groups (loading ``lam``) to inject redundancy; the rest are
    independent. Discriminations are spread over [disc_low, disc_high] so the
    discrimination lever is also exercised. Neutral filler items sit far from
    the midpoint (they rarely differentiate the focal pair).

    The two focal models m1, m2 are given near-equal abilities (separated by
    ``ability_gap``) so the realized contest is tight, mirroring the fragile
    top-of-leaderboard regime studied in the paper.
    """
    rng = np.random.default_rng(seed)

    # Abilities: a spread of "other" models plus two close focal models on top.
    theta = rng.normal(0.0, 1.0, size=n_models)
    theta_star = float(np.quantile(theta, 0.85))
    m1, m2 = 0, 1
    theta[m1] = theta_star + ability_gap / 2.0
    theta[m2] = theta_star - ability_gap / 2.0

    N = n_decisive + n_neutral
    a = np.empty(N)
    b = np.empty(N)
    group = np.full(N, -1, dtype=int)
    lam_vec = np.zeros(N)

    # Decisive items: difficulty at the runner-up's ability (so m2 is ~50% and
    # the higher-ability winner m1 is reliably above it -> E[d_i] > 0, a real
    # decision signal), discrimination spread over the requested range.
    dec_idx = np.arange(n_decisive)
    a[dec_idx] = np.linspace(disc_low, disc_high, n_decisive)
    b[dec_idx] = theta[m2]
    rng.shuffle(a[:n_decisive])  # decouple discrimination order from index

    # Inject redundancy: pair up the first 2*redundant_pairs decisive items
    # into shared testlet groups.
    g = 0
    for pr in range(redundant_pairs):
        i, j = 2 * pr, 2 * pr + 1
        if j >= n_decisive:
            break
        group[i] = group[j] = g
        lam_vec[i] = lam_vec[j] = lam
        g += 1

    # Neutral filler: difficulty away from the midpoint, modest discrimination.
    neu_idx = np.arange(n_decisive, N)
    a[neu_idx] = rng.uniform(0.3, 0.8, size=n_neutral)
    b[neu_idx] = theta_star + rng.choice([-1.0, 1.0], size=n_neutral) * rng.uniform(1.5, 3.0, size=n_neutral)

    return ContestParams(
        theta=theta, a=a, b=b, group=group, lam=lam_vec,
        decisive=dec_idx, m1=m1, m2=m2,
        meta={"theta_star": theta_star, "n_groups": g, "lam": lam,
              "redundant_pairs": redundant_pairs, "seed": seed},
    )


def _exact_shapley_from_values(v: np.ndarray, K: int) -> np.ndarray:
    """Exact Shapley values over K players given v[mask] for every coalition.

    v is indexed by bitmask (player i in coalition iff bit i set), length 2**K.
    """
    from math import factorial

    phi = np.zeros(K, dtype=np.float64)
    # Precompute Shapley weights by coalition size s = |S| (S excludes player i).
    w = np.array([factorial(s) * factorial(K - s - 1) / factorial(K)
                  for s in range(K)], dtype=np.float64)
    popcount = np.array([bin(m).count("1") for m in range(1 << K)])
    for i in range(K):
        bit = 1 << i
        for mask in range(1 << K):
            if mask & bit:
                continue
            s = popcount[mask]
            phi[i] += w[s] * (v[mask | bit] - v[mask])
    return phi


def population_decision_shapley(
    p: ContestParams, n_sims: int = 3000, seed: int = 12345,
) -> dict:
    """Value-function-free ground truth: Shapley of the TRUE decision probability.

    We estimate the population decision-probability value function
        v*(S) = P_pop( sum_{i in S} d_i > 0 )      (v*(empty) = 1/2, ties = 1/2)
    over the decisive items by Monte-Carlo over fresh draws of the generative
    model, then take its EXACT Shapley value per decisive item. This uses no
    modelling assumption about the value function (unlike v_conf's Gaussian
    form); it is the credit each item deserves for the true decision, against
    which any single-sample estimator can be scored.

    Because v* is the genuine population probability, it splits credit among
    redundant (testlet) items: a near-duplicate adds little marginal decision
    probability once its partner is in the coalition. This is the gradation a
    correlation-aware attribution should recover.

    Returns dict with:
        shapley:    (K,) ground-truth Shapley over decisive items.
        decisive:   the decisive item indices (for alignment).
        e_d:        (K,) population mean differential E[d_i] per decisive item.
        v_all:      v*(full decisive set).
    """
    dec = p.decisive
    K = len(dec)
    if K > 16:
        raise ValueError(f"exact ground-truth Shapley is 2**K; K={K} too large")

    rng = np.random.default_rng(seed)
    d_draws = np.empty((n_sims, K), dtype=np.int8)
    for t in range(n_sims):
        R = sample_responses(p, rng)
        d_draws[t] = (R[p.m1].astype(np.int8) - R[p.m2].astype(np.int8))[dec]

    # Coalition value v*[mask] = mean over draws of flip-indicator(sum_{i in S} d_i).
    idx_lists = [np.array([i for i in range(K) if (m >> i) & 1], dtype=int)
                 for m in range(1 << K)]
    v = np.empty(1 << K, dtype=np.float64)
    v[0] = 0.5
    for m in range(1, 1 << K):
        s = d_draws[:, idx_lists[m]].sum(axis=1)
        v[m] = float(np.mean(np.where(s > 0, 1.0, np.where(s < 0, 0.0, 0.5))))

    phi = _exact_shapley_from_values(v, K)
    return {
        "shapley": phi,
        "decisive": dec,
        "e_d": d_draws.mean(axis=0),
        "v_all": float(v[(1 << K) - 1]),
        "n_sims": n_sims,
    }


def realize_contest(p: ContestParams, seed: int,
                    target_margin=(1, 60), max_tries: int = 500):
    """Draw a single realized R whose decisive-set margin lands in a tight band.

    Mirrors the paper's fragile regime (margin 1-3). Returns (R, d_full, info)
    where d_full is the full per-item differential for the focal pair. Raises if
    a margin in band is not found within max_tries.
    """
    rng = np.random.default_rng(seed)
    lo, hi = target_margin
    for t in range(max_tries):
        R = sample_responses(p, rng)
        d_full = R[p.m1].astype(np.int8) - R[p.m2].astype(np.int8)
        margin = int(d_full[p.decisive].sum())
        if lo <= margin <= hi:
            return R, d_full, {"margin": margin, "tries": t + 1}
    raise RuntimeError(
        f"No realized margin in {target_margin} after {max_tries} tries "
        f"(seed {seed}); widen the band or adjust ability_gap.")
