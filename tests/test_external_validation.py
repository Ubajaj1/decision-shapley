"""Tests for the model-free external validation measurement layer (Step 4).

Pre-registered in notes/information-vf-preregistration.md Amendment 1 (+ calibration
update). The primary external measure is near-ability-twin pivotality:
  - abilities theta_m = logit(mean score)            [VF-free]
  - twin pairs = k = ceil(sqrt(n_models)) real model pairs nearest (theta_m1, theta_m2)
  - item i is pivotal for a pair iff removing it flips whether p beats q;
    pivotality_i = fraction of the k pairs for which i is pivotal.
"""

import math

import numpy as np
import pytest

from _helpers import make_2pl_matrix
from external_validation import (
    model_abilities,
    default_twin_k,
    nearest_twin_pairs,
    item_pivotality,
    external_pivotality,
    simulate_2pl,
)


# --------------------------------------------------------------------------- #
# Abilities
# --------------------------------------------------------------------------- #

def test_model_abilities_monotone_in_score():
    R, _ = make_2pl_matrix(120, [(1.0, b) for b in np.linspace(-1, 1, 10)], seed=1)
    theta = model_abilities(R)
    assert theta.shape == (R.shape[0],)
    # ability ordering matches mean-score ordering
    np.testing.assert_array_equal(np.argsort(theta), np.argsort(R.mean(axis=1)))


def test_model_abilities_is_logit_of_mean():
    R, _ = make_2pl_matrix(50, [(1.0, 0.0), (1.0, 0.5), (1.0, -0.5)], seed=2)
    theta = model_abilities(R)
    p = np.clip(R.mean(axis=1), 0.001, 0.999)
    np.testing.assert_allclose(theta, np.log(p / (1 - p)))


# --------------------------------------------------------------------------- #
# Twin neighborhood
# --------------------------------------------------------------------------- #

def test_default_twin_k():
    assert default_twin_k(160) == 13          # ceil(sqrt(160)) = 13
    assert default_twin_k(100) == 10
    assert default_twin_k(168) == 13
    assert default_twin_k(4) == 2


def test_nearest_twin_pairs_count_and_anchor():
    theta = np.array([0.0, 1.0, 2.0, 3.0, 0.1, 0.9])
    k = default_twin_k(len(theta))            # ceil(sqrt(6)) = 3
    pairs = nearest_twin_pairs(theta, m1_idx=0, m2_idx=1, k=k)
    assert len(pairs) == k
    # the actual contest pair has distance 0 -> must be present and first
    assert pairs[0] == (0, 1)
    # no degenerate self-pairs
    assert all(p != q for (p, q) in pairs)


def test_nearest_twin_pairs_sorted_by_distance():
    rng = np.random.default_rng(0)
    theta = rng.normal(0, 1, 40)
    pairs = nearest_twin_pairs(theta, m1_idx=3, m2_idx=7, k=10)
    t0, t1 = theta[3], theta[7]
    dists = [(theta[p] - t0) ** 2 + (theta[q] - t1) ** 2 for (p, q) in pairs]
    assert dists == sorted(dists)
    assert len(set(pairs)) == len(pairs)      # no duplicate pairs


def test_nearest_twin_pairs_large_n_matches_bruteforce():
    # n large enough to trigger the O(n) fast path (n^2 > 4M); the returned set of k
    # nearest non-self pairs must match the full-matrix brute force, and the distances
    # must be ascending.
    rng = np.random.default_rng(1)
    n = 2500
    theta = rng.normal(0, 1, n)
    m1, m2, k = 11, 23, 12
    pairs = nearest_twin_pairs(theta, m1, m2, k)
    assert len(pairs) == k
    assert pairs[0] == (m1, m2)               # contest pair, distance 0, first
    assert all(p != q for (p, q) in pairs)
    dp = (theta - theta[m1]) ** 2
    dq = (theta - theta[m2]) ** 2
    dist = dp[:, None] + dq[None, :]
    np.fill_diagonal(dist, np.inf)
    brute = set(map(tuple, np.argwhere(dist <= np.sort(dist.ravel())[k - 1])))
    assert set(pairs) <= brute                # all chosen pairs are among the k nearest
    fast_d = [dp[p] + dq[q] for (p, q) in pairs]
    assert fast_d == sorted(fast_d)


# --------------------------------------------------------------------------- #
# Item pivotality
# --------------------------------------------------------------------------- #

def test_item_pivotality_hand_computed_single_pair():
    # R[0] beats R[1] with margin M=1; only the items that tip M across 0 are pivotal
    R = np.array([
        [1, 1, 0, 1],   # model 0
        [0, 1, 1, 0],   # model 1
    ], dtype=float)
    # pair (0,1): d = [1,0,-1,1], M = 1.
    #   item0: M-1=0 -> not >0 -> flips (pivotal)
    #   item1: M-0=1 -> still wins -> no
    #   item2: M-(-1)=2 -> still wins -> no
    #   item3: M-1=0 -> flips (pivotal)
    piv = item_pivotality(R, [(0, 1)])
    np.testing.assert_array_equal(piv, [1.0, 0.0, 0.0, 1.0])


def test_item_pivotality_averages_over_pairs():
    R = np.array([
        [1, 1, 0, 1],
        [0, 1, 1, 0],
    ], dtype=float)
    # pair (1,0): d=[-1,0,1,-1], M=-1 (model1 loses); no single item can lift it >0
    piv = item_pivotality(R, [(0, 1), (1, 0)])
    np.testing.assert_array_equal(piv, [0.5, 0.0, 0.0, 0.5])


def test_item_pivotality_in_unit_range():
    R, _ = make_2pl_matrix(40, [(1.0, b) for b in np.linspace(-1, 1, 12)], seed=3)
    pairs = nearest_twin_pairs(model_abilities(R), 0, 1, k=default_twin_k(40))
    piv = item_pivotality(R, pairs)
    assert piv.shape == (R.shape[1],)
    assert np.all((piv >= 0.0) & (piv <= 1.0))


# --------------------------------------------------------------------------- #
# End-to-end + simulator
# --------------------------------------------------------------------------- #

def test_external_pivotality_end_to_end():
    R, _ = make_2pl_matrix(160, [(1.0, b) for b in np.linspace(-1.5, 1.5, 20)], seed=5)
    res = external_pivotality(R, m1_idx=0, m2_idx=1)
    assert res["k"] == default_twin_k(160)
    assert len(res["twin_pairs"]) == res["k"]
    assert res["pivotality"].shape == (R.shape[1],)
    assert np.all((res["pivotality"] >= 0.0) & (res["pivotality"] <= 1.0))


def test_simulate_2pl_recovers_probabilities():
    n_models, n_items = 4000, 5
    rng = np.random.default_rng(0)
    theta = rng.normal(0, 1, n_models)
    a = np.array([1.5, 0.5, 2.0, 1.0, 0.3])
    b = np.array([0.0, -0.5, 0.5, 1.0, -1.0])
    R = simulate_2pl(a, b, theta, rng)
    assert R.shape == (n_models, n_items)
    assert set(np.unique(R)).issubset({0.0, 1.0})
    # empirical correct-rate per item ~ mean of the 2PL probabilities
    p = 1.0 / (1.0 + np.exp(-a * (theta[:, None] - b)))
    np.testing.assert_allclose(R.mean(axis=0), p.mean(axis=0), atol=0.03)
