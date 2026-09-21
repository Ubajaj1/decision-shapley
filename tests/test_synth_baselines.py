"""Smoke + correctness tests for the attribution-validation harness (Step #1).

Covers src/synth.py (generative model, population pivotality) and
src/baselines.py (Banzhaf, LOO, dispatcher). These guard the contracts the
e18 experiment relies on; they are fast (small matrices, few sims).
"""

import numpy as np
import pytest

from synth import (
    make_contest, sample_responses, population_decision_shapley, realize_contest,
)
from baselines import (
    banzhaf_flip, loo_indicator, raw_differential, attribution_scores,
    ALL_ATTRIBUTIONS,
)


# --------------------------------------------------------------------------- #
# Generative model
# --------------------------------------------------------------------------- #

def test_sample_responses_shape_and_binary():
    p = make_contest(n_models=40, n_decisive=8, n_neutral=20, seed=1)
    R = sample_responses(p, np.random.default_rng(0))
    assert R.shape == (40, 28)
    assert set(np.unique(R)).issubset({0, 1})


def test_focal_winner_outscores_on_average():
    # m1 has the higher ability, so over many draws it should win on average.
    p = make_contest(seed=2)
    rng = np.random.default_rng(0)
    diffs = []
    for _ in range(200):
        R = sample_responses(p, rng)
        diffs.append(R[p.m1].sum() - R[p.m2].sum())
    assert np.mean(diffs) > 0


def test_redundant_items_get_less_true_credit_than_independent():
    """The core ground-truth property: with discrimination held equal, testlet
    (redundant) decisive items receive lower true-decision Shapley than
    independent ones -- the population value function splits their credit."""
    # Equal discrimination isolates the redundancy lever from the disc lever.
    p = make_contest(n_decisive=10, redundant_pairs=2, lam=3.0,
                     disc_low=1.0, disc_high=1.0, ability_gap=1.1, seed=3)
    gt = population_decision_shapley(p, n_sims=5000, seed=7)
    phi = gt["shapley"]
    # Items 0..3 are the two redundant pairs; 4..9 are independent.
    redundant = phi[:4]
    independent = phi[4:]
    assert independent.mean() > redundant.mean()


def test_realize_contest_hits_target_margin_band():
    p = make_contest(seed=4)
    R, d_full, info = realize_contest(p, seed=11, target_margin=(1, 60))
    assert 1 <= info["margin"] <= 60
    assert R.shape[0] == p.theta.shape[0]


# --------------------------------------------------------------------------- #
# Baselines
# --------------------------------------------------------------------------- #

def test_banzhaf_exact_symmetry_for_same_sign_items():
    # All +1 items are interchangeable -> equal Banzhaf swing (the degeneracy).
    d = np.array([1, 1, 1, -1, 0, 0])
    res = banzhaf_flip(d, exact_max=18)
    phi = res["phi"]
    assert res["mode"] == "exact"
    pos = phi[d == 1]
    assert np.allclose(pos, pos[0])           # symmetric
    assert np.allclose(phi[d == 0], 0.0)      # neutral items get nothing


def test_banzhaf_mc_matches_exact():
    d = np.array([1, 1, -1, 1, -1, 1, 0, 0, 1, -1])
    exact = banzhaf_flip(d, exact_max=99)["phi"]
    mc = banzhaf_flip(d, T=60_000, seed=1, exact_max=0)["phi"]
    assert np.allclose(exact, mc, atol=0.02)


def test_loo_indicator_margin_one_flags_all_plus():
    d = np.array([1, 1, 1, -1, -1, 0])         # margin = 1
    phi = loo_indicator(d)["phi"]
    assert np.allclose(phi[d == 1], 1.0)       # every +1 ties on removal
    assert np.allclose(phi[d != 1], 0.0)


def test_raw_differential_returns_d():
    d = np.array([1, -1, 0, 1])
    assert np.allclose(raw_differential(d)["phi"], d)


def test_dispatcher_returns_full_length_for_all():
    p = make_contest(n_models=40, n_decisive=8, n_neutral=15, seed=5)
    R, d_full, _ = realize_contest(p, seed=9, target_margin=(1, 60))
    for name in ALL_ATTRIBUTIONS:
        phi = attribution_scores(name, d_full, R, p.m1, p.m2, T=500, seed=0)
        assert phi.shape == d_full.shape
        # inactive items always score zero under every attribution here
        assert np.allclose(phi[d_full == 0], 0.0)
