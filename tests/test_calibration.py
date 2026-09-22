"""Tests for the Step-4/5 within-contest external-validation gate.

Re-architected (2026-06-22) after the LiveBench pilot showed the pooled-Spearman
gate was dominated by between-contest structure (an ecological confound) and its
two-reference verdict inverted on real data. The gate is now:

  - PRIMARY statistic = within-contest Spearman(J_i, pivotality_i) among same-sign
    active items, aggregated (median) across qualifying contests. This isolates the
    genuine "does discrimination predict pivotality, holding the contest fixed"
    signal from between-contest leakage.
  - NULL = shuffle J labels WITHIN each contest, recompute per-contest rho, re-
    aggregate (R_perm reps). Floor = 95th pct; permutation p = P(null >= observed).
  - VERDICT = Supported if aggregate > floor, else No support. The 2PL ceiling is
    dropped (it sat below the floor and added nothing).

The pooled_spearman descriptive is retained only to document the confound.
"""

import numpy as np
import pytest

from _helpers import find_qualifying_pair, make_2pl_matrix
from decision import top_two
from calibration import (
    MIN_SAME_SIGN,
    contest_sample,
    pooled_spearman,
    within_contest_rhos,
    aggregate_signal,
    within_permutation_null,
    within_verdict,
    run_calibration,
    k_sensitivity_report,
)


# --------------------------------------------------------------------------- #
# pooled_spearman (descriptive only, kept)
# --------------------------------------------------------------------------- #

def test_pooled_spearman_perfect_monotone():
    a = (np.array([1.0, 2.0, 3.0, 4.0]), np.array([0.1, 0.2, 0.3, 0.4]))
    b = (np.array([5.0, 6.0, 7.0]), np.array([0.5, 0.6, 0.7]))
    assert pooled_spearman([a, b]) == pytest.approx(1.0)


def test_pooled_spearman_degenerate_is_nan():
    flat = (np.array([1.0, 2.0, 3.0]), np.array([0.5, 0.5, 0.5]))
    assert np.isnan(pooled_spearman([flat]))


# --------------------------------------------------------------------------- #
# contest_sample (same-sign filtering + qualification)
# --------------------------------------------------------------------------- #

def test_contest_sample_returns_same_sign_only():
    R, _ = make_2pl_matrix(150, [(1.0, b) for b in np.linspace(-1.5, 1.5, 24)], seed=7)
    c = find_qualifying_pair(R)
    sample = contest_sample(R, c["m1_idx"], c["m2_idx"], sign=1)
    assert sample is not None
    J, piv = sample
    assert J.shape[0] == c["n_plus"]
    assert piv.shape[0] == c["n_plus"]
    assert np.all((piv >= 0.0) & (piv <= 1.0))


def test_contest_sample_disqualifies_too_few_same_sign():
    R = np.array([
        [1, 1, 0, 0, 1],
        [0, 0, 1, 1, 0],   # d = [1,1,-1,-1,1] -> n_plus = 3 < 4
    ], dtype=float)
    assert contest_sample(R, 0, 1, sign=1) is None


def test_contest_sample_info_phi_weight():
    # secondary target: weight by info-VF Shapley phi instead of raw Fisher J
    R, _ = make_2pl_matrix(150, [(1.0, b) for b in np.linspace(-1.5, 1.5, 24)], seed=7)
    c = find_qualifying_pair(R)
    fisher = contest_sample(R, c["m1_idx"], c["m2_idx"], weight="fisher")
    phi = contest_sample(R, c["m1_idx"], c["m2_idx"], weight="info_phi", T=2000)
    assert phi is not None
    Jw, piv_j = fisher
    pw, piv_p = phi
    assert pw.shape == (c["n_plus"],)
    np.testing.assert_array_equal(piv_j, piv_p)      # pivotality unchanged by weight
    assert not np.allclose(Jw, pw)                   # phi is a different weighting


# --------------------------------------------------------------------------- #
# within_contest_rhos + aggregate_signal
# --------------------------------------------------------------------------- #

def test_within_contest_rhos_one_per_varying_contest():
    monotone = (np.array([1.0, 2.0, 3.0, 4.0]), np.array([0.1, 0.2, 0.3, 0.4]))
    anti = (np.array([1.0, 2.0, 3.0, 4.0]), np.array([0.4, 0.3, 0.2, 0.1]))
    rhos = within_contest_rhos([monotone, anti])
    np.testing.assert_allclose(np.sort(rhos), [-1.0, 1.0])


def test_within_contest_rhos_drops_constant_contest():
    monotone = (np.array([1.0, 2.0, 3.0, 4.0]), np.array([0.1, 0.2, 0.3, 0.4]))
    flat = (np.array([1.0, 2.0, 3.0]), np.array([0.5, 0.5, 0.5]))   # no variance
    rhos = within_contest_rhos([monotone, flat])
    assert rhos.shape == (1,)            # the constant contest is excluded
    assert rhos[0] == pytest.approx(1.0)


def test_aggregate_signal_median_and_mean():
    rhos = np.array([0.1, 0.2, 0.6])
    assert aggregate_signal(rhos, stat="median") == pytest.approx(0.2)
    assert aggregate_signal(rhos, stat="mean") == pytest.approx(0.3)


# --------------------------------------------------------------------------- #
# within_permutation_null
# --------------------------------------------------------------------------- #

def test_within_permutation_null_centered_near_zero():
    # strong within-contest signal -> the shuffled-null aggregate sits well below it
    rng = np.random.default_rng(0)
    samples = []
    for _ in range(4):
        J = np.arange(1.0, 9.0)
        piv = J / 10.0 + rng.normal(0, 0.01, J.shape)
        samples.append((J, piv))
    observed = aggregate_signal(within_contest_rhos(samples), stat="median")
    null = within_permutation_null(samples, R_perm=300, seed=1, stat="median")
    assert null.shape == (300,)
    assert np.nanpercentile(null, 95) < observed       # signal clears the floor
    assert abs(np.nanmedian(null)) < 0.3               # null centered near zero


# --------------------------------------------------------------------------- #
# within_verdict (deterministic decision logic)
# --------------------------------------------------------------------------- #

def test_within_verdict_supported():
    assert within_verdict(0.30, floor=0.12) == "Supported"


def test_within_verdict_no_support():
    assert within_verdict(0.05, floor=0.12) == "No support"


def test_within_verdict_boundary_is_no_support():
    # at-or-below the floor is not distinguishable from chance
    assert within_verdict(0.12, floor=0.12) == "No support"


# --------------------------------------------------------------------------- #
# run_calibration end-to-end + k-sensitivity
# --------------------------------------------------------------------------- #

def _tight_contest_R(seed=0):
    """Items clustered near the top ability -> tight leading contest with >= 4
    same-sign items and genuine across-item pivotality variance."""
    specs = [(a, b) for a, b in zip(np.linspace(0.6, 2.2, 30),
                                    np.linspace(0.5, 2.4, 30))]
    return make_2pl_matrix(150, specs, seed=seed)[0]


def test_run_calibration_end_to_end():
    R = _tight_contest_R(seed=0)
    contests = []
    ids = [str(i) for i in range(R.shape[0])]
    for r1 in (1, 3, 5, 7, 9):
        c = top_two(R, ids, rank1=r1)
        if contest_sample(R, c["m1_idx"], c["m2_idx"]) is not None:
            contests.append((R, c["m1_idx"], c["m2_idx"]))
    res = run_calibration(contests, R_perm=300, seed=5)
    assert set(res) >= {"aggregate", "floor", "p_value", "verdict",
                        "n_contests", "n_with_variance", "k", "stat",
                        "pooled_rho"}
    assert res["verdict"] in {"Supported", "No support"}
    assert res["n_contests"] == len(contests)
    assert 0.0 <= res["p_value"] <= 1.0
    # consistency: Supported iff aggregate clears the floor
    assert (res["verdict"] == "Supported") == (res["aggregate"] > res["floor"])


def test_k_sensitivity_reports_three_settings():
    R = _tight_contest_R(seed=3)
    ids = [str(i) for i in range(R.shape[0])]
    contests = []
    for r1 in (1, 3, 5, 7, 9):
        c = top_two(R, ids, rank1=r1)
        if contest_sample(R, c["m1_idx"], c["m2_idx"]) is not None:
            contests.append((R, c["m1_idx"], c["m2_idx"]))
    report = k_sensitivity_report(contests, R_perm=200, seed=6)
    assert set(report) == {"half_k", "k", "double_k"}
    for entry in report.values():
        assert entry["verdict"] in {"Supported", "No support"}
        assert entry["k"] >= 1
