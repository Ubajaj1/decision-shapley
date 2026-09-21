"""Tests for the correlation-aware confidence value function (revision Step 1).

Covers:
  - compute_item_weights        (1.1) — 2PL discrimination/difficulty/Fisher
  - compute_active_correlation  (1.2) — phi correlation among active items
  - mc_shapley_confidence       (1.3/1.4) — confidence-Shapley with O(n_active) walk

Acceptance criteria are drawn from memory/project_revision-plan.md.
"""

from itertools import permutations
from math import factorial

import numpy as np
import pytest
from scipy.stats import norm

from shapley import (
    compute_item_weights,
    compute_active_correlation,
    mc_shapley_confidence,
    _mc_shapley_confidence_core,
    mc_shapley_flip,
)


# --------------------------------------------------------------------------- #
# Synthetic data helpers
# --------------------------------------------------------------------------- #

def make_2pl_matrix(n_models, item_specs, seed=0):
    """Generate a binary response matrix from a 2PL model.

    item_specs: list of (a, b) discrimination/difficulty pairs.
    Returns (R, theta) where R is (n_models x n_items) of {0,1} floats.
    """
    rng = np.random.default_rng(seed)
    theta = rng.normal(0.0, 1.0, n_models)
    R = np.zeros((n_models, len(item_specs)), dtype=np.float64)
    for j, (a, b) in enumerate(item_specs):
        p = 1.0 / (1.0 + np.exp(-a * (theta - b)))
        R[:, j] = (rng.random(n_models) < p).astype(np.float64)
    return R, theta


def exact_confidence_shapley(d_active, a_active, C, link="probit"):
    """Brute-force exact Shapley of the confidence value function (independent
    reference implementation for cross-checking the MC core)."""
    w = a_active.astype(float) * d_active.astype(float)
    n = len(w)
    phi = np.zeros(n)

    def v(S):
        if not S:
            return 0.5
        S = list(S)
        wS = w[S]
        CS = C[np.ix_(S, S)]
        mu = wS.sum()
        var = wS @ CS @ wS
        z = mu / np.sqrt(max(var, 1e-10))
        if link == "probit":
            return float(norm.cdf(z))
        return 1.0 if z > 0 else (0.0 if z < 0 else 0.5)

    for perm in permutations(range(n)):
        S, prev = [], 0.5
        for i in perm:
            S.append(i)
            cur = v(S)
            phi[i] += cur - prev
            prev = cur
    return phi / factorial(n)


# --------------------------------------------------------------------------- #
# 1.1 compute_item_weights
# --------------------------------------------------------------------------- #

def test_compute_item_weights_matches_e5():
    """Acceptance 1.1: values match the existing E5 fit_2pl_and_fisher code."""
    from e5_localized_fisher import fit_2pl_and_fisher

    R, _ = make_2pl_matrix(
        120,
        [(1.5, 0.0), (0.3, -0.5), (1.0, 0.8), (0.7, -1.0), (2.0, 0.2)],
        seed=7,
    )
    m1_idx, m2_idx = 3, 9  # arbitrary two models
    w = compute_item_weights(R, m1_idx, m2_idx)
    irt = fit_2pl_and_fisher(R, m1_idx, m2_idx)

    np.testing.assert_allclose(w["a"], irt["discrimination"])
    np.testing.assert_allclose(w["b"], irt["difficulty"])
    np.testing.assert_allclose(w["J"], irt["fisher_info"])
    assert w["theta_star"] == pytest.approx(irt["theta_star"])


def test_compute_item_weights_properties():
    """Difficulty decreases with item easiness; high-discrimination item
    tracks total score more strongly; Fisher info is non-negative."""
    # Item 0 strongly tracks ability (high disc); item 1 is near-noise (low disc).
    R, _ = make_2pl_matrix(
        300,
        [(2.5, 0.0), (0.15, 0.0)] + [(1.0, b) for b in np.linspace(-1, 1, 6)],
        seed=3,
    )
    w = compute_item_weights(R, 0, 1)

    # high-discrimination item has larger estimated discrimination
    assert w["a"][0] > w["a"][1]

    # difficulty is a decreasing function of item easiness (mean correctness)
    p = R.mean(axis=0)
    order = np.argsort(p)            # hardest -> easiest
    b_sorted = w["b"][order]
    assert np.all(np.diff(b_sorted) <= 1e-9)

    # Fisher information is non-negative everywhere
    assert np.all(w["J"] >= 0.0)


# --------------------------------------------------------------------------- #
# 1.2 compute_active_correlation
# --------------------------------------------------------------------------- #

def test_compute_active_correlation_basic():
    """Acceptance 1.2: symmetric PSD matrix, unit diagonal, no NaN, correct
    active indexing."""
    R, _ = make_2pl_matrix(200, [(1.0, b) for b in np.linspace(-1, 1, 8)], seed=1)
    d = np.array([1, 0, -1, 1, 0, -1, 1, 0])

    C, active = compute_active_correlation(R, d)

    expected_active = np.where(d != 0)[0]
    np.testing.assert_array_equal(active, expected_active)
    assert C.shape == (len(expected_active), len(expected_active))
    np.testing.assert_allclose(np.diag(C), 1.0)
    np.testing.assert_allclose(C, C.T)              # symmetric
    assert not np.isnan(C).any()                    # no NaN
    eig = np.linalg.eigvalsh(C)
    assert eig.min() > -1e-8                         # PSD


def test_compute_active_correlation_identical_items():
    """Two identical active columns are perfectly correlated."""
    base, _ = make_2pl_matrix(200, [(1.2, 0.0)], seed=2)
    col = base[:, 0]
    R = np.column_stack([col, col, 1.0 - col])      # item1 == item0, item2 anti
    d = np.array([1, 1, 1])

    C, active = compute_active_correlation(R, d)
    assert C[0, 1] == pytest.approx(1.0, abs=1e-9)
    assert C[0, 2] == pytest.approx(-1.0, abs=1e-9)


def test_compute_active_correlation_constant_column_no_nan():
    """A zero-variance active item must not produce NaN; diagonal stays 1."""
    R, _ = make_2pl_matrix(150, [(1.0, 0.0), (1.0, 0.5)], seed=4)
    R = np.column_stack([R, np.ones(R.shape[0])])   # constant active item
    d = np.array([1, -1, 1])

    C, active = compute_active_correlation(R, d)
    assert not np.isnan(C).any()
    np.testing.assert_allclose(np.diag(C), 1.0)
    # the constant column has no linear relationship -> zero off-diagonal
    assert C[2, 0] == pytest.approx(0.0, abs=1e-9)
    assert C[2, 1] == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# 1.3 / 1.4 mc_shapley_confidence — core
# --------------------------------------------------------------------------- #

def test_core_efficiency_axiom():
    """Acceptance 1.3: sum of phi equals v(all) - v(empty) = Phi(mu/sigma) - 0.5."""
    d_active = np.array([1, 1, -1, 1])
    a_active = np.array([1.5, 0.4, 0.9, 1.1])
    C = np.eye(4)

    res = _mc_shapley_confidence_core(d_active, a_active, C, T=6000, seed=0)

    w = a_active * d_active
    mu = w.sum()
    sigma = np.sqrt(w @ C @ w)
    v_all = norm.cdf(mu / sigma)
    expected = v_all - 0.5

    assert res["phi"].sum() == pytest.approx(expected, abs=0.01)
    assert res["efficiency_expected"] == pytest.approx(expected, abs=1e-9)
    assert res["efficiency_error"] < 0.01


def test_core_single_active_item():
    """1.4 edge case: one active item -> v = Phi(+/-1) - 0.5."""
    pos = _mc_shapley_confidence_core(np.array([1]), np.array([1.0]),
                                      np.array([[1.0]]), T=2000, seed=0)
    neg = _mc_shapley_confidence_core(np.array([-1]), np.array([1.0]),
                                      np.array([[1.0]]), T=2000, seed=0)
    assert pos["phi"][0] == pytest.approx(norm.cdf(1.0) - 0.5, abs=1e-9)
    assert neg["phi"][0] == pytest.approx(norm.cdf(-1.0) - 0.5, abs=1e-9)


def test_core_symmetry_breaking_via_correlation():
    """Acceptance: items with the SAME d and SAME discrimination get DIFFERENT
    phi when their correlation structure differs (impossible under the flip
    indicator). Item 0 is independent; items 1 and 2 are correlated, so they
    split credit and each earns strictly less than the independent item.

    Note: discrimination alone is a weak lever here — a singleton's value
    Phi(w/|w|)=Phi(+/-1) is scale-invariant — so correlation is the robust
    symmetry-breaking mechanism (see exact_confidence_shapley cross-check)."""
    d_active = np.array([1, 1, 1])
    a_active = np.array([1.0, 1.0, 1.0])
    C = np.array([[1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.95],
                  [0.0, 0.95, 1.0]])

    res = _mc_shapley_confidence_core(d_active, a_active, C, T=10000, seed=0)
    phi, se = res["phi"], res["se"]
    tol = 5 * max(se.max(), 1e-6)

    # the independent decisive item earns more than each correlated one
    assert phi[0] > phi[1] + tol
    assert phi[0] > phi[2] + tol
    # the two correlated items are symmetric players -> equal phi
    assert abs(phi[1] - phi[2]) < tol


def test_core_matches_exact_brute_force():
    """The MC core agrees with an independent brute-force exact Shapley of the
    confidence value function on a mixed correlated example."""
    d_active = np.array([1, 1, -1, 1])
    a_active = np.array([1.5, 0.4, 0.9, 1.1])
    C = np.array([[1.0, 0.3, 0.0, 0.2],
                  [0.3, 1.0, 0.1, 0.0],
                  [0.0, 0.1, 1.0, 0.4],
                  [0.2, 0.0, 0.4, 1.0]])

    mc = _mc_shapley_confidence_core(d_active, a_active, C, T=40000, seed=1)
    exact = exact_confidence_shapley(d_active, a_active, C)

    np.testing.assert_allclose(mc["phi"], exact, atol=3e-3)


def test_core_degeneracy_reduces_to_flip():
    """Acceptance / Proposition 1: equal discrimination + identity correlation +
    hard step link reduces exactly to the flip-indicator Shapley values."""
    d = np.array([1, 1, -1, 1, 0, -1, 1])
    active = d != 0
    d_active = d[active]
    a_active = np.ones(d_active.shape[0])
    C = np.eye(d_active.shape[0])

    conf = _mc_shapley_confidence_core(d_active, a_active, C,
                                       link="step", T=5000, seed=42)
    flip = mc_shapley_flip(d, T=5000, seed=42)

    np.testing.assert_allclose(conf["phi"], flip["phi"][active], atol=1e-9)


def test_core_reports_standard_errors():
    """Acceptance 1.3: standard errors are reported and positive for active items."""
    res = _mc_shapley_confidence_core(np.array([1, -1, 1]),
                                      np.array([1.0, 0.8, 1.2]),
                                      np.eye(3), T=3000, seed=0)
    assert "se" in res
    assert res["se"].shape == (3,)
    assert np.all(res["se"] >= 0.0)
    assert res["se"].max() > 0.0


def test_core_correlated_items_split_credit():
    """Two perfectly correlated decisive items together earn less than two
    independent ones (credit splitting via the correlation term)."""
    a = np.array([1.0, 1.0])
    indep = _mc_shapley_confidence_core(np.array([1, 1]), a, np.eye(2),
                                        T=8000, seed=0)
    corr = _mc_shapley_confidence_core(
        np.array([1, 1]), a, np.array([[1.0, 0.99], [0.99, 1.0]]),
        T=8000, seed=0)
    # correlated duplicates carry less total decisive weight
    assert corr["phi"].sum() < indep["phi"].sum()


# --------------------------------------------------------------------------- #
# 1.3 mc_shapley_confidence — public end-to-end
# --------------------------------------------------------------------------- #

def test_public_symmetry_breaking_end_to_end():
    """Acceptance 1.3: on a real-shaped R built through the full pipeline
    (compute_item_weights -> compute_active_correlation -> MC core), two
    duplicated (highly correlated) decisive items split credit, each earning
    strictly less than an independent decisive item. The flip indicator gives
    all three the same phi."""
    rng = np.random.default_rng(11)
    n_models = 250
    theta = rng.normal(0.0, 1.0, n_models)

    def item(a, b, seed):
        r = np.random.default_rng(seed)
        p = 1.0 / (1.0 + np.exp(-a * (theta - b)))
        return (r.random(n_models) < p).astype(np.float64)

    base = item(1.2, 0.0, 100)
    col_a = base.copy()                 # duplicate decisive item
    col_b = base.copy()                 # duplicate of col_a -> correlation ~1
    col_c = item(1.2, 0.0, 200)         # independent decisive item
    fillers = np.column_stack([item(1.0, b, 300 + i)
                               for i, b in enumerate(np.linspace(-1, 1, 6))])
    R = np.column_stack([col_a, col_b, col_c, fillers])

    d = np.zeros(R.shape[1], dtype=np.int64)
    d[0] = d[1] = d[2] = 1

    res = mc_shapley_confidence(d, R, m1_idx=0, m2_idx=1, T=8000, seed=0)
    phi, se = res["phi"], res["se"]
    tol = 5 * max(se.max(), 1e-6)

    assert phi.shape == (R.shape[1],)
    assert abs(phi[0] - phi[1]) < tol               # duplicates are symmetric
    assert phi[2] > phi[0] + tol                    # independent item earns more
    assert phi[2] > phi[1] + tol
    assert res["efficiency_error"] < 0.01           # efficiency axiom holds


def test_public_empty_active_returns_zeros():
    """All-zero d: no active items, phi all zero, efficiency zero."""
    R, _ = make_2pl_matrix(80, [(1.0, 0.0), (1.0, 0.5)], seed=6)
    d = np.zeros(R.shape[1], dtype=np.int64)
    res = mc_shapley_confidence(d, R, m1_idx=0, m2_idx=1, T=1000, seed=0)
    assert np.all(res["phi"] == 0.0)
    assert res["n_active"] == 0
    assert res["efficiency_sum"] == pytest.approx(0.0)
