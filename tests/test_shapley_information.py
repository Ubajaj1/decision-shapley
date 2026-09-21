"""Tests for the information value function (revision Step 2).

Pre-registered in notes/information-vf-preregistration.md. The information VF is
  v(S) = Φ(z_S),  z_S = (dᵀ Σ_cov⁻¹ 1) / √(1ᵀ Σ_cov⁻¹ 1)
with Σ_cov = D C D, D = diag(1/√J_i), C = Ledoit-Wolf-shrunk phi correlation.
Equivalently, with g_i = √J_i and h_i = d_i √J_i,
  z_S = (h_Sᵀ C_S⁻¹ g_S) / √(g_Sᵀ C_S⁻¹ g_S).

Required passing tests (note §D): efficiency axiom, degeneracy → flip indicator,
brute-force exact cross-check, and discrimination monotonicity (the assertion that
fails under the confidence VF must pass here).
"""

import numpy as np
import pytest
from scipy.stats import norm

from _helpers import make_2pl_matrix, exact_shapley
from shapley import (
    ledoit_wolf_correlation,
    mc_shapley_information,
    _mc_shapley_information_core,
    mc_shapley_flip,
)


# --------------------------------------------------------------------------- #
# Reference value function (independent of the production implementation)
# --------------------------------------------------------------------------- #

def info_value_fn(J_active, d_active, C, link="probit"):
    g = np.sqrt(np.maximum(J_active, 1e-12))
    h = d_active.astype(float) * g

    def v(S):
        if len(S) == 0:
            return 0.5
        S = list(S)
        CS = C[np.ix_(S, S)]
        y = np.linalg.solve(CS, g[S])
        z = (h[S] @ y) / np.sqrt(max(g[S] @ y, 1e-10))
        if link == "probit":
            return float(norm.cdf(z))
        return 1.0 if z > 0 else (0.0 if z < 0 else 0.5)

    return v


# --------------------------------------------------------------------------- #
# Ledoit-Wolf shrinkage
# --------------------------------------------------------------------------- #

def test_ledoit_wolf_intensity_in_range():
    R, _ = make_2pl_matrix(120, [(1.0, b) for b in np.linspace(-1, 1, 6)], seed=1)
    d = np.array([1, -1, 1, 1, -1, 1])
    C, active, delta = ledoit_wolf_correlation(R, d)
    assert 0.0 <= delta <= 1.0


def test_ledoit_wolf_diagonal_symmetric_psd():
    R, _ = make_2pl_matrix(150, [(1.0, b) for b in np.linspace(-1, 1, 7)], seed=2)
    d = np.array([1, 0, -1, 1, 0, -1, 1])
    C, active, delta = ledoit_wolf_correlation(R, d)
    np.testing.assert_array_equal(active, np.where(d != 0)[0])
    assert C.shape == (len(active), len(active))
    np.testing.assert_allclose(np.diag(C), 1.0, atol=1e-9)
    np.testing.assert_allclose(C, C.T, atol=1e-9)
    assert not np.isnan(C).any()
    assert np.linalg.eigvalsh(C).min() > -1e-8


def test_ledoit_wolf_improves_conditioning_on_collinear_items():
    """With near-duplicate items the raw correlation is near-singular; shrinkage
    must produce a strictly better-conditioned (invertible) matrix."""
    base, _ = make_2pl_matrix(80, [(1.2, 0.0)], seed=3)
    col = base[:, 0]
    noise = (np.random.default_rng(0).random(col.shape[0]) < 0.02)
    near_dup = np.where(noise, 1.0 - col, col)          # ~98% identical
    R = np.column_stack([col, near_dup, 1.0 - col])
    d = np.array([1, 1, -1])

    raw = np.corrcoef(R.T)
    C, active, delta = ledoit_wolf_correlation(R, d)

    assert delta > 0.0
    assert np.linalg.eigvalsh(C).min() > np.linalg.eigvalsh(raw).min()
    assert np.linalg.eigvalsh(C).min() > 1e-6           # invertible


# --------------------------------------------------------------------------- #
# Information VF core
# --------------------------------------------------------------------------- #

def test_info_singleton_depends_on_discrimination():
    """v({i}) = Φ(d_i · √J_i): discrimination does NOT cancel (the cure)."""
    for J in (4.0, 1.0, 0.25):
        pos = _mc_shapley_information_core(np.array([1]), np.array([J]),
                                           np.array([[1.0]]), T=1500, seed=0)
        assert pos["phi"][0] == pytest.approx(norm.cdf(np.sqrt(J)) - 0.5, abs=1e-9)


def test_info_core_efficiency_axiom():
    d = np.array([1, 1, -1, 1])
    J = np.array([2.0, 0.5, 1.3, 0.8])
    C = np.eye(4)
    res = _mc_shapley_information_core(d, J, C, T=6000, seed=0)

    v = info_value_fn(J, d, C)
    expected = v(tuple(range(4))) - 0.5
    assert res["phi"].sum() == pytest.approx(expected, abs=0.01)
    assert res["efficiency_error"] < 0.01


def test_info_core_degeneracy_reduces_to_flip():
    """Equal J + identity correlation + step link == flip-indicator Shapley."""
    d = np.array([1, 1, -1, 1, 0, -1, 1])
    active = d != 0
    d_a = d[active]
    J = np.ones(d_a.shape[0]) * 1.7          # equal discrimination
    C = np.eye(d_a.shape[0])

    conf = _mc_shapley_information_core(d_a, J, C, link="step", T=5000, seed=42)
    flip = mc_shapley_flip(d, T=5000, seed=42)
    np.testing.assert_allclose(conf["phi"], flip["phi"][active], atol=1e-9)


def test_info_core_matches_exact_brute_force():
    d = np.array([1, 1, -1, 1])
    J = np.array([3.0, 0.4, 1.1, 0.7])
    C = np.array([[1.0, 0.3, 0.0, 0.2],
                  [0.3, 1.0, 0.1, 0.0],
                  [0.0, 0.1, 1.0, 0.4],
                  [0.2, 0.0, 0.4, 1.0]])
    mc = _mc_shapley_information_core(d, J, C, T=40000, seed=1)
    exact = exact_shapley(4, info_value_fn(J, d, C))
    np.testing.assert_allclose(mc["phi"], exact, atol=3e-3)


def test_info_core_discrimination_monotonicity():
    """THE new behavioral test (failed under the confidence VF, must pass here):
    among same-sign items with identity correlation, higher Fisher information
    earns strictly higher φ."""
    d = np.array([1, 1, 1])
    J = np.array([4.0, 1.0, 0.25])          # strictly decreasing discrimination
    C = np.eye(3)
    res = _mc_shapley_information_core(d, J, C, T=12000, seed=0)
    phi, se = res["phi"], res["se"]
    tol = 3 * max(se.max(), 1e-6)
    assert phi[0] > phi[1] + tol
    assert phi[1] > phi[2] + tol


def test_info_core_reports_standard_errors():
    res = _mc_shapley_information_core(np.array([1, -1, 1]),
                                      np.array([1.0, 0.8, 1.2]),
                                      np.eye(3), T=3000, seed=0)
    assert res["se"].shape == (3,)
    assert np.all(res["se"] >= 0.0)
    assert res["se"].max() > 0.0


# --------------------------------------------------------------------------- #
# Public end-to-end
# --------------------------------------------------------------------------- #

def test_info_public_end_to_end():
    """Full pipeline (compute_item_weights -> ledoit_wolf_correlation -> core):
    among same-sign decisive items, the higher-discrimination one earns more."""
    # item 0 strongly tracks ability (high J); item 1 near-noise (low J)
    specs = [(2.5, 0.0), (0.2, 0.0)] + [(1.0, b) for b in np.linspace(-1, 1, 8)]
    R, _ = make_2pl_matrix(300, specs, seed=5)
    d = np.zeros(R.shape[1], dtype=np.int64)
    d[0] = 1
    d[1] = 1

    res = mc_shapley_information(d, R, m1_idx=0, m2_idx=1, T=8000, seed=0)
    phi, se = res["phi"], res["se"]
    tol = 5 * max(se.max(), 1e-6)

    assert res["J"][0] > res["J"][1]                    # higher discrimination
    assert phi[0] > phi[1] + tol                        # -> higher attribution
    assert res["efficiency_error"] < 0.01


def test_info_planted_structure_matches_exact():
    """Step 3 anchor: a planted case with all three structures — two high-J
    items, a correlated cluster, a counter-evidential item, two low-J items —
    matches exact brute-force Shapley."""
    d = np.array([1, 1, 1, 1, 1, -1, 1, 1])
    J = np.array([4.0, 3.5, 1.0, 1.0, 1.0, 1.2, 0.3, 0.25])
    C = np.eye(8)
    # correlated cluster over items {2,3,4}
    for i in (2, 3, 4):
        for j in (2, 3, 4):
            if i != j:
                C[i, j] = 0.85
    assert np.linalg.eigvalsh(C).min() > 0           # planted C is valid (PSD)

    mc = _mc_shapley_information_core(d, J, C, T=60000, seed=2)
    exact = exact_shapley(8, info_value_fn(J, d, C))
    np.testing.assert_allclose(mc["phi"], exact, atol=3e-3)
    # efficiency holds on the planted case
    assert mc["efficiency_error"] < 0.01


def test_info_convergence_independent_of_seed():
    """A larger planted case (16 active items) converges: two seeds agree within
    combined Monte-Carlo error."""
    rng = np.random.default_rng(0)
    d = rng.choice([1, -1], size=16)
    J = rng.uniform(0.2, 4.0, size=16)
    C = np.eye(16)

    a = _mc_shapley_information_core(d, J, C, T=20000, seed=1)
    b = _mc_shapley_information_core(d, J, C, T=20000, seed=2)
    combined_se = np.sqrt(a["se"] ** 2 + b["se"] ** 2)
    assert np.all(np.abs(a["phi"] - b["phi"]) < 5 * np.maximum(combined_se, 1e-6))


def test_info_public_empty_active_returns_zeros():
    R, _ = make_2pl_matrix(60, [(1.0, 0.0), (1.0, 0.5)], seed=6)
    d = np.zeros(R.shape[1], dtype=np.int64)
    res = mc_shapley_information(d, R, m1_idx=0, m2_idx=1, T=800, seed=0)
    assert np.all(res["phi"] == 0.0)
    assert res["n_active"] == 0
