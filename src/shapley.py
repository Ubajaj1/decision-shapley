"""Monte-Carlo Shapley attribution for the flip-indicator value function (§6.3).

Key insight from §5 Gap 1: the margin (additive) value function makes Shapley
degenerate to LOO. The flip-indicator is nonlinear, so Shapley properly handles
duplicate/correlated items by splitting credit.
"""

import numpy as np
from scipy.stats import norm


def mc_shapley_flip(d: np.ndarray, T: int = 20_000, seed: int = 42,
                    se_target: float = 1e-3) -> dict:
    """Compute Monte-Carlo Shapley values under the flip-indicator value function.

    Args:
        d: per-item differential array (d_i ∈ {-1, 0, +1}).
        T: number of random permutations.
        seed: RNG seed.
        se_target: early-stop if max SE drops below this (not implemented yet).

    Returns dict with phi (Shapley values), se (standard errors), and metadata.
    """
    rng = np.random.default_rng(seed)
    n = len(d)

    # Items with d_i = 0 contribute marginal 0 always — skip them
    active_mask = d != 0
    active_indices = np.where(active_mask)[0]
    n_active = len(active_indices)
    d_active = d[active_indices].astype(np.int64)

    if n_active == 0:
        return _empty_result(n, T, seed)

    # Accumulate marginals per active item
    marginal_sum = np.zeros(n_active, dtype=np.float64)
    marginal_sum_sq = np.zeros(n_active, dtype=np.float64)

    for t in range(T):
        perm = rng.permutation(n_active)
        _walk_permutation(d_active, perm, marginal_sum, marginal_sum_sq)

    phi_active = marginal_sum / T
    var_active = marginal_sum_sq / T - phi_active ** 2
    se_active = np.sqrt(np.maximum(var_active, 0.0) / T)

    # Map back to full item set
    phi = np.zeros(n, dtype=np.float64)
    se = np.zeros(n, dtype=np.float64)
    phi[active_indices] = phi_active
    se[active_indices] = se_active

    efficiency_sum = phi.sum()
    expected_efficiency = 0.5  # v(all) - v({}) = 1.0 - 0.5

    return {
        "phi": phi,
        "se": se,
        "T": T,
        "seed": seed,
        "n_active": n_active,
        "n_total": n,
        "efficiency_sum": efficiency_sum,
        "efficiency_expected": expected_efficiency,
        "efficiency_error": abs(efficiency_sum - expected_efficiency),
        "max_se": float(se.max()) if n > 0 else 0.0,
    }


def _walk_permutation(d_active: np.ndarray, perm: np.ndarray,
                      marginal_sum: np.ndarray, marginal_sum_sq: np.ndarray):
    """Walk one permutation, accumulate marginals."""
    running_sum = 0
    prev_v = 0.5  # v({}) = 0.5

    for pos in range(len(perm)):
        idx = perm[pos]
        running_sum += d_active[idx]
        curr_v = _v_from_sum(running_sum)
        marginal = curr_v - prev_v
        marginal_sum[idx] += marginal
        marginal_sum_sq[idx] += marginal * marginal
        prev_v = curr_v


def _v_from_sum(s: int) -> float:
    if s > 0:
        return 1.0
    elif s == 0:
        return 0.5
    else:
        return 0.0


def mc_shapley_margin(d: np.ndarray, T: int = 20_000, seed: int = 42) -> dict:
    """Shapley under the additive margin value function (for E1 comparison).

    With v(S) = Σ_{i∈S} d_i, Shapley values equal d_i exactly (additive game).
    We compute via MC anyway to demonstrate they match LOO, validating Gap 1.
    """
    rng = np.random.default_rng(seed)
    n = len(d)

    active_mask = d != 0
    active_indices = np.where(active_mask)[0]
    n_active = len(active_indices)
    d_active = d[active_indices].astype(np.int64).astype(np.float64)

    if n_active == 0:
        return _empty_result(n, T, seed)

    marginal_sum = np.zeros(n_active, dtype=np.float64)
    marginal_sum_sq = np.zeros(n_active, dtype=np.float64)

    for t in range(T):
        perm = rng.permutation(n_active)
        # For additive v(S) = sum(d), marginal of item i is always d_i
        for pos in range(n_active):
            idx = perm[pos]
            m = d_active[idx]
            marginal_sum[idx] += m
            marginal_sum_sq[idx] += m * m

    phi_active = marginal_sum / T
    var_active = marginal_sum_sq / T - phi_active ** 2
    se_active = np.sqrt(np.maximum(var_active, 0.0) / T)

    phi = np.zeros(n, dtype=np.float64)
    se = np.zeros(n, dtype=np.float64)
    phi[active_indices] = phi_active
    se[active_indices] = se_active

    return {
        "phi": phi,
        "se": se,
        "T": T,
        "seed": seed,
        "n_active": n_active,
        "n_total": n,
        "efficiency_sum": phi.sum(),
        "efficiency_expected": float(d.sum()),
        "efficiency_error": abs(phi.sum() - float(d.sum())),
        "max_se": float(se.max()) if n > 0 else 0.0,
    }


def _empty_result(n: int, T: int, seed: int) -> dict:
    return {
        "phi": np.zeros(n, dtype=np.float64),
        "se": np.zeros(n, dtype=np.float64),
        "T": T,
        "seed": seed,
        "n_active": 0,
        "n_total": n,
        "efficiency_sum": 0.0,
        "efficiency_expected": 0.5,
        "efficiency_error": 0.5,
        "max_se": 0.0,
    }


# --------------------------------------------------------------------------- #
# Correlation-aware confidence value function (revision Step 1)
#
# The flip indicator v(S) = sign(Σ d_i) makes every same-sign item a symmetric
# player, collapsing to the Shapley-Shubik power index. The confidence value
# function v(S) = Φ(μ_S / σ_S) weights each item by its discrimination and
# discounts correlated items, genuinely breaking that symmetry.
# --------------------------------------------------------------------------- #


def compute_item_weights(R: np.ndarray, m1_idx: int, m2_idx: int) -> dict:
    """Per-item 2PL discrimination/difficulty and localized Fisher information.

    Closed-form 2PL approximation (identical to the E5 baseline, extracted here
    for reuse): difficulty b_i = -logit(p_i), discrimination a_i ∝ point-biserial
    correlation of the item with total score, abilities θ_m = logit(score_m), and
    Fisher information J_i evaluated at the ability midpoint θ* between m1 and m2.

    Returns a dict with arrays ``a`` (discrimination), ``b`` (difficulty),
    ``J`` (Fisher info at θ*), ``p_star`` (success prob at θ*), and the scalar
    abilities ``theta_m1``, ``theta_m2``, ``theta_star``.
    """
    n_models, n_items = R.shape

    p_i = R.mean(axis=0)
    p_i = np.clip(p_i, 0.001, 0.999)
    difficulty = -np.log(p_i / (1.0 - p_i))

    # Point-biserial discrimination = Pearson r of each item with the total score,
    # vectorized over all items (identical to the per-item np.corrcoef it replaced, but
    # O(n_models * n_items) instead of an n_items-long Python loop -- essential for the
    # 5000-model open-leaderboard matrices).
    Rf = R.astype(np.float64)
    total_scores = Rf.sum(axis=1)
    item_std = Rf.std(axis=0)
    totc = total_scores - total_scores.mean()
    Rc = Rf - Rf.mean(axis=0)
    denom = np.sqrt((Rc ** 2).sum(axis=0) * (totc ** 2).sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        r_pb = (Rc * totc[:, None]).sum(axis=0) / denom
    r_pb = np.nan_to_num(r_pb, nan=0.0)
    disc = np.maximum(0.01, r_pb * 2.0)
    disc[item_std == 0] = 0.01            # zero-variance items get the floor

    scores = R.mean(axis=1)
    scores = np.clip(scores, 0.001, 0.999)
    theta = np.log(scores / (1.0 - scores))

    theta_m1 = float(theta[m1_idx])
    theta_m2 = float(theta[m2_idx])
    theta_star = (theta_m1 + theta_m2) / 2.0

    p_star = 1.0 / (1.0 + np.exp(-disc * (theta_star - difficulty)))
    fisher_info = disc**2 * p_star * (1.0 - p_star)

    return {
        "a": disc,
        "b": difficulty,
        "J": fisher_info,
        "p_star": p_star,
        "theta_m1": theta_m1,
        "theta_m2": theta_m2,
        "theta_star": theta_star,
    }


def compute_active_correlation(R: np.ndarray, d: np.ndarray):
    """Phi (= Pearson on binary data) correlation matrix among active items.

    Active items are those with d_i != 0. Items with zero variance produce NaN
    correlations, which are clipped to 0; the diagonal is forced to 1.

    Returns ``(C, active_indices)`` where C is a symmetric PSD matrix of shape
    (n_active, n_active) with unit diagonal and no NaN values.
    """
    active_indices = np.where(d != 0)[0]
    n_active = len(active_indices)

    if n_active == 0:
        return np.zeros((0, 0), dtype=np.float64), active_indices
    if n_active == 1:
        return np.ones((1, 1), dtype=np.float64), active_indices

    sub = R[:, active_indices]
    with np.errstate(invalid="ignore", divide="ignore"):
        C = np.corrcoef(sub.T)          # zero-variance items yield NaN (handled below)
    C = np.nan_to_num(C, nan=0.0)       # zero-variance items -> 0 correlation
    np.fill_diagonal(C, 1.0)
    C = (C + C.T) / 2.0                  # enforce exact symmetry
    return C, active_indices


def _confidence_value(signal: float, var: float, link: str) -> float:
    """Value function v = link(signal / sigma), with sigma = sqrt(var)."""
    sigma = np.sqrt(max(var, 1e-10))
    z = signal / sigma
    if link == "probit":
        return float(norm.cdf(z))
    if link == "step":
        if z > 0:
            return 1.0
        if z < 0:
            return 0.0
        return 0.5
    raise ValueError(f"unknown link: {link!r}")


def _mc_shapley_confidence_core(d_active: np.ndarray, a_active: np.ndarray,
                                C: np.ndarray, link: str = "probit",
                                T: int = 20_000, seed: int = 42) -> dict:
    """Monte-Carlo Shapley for the confidence value function over active items.

    Maintains, per permutation prefix S:
      - running_signal: μ_S = Σ_{i∈S} w_i        (w_i = a_i · d_i)
      - running_var:    σ²_S = w_Sᵀ C_S w_S
      - running_cw[k]:  Σ_{i∈S} C[k,i] · w_i      (cross-correlation accumulator)

    Adding item i costs O(n_active): σ² gains w_i² + 2·w_i·running_cw[i].
    """
    n_active = len(d_active)
    if n_active == 0:
        return {
            "phi": np.zeros(0, dtype=np.float64),
            "se": np.zeros(0, dtype=np.float64),
            "T": T, "seed": seed, "link": link, "n_active": 0,
            "v_all": 0.5, "efficiency_sum": 0.0,
            "efficiency_expected": 0.0, "efficiency_error": 0.0, "max_se": 0.0,
        }

    w = a_active.astype(np.float64) * d_active.astype(np.float64)
    C = np.asarray(C, dtype=np.float64)

    rng = np.random.default_rng(seed)
    marginal_sum = np.zeros(n_active, dtype=np.float64)
    marginal_sum_sq = np.zeros(n_active, dtype=np.float64)

    for _ in range(T):
        perm = rng.permutation(n_active)
        running_signal = 0.0
        running_var = 0.0
        running_cw = np.zeros(n_active, dtype=np.float64)
        prev_v = 0.5  # v(∅) = Φ(0) = 0.5

        for idx in perm:
            cross = running_cw[idx]                  # Σ_{j∈S} C[idx,j] w_j
            running_var += w[idx] ** 2 + 2.0 * w[idx] * cross
            running_signal += w[idx]
            running_cw += C[:, idx] * w[idx]         # fold item idx into S
            curr_v = _confidence_value(running_signal, running_var, link)
            marginal = curr_v - prev_v
            marginal_sum[idx] += marginal
            marginal_sum_sq[idx] += marginal * marginal
            prev_v = curr_v

    phi = marginal_sum / T
    var = marginal_sum_sq / T - phi ** 2
    se = np.sqrt(np.maximum(var, 0.0) / T)

    v_all = _confidence_value(float(w.sum()), float(w @ C @ w), link)
    efficiency_expected = v_all - 0.5

    return {
        "phi": phi,
        "se": se,
        "T": T,
        "seed": seed,
        "link": link,
        "n_active": n_active,
        "v_all": v_all,
        "efficiency_sum": float(phi.sum()),
        "efficiency_expected": efficiency_expected,
        "efficiency_error": abs(float(phi.sum()) - efficiency_expected),
        "max_se": float(se.max()),
    }


def mc_shapley_confidence(d: np.ndarray, R: np.ndarray, m1_idx: int, m2_idx: int,
                          T: int = 20_000, seed: int = 42,
                          link: str = "probit") -> dict:
    """Shapley values under the correlation-aware confidence value function.

    v(S) = Φ(μ_S / σ_S) with weighted votes w_i = a_i·d_i, total signal
    μ_S = Σ w_i, and variance σ²_S = w_Sᵀ C_S w_S where C is the phi correlation
    matrix among active items. Unlike the flip indicator, items sharing the same
    d_i receive different φ according to their discrimination and correlation.

    ``link="step"`` substitutes a hard step for Φ, recovering the flip-indicator
    Shapley-Shubik values when discrimination is equal and items are uncorrelated
    (Proposition 1).
    """
    n = len(d)
    active_indices = np.where(d != 0)[0]
    n_active = len(active_indices)

    if n_active == 0:
        res = _empty_result(n, T, seed)
        res["link"] = link
        res["v_all"] = 0.5
        res["efficiency_expected"] = 0.0
        res["efficiency_error"] = 0.0
        return res

    weights = compute_item_weights(R, m1_idx, m2_idx)
    a_active = weights["a"][active_indices]
    C, _ = compute_active_correlation(R, d)
    d_active = d[active_indices].astype(np.int64)

    core = _mc_shapley_confidence_core(d_active, a_active, C,
                                       link=link, T=T, seed=seed)

    phi = np.zeros(n, dtype=np.float64)
    se = np.zeros(n, dtype=np.float64)
    phi[active_indices] = core["phi"]
    se[active_indices] = core["se"]

    return {
        "phi": phi,
        "se": se,
        "T": T,
        "seed": seed,
        "link": link,
        "n_active": n_active,
        "n_total": n,
        "v_all": core["v_all"],
        "efficiency_sum": core["efficiency_sum"],
        "efficiency_expected": core["efficiency_expected"],
        "efficiency_error": core["efficiency_error"],
        "max_se": core["max_se"],
        "discrimination": weights["a"],
    }


# --------------------------------------------------------------------------- #
# Information value function (revision Step 2)
#
# The confidence VF normalizes each item by its own variance, so a singleton's
# value Φ(w_i/|w_i|)=Φ(±1) is scale-invariant and discrimination cancels. The
# information VF instead pools items by Fisher-information precision, so a
# singleton is worth Φ(d_i·√J_i) — discrimination no longer cancels. See
# notes/information-vf-preregistration.md for the full derivation.
# --------------------------------------------------------------------------- #


def ledoit_wolf_correlation(R: np.ndarray, d: np.ndarray):
    """Ledoit-Wolf (2004) analytically-shrunk phi correlation over active items.

    Shrinks the sample correlation toward the scaled-identity target with the
    closed-form optimal intensity δ* (data-determined, no free parameter), then
    renormalizes to unit diagonal. This well-conditions C so that Σ_cov⁻¹ in the
    information VF is stable even with few models or collinear items.

    Returns ``(C, active_indices, delta)``.
    """
    active_indices = np.where(d != 0)[0]
    p = len(active_indices)

    if p == 0:
        return np.zeros((0, 0), dtype=np.float64), active_indices, 0.0
    if p == 1:
        return np.ones((1, 1), dtype=np.float64), active_indices, 0.0

    X = R[:, active_indices].astype(np.float64)
    n = X.shape[0]

    # Standardize columns -> sample covariance of Xs is the sample correlation.
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    nz = std > 0
    Xs = np.zeros_like(X)
    Xs[:, nz] = (X[:, nz] - mean[nz]) / std[nz]

    S = (Xs.T @ Xs) / n                          # sample correlation (diag ~1)
    mu = np.trace(S) / p                         # mean eigenvalue (target scale)
    d2 = np.sum((S - mu * np.eye(p)) ** 2) / p   # dispersion of S around target

    # b̄²: average squared Frobenius error of per-sample rank-1 estimates.
    b_bar2 = 0.0
    for k in range(n):
        xk = Xs[k][:, None]
        b_bar2 += np.sum((xk @ xk.T - S) ** 2)
    b_bar2 = b_bar2 / (n ** 2) / p
    b2 = min(b_bar2, d2)                          # clamp (b² ≤ d²)

    delta = (b2 / d2) if d2 > 0 else 0.0
    C = (1.0 - delta) * S + delta * mu * np.eye(p)

    # Renormalize to a proper correlation matrix (unit diagonal, symmetric).
    dsqrt = np.sqrt(np.clip(np.diag(C), 1e-12, None))
    C = C / np.outer(dsqrt, dsqrt)
    C = (C + C.T) / 2.0
    np.fill_diagonal(C, 1.0)
    return C, active_indices, float(delta)


def _info_value(g_S: np.ndarray, h_S: np.ndarray, C_S: np.ndarray,
                link: str) -> float:
    """Information value of a coalition: v = link(z), z = (hᵀC⁻¹g)/√(gᵀC⁻¹g).

    g_i = √J_i, h_i = d_i·√J_i. Solves C_S y = g_S once (y = C_S⁻¹ g_S) and
    forms both quadratic forms from y.
    """
    try:
        y = np.linalg.solve(C_S, g_S)
    except np.linalg.LinAlgError:
        y = np.linalg.lstsq(C_S, g_S, rcond=None)[0]
    denom = float(g_S @ y)
    z = float(h_S @ y) / np.sqrt(max(denom, 1e-10))
    if link == "probit":
        return float(norm.cdf(z))
    if link == "step":
        # Linear solves and dot products can leave an epsilon-sized residual
        # for an analytically exact tie. Preserve the step game's v(tie)=0.5
        # semantics across BLAS/NumPy implementations.
        if np.isclose(z, 0.0, rtol=0.0, atol=1e-12):
            return 0.5
        if z > 0:
            return 1.0
        return 0.0
    raise ValueError(f"unknown link: {link!r}")


def _mc_shapley_information_core(d_active: np.ndarray, J_active: np.ndarray,
                                 C: np.ndarray, link: str = "probit",
                                 T: int = 20_000, seed: int = 42) -> dict:
    """Monte-Carlo Shapley for the information value function over active items.

    For each permutation prefix S, evaluates v(S) = Φ(z_S) with a fresh GLS solve
    on the correlation submatrix C_S (O(|S|³); fine for the modest active sets of
    real contests and the small synthetic cases used for validation).
    """
    n_active = len(d_active)
    if n_active == 0:
        return {
            "phi": np.zeros(0), "se": np.zeros(0), "T": T, "seed": seed,
            "link": link, "n_active": 0, "v_all": 0.5, "efficiency_sum": 0.0,
            "efficiency_expected": 0.0, "efficiency_error": 0.0, "max_se": 0.0,
        }

    g = np.sqrt(np.maximum(J_active.astype(np.float64), 1e-12))
    h = d_active.astype(np.float64) * g
    C = np.asarray(C, dtype=np.float64)

    rng = np.random.default_rng(seed)
    marginal_sum = np.zeros(n_active, dtype=np.float64)
    marginal_sum_sq = np.zeros(n_active, dtype=np.float64)

    for _ in range(T):
        perm = rng.permutation(n_active)
        prev_v = 0.5  # v(∅) = Φ(0) = 0.5
        for pos in range(n_active):
            S = perm[: pos + 1]
            sub = np.ix_(S, S)
            curr_v = _info_value(g[S], h[S], C[sub], link)
            idx = perm[pos]
            marginal = curr_v - prev_v
            marginal_sum[idx] += marginal
            marginal_sum_sq[idx] += marginal * marginal
            prev_v = curr_v

    phi = marginal_sum / T
    var = marginal_sum_sq / T - phi ** 2
    se = np.sqrt(np.maximum(var, 0.0) / T)

    all_idx = np.arange(n_active)
    v_all = _info_value(g, h, C[np.ix_(all_idx, all_idx)], link)
    efficiency_expected = v_all - 0.5

    return {
        "phi": phi,
        "se": se,
        "T": T,
        "seed": seed,
        "link": link,
        "n_active": n_active,
        "v_all": v_all,
        "efficiency_sum": float(phi.sum()),
        "efficiency_expected": efficiency_expected,
        "efficiency_error": abs(float(phi.sum()) - efficiency_expected),
        "max_se": float(se.max()),
    }


def mc_shapley_information(d: np.ndarray, R: np.ndarray, m1_idx: int, m2_idx: int,
                          T: int = 20_000, seed: int = 42,
                          link: str = "probit") -> dict:
    """Shapley values under the information (Fisher-precision) value function.

    v(S) = Φ(z_S), z_S = (dᵀ Σ_cov⁻¹ 1)/√(1ᵀ Σ_cov⁻¹ 1), with Σ_cov = D C D,
    D = diag(1/√J_i), and C the Ledoit-Wolf-shrunk phi correlation among active
    items. Unlike the confidence VF, discrimination does not cancel: more
    informative items (higher Fisher info J_i) earn higher φ, all else equal.
    """
    n = len(d)
    active_indices = np.where(d != 0)[0]
    n_active = len(active_indices)

    weights = compute_item_weights(R, m1_idx, m2_idx)

    if n_active == 0:
        res = _empty_result(n, T, seed)
        res["link"] = link
        res["v_all"] = 0.5
        res["efficiency_expected"] = 0.0
        res["efficiency_error"] = 0.0
        res["delta"] = 0.0
        res["J"] = weights["J"]
        res["discrimination"] = weights["a"]
        return res

    J_active = weights["J"][active_indices]
    C, _, delta = ledoit_wolf_correlation(R, d)
    d_active = d[active_indices].astype(np.int64)

    core = _mc_shapley_information_core(d_active, J_active, C,
                                        link=link, T=T, seed=seed)

    phi = np.zeros(n, dtype=np.float64)
    se = np.zeros(n, dtype=np.float64)
    phi[active_indices] = core["phi"]
    se[active_indices] = core["se"]

    return {
        "phi": phi,
        "se": se,
        "T": T,
        "seed": seed,
        "link": link,
        "n_active": n_active,
        "n_total": n,
        "v_all": core["v_all"],
        "efficiency_sum": core["efficiency_sum"],
        "efficiency_expected": core["efficiency_expected"],
        "efficiency_error": core["efficiency_error"],
        "max_se": core["max_se"],
        "delta": delta,
        "J": weights["J"],
        "discrimination": weights["a"],
    }
