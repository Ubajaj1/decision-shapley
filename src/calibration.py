"""Within-contest external-validation gate (Step 4/5).

Re-architected 2026-06-22 after the LiveBench pilot (experiments/e12) showed the
original pooled-Spearman gate was an ecological confound: pooling (J_i, pivotality_i)
across contests let between-contest covariance dominate, so the permutation floor
(which shuffles only WITHIN contests) landed at ~0.22 while the 2PL "ceiling" sat near
0 -- inverting the intended references and making the verdict meaningless.

The gate now measures the genuine within-contest signal:

  PRIMARY  : per-contest Spearman(J_i, pivotality_i) over same-sign active items,
             aggregated (median) across qualifying contests.
  NULL     : shuffle J labels WITHIN each contest, recompute per-contest rho,
             re-aggregate (R_perm reps). floor = 95th pct; p = P(null >= observed).
  VERDICT  : Supported if aggregate > floor, else No support.

The fitted-2PL ceiling is dropped. pooled_spearman is retained as a descriptive only,
to report (and caveat) the between-contest-confounded number.

Standing caveat (paper): J_i and pivotality are both functions of the response
patterns, so even a within-contest association cannot fully separate "discrimination
matters" from mechanical coupling.
"""

import numpy as np
from scipy import stats

from shapley import compute_item_weights, mc_shapley_information
from external_validation import default_twin_k, external_pivotality, simulate_2pl

# Pre-registered constants.
MIN_SAME_SIGN = 4          # a contest qualifies with >= 4 same-sign active items
R_PERM = 1000              # within-contest permutation replicates
FLOOR_PCT = 95             # floor = 95th pct of the no-signal null


def contest_sample(R, m1_idx, m2_idx, sign=1, k=None, weight="fisher",
                   T=2000, seed=42):
    """Same-sign (weight_i, pivotality_i) arrays for one contest, or None if it fails
    to qualify (fewer than MIN_SAME_SIGN items with d_i == sign).

    weight="fisher"  -> raw, un-riggable 2PL Fisher information J_i (PRIMARY target).
    weight="info_phi" -> info-VF Shapley value phi_i (SECONDARY, downstream pipeline
                         check; depends on the modeling choice, so not headline).
    """
    R = np.asarray(R, dtype=float)
    d = R[m1_idx] - R[m2_idx]
    mask = d == sign
    if int(mask.sum()) < MIN_SAME_SIGN:
        return None
    if weight == "fisher":
        w = compute_item_weights(R, m1_idx, m2_idx)["J"]
    elif weight == "info_phi":
        w = mc_shapley_information(d, R, m1_idx, m2_idx, T=T, seed=seed)["phi"]
    else:
        raise ValueError(f"unknown weight {weight!r}")
    piv = external_pivotality(R, m1_idx, m2_idx, k=k)["pivotality"]
    return w[mask], piv[mask]


def _spearman(x, y):
    """Spearman rho with an exact-constant guard (np.ptp, not np.std, to dodge the
    ~1e-17 float dust np.std reports on an all-equal array)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan
    return float(stats.spearmanr(x, y).statistic)


def pooled_spearman(samples):
    """DESCRIPTIVE ONLY: Spearman of (J, pivotality) pooled across contests. Reported
    to document the between-contest confound; not the gate statistic."""
    J = np.concatenate([s[0] for s in samples])
    piv = np.concatenate([s[1] for s in samples])
    return _spearman(J, piv)


def within_contest_rhos(samples):
    """Per-contest Spearman(J, pivotality); contests with no variance are dropped."""
    rhos = np.array([_spearman(J, piv) for (J, piv) in samples])
    return rhos[np.isfinite(rhos)]


def aggregate_signal(rhos, stat="median"):
    """Aggregate per-contest rhos into the gate statistic (median by default)."""
    rhos = np.asarray(rhos, dtype=float)
    if rhos.size == 0:
        return np.nan
    return float(np.median(rhos) if stat == "median" else np.mean(rhos))


def within_permutation_null(samples, R_perm=R_PERM, seed=0, stat="median"):
    """No-signal null: shuffle J within each contest, re-aggregate per-contest rho.

    Pivotality is held fixed (it never depends on J), so each replicate just breaks
    the within-contest J->pivotality link without disturbing contest structure.
    """
    rng = np.random.default_rng(seed)
    out = np.empty(R_perm)
    for r in range(R_perm):
        shuffled = [(rng.permutation(J), piv) for (J, piv) in samples]
        out[r] = aggregate_signal(within_contest_rhos(shuffled), stat=stat)
    return out


def within_verdict(aggregate, floor):
    """Supported iff the observed aggregate clears the no-signal floor."""
    if not (np.isfinite(aggregate) and np.isfinite(floor)):
        return "No support"
    return "Supported" if aggregate > floor else "No support"


def _contest_k(n_models, k, k_mult):
    """Per-contest twin-neighborhood size. Absolute k overrides; otherwise scale the
    pre-registered ceil(sqrt(n_models)) rule by k_mult (so a mixed-n pool keeps each
    contest's own sqrt-n rule under k-sensitivity)."""
    if k is not None:
        return k
    return max(1, round(k_mult * default_twin_k(n_models)))


def run_calibration(contests, sign=1, k=None, k_mult=1.0, R_perm=R_PERM, seed=0,
                    stat="median", weight="fisher"):
    """Run the within-contest gate over a list of (R, m1_idx, m2_idx) contests.

    Neighborhood size is per-contest ceil(sqrt(n_models)) * k_mult unless an absolute
    `k` is given. weight selects the item importance measure ("fisher" = primary raw
    J_i; "info_phi" = secondary info-VF Shapley). Non-qualifying contests are dropped.
    Returns the aggregate signal, the no-signal floor and permutation p-value, the
    verdict, and the confounded pooled rho for reference.
    """
    samples, k_used_list = [], []
    for (R, m1, m2) in contests:
        k_i = _contest_k(R.shape[0], k, k_mult)
        s = contest_sample(R, m1, m2, sign=sign, k=k_i, weight=weight)
        if s is not None:
            samples.append(s)
            k_used_list.append(k_i)
    # report the modal per-contest k (contests can mix n_models)
    k_used = int(stats.mode(k_used_list, keepdims=False).mode) if k_used_list else 0

    rhos = within_contest_rhos(samples) if samples else np.array([])
    aggregate = aggregate_signal(rhos, stat=stat)
    if samples and rhos.size:
        null = within_permutation_null(samples, R_perm=R_perm, seed=seed, stat=stat)
        floor = float(np.nanpercentile(null, FLOOR_PCT))
        finite_null = null[np.isfinite(null)]
        p_value = (float(np.mean(finite_null >= aggregate))
                   if finite_null.size else np.nan)
    else:
        floor, p_value = np.nan, np.nan

    return {
        "stat": stat,
        "aggregate": aggregate,
        "floor": floor,
        "p_value": p_value,
        "verdict": within_verdict(aggregate, floor),
        "n_contests": len(samples),
        "n_with_variance": int(rhos.size),
        "pooled_rho": pooled_spearman(samples) if samples else np.nan,
        "k": k_used,
        "weight": weight,
    }


# --------------------------------------------------------------------------- #
# Task-clustered gate (Problem 1 fix: pseudoreplication)
#
# Every adjacent-rank contest within a LiveBench task is built from the SAME item
# set, and adjacent ranks share a model, so the ~120 contests per task are one
# correlated block, not 120 independent draws. The within-contest null above shuffles
# each contest's J independently and so assumes ~n_contests independent units --
# anti-conservative. The clustered gate instead draws ONE item permutation per TASK
# per replicate and applies it to every contest in that task, so the null inherits the
# real cross-contest correlation and its spread reflects the true ~n_tasks effective
# units. We also expose a per-task table (the honest n=#tasks view) and a task-level
# cluster bootstrap. NOTE: passing the clustered floor still only rejects "zero
# association"; it does not separate genuine discrimination signal from the mechanical
# coupling of J and pivotality (both functions of R) -- that is Problem 2's ceiling.
# --------------------------------------------------------------------------- #

def contest_record(R, m1_idx, m2_idx, sign=1, k=None, weight="fisher",
                   T=2000, seed=42):
    """Like contest_sample, but keeps the FULL per-item J/pivotality vectors plus the
    same-sign mask, so a task-level item permutation can be applied consistently across
    contests that share an item set. None if the contest fails to qualify."""
    R = np.asarray(R, dtype=float)
    d = R[m1_idx] - R[m2_idx]
    mask = d == sign
    if int(mask.sum()) < MIN_SAME_SIGN:
        return None
    if weight == "fisher":
        w = compute_item_weights(R, m1_idx, m2_idx)["J"]
    elif weight == "info_phi":
        w = mc_shapley_information(d, R, m1_idx, m2_idx, T=T, seed=seed)["phi"]
    else:
        raise ValueError(f"unknown weight {weight!r}")
    piv = external_pivotality(R, m1_idx, m2_idx, k=k)["pivotality"]
    return {"J": np.asarray(w, dtype=float), "piv": np.asarray(piv, dtype=float),
            "mask": np.asarray(mask, dtype=bool)}


def _record_rho(rec, J=None):
    """Within-contest Spearman over the same-sign items; J overridable for the null."""
    J = rec["J"] if J is None else J
    m = rec["mask"]
    return _spearman(J[m], rec["piv"][m])


def records_rhos(records):
    """Per-record within-contest rho; records with no variance are dropped."""
    rhos = np.array([_record_rho(r) for r in records])
    return rhos[np.isfinite(rhos)]


def synchronized_null(records_by_task, R_perm=R_PERM, seed=0, stat="median"):
    """Cluster-respecting no-signal null.

    For each replicate, draw ONE item permutation per task and apply it to every
    contest in that task (the contests share an item index space because they share R),
    then aggregate per-contest rhos across all tasks. Because the same permutation hits
    every contest in a task, a "lucky" shuffle perturbs the whole block coherently --
    exactly as a real (or mechanical) item-level signal would -- so the null spread
    reflects the true ~n_tasks effective sample size rather than n_contests.
    """
    rng = np.random.default_rng(seed)
    out = np.empty(R_perm)
    for r in range(R_perm):
        rhos = []
        for recs in records_by_task.values():
            if not recs:
                continue
            n_items = recs[0]["J"].shape[0]
            pi = rng.permutation(n_items)
            for rec in recs:
                val = _record_rho(rec, J=rec["J"][pi])
                if np.isfinite(val):
                    rhos.append(val)
        out[r] = aggregate_signal(np.asarray(rhos), stat=stat)
    return out


def per_task_table(records_by_task, stat="median"):
    """Collapse each task to one aggregate rho -- the honest n=#tasks view."""
    table = {}
    for task, recs in records_by_task.items():
        rhos = records_rhos(recs)
        table[task] = {
            "aggregate": aggregate_signal(rhos, stat=stat),
            "n_contests": len(recs),
            "n_with_variance": int(rhos.size),
        }
    return table


def cluster_bootstrap_ci(records_by_task, B=2000, seed=0, stat="median", alpha=0.05):
    """Percentile CI for the aggregate, resampling whole TASKS with replacement.

    With only a handful of tasks this interval is coarse and lumpy by construction --
    that coarseness is the honest signal that the effective sample size is #tasks.
    """
    tasks = list(records_by_task.keys())
    n_t = len(tasks)
    rng = np.random.default_rng(seed)
    boot = np.empty(B)
    for b in range(B):
        chosen = rng.integers(0, n_t, size=n_t)
        recs = []
        for idx in chosen:
            recs.extend(records_by_task[tasks[idx]])
        boot[b] = aggregate_signal(records_rhos(recs), stat=stat)
    finite = boot[np.isfinite(boot)]
    lo = float(np.percentile(finite, 100 * alpha / 2)) if finite.size else np.nan
    hi = float(np.percentile(finite, 100 * (1 - alpha / 2))) if finite.size else np.nan
    return {"ci_lo": lo, "ci_hi": hi, "n_tasks": n_t, "B": B}


def clustered_gate(records_by_task, R_perm=R_PERM, seed=0, stat="median"):
    """Task-clustered within-contest gate. Same statistic as run_calibration (median
    per-contest rho) but judged against the synchronized (cluster-respecting) null."""
    all_recs = [r for recs in records_by_task.values() for r in recs]
    rhos = records_rhos(all_recs)
    aggregate = aggregate_signal(rhos, stat=stat)
    if all_recs and rhos.size:
        null = synchronized_null(records_by_task, R_perm=R_perm, seed=seed, stat=stat)
        finite = null[np.isfinite(null)]
        floor = float(np.nanpercentile(null, FLOOR_PCT)) if finite.size else np.nan
        p_value = (float(np.mean(finite >= aggregate))
                   if finite.size else np.nan)
    else:
        floor, p_value = np.nan, np.nan
    return {
        "stat": stat,
        "aggregate": aggregate,
        "floor": floor,
        "p_value": p_value,
        "verdict": within_verdict(aggregate, floor),
        "n_contests": len(all_recs),
        "n_with_variance": int(rhos.size),
        "n_tasks": len(records_by_task),
        "pooled_rho": pooled_spearman(
            [(r["J"][r["mask"]], r["piv"][r["mask"]]) for r in all_recs]
        ) if all_recs else np.nan,
    }


# --------------------------------------------------------------------------- #
# Cluster-level meta-gate (each cluster = one unit, size-balanced)
#
# clustered_gate pools every contest and takes one median, so a benchmark with 5000
# contests would dominate a LiveBench task with 100. When we expand to many clusters of
# very different sizes (3 LiveBench tasks + classic benchmarks + 57 MMLU subjects), the
# statistically sound unit is the CLUSTER: compute one rho per cluster, then aggregate
# across clusters (a meta-analysis). The synchronized null and the bootstrap then both
# operate at the cluster level, so effective n = number of clusters.
# --------------------------------------------------------------------------- #

def cluster_aggregates(records_by_cluster, stat="median"):
    """One within-cluster rho per cluster (clusters with no variance are dropped)."""
    out = {}
    for c, recs in records_by_cluster.items():
        a = aggregate_signal(records_rhos(recs), stat=stat)
        if np.isfinite(a):
            out[c] = a
    return out


def meta_synchronized_null(records_by_cluster, R_perm=R_PERM, seed=0, stat="median"):
    """Cluster-level no-signal null: one item permutation per cluster per replicate,
    aggregate within each cluster, then aggregate the cluster rhos. The cross-cluster
    spread reflects the true number of independent clusters."""
    rng = np.random.default_rng(seed)
    clusters = list(records_by_cluster.values())
    out = np.empty(R_perm)
    for r in range(R_perm):
        cluster_rhos = []
        for recs in clusters:
            if not recs:
                continue
            n_items = recs[0]["J"].shape[0]
            pi = rng.permutation(n_items)
            rhos = [_record_rho(rec, J=rec["J"][pi]) for rec in recs]
            a = aggregate_signal(np.asarray([v for v in rhos if np.isfinite(v)]),
                                 stat=stat)
            if np.isfinite(a):
                cluster_rhos.append(a)
        out[r] = aggregate_signal(np.asarray(cluster_rhos), stat=stat)
    return out


def meta_cluster_bootstrap_ci(records_by_cluster, B=2000, seed=0, stat="median",
                              alpha=0.05):
    """Percentile CI resampling the per-cluster rhos with replacement (cluster = unit)."""
    vals = np.array(list(cluster_aggregates(records_by_cluster, stat=stat).values()))
    if vals.size == 0:
        return {"ci_lo": np.nan, "ci_hi": np.nan, "n_clusters": 0, "B": B}
    rng = np.random.default_rng(seed)
    boot = np.array([aggregate_signal(rng.choice(vals, size=vals.size, replace=True),
                                      stat=stat) for _ in range(B)])
    finite = boot[np.isfinite(boot)]
    return {
        "ci_lo": float(np.percentile(finite, 100 * alpha / 2)),
        "ci_hi": float(np.percentile(finite, 100 * (1 - alpha / 2))),
        "n_clusters": int(vals.size),
        "B": B,
    }


def cluster_meta_gate(records_by_cluster, R_perm=R_PERM, seed=0, stat="median"):
    """Cluster-as-unit meta gate: median of per-cluster rhos, judged against the
    cluster-level synchronized null. Each cluster counts once regardless of size."""
    cagg = cluster_aggregates(records_by_cluster, stat=stat)
    vals = np.array(list(cagg.values()))
    aggregate = aggregate_signal(vals, stat=stat)
    if vals.size:
        null = meta_synchronized_null(records_by_cluster, R_perm=R_perm, seed=seed,
                                      stat=stat)
        finite = null[np.isfinite(null)]
        floor = float(np.nanpercentile(null, FLOOR_PCT)) if finite.size else np.nan
        p_value = float(np.mean(finite >= aggregate)) if finite.size else np.nan
    else:
        floor, p_value = np.nan, np.nan
    return {
        "stat": stat,
        "aggregate": aggregate,
        "floor": floor,
        "p_value": p_value,
        "verdict": within_verdict(aggregate, floor),
        "n_clusters": int(vals.size),
        "n_clusters_total": len(records_by_cluster),
        "cluster_rhos": {c: float(v) for c, v in cagg.items()},
    }


# --------------------------------------------------------------------------- #
# Mechanical-coupling ceiling (Problem 2)
#
# The synchronized floor only rejects "zero association." But J_i and pivotality are
# both deterministic functions of the same response matrix, so SOME rank correlation is
# expected purely by construction (e.g. difficulty drives both). To bound that, we
# simulate equal-discrimination (1PL/Rasch) data matched to each cluster's empirical item
# difficulty and model ability -- i.e. data with NO genuine discrimination variation --
# and rerun the whole J + pivotality pipeline. The resulting rho distribution is the
# mechanical-coupling level. Observed rho ABOVE this ceiling is evidence of genuine
# discrimination signal beyond mechanical coupling; between floor and ceiling is "partial
# / consistent with mechanical coupling"; at/below the floor is no support.
# --------------------------------------------------------------------------- #

def _difficulty_ability(R):
    """Empirical 1PL marginals, matching compute_item_weights / model_abilities:
    difficulty b_i = -logit(mean item score), ability theta_m = logit(mean model score)."""
    R = np.asarray(R, dtype=float)
    p_i = np.clip(R.mean(axis=0), 0.001, 0.999)
    p_m = np.clip(R.mean(axis=1), 0.001, 0.999)
    return -np.log(p_i / (1.0 - p_i)), np.log(p_m / (1.0 - p_m))


def mechanical_ceiling(clusters, sign=1, R_sim=100, seed=0, stat="median",
                       contest_cap=None, k=None):
    """1PL/Rasch parametric ceiling for the J->pivotality correlation.

    clusters: {name: (R, contest_pairs)} with contest_pairs a list of (m1_idx, m2_idx).
    Each replicate simulates equal-discrimination responses matched to each cluster's
    difficulty/ability, recomputes per-cluster rho through the real pipeline, and
    aggregates across clusters. Returns the array (length R_sim) of aggregate rhos -- the
    distribution expected with no genuine discrimination variation."""
    rng = np.random.default_rng(seed)
    fitted = {}
    for name, (R, contests) in clusters.items():
        b, theta = _difficulty_ability(R)
        cs = contests if contest_cap is None else contests[:contest_cap]
        fitted[name] = (np.ones_like(b), b, theta, cs)
    out = np.empty(R_sim)
    for r in range(R_sim):
        cluster_rhos = []
        for (a, b, theta, cs) in fitted.values():
            R_sim_mat = simulate_2pl(a, b, theta, rng)        # a == 1 -> 1PL
            rhos = []
            for (m1, m2) in cs:
                rec = contest_record(R_sim_mat, m1, m2, sign=sign, k=k)
                if rec is not None:
                    v = _record_rho(rec)
                    if np.isfinite(v):
                        rhos.append(v)
            agg = aggregate_signal(np.asarray(rhos), stat=stat)
            if np.isfinite(agg):
                cluster_rhos.append(agg)
        out[r] = aggregate_signal(np.asarray(cluster_rhos), stat=stat)
    return out


def ceiling_verdict(observed, floor, ceiling_hi):
    """Three-way verdict against the no-signal floor and the mechanical-coupling ceiling."""
    if not all(np.isfinite(x) for x in (observed, floor, ceiling_hi)):
        return "No support"
    if observed <= floor:
        return "No support"
    if observed > ceiling_hi:
        return "Supported (beyond mechanical coupling)"
    return "Partial (within mechanical coupling)"


def k_sensitivity_report(contests, sign=1, R_perm=R_PERM, seed=0, stat="median"):
    """Re-run the gate at half, 1x, and double the per-contest sqrt-n rule. Scaling is
    applied per contest (k_mult), so the 1x row reproduces the headline exactly even
    when the pool mixes contests with different n_models."""
    settings = {"half_k": 0.5, "k": 1.0, "double_k": 2.0}
    return {
        name: run_calibration(contests, sign=sign, k_mult=mult,
                              R_perm=R_perm, seed=seed, stat=stat)
        for name, mult in settings.items()
    }
