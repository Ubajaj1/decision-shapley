"""Tests for the task-clustered external-validation gate (Problem 1 fix).

The original within-contest gate (src/calibration.py) shuffles J independently in
each contest and aggregates across ALL adjacent-rank contests as if they were
independent. They are not: within one LiveBench task every contest is built from the
same item set (and adjacent ranks share a model), so the ~120 contests per task are
one correlated block, not 120 independent draws. That pseudoreplication makes the
permutation null far too tight and the p-value anti-conservative.

The clustered gate fixes this by:
  - synchronized_null: draw ONE item permutation per TASK per replicate and apply it
    to every contest in that task, so the null inherits the real cross-contest
    correlation (the null spread widens to reflect the true ~n_tasks effective units).
  - per_task_table: collapse each task to one number, exposing the real sample size.
  - cluster_bootstrap_ci: resample whole tasks for an honest (coarse) interval.

These tests pin the contract and the central behaviour: for item-sharing contests the
synchronized null is materially WIDER than the naive independent null.
"""

import numpy as np
import pytest

from _helpers import find_qualifying_pair, make_2pl_matrix
from calibration import (
    MIN_SAME_SIGN,
    contest_record,
    records_rhos,
    aggregate_signal,
    synchronized_null,
    within_permutation_null,
    per_task_table,
    cluster_bootstrap_ci,
    clustered_gate,
    cluster_aggregates,
    meta_synchronized_null,
    cluster_meta_gate,
    mechanical_ceiling,
    ceiling_verdict,
)


# --------------------------------------------------------------------------- #
# contest_record (full per-item vectors + same-sign mask)
# --------------------------------------------------------------------------- #

def test_contest_record_returns_full_vectors_and_mask():
    R, _ = make_2pl_matrix(150, [(1.0, b) for b in np.linspace(-1.5, 1.5, 24)], seed=7)
    c = find_qualifying_pair(R)
    rec = contest_record(R, c["m1_idx"], c["m2_idx"], sign=1)
    assert rec is not None
    n_items = R.shape[1]
    assert rec["J"].shape == (n_items,)
    assert rec["piv"].shape == (n_items,)
    assert rec["mask"].shape == (n_items,)
    assert int(rec["mask"].sum()) == c["n_plus"]
    assert np.all((rec["piv"] >= 0.0) & (rec["piv"] <= 1.0))


def test_contest_record_disqualifies_too_few_same_sign():
    R = np.array([
        [1, 1, 0, 0, 1],
        [0, 0, 1, 1, 0],   # d = [1,1,-1,-1,1] -> n_plus = 3 < 4
    ], dtype=float)
    assert contest_record(R, 0, 1, sign=1) is None


# --------------------------------------------------------------------------- #
# records_rhos matches the masked within-contest Spearman
# --------------------------------------------------------------------------- #

def test_records_rhos_one_per_varying_contest():
    rec_up = {"J": np.array([1.0, 2.0, 3.0, 4.0]),
              "piv": np.array([0.1, 0.2, 0.3, 0.4]),
              "mask": np.array([True, True, True, True])}
    rec_down = {"J": np.array([1.0, 2.0, 3.0, 4.0]),
                "piv": np.array([0.4, 0.3, 0.2, 0.1]),
                "mask": np.array([True, True, True, True])}
    rhos = records_rhos([rec_up, rec_down])
    np.testing.assert_allclose(np.sort(rhos), [-1.0, 1.0])


def test_records_rhos_respects_mask():
    # only the masked items participate; the off-mask item would invert the sign
    rec = {"J": np.array([1.0, 2.0, 3.0, 100.0]),
           "piv": np.array([0.1, 0.2, 0.3, 0.0]),
           "mask": np.array([True, True, True, False])}
    rhos = records_rhos([rec])
    assert rhos[0] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# THE central property: clustering widens the null for item-sharing contests
# --------------------------------------------------------------------------- #

def _shared_item_task(n_contests=8, n_items=12, seed=0):
    """One task: many contests over the SAME item set with a shared J ranking,
    i.e. the pseudoreplication structure in miniature."""
    rng = np.random.default_rng(seed)
    J = np.arange(1.0, n_items + 1.0)
    mask = np.ones(n_items, dtype=bool)
    recs = []
    for _ in range(n_contests):
        piv = J / (n_items + 1.0) + rng.normal(0, 0.02, n_items)
        recs.append({"J": J.copy(), "piv": piv, "mask": mask.copy()})
    return recs


def test_synchronized_null_wider_than_independent_for_shared_items():
    recs = _shared_item_task(n_contests=8, n_items=12, seed=1)
    # naive (independent per-contest) null: feed the masked (J, piv) pairs
    indep_samples = [(r["J"][r["mask"]], r["piv"][r["mask"]]) for r in recs]
    indep = within_permutation_null(indep_samples, R_perm=400, seed=2, stat="median")
    sync = synchronized_null({"task": recs}, R_perm=400, seed=2, stat="median")
    # the cluster-respecting null must be materially wider (the whole point of the fix)
    assert np.nanstd(sync) > 1.5 * np.nanstd(indep)
    assert np.nanpercentile(sync, 95) > np.nanpercentile(indep, 95)


def test_synchronized_null_centered_near_zero():
    recs = _shared_item_task(n_contests=6, n_items=12, seed=3)
    null = synchronized_null({"task": recs}, R_perm=400, seed=4, stat="median")
    assert null.shape == (400,)
    assert abs(np.nanmedian(null)) < 0.3


# --------------------------------------------------------------------------- #
# per_task_table + cluster_bootstrap_ci
# --------------------------------------------------------------------------- #

def test_per_task_table_one_entry_per_task():
    t1 = _shared_item_task(n_contests=5, n_items=10, seed=5)
    t2 = _shared_item_task(n_contests=7, n_items=10, seed=6)
    table = per_task_table({"a": t1, "b": t2})
    assert set(table) == {"a", "b"}
    assert table["a"]["n_contests"] == 5
    assert table["b"]["n_contests"] == 7
    assert np.isfinite(table["a"]["aggregate"])


def test_cluster_bootstrap_ci_ordered():
    by_task = {"a": _shared_item_task(seed=7),
               "b": _shared_item_task(seed=8),
               "c": _shared_item_task(seed=9)}
    ci = cluster_bootstrap_ci(by_task, B=500, seed=0)
    assert ci["ci_lo"] <= ci["ci_hi"]
    assert ci["n_tasks"] == 3


# --------------------------------------------------------------------------- #
# clustered_gate end-to-end
# --------------------------------------------------------------------------- #

def test_clustered_gate_contract():
    by_task = {"a": _shared_item_task(seed=10),
               "b": _shared_item_task(seed=11),
               "c": _shared_item_task(seed=12)}
    res = clustered_gate(by_task, R_perm=300, seed=0)
    assert set(res) >= {"aggregate", "floor", "p_value", "verdict",
                        "n_contests", "n_with_variance", "n_tasks", "stat"}
    assert res["n_tasks"] == 3
    assert 0.0 <= res["p_value"] <= 1.0
    assert res["verdict"] in {"Supported", "No support"}
    assert (res["verdict"] == "Supported") == (res["aggregate"] > res["floor"])


def test_clustered_gate_p_larger_than_naive_for_shared_items():
    # the corrected p-value must be >= the anti-conservative naive one (clustering
    # can only cost significance for pseudoreplicated data, never manufacture it).
    recs = _shared_item_task(n_contests=10, n_items=14, seed=13)
    indep_samples = [(r["J"][r["mask"]], r["piv"][r["mask"]]) for r in recs]
    observed = aggregate_signal(records_rhos(recs), stat="median")
    indep_null = within_permutation_null(indep_samples, R_perm=500, seed=1)
    indep_p = float(np.mean(indep_null[np.isfinite(indep_null)] >= observed))
    clustered = clustered_gate({"task": recs}, R_perm=500, seed=1)
    assert clustered["p_value"] >= indep_p


# --------------------------------------------------------------------------- #
# cluster-level meta gate (each cluster = one unit, size-balanced)
# --------------------------------------------------------------------------- #

def _signal_cluster(n_contests, n_items, rho_sign=+1, seed=0):
    """A cluster whose items carry a consistent (signed) J->pivotality relationship."""
    rng = np.random.default_rng(seed)
    J = np.arange(1.0, n_items + 1.0)
    mask = np.ones(n_items, dtype=bool)
    recs = []
    for _ in range(n_contests):
        ranked = J if rho_sign > 0 else (n_items + 1.0 - J)   # reverse, don't negate
        piv = np.clip(ranked / (n_items + 1.0) + rng.normal(0, 0.05, n_items), 0, 1)
        recs.append({"J": J.copy(), "piv": piv, "mask": mask.copy()})
    return recs


def test_cluster_aggregates_one_value_per_cluster():
    by = {"a": _signal_cluster(5, 10, +1, seed=1),
          "b": _signal_cluster(5, 10, -1, seed=2)}
    cagg = cluster_aggregates(by)
    assert set(cagg) == {"a", "b"}
    assert cagg["a"] > 0 and cagg["b"] < 0


def test_cluster_meta_gate_balances_cluster_sizes():
    # one huge cluster of noise + many small consistent-signal clusters: the meta gate
    # (cluster = one unit) should reflect the many signal clusters, not the big noisy one.
    rng = np.random.default_rng(0)
    big_noise = []
    J = np.arange(1.0, 12.0)
    for _ in range(400):
        big_noise.append({"J": J.copy(), "piv": rng.random(11), "mask": np.ones(11, bool)})
    by = {"noise": big_noise}
    for s in range(12):
        by[f"sig{s}"] = _signal_cluster(4, 11, +1, seed=100 + s)
    res = cluster_meta_gate(by, R_perm=300, seed=1)
    assert res["n_clusters"] == 13                 # noise + 12 signal
    assert res["aggregate"] > 0                     # signal clusters carry it
    assert 0.0 <= res["p_value"] <= 1.0


def test_cluster_meta_gate_detects_consistent_signal():
    by = {f"c{i}": _signal_cluster(4, 12, +1, seed=i) for i in range(15)}
    res = cluster_meta_gate(by, R_perm=400, seed=2)
    assert res["verdict"] == "Supported"
    assert res["p_value"] < 0.05


def test_cluster_meta_gate_null_when_no_signal():
    rng = np.random.default_rng(7)
    by = {}
    J = np.arange(1.0, 13.0)
    for c in range(15):
        recs = [{"J": J.copy(), "piv": rng.random(12), "mask": np.ones(12, bool)}
                for _ in range(4)]
        by[f"c{c}"] = recs
    res = cluster_meta_gate(by, R_perm=400, seed=3)
    assert res["verdict"] == "No support"


def test_meta_synchronized_null_shape():
    by = {f"c{i}": _signal_cluster(3, 10, +1, seed=i) for i in range(6)}
    null = meta_synchronized_null(by, R_perm=250, seed=1)
    assert null.shape == (250,)


# --------------------------------------------------------------------------- #
# mechanical-coupling ceiling (Problem 2)
# --------------------------------------------------------------------------- #

def test_ceiling_verdict_logic():
    # at/below floor -> no support; above ceiling -> beyond mechanical; between -> partial
    assert ceiling_verdict(0.01, floor=0.02, ceiling_hi=0.05) == "No support"
    assert ceiling_verdict(0.02, floor=0.02, ceiling_hi=0.05) == "No support"
    assert ceiling_verdict(0.09, floor=0.02, ceiling_hi=0.05) == \
        "Supported (beyond mechanical coupling)"
    assert ceiling_verdict(0.04, floor=0.02, ceiling_hi=0.05) == \
        "Partial (within mechanical coupling)"
    assert ceiling_verdict(np.nan, 0.02, 0.05) == "No support"


def _onepl_cluster_matrices(n_clusters, n_models=600, n_items=30, seed=0):
    """1PL (equal-discrimination) clusters with their qualifying contest pairs."""
    from _helpers import make_2pl_matrix
    clusters, by_records = {}, {}
    for c in range(n_clusters):
        specs = [(1.0, b) for b in np.linspace(-1.6, 1.8, n_items)]
        R, _ = make_2pl_matrix(n_models, specs, seed=seed + c)
        scores = R.mean(1)
        order = np.argsort(-scores)
        pairs = []
        for i in range(len(order) - 1):
            m1, m2 = int(order[i]), int(order[i + 1])
            if scores[m1] != scores[m2] and int((R[m1] - R[m2] == 1).sum()) >= MIN_SAME_SIGN:
                pairs.append((m1, m2))
        if not pairs:
            continue
        recs = [contest_record(R, m1, m2, sign=1) for (m1, m2) in pairs]
        recs = [r for r in recs if r is not None]
        if recs:
            clusters[f"c{c}"] = (R, pairs)
            by_records[f"c{c}"] = recs
    return clusters, by_records


def test_mechanical_ceiling_shape_and_finite():
    clusters, _ = _onepl_cluster_matrices(5, seed=10)
    ceil = mechanical_ceiling(clusters, R_sim=40, seed=1, contest_cap=10)
    assert ceil.shape == (40,)
    assert np.isfinite(np.nanmedian(ceil))


def test_mechanical_ceiling_covers_1pl_data():
    # FALSE-POSITIVE CONTROL: on equal-discrimination (1PL) data there is no genuine
    # discrimination signal, so the observed correlation must NOT clear the mechanical
    # ceiling -- the verdict must not be "beyond mechanical coupling".
    clusters, by_records = _onepl_cluster_matrices(8, seed=20)
    observed = aggregate_signal(
        np.array(list(cluster_aggregates(by_records).values())), stat="median")
    ceil = mechanical_ceiling(clusters, R_sim=80, seed=2, contest_cap=12)
    ceiling_hi = float(np.nanpercentile(ceil, 95))
    assert observed <= ceiling_hi          # mechanical ceiling covers mechanical-only data
