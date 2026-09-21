"""E14 -- Multi-benchmark, cluster-level external validation (Problem 1 power fix).

e13 showed the discrimination->pivotality signal does not survive once contests are
clustered by task -- but with only 3 LiveBench tasks the gate had effective n=3 and was
badly underpowered. This experiment expands the evidence base to MANY independent
item-sets so the cluster-level test actually has power:

  CLUSTERS (each = one distinct item-set, the unit of the meta-analysis):
    - livebench : LCB_generation, coding_completion, typos          (3, frontier models)
    - classic   : arc, gsm8k, hellaswag, truthfulqa, winogrande     (5, open-leaderboard)
    - mmlu      : MMLU split into its ~57 subjects                   (~57, distinct items)

Each cluster yields one within-cluster Spearman(J_i, pivotality_i) (median over a capped
sample of its tight contests). cluster_meta_gate then aggregates ACROSS clusters and
judges the result against a cluster-level synchronized null (one item permutation per
cluster). We report per family and pooled, with a per-cluster table and a
cluster-bootstrap CI, and contrast against the n=3 LiveBench-only result.

CAVEATS (must stay in the paper):
  - Two populations: LiveBench = ~150 frontier models; classic/mmlu = ~5000 open-
    leaderboard models (heavy fine-tune families) on saturated/contamination-prone
    benchmarks. We keep the families separate and pool only as a sensitivity view.
  - MMLU subjects share models (distinct items though), so subjects are not perfectly
    independent; treat ~57 as an upper bound on independent units.
  - Passing the floor still only rejects "zero association," not mechanical coupling of
    J and pivotality (Problem 2).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import json
import numpy as np

from livebench_io import load_livebench_task, LIVEBENCH_BINARY_TASKS
from data_io import (
    BENCHMARKS, MMLU_SUBJECTS, RAW_DIR, load_long_csv, long_to_matrix, load_benchmark,
)
from calibration import (
    contest_record,
    cluster_meta_gate,
    per_task_table,
    meta_cluster_bootstrap_ci,
    MIN_SAME_SIGN,
    R_PERM,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
SEED = 42
CLASSIC_WHOLE = ["arc", "gsm8k", "hellaswag", "truthfulqa", "winogrande"]
MAX_CONTESTS = 40          # cap tight contests sampled per cluster (bounds compute)


def harvest_cluster_records(R, sign=1, max_contests=MAX_CONTESTS):
    """Records for one cluster. Cheap qualification pass over adjacent ranks first
    (no value-function work), then build contest_records for an evenly-spaced sample of
    qualifying contests -- so the expensive J/pivotality computation runs <= max_contests
    times even for 5000-model benchmarks."""
    R = np.asarray(R, dtype=float)
    scores = R.mean(axis=1)
    order = np.argsort(-scores)
    qualifying = []
    for i in range(len(order) - 1):
        m1, m2 = int(order[i]), int(order[i + 1])
        if scores[m1] == scores[m2]:
            continue
        d = R[m1] - R[m2]
        if int((d == sign).sum()) >= MIN_SAME_SIGN:
            qualifying.append((m1, m2))
    if not qualifying:
        return []
    if len(qualifying) > max_contests:
        pick = np.unique(np.linspace(0, len(qualifying) - 1, max_contests).round().astype(int))
        qualifying = [qualifying[j] for j in pick]
    records = []
    for (m1, m2) in qualifying:
        rec = contest_record(R, m1, m2, sign=sign)
        if rec is not None:
            records.append(rec)
    return records


def load_clusters():
    """Return {family: {cluster_name: R}}."""
    families = {"livebench": {}, "classic": {}, "mmlu": {}}

    for t in LIVEBENCH_BINARY_TASKS:
        R, _models, _items = load_livebench_task(t)
        families["livebench"][t] = R.astype(float)

    for b in CLASSIC_WHOLE:
        R, _models, _items = load_benchmark(b)
        families["classic"][b] = R.astype(float)

    for subj in MMLU_SUBJECTS:
        df = load_long_csv(RAW_DIR / f"{subj}.csv")
        R, _models, _items = long_to_matrix(df)
        if R.shape[1] >= MIN_SAME_SIGN and R.shape[0] >= 3:
            families["mmlu"][subj] = R.astype(float)

    return families


def build_records(family_clusters):
    """{cluster_name: records} for one family, dropping clusters with no qualifying
    contests."""
    out = {}
    for name, R in family_clusters.items():
        recs = harvest_cluster_records(R, sign=1)
        if recs:
            out[name] = recs
    return out


def _r(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 4)


def run_family(label, records_by_cluster, R_perm):
    if not records_by_cluster:
        print(f"\n  {label}: no qualifying clusters")
        return None
    gate = cluster_meta_gate(records_by_cluster, R_perm=R_perm, seed=SEED)
    ci = meta_cluster_bootstrap_ci(records_by_cluster, B=2000, seed=SEED)
    pos = sum(1 for v in gate["cluster_rhos"].values() if v > 0)
    print(f"\n{'='*72}\n  {label}\n{'='*72}")
    print(f"  clusters with variance = {gate['n_clusters']} / {gate['n_clusters_total']}"
          f"   ({pos} of {gate['n_clusters']} have rho>0)")
    print(f"  median cluster-rho = {_fmt(gate['aggregate'])}   "
          f"floor(p95) = {_fmt(gate['floor'])}   p = {_fmt(gate['p_value'])}  "
          f"-> {gate['verdict']}")
    print(f"  cluster-bootstrap 95% CI = [{_fmt(ci['ci_lo'])}, {_fmt(ci['ci_hi'])}]")
    return {
        "median_cluster_rho": _r(gate["aggregate"]),
        "floor_p95": _r(gate["floor"]),
        "p_value": _r(gate["p_value"]),
        "verdict": gate["verdict"],
        "n_clusters_with_variance": gate["n_clusters"],
        "n_clusters_total": gate["n_clusters_total"],
        "n_clusters_rho_positive": pos,
        "cluster_bootstrap_ci": [_r(ci["ci_lo"]), _r(ci["ci_hi"])],
        "cluster_rhos": {c: _r(v) for c, v in gate["cluster_rhos"].items()},
    }


def _fmt(x):
    return "nan" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.4f}"


def run_e14(R_perm=R_PERM):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading clusters (livebench + classic + mmlu subjects)...")
    families = load_clusters()
    for fam, cl in families.items():
        print(f"  {fam}: {len(cl)} candidate clusters")

    recs = {fam: build_records(cl) for fam, cl in families.items()}
    summary = {"config": {"R_perm": R_perm, "min_same_sign": MIN_SAME_SIGN,
                          "max_contests_per_cluster": MAX_CONTESTS, "seed": SEED,
                          "gate": "cluster_meta_synchronized_null", "sign": "+1"}}

    summary["livebench"] = run_family("FAMILY: LiveBench (frontier, 3 tasks)",
                                      recs["livebench"], R_perm)
    summary["classic"] = run_family("FAMILY: classic benchmarks (open-leaderboard, whole)",
                                    recs["classic"], R_perm)
    summary["mmlu_subjects"] = run_family("FAMILY: MMLU subjects",
                                          recs["mmlu"], R_perm)

    # pooled sensitivity view (all clusters across families)
    pooled = {}
    for fam in recs.values():
        pooled.update(fam)
    summary["pooled_all"] = run_family(
        "POOLED (all families -- sensitivity only, mixes populations)", pooled, R_perm)

    out = RESULTS_DIR / "e14_multibench_validation.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved {out}")
    print("\nNOTE: families kept separate by design; 'pooled_all' mixes frontier and "
          "open-leaderboard populations and is a sensitivity view only.")
    return summary


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="fast smoke run (small R_perm)")
    args = ap.parse_args()
    run_e14(R_perm=200 if args.quick else R_PERM)
