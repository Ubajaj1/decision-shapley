"""E16 -- Secondary target: info-VF Shapley phi_i through the clustered gate.

The primary external check (e14/e15) used the raw, un-riggable Fisher discrimination J_i.
The info-VF Shapley value phi_i is the modeling-dependent downstream quantity; comparing
its external correlation to J_i's is an anti-circularity test: if a value function were
manufacturing the signal, phi would correlate with pivotality MORE strongly than raw J.
The e12 pilot found phi ~ J on LiveBench, but pre-clustering. This re-runs the comparison
through the cluster-level meta gate.

Scope: LiveBench (frontier, n_active small) + MMLU subjects. Classic whole benchmarks are
omitted because phi's O(|S|^3)-per-prefix cost is infeasible there (same-sign item counts
run to >1600). To keep phi and J strictly comparable we compute BOTH on the SAME contests,
skipping contests whose same-sign item count exceeds N_ACTIVE_CAP.

Interpretation: if rho_phi ~ rho_J, phi adds no signal beyond raw discrimination (no
circularity), and phi inherits J's mechanical-coupling verdict from e15.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import json
import numpy as np

from livebench_io import load_livebench_task, LIVEBENCH_BINARY_TASKS
from data_io import MMLU_SUBJECTS, RAW_DIR, load_long_csv, long_to_matrix
from calibration import contest_record, cluster_meta_gate, MIN_SAME_SIGN

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
SEED = 42
N_ACTIVE_CAP = 60      # skip contests with more same-sign items (phi cost ~ n^3)
MAX_CONTESTS = 15      # per cluster
PHI_T = 800            # MC permutations for the info-VF Shapley (rank-corr tolerant)


def dual_records(R, sign=1):
    """(J-records, phi-records) on the SAME qualifying, n_active-capped contests."""
    R = np.asarray(R, dtype=float)
    scores = R.mean(axis=1)
    order = np.argsort(-scores)
    pairs = []
    for i in range(len(order) - 1):
        m1, m2 = int(order[i]), int(order[i + 1])
        if scores[m1] == scores[m2]:
            continue
        nplus = int((R[m1] - R[m2] == sign).sum())
        if MIN_SAME_SIGN <= nplus <= N_ACTIVE_CAP:
            pairs.append((m1, m2))
    if not pairs:
        return [], []
    if len(pairs) > MAX_CONTESTS:
        pick = np.unique(np.linspace(0, len(pairs) - 1, MAX_CONTESTS).round().astype(int))
        pairs = [pairs[j] for j in pick]
    j_recs, phi_recs = [], []
    for (m1, m2) in pairs:
        jr = contest_record(R, m1, m2, sign=sign, weight="fisher")
        pr = contest_record(R, m1, m2, sign=sign, weight="info_phi", T=PHI_T)
        if jr is not None and pr is not None:
            j_recs.append(jr)
            phi_recs.append(pr)
    return j_recs, phi_recs


def build_family(matrices):
    j_by, phi_by = {}, {}
    for name, R in matrices.items():
        jr, pr = dual_records(R, sign=1)
        if jr and pr:
            j_by[name] = jr
            phi_by[name] = pr
    return j_by, phi_by


def _fmt(x):
    return "nan" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.4f}"


def run_family(label, matrices, R_perm):
    j_by, phi_by = build_family(matrices)
    if not j_by:
        print(f"\n  {label}: no qualifying clusters")
        return None
    gj = cluster_meta_gate(j_by, R_perm=R_perm, seed=SEED)
    gp = cluster_meta_gate(phi_by, R_perm=R_perm, seed=SEED)
    print(f"\n{'='*74}\n  {label}  ({gj['n_clusters']} clusters, same contests)\n{'='*74}")
    print(f"  PRIMARY  raw J_i : median rho = {_fmt(gj['aggregate'])}  "
          f"floor={_fmt(gj['floor'])}  p={_fmt(gj['p_value'])}  -> {gj['verdict']}")
    print(f"  SECONDARY phi_i  : median rho = {_fmt(gp['aggregate'])}  "
          f"floor={_fmt(gp['floor'])}  p={_fmt(gp['p_value'])}  -> {gp['verdict']}")
    print(f"  phi - J = {_fmt(gp['aggregate'] - gj['aggregate'])}  "
          f"(near 0 => phi tracks raw J, no circular inflation)")
    return {
        "n_clusters": gj["n_clusters"],
        "J_median_rho": round(float(gj["aggregate"]), 4),
        "J_floor_p95": round(float(gj["floor"]), 4),
        "J_p_value": round(float(gj["p_value"]), 4),
        "J_verdict": gj["verdict"],
        "phi_median_rho": round(float(gp["aggregate"]), 4),
        "phi_floor_p95": round(float(gp["floor"]), 4),
        "phi_p_value": round(float(gp["p_value"]), 4),
        "phi_verdict": gp["verdict"],
        "phi_minus_J": round(float(gp["aggregate"] - gj["aggregate"]), 4),
    }


def run_e16(R_perm=1000):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = {"config": {"R_perm": R_perm, "n_active_cap": N_ACTIVE_CAP,
                          "max_contests": MAX_CONTESTS, "phi_T": PHI_T, "seed": SEED,
                          "note": "classic benchmarks omitted: phi O(n^3) infeasible "
                                  "(n_active up to 1603)"}}

    print("Loading LiveBench...")
    lb = {t: load_livebench_task(t)[0].astype(float) for t in LIVEBENCH_BINARY_TASKS}
    summary["livebench"] = run_family("LiveBench (frontier)", lb, R_perm)

    print("\nLoading MMLU subjects...")
    mmlu = {}
    for subj in MMLU_SUBJECTS:
        R = long_to_matrix(load_long_csv(RAW_DIR / f"{subj}.csv"))[0]
        if R.shape[1] >= MIN_SAME_SIGN and R.shape[0] >= 3:
            mmlu[subj] = R.astype(float)
    summary["mmlu_subjects"] = run_family("MMLU subjects", mmlu, R_perm)

    out = RESULTS_DIR / "e16_phi_clustered.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved {out}")
    return summary


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    run_e16(R_perm=200 if args.quick else 1000)
