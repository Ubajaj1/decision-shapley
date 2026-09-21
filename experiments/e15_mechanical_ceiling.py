"""E15 -- Mechanical-coupling ceiling for the discrimination->pivotality signal (Problem 2).

e14 showed discrimination (J_i) predicts model-free pivotality across the Open-LLM and
MMLU populations once we cluster honestly (Problem 1). But J_i and pivotality are both
functions of the same response matrix, so SOME rank correlation is expected purely by
construction. This experiment bounds that with a 1PL/Rasch parametric ceiling: for each
cluster we simulate equal-discrimination data matched to its empirical item difficulty
and model ability (so there is NO genuine discrimination variation), rerun the full
J + pivotality pipeline, and aggregate per-cluster rho across clusters. The ceiling band
(5th-95th pct over R_sim replicates) is the mechanical-coupling level.

VERDICT per family:
  observed <= floor                      -> No support
  floor < observed <= ceiling(95th)      -> Partial (consistent with mechanical coupling)
  observed > ceiling(95th)               -> Supported beyond mechanical coupling

We run it on the families where e14 found support (classic Open-LLM + MMLU subjects), and
on LiveBench for completeness. Floor/observed come from the cluster-level gate (e14).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import json
import numpy as np

from livebench_io import load_livebench_task, LIVEBENCH_BINARY_TASKS
from data_io import MMLU_SUBJECTS, RAW_DIR, load_long_csv, long_to_matrix, load_benchmark
from calibration import (
    contest_record,
    cluster_meta_gate,
    mechanical_ceiling,
    ceiling_verdict,
    MIN_SAME_SIGN,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
SEED = 42
CLASSIC_WHOLE = ["arc", "gsm8k", "hellaswag", "truthfulqa", "winogrande"]
MAX_CONTESTS = 40           # for observed/floor (matches e14)
CEIL_CONTEST_CAP = 15       # per-cluster contests used in each ceiling replicate
R_SIM = 100                 # ceiling replicates


def qualifying_pairs(R, sign=1, max_contests=MAX_CONTESTS):
    R = np.asarray(R, dtype=float)
    scores = R.mean(axis=1)
    order = np.argsort(-scores)
    pairs = []
    for i in range(len(order) - 1):
        m1, m2 = int(order[i]), int(order[i + 1])
        if scores[m1] == scores[m2]:
            continue
        if int((R[m1] - R[m2] == sign).sum()) >= MIN_SAME_SIGN:
            pairs.append((m1, m2))
    if len(pairs) > max_contests:
        pick = np.unique(np.linspace(0, len(pairs) - 1, max_contests).round().astype(int))
        pairs = [pairs[j] for j in pick]
    return pairs


def build_family(matrices):
    """Return (clusters={name:(R,pairs)}, records_by_cluster={name:[recs]})."""
    clusters, recs_by = {}, {}
    for name, R in matrices.items():
        pairs = qualifying_pairs(R, sign=1)
        if not pairs:
            continue
        recs = [contest_record(R, m1, m2, sign=1) for (m1, m2) in pairs]
        recs = [r for r in recs if r is not None]
        if recs:
            clusters[name] = (R, pairs)
            recs_by[name] = recs
    return clusters, recs_by


def load_family_matrices():
    fam = {"livebench": {}, "classic": {}, "mmlu": {}}
    for t in LIVEBENCH_BINARY_TASKS:
        fam["livebench"][t] = load_livebench_task(t)[0].astype(float)
    for b in CLASSIC_WHOLE:
        fam["classic"][b] = load_benchmark(b)[0].astype(float)
    for subj in MMLU_SUBJECTS:
        R = long_to_matrix(load_long_csv(RAW_DIR / f"{subj}.csv"))[0]
        if R.shape[1] >= MIN_SAME_SIGN and R.shape[0] >= 3:
            fam["mmlu"][subj] = R.astype(float)
    return fam


def _fmt(x):
    return "nan" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.4f}"


def run_family(label, matrices, R_perm):
    clusters, recs_by = build_family(matrices)
    if not clusters:
        print(f"\n  {label}: no qualifying clusters")
        return None
    gate = cluster_meta_gate(recs_by, R_perm=R_perm, seed=SEED)
    observed, floor = gate["aggregate"], gate["floor"]
    ceil = mechanical_ceiling(clusters, sign=1, R_sim=R_SIM, seed=SEED,
                              contest_cap=CEIL_CONTEST_CAP)
    ceil_lo = float(np.nanpercentile(ceil, 5))
    ceil_md = float(np.nanpercentile(ceil, 50))
    ceil_hi = float(np.nanpercentile(ceil, 95))
    verdict = ceiling_verdict(observed, floor, ceil_hi)
    print(f"\n{'='*74}\n  {label}\n{'='*74}")
    print(f"  clusters={gate['n_clusters']}   observed median rho = {_fmt(observed)}")
    print(f"  no-signal floor (p95)        = {_fmt(floor)}   (clustered permutation)")
    print(f"  mechanical ceiling [5,50,95] = [{_fmt(ceil_lo)}, {_fmt(ceil_md)}, {_fmt(ceil_hi)}]   (1PL sim)")
    print(f"  -> {verdict}")
    return {
        "n_clusters": gate["n_clusters"],
        "observed_median_rho": round(float(observed), 4),
        "floor_p95": round(float(floor), 4),
        "ceiling_p5": round(ceil_lo, 4),
        "ceiling_p50": round(ceil_md, 4),
        "ceiling_p95": round(ceil_hi, 4),
        "verdict": verdict,
    }


def run_e15(R_perm=1000):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading family matrices...")
    fam = load_family_matrices()
    summary = {"config": {"R_perm": R_perm, "R_sim": R_SIM,
                          "ceil_contest_cap": CEIL_CONTEST_CAP, "seed": SEED,
                          "ceiling_model": "1PL/Rasch (equal discrimination)"}}
    summary["livebench"] = run_family("LiveBench (frontier)", fam["livebench"], R_perm)
    summary["classic"] = run_family("Classic Open-LLM benchmarks", fam["classic"], R_perm)
    summary["mmlu_subjects"] = run_family("MMLU subjects", fam["mmlu"], R_perm)

    out = RESULTS_DIR / "e15_mechanical_ceiling.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved {out}")
    return summary


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    run_e15(R_perm=200 if args.quick else 1000)
