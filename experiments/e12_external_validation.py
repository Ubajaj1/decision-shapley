"""E12 -- Model-free external-validation gate on LiveBench (Step 5 pilot).

Within-contest gate (re-architected 2026-06-22; see src/calibration.py): per-contest
Spearman(J_i, pivotality_i) over same-sign active items, aggregated (median) across
qualifying contests, judged against a within-contest permutation null. This replaces
the original pooled-Spearman gate, which the first pilot showed was an ecological
confound (between-contest covariance dominated; the references inverted).

Primary target: the raw, un-riggable J_i (sign d=+1). We also report the d=-1 side
separately, the between-contest-confounded pooled rho for transparency, and a
k-sensitivity (k/2, k, 2k) robustness check. Every qualifying contest (>= 4 same-sign
active items) across ranks and tasks is harvested for statistical power.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import json
import numpy as np

from livebench_io import load_livebench_task, LIVEBENCH_BINARY_TASKS
from decision import top_two
from calibration import (
    contest_sample,
    run_calibration,
    k_sensitivity_report,
    MIN_SAME_SIGN,
    R_PERM,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
SEED = 42


def harvest_contests(R, models, sign=1):
    """Every adjacent-rank contest that qualifies (>= MIN_SAME_SIGN same-sign items)."""
    contests = []
    for rank1 in range(1, R.shape[0] - 1):
        try:
            c = top_two(R, models, rank1=rank1)
        except ValueError:
            break
        if contest_sample(R, c["m1_idx"], c["m2_idx"], sign=sign) is not None:
            contests.append((R, c["m1_idx"], c["m2_idx"]))
    return contests


def _round_gate(res):
    return {
        "aggregate_within_rho": round(res["aggregate"], 4),
        "floor_p95": round(res["floor"], 4),
        "p_value": round(res["p_value"], 4),
        "verdict": res["verdict"],
        "pooled_rho_confounded": round(res["pooled_rho"], 4),
        "n_contests": res["n_contests"],
        "n_with_variance": res["n_with_variance"],
        "stat": res["stat"],
        "k": res["k"],
    }


def _show(label, res):
    print(f"  [{label}] median within-rho={res['aggregate']:.4f}  "
          f"floor(p95)={res['floor']:.4f}  p={res['p_value']:.4f}  "
          f"(pooled rho={res['pooled_rho']:.4f}, confounded)  -> {res['verdict']}")


def run_e12(R_perm=R_PERM):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    tasks = {t: tuple(load_livebench_task(t)) for t in LIVEBENCH_BINARY_TASKS}
    for t, (R, m, i) in tasks.items():
        tasks[t] = (R.astype(float), m, i)

    summary = {"config": {"R_perm": R_perm, "min_same_sign": MIN_SAME_SIGN,
                          "seed": SEED, "gate": "within_contest_median"}}

    # ---- per-task primary gate (d = +1) --------------------------------- #
    pooled_pos = []
    for task, (R, models, _items) in tasks.items():
        contests = harvest_contests(R, models, sign=1)
        pooled_pos.extend(contests)
        print(f"\n{'='*70}\n  {task}: {len(contests)} qualifying contests "
              f"({R.shape[0]} models x {R.shape[1]} items)\n{'='*70}")
        res = run_calibration(contests, sign=1, R_perm=R_perm, seed=SEED)
        summary[task] = _round_gate(res)
        _show(task, res)

    # ---- pooled-across-all primary gate (the headline) ------------------ #
    print(f"\n{'#'*70}\n  ALL TASKS (primary, d=+1): {len(pooled_pos)} contests\n{'#'*70}")
    pooled_res = run_calibration(pooled_pos, sign=1, R_perm=R_perm, seed=SEED)
    summary["pooled_primary"] = _round_gate(pooled_res)
    _show("all/d=+1", pooled_res)

    # ---- secondary: d = -1 side, reported separately -------------------- #
    pooled_neg = []
    for task, (R, models, _items) in tasks.items():
        pooled_neg.extend(harvest_contests(R, models, sign=-1))
    print(f"\n{'#'*70}\n  ALL TASKS (secondary, d=-1): {len(pooled_neg)} contests\n{'#'*70}")
    neg_res = run_calibration(pooled_neg, sign=-1, R_perm=R_perm, seed=SEED)
    summary["pooled_negative"] = _round_gate(neg_res)
    _show("all/d=-1", neg_res)

    # ---- k-sensitivity on the pooled primary set ----------------------- #
    print(f"\n{'#'*70}\n  k-SENSITIVITY (pooled primary)\n{'#'*70}")
    ksens = k_sensitivity_report(pooled_pos, sign=1, R_perm=R_perm, seed=SEED)
    summary["k_sensitivity"] = {name: _round_gate(r) for name, r in ksens.items()}
    for name, r in ksens.items():
        print(f"  {name:9s} k={r['k']:<3d} median within-rho={r['aggregate']:.4f}  "
              f"p={r['p_value']:.4f} -> {r['verdict']}")

    # ---- SECONDARY target: info-VF phi_i vs pivotality (d=+1) ----------- #
    # Downstream pipeline check, not headline (phi is built to surface discrimination).
    print(f"\n{'#'*70}\n  SECONDARY: info-VF phi vs pivotality (d=+1)\n{'#'*70}")
    phi_res = run_calibration(pooled_pos, sign=1, R_perm=R_perm, seed=SEED,
                              weight="info_phi")
    summary["pooled_primary_info_phi"] = _round_gate(phi_res)
    _show("all/d=+1 [phi]", phi_res)

    out = RESULTS_DIR / "e12_external_validation.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved {out}")
    return summary


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="fast smoke run with small R_perm")
    args = ap.parse_args()
    run_e12(R_perm=200 if args.quick else R_PERM)
