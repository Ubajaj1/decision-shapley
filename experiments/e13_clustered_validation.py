"""E13 -- Task-clustered external-validation gate (Problem 1 fix: pseudoreplication).

The e12 gate pooled every adjacent-rank contest and shuffled each one independently,
treating the ~319 contests as independent units. They are not: within a LiveBench task
every contest is built from the same item set, so each task is one correlated block
(effective n ~ #tasks, not #contests). That made the e12 permutation null too tight and
its p-value anti-conservative.

E13 corrects this with a TIERED design, each tier judged by the cluster-respecting
synchronized null (one item permutation per task per replicate):

  TIER 1 (PRIMARY, the headline)  : top-of-leaderboard contests only -- this is the
      claim the paper actually makes ("is the leaderboard WINNER trustworthy?").
      Reported for #1-vs-#2 only and for the top-5 ranks. Small n; read as suggestive.
  TIER 2 (SECONDARY, exploratory) : all adjacent-rank contests, reframed as "across the
      full rank ladder" -- NOT the headline.

For each tier we report the clustered gate, the honest per-task table (the n=#tasks
view), and a task-level cluster-bootstrap CI. For Tier 2 we also print the OLD e12-style
independent-shuffle p side by side with the corrected clustered p, to show how much the
pseudoreplication inflated significance.

Caveat carried forward (Problem 2): clearing the clustered floor rejects only "zero
association"; it does NOT separate genuine discrimination signal from the mechanical
coupling of J and pivotality (both are functions of the same response matrix).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import json
import numpy as np

from livebench_io import load_livebench_task, LIVEBENCH_BINARY_TASKS
from decision import top_two
from calibration import (
    contest_record,
    contest_sample,
    clustered_gate,
    per_task_table,
    cluster_bootstrap_ci,
    run_calibration,          # old independent-shuffle gate, for the comparison
    MIN_SAME_SIGN,
    R_PERM,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
SEED = 42


def harvest_records(R, models, sign=1, max_rank=None):
    """Qualifying contest_records for a task. max_rank=None -> all adjacent ranks;
    max_rank=K -> only the top-K leading contests (#1v2 ... #KvK+1)."""
    records, contests = [], []
    last_rank = R.shape[0] - 1 if max_rank is None else min(max_rank, R.shape[0] - 1)
    for rank1 in range(1, last_rank + 1):
        try:
            c = top_two(R, models, rank1=rank1)
        except ValueError:
            break
        rec = contest_record(R, c["m1_idx"], c["m2_idx"], sign=sign)
        if rec is not None:
            records.append(rec)
            contests.append((R, c["m1_idx"], c["m2_idx"]))
    return records, contests


def _round_clustered(res, ci=None):
    out = {
        "aggregate_within_rho": _r(res["aggregate"]),
        "clustered_floor_p95": _r(res["floor"]),
        "clustered_p_value": _r(res["p_value"]),
        "verdict": res["verdict"],
        "n_contests": res["n_contests"],
        "n_with_variance": res["n_with_variance"],
        "n_tasks": res["n_tasks"],
        "pooled_rho_confounded": _r(res["pooled_rho"]),
    }
    if ci is not None:
        out["cluster_bootstrap_ci"] = [_r(ci["ci_lo"]), _r(ci["ci_hi"])]
    return out


def _r(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 4)


def _round_table(table):
    return {t: {"aggregate": _r(v["aggregate"]), "n_contests": v["n_contests"],
                "n_with_variance": v["n_with_variance"]} for t, v in table.items()}


def run_tier(label, records_by_task, R_perm, with_ci=True):
    gate = clustered_gate(records_by_task, R_perm=R_perm, seed=SEED)
    table = per_task_table(records_by_task)
    ci = cluster_bootstrap_ci(records_by_task, B=2000, seed=SEED) if with_ci else None
    block = _round_clustered(gate, ci)
    block["per_task"] = _round_table(table)
    print(f"\n{'='*72}\n  {label}\n{'='*72}")
    print(f"  contests={gate['n_contests']} across {gate['n_tasks']} tasks "
          f"(n_with_variance={gate['n_with_variance']})")
    print(f"  median within-rho = {_fmt(gate['aggregate'])}   "
          f"clustered floor(p95) = {_fmt(gate['floor'])}   "
          f"clustered p = {_fmt(gate['p_value'])}  -> {gate['verdict']}")
    if ci is not None:
        print(f"  cluster-bootstrap 95% CI (resampling tasks): "
              f"[{_fmt(ci['ci_lo'])}, {_fmt(ci['ci_hi'])}]")
    print("  per-task (the honest n=#tasks view):")
    for t, v in table.items():
        print(f"    {t:18s} rho={_fmt(v['aggregate'])}  "
              f"(contests={v['n_contests']}, with_var={v['n_with_variance']})")
    return block


def _fmt(x):
    return "nan" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.4f}"


def run_e13(R_perm=R_PERM):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tasks = {}
    for t in LIVEBENCH_BINARY_TASKS:
        R, models, items = load_livebench_task(t)
        tasks[t] = (R.astype(float), models)

    summary = {"config": {"R_perm": R_perm, "min_same_sign": MIN_SAME_SIGN,
                          "seed": SEED, "gate": "task_clustered_synchronized_null"}}

    # ---- TIER 1 PRIMARY: top-of-leaderboard ----------------------------- #
    top1, top5, allranks = {}, {}, {}
    contests_all = {}                      # for the old-gate comparison (flat per task)
    for t, (R, models) in tasks.items():
        r1, _ = harvest_records(R, models, sign=1, max_rank=1)
        r5, _ = harvest_records(R, models, sign=1, max_rank=5)
        ra, ca = harvest_records(R, models, sign=1, max_rank=None)
        if r1:
            top1[t] = r1
        if r5:
            top5[t] = r5
        if ra:
            allranks[t] = ra
            contests_all[t] = ca

    print("\n" + "#" * 72 + "\n  TIER 1 -- PRIMARY (top-of-leaderboard, d=+1)\n" + "#" * 72)
    summary["tier1_top1"] = run_tier("Tier 1a: #1 vs #2 only", top1, R_perm)
    summary["tier1_top5"] = run_tier("Tier 1b: top-5 ranks", top5, R_perm)

    # ---- TIER 2 SECONDARY: all ranks ------------------------------------ #
    print("\n" + "#" * 72 + "\n  TIER 2 -- SECONDARY/EXPLORATORY (all ranks, d=+1)\n" + "#" * 72)
    summary["tier2_all_ranks"] = run_tier("Tier 2: all adjacent ranks", allranks, R_perm)

    # ---- OLD (e12) independent-shuffle gate on the SAME all-ranks set ---- #
    flat_contests = [c for cs in contests_all.values() for c in cs]
    old = run_calibration(flat_contests, sign=1, R_perm=R_perm, seed=SEED)
    print("\n" + "#" * 72 + "\n  COMPARISON: old independent shuffle vs corrected clustering "
          "(all ranks)\n" + "#" * 72)
    print(f"  OLD (e12, treats {len(flat_contests)} contests as independent):  "
          f"p = {_fmt(old['p_value'])}  floor = {_fmt(old['floor'])}  -> {old['verdict']}")
    cg = summary["tier2_all_ranks"]
    print(f"  NEW (e13, clusters by task, effective n ~ {cg['n_tasks']}):       "
          f"p = {_fmt(cg['clustered_p_value'])}  floor = {_fmt(cg['clustered_floor_p95'])}  "
          f"-> {cg['verdict']}")
    summary["comparison_all_ranks"] = {
        "old_independent_p": _r(old["p_value"]),
        "old_independent_floor": _r(old["floor"]),
        "old_independent_verdict": old["verdict"],
        "old_n_contests_treated_independent": len(flat_contests),
        "new_clustered_p": cg["clustered_p_value"],
        "new_clustered_floor": cg["clustered_floor_p95"],
        "new_clustered_verdict": cg["verdict"],
        "new_effective_n_tasks": cg["n_tasks"],
    }

    out = RESULTS_DIR / "e13_clustered_validation.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved {out}")
    return summary


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="fast smoke run (small R_perm)")
    args = ap.parse_args()
    run_e13(R_perm=200 if args.quick else R_PERM)
