"""E19 — Per-item bootstrap CIs for the LiveCodeBench showcase (Table II).

Addresses the reviewer's request: report bootstrap CIs on each phi_conf in the
seven-item attribution table, and state whether the decisive-item ordering
D1 > D2 > D3 > D4 survives resampling.

Design (model-population bootstrap):
  The per-item confidence attribution depends on two quantities estimated from
  the M models -- each item's discrimination a_i (point-biserial with total
  score) and the active-item correlation matrix C. The realized differential d
  is fixed (it depends only on the two contestants). We therefore resample the
  model population with replacement, ALWAYS retaining the two contestants
  (m1, m2) so the contest itself is unchanged, re-estimate a_i and C, and
  recompute phi_conf for the seven (fixed) active items. Percentile intervals
  over B_BOOT resamples give each item's CI; the fraction of resamples that
  preserve the decisive ordering measures ranking stability.

This isolates input-estimation noise (a_i, C from a finite model sample); the
MC-estimator noise is separately negligible here (max SE < 0.002 at T=20000,
far below the phi gaps), so a stable ordering under this bootstrap means the
ranking is signal, not noise.

Outputs:
  results/e19_showcase_cis.json
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from livebench_io import load_livebench_task
from decision import top_two
from shapley import mc_shapley_confidence, compute_item_weights, compute_active_correlation

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

TASK = "LCB_generation"
T_POINT = 20_000     # high-T point estimate (matches Table II)
T_BOOT = 6_000       # per-resample MC budget (SE small vs bootstrap spread)
B_BOOT = 2_000       # model-population resamples
SEED = 42


def _same_sign_mean_corr(C, active, d, pos):
    sign = d[active[pos]]
    same = [j for j in range(len(active))
            if j != pos and d[active[j]] == sign]
    if not same:
        return 0.0
    return float(np.mean([abs(C[pos, j]) for j in same]))


def run_e19():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    R, models, items = load_livebench_task(TASK)
    c = top_two(R, models)
    d = c["d"]
    m1, m2 = c["m1_idx"], c["m2_idx"]
    M = R.shape[0]

    weights = compute_item_weights(R, m1, m2)
    a = weights["a"]
    C, active = compute_active_correlation(R, d)
    active = np.asarray(active)

    # Point estimate at high T, then assign labels D1..D4 (decisive, phi desc)
    # and C5..C7 (counter, phi asc) -- the Table II ordering.
    conf_pt = mc_shapley_confidence(d, R, m1, m2, T=T_POINT, seed=SEED)
    phi_pt = conf_pt["phi"]

    decisive = sorted([p for p in range(len(active)) if d[active[p]] == 1],
                      key=lambda p: -phi_pt[active[p]])
    counter = sorted([p for p in range(len(active)) if d[active[p]] == -1],
                     key=lambda p: phi_pt[active[p]])
    ordered_pos = decisive + counter
    labels = ([f"D{i+1}" for i in range(len(decisive))]
              + [f"C{len(decisive)+i+1}" for i in range(len(counter))])

    # ---- model-population bootstrap ----
    others = np.array([i for i in range(M) if i not in (m1, m2)])
    n_active = len(active)
    boot_phi = np.full((B_BOOT, n_active), np.nan)  # phi for each active item

    for b in range(B_BOOT):
        # Resample the non-contestant rows with replacement; keep m1, m2 at 0,1.
        resampled = rng.choice(others, size=len(others), replace=True)
        rows = np.concatenate(([m1, m2], resampled))
        R_b = R[rows]
        # d is unchanged (rows 0,1 are m1,m2). Recompute a_i, C from R_b inside.
        try:
            res = mc_shapley_confidence(d, R_b, 0, 1, T=T_BOOT, seed=int(rng.integers(1 << 31)))
            phi_b = res["phi"]
            boot_phi[b] = [phi_b[active[p]] for p in range(n_active)]
        except Exception:
            continue  # degenerate resample (e.g. zero-variance item); skip

    valid = ~np.isnan(boot_phi[:, 0])
    boot_phi = boot_phi[valid]
    n_valid = int(boot_phi.shape[0])

    # ---- per-item summaries ----
    rows_out = []
    for label, pos in zip(labels, ordered_pos):
        idx = active[pos]
        col = boot_phi[:, pos]
        rows_out.append({
            "label": label,
            "item_index": int(idx),
            "d": int(d[idx]),
            "discrimination_a": round(float(a[idx]), 3),
            "redundancy_mean_corr_same_sign": round(_same_sign_mean_corr(C, active, d, pos), 3),
            "phi_point": round(float(phi_pt[idx]), 4),
            "phi_boot_mean": round(float(np.mean(col)), 4),
            "ci_lo": round(float(np.percentile(col, 2.5)), 4),
            "ci_hi": round(float(np.percentile(col, 97.5)), 4),
            "boot_sd": round(float(np.std(col)), 4),
        })

    # ---- ordering stability (decisive side) ----
    dec_cols = [boot_phi[:, p] for p in decisive]  # in D1..D4 order
    dec_mat = np.column_stack(dec_cols)
    strict_monotone = np.all(np.diff(dec_mat, axis=1) < 0, axis=1)  # D1>D2>D3>D4
    endpoints = (dec_mat[:, 0] == dec_mat.max(axis=1)) & (dec_mat[:, -1] == dec_mat.min(axis=1))
    # D1 vs D4 pairwise separation (the crux of "ranking adds information")
    d1_gt_d4 = dec_mat[:, 0] > dec_mat[:, -1]

    out = {
        "task": f"livebench/{TASK}",
        "method": "model-population bootstrap (retain contestants), recompute a_i, C, phi_conf",
        "B_boot": B_BOOT,
        "n_valid": n_valid,
        "T_boot": T_BOOT,
        "T_point": T_POINT,
        "contest": {"m1": c["m1_id"], "m2": c["m2_id"], "margin": c["margin"],
                    "M_models": int(M), "n_items": int(len(d))},
        "items": rows_out,
        "ordering_stability": {
            "decisive_labels": [labels[ordered_pos.index(p)] for p in decisive],
            "strict_D1_D2_D3_D4_frac": round(float(np.mean(strict_monotone)), 4),
            "D1_top_and_D4_bottom_frac": round(float(np.mean(endpoints)), 4),
            "D1_gt_D4_frac": round(float(np.mean(d1_gt_d4)), 4),
        },
    }

    json_path = RESULTS_DIR / "e19_showcase_cis.json"
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)

    # ---- console summary ----
    print(f"Contest: {c['m1_id']} vs {c['m2_id']}  (margin {c['margin']}, "
          f"M={M} models, N={len(d)} items)")
    print(f"Valid resamples: {n_valid}/{B_BOOT}\n")
    print(f"{'item':>4} {'d':>3} {'a':>5} {'phi':>8} "
          f"{'boot_mu':>8} {'95% CI':>18} {'sd':>7}")
    for r in rows_out:
        print(f"{r['label']:>4} {r['d']:>+3} {r['discrimination_a']:>5.2f} "
              f"{r['phi_point']:>8.4f} {r['phi_boot_mean']:>8.4f} "
              f"[{r['ci_lo']:>7.4f},{r['ci_hi']:>7.4f}] {r['boot_sd']:>7.4f}")
    s = out["ordering_stability"]
    print(f"\nOrdering stability over {n_valid} resamples:")
    print(f"  strict D1>D2>D3>D4 : {s['strict_D1_D2_D3_D4_frac']:.1%}")
    print(f"  D1 top & D4 bottom : {s['D1_top_and_D4_bottom_frac']:.1%}")
    print(f"  D1 > D4            : {s['D1_gt_D4_frac']:.1%}")
    print(f"\nSaved {json_path}")


if __name__ == "__main__":
    run_e19()
