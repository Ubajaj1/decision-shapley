"""E17 — Real-data per-item attribution showcase with the confidence VF.

Takes the LiveCodeBench (LCB_generation) #1-vs-#2 frontier contest
(Gemini 2.5 Pro vs QwQ-32B, margin 1) and contrasts three value functions
on the SAME contest:

  * flip indicator  -> Shapley-Shubik: all same-sign items get one value
                       (the degeneracy; no per-item attribution)
  * confidence VF   -> correlation-aware credit-splitting (primary method)
  * information VF   -> Fisher/discrimination-weighted attribution

Demonstrates on real frontier data that the confidence VF breaks the
Shapley-Shubik symmetry: the least-discriminating decisive item, which is
also correlated with the other decisive items, earns markedly less credit.

Outputs:
  results/e17_confidence_showcase.json
  figures/e17_confidence_showcase.png
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from livebench_io import load_livebench_task
from decision import top_two
from shapley import (
    mc_shapley_flip,
    mc_shapley_confidence,
    mc_shapley_information,
    compute_item_weights,
    compute_active_correlation,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"

TASK = "LCB_generation"
T = 20_000
SEED = 42


def run_e17():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    R, models, items = load_livebench_task(TASK)
    c = top_two(R, models)
    d = c["d"]
    m1, m2 = c["m1_idx"], c["m2_idx"]

    weights = compute_item_weights(R, m1, m2)
    a = weights["a"]

    flip = mc_shapley_flip(d, T=T, seed=SEED)
    conf = mc_shapley_confidence(d, R, m1, m2, T=T, seed=SEED)
    info = mc_shapley_information(d, R, m1, m2, T=T, seed=SEED)

    C, active = compute_active_correlation(R, d)
    active = np.asarray(active)

    # Mean absolute off-diagonal correlation of each active item with the
    # OTHER same-sign active items (a simple "redundancy" summary).
    def same_sign_mean_corr(pos_in_active):
        sign = d[active[pos_in_active]]
        same = [j for j in range(len(active))
                if j != pos_in_active and d[active[j]] == sign]
        if not same:
            return 0.0
        return float(np.mean([abs(C[pos_in_active, j]) for j in same]))

    rows = []
    for pos, idx in enumerate(active):
        rows.append({
            "item_index": int(idx),
            "d": int(d[idx]),
            "discrimination_a": round(float(a[idx]), 3),
            "redundancy_mean_corr_same_sign": round(same_sign_mean_corr(pos), 3),
            "phi_flip": round(float(flip["phi"][idx]), 4),
            "phi_confidence": round(float(conf["phi"][idx]), 4),
            "phi_information": round(float(info["phi"][idx]), 4),
        })

    # Sort: decisive (d=+1) by confidence desc, then counter (d=-1) by confidence asc
    decisive = sorted([r for r in rows if r["d"] == 1],
                      key=lambda r: -r["phi_confidence"])
    counter = sorted([r for r in rows if r["d"] == -1],
                     key=lambda r: r["phi_confidence"])
    ordered = decisive + counter

    out = {
        "task": f"livebench/{TASK}",
        "contest": {
            "m1": c["m1_id"], "m2": c["m2_id"],
            "m1_score": round(c["m1_score"], 4),
            "m2_score": round(c["m2_score"], 4),
            "margin": c["margin"],
            "n_plus": c["n_plus"], "n_minus": c["n_minus"],
            "n_zero": c["n_zero"], "n_items": int(len(d)),
        },
        "value_functions": {
            "flip": {"efficiency_sum": round(float(flip["efficiency_sum"]), 4),
                     "max_se": round(float(flip["max_se"]), 5)},
            "confidence": {"v_all": round(float(conf["v_all"]), 4),
                           "max_se": round(float(conf["max_se"]), 5)},
            "information": {"v_all": round(float(info["v_all"]), 4),
                            "max_se": round(float(info["max_se"]), 5)},
        },
        "items": ordered,
    }

    json_path = RESULTS_DIR / "e17_confidence_showcase.json"
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)

    # ---- console summary ----
    print(f"Contest: {c['m1_id']} vs {c['m2_id']}  (margin {c['margin']}, "
          f"{c['n_plus']}+ / {c['n_minus']}- / {c['n_zero']}0, N={len(d)})")
    print(f"{'item':>5} {'d':>3} {'a':>6} {'corr':>6} "
          f"{'flip':>8} {'conf':>8} {'info':>8}")
    for r in ordered:
        print(f"{r['item_index']:>5} {r['d']:>+3} {r['discrimination_a']:>6.2f} "
              f"{r['redundancy_mean_corr_same_sign']:>6.2f} "
              f"{r['phi_flip']:>8.4f} {r['phi_confidence']:>8.4f} "
              f"{r['phi_information']:>8.4f}")
    print(f"\nSaved {json_path}")

    _plot(ordered, c, FIGURES_DIR / "e17_confidence_showcase.png")


def _plot(ordered, contest, path):
    labels = []
    for i, r in enumerate(ordered):
        tag = "D" if r["d"] == 1 else "C"
        labels.append(f"{tag}{i+1}\n(a={r['discrimination_a']:.2f})")

    flip = [r["phi_flip"] for r in ordered]
    conf = [r["phi_confidence"] for r in ordered]
    info = [r["phi_information"] for r in ordered]

    x = np.arange(len(ordered))
    w = 0.27
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.bar(x - w, flip, w, label="flip indicator (Shapley–Shubik)",
           color="#bbbbbb")
    ax.bar(x, conf, w, label="confidence VF (correlation-aware)",
           color="#2c7fb8")
    ax.bar(x + w, info, w, label="information VF (Fisher-weighted)",
           color="#7fcdbb")

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Decision-Shapley value  $\\phi_i$", fontsize=11)
    ax.set_title("Per-item attribution: Gemini 2.5 Pro vs. QwQ-32B\n"
                 "(LiveCodeBench #1 vs. #2, margin 1)",
                 fontsize=10)

    # Headroom so the legend clears the tallest bars instead of sitting on them.
    all_vals = flip + conf + info
    ymin, ymax = min(all_vals), max(all_vals)
    pad = 0.12 * (ymax - ymin)
    ax.set_ylim(ymin - pad, ymax + 2.2 * pad)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.95)
    plt.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Figure: {path}")


if __name__ == "__main__":
    run_e17()
