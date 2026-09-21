"""E4 — Shapley-vs-LOO on real data.

Quantify where flip-Shapley and LOO disagree, especially near duplicate
clusters. Show concrete cases where LOO under-counts decisive items vs Shapley.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import json
import numpy as np
import matplotlib.pyplot as plt
from data_io import load_benchmark, BENCHMARKS
from decision import top_two
from shapley import mc_shapley_flip
from loo import loo_flips

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"

T = 20_000
SEED = 42


def run_e4():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    results = {}

    for bench in BENCHMARKS:
        print(f"\n{'='*60}")
        print(f"  {bench.upper()}")
        print(f"{'='*60}")

        R, models, items = load_benchmark(bench)
        contest = top_two(R, models)
        d = contest["d"]
        margin = contest["margin"]

        # Flip-Shapley
        shap = mc_shapley_flip(d, T=T, seed=SEED)
        phi = shap["phi"]

        # LOO analysis
        loo = loo_flips(d)

        # LOO importance: for binary d, LOO "value" of item i is just d_i
        # (removing it changes margin by d_i). LOO can only cause flip when margin=1.
        # For comparison with Shapley, define LOO_value_i = d_i / margin
        # (fractional contribution to the margin).
        loo_value = d.astype(np.float64) / margin if margin > 0 else np.zeros(len(d))

        # Key comparison: rank correlation between Shapley and LOO on active items
        active = d != 0
        phi_active = phi[active]
        loo_active = loo_value[active]
        d_active = d[active]

        # Rank correlation (Spearman)
        from scipy.stats import spearmanr, kendalltau
        if len(phi_active) > 2:
            spearman_r, spearman_p = spearmanr(phi_active, loo_active)
            kendall_t, kendall_p = kendalltau(phi_active, loo_active)
        else:
            spearman_r = spearman_p = kendall_t = kendall_p = float("nan")

        # Where do they disagree? Items where Shapley rank differs from LOO rank
        shap_rank = np.argsort(np.argsort(-phi_active))
        loo_rank = np.argsort(np.argsort(-loo_active))
        rank_diff = np.abs(shap_rank - loo_rank)
        max_rank_diff = int(rank_diff.max()) if len(rank_diff) > 0 else 0
        mean_rank_diff = float(rank_diff.mean()) if len(rank_diff) > 0 else 0.0

        # Key structural comparison:
        # With additive value function, all d=+1 items get identical Shapley = 1
        # and all d=-1 items get identical Shapley = -1. The flip indicator
        # differentiates them by position-sensitivity near the decision boundary.
        plus_phi = phi[d == 1]
        minus_phi = phi[d == -1]
        plus_spread = float(plus_phi.std()) if len(plus_phi) > 1 else 0.0
        minus_spread = float(minus_phi.std()) if len(minus_phi) > 1 else 0.0

        print(f"  Margin: {margin}, Active: {active.sum()}")
        print(f"  Spearman r: {spearman_r:.4f} (p={spearman_p:.2e})")
        print(f"  Kendall τ:  {kendall_t:.4f} (p={kendall_p:.2e})")
        print(f"  Mean rank diff: {mean_rank_diff:.2f}, Max: {max_rank_diff}")
        print(f"  LOO-vulnerable: {loo['loo_vulnerable']}")
        print(f"  φ spread among d=+1 items: σ={plus_spread:.6f} (LOO: all identical)")
        print(f"  φ spread among d=-1 items: σ={minus_spread:.6f} (LOO: all identical)")

        results[bench] = {
            "margin": margin,
            "n_active": int(active.sum()),
            "n_plus": contest["n_plus"],
            "n_minus": contest["n_minus"],
            "spearman_r": float(spearman_r),
            "spearman_p": float(spearman_p),
            "kendall_tau": float(kendall_t),
            "kendall_p": float(kendall_p),
            "mean_rank_diff": mean_rank_diff,
            "max_rank_diff": max_rank_diff,
            "loo_vulnerable": loo["loo_vulnerable"],
            "plus_phi_spread": plus_spread,
            "minus_phi_spread": minus_spread,
            "plus_phi_range": [float(plus_phi.min()), float(plus_phi.max())] if len(plus_phi) > 0 else [],
            "minus_phi_range": [float(minus_phi.min()), float(minus_phi.max())] if len(minus_phi) > 0 else [],
        }

    # Save
    report_path = RESULTS_DIR / "e4_shapley_vs_loo.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {report_path}")

    # Summary
    print(f"\n{'='*100}")
    print(f"{'Bench':<12} {'Margin':>7} {'d+':>5} {'d-':>5} {'Spear':>7} {'Kendall':>8} "
          f"{'MeanΔR':>7} {'σ(φ+)':>9} {'σ(φ-)':>9}")
    print(f"{'='*100}")
    for bench in BENCHMARKS:
        r = results[bench]
        print(f"{bench:<12} {r['margin']:>7} {r['n_plus']:>5} {r['n_minus']:>5} "
              f"{r['spearman_r']:>7.4f} {r['kendall_tau']:>8.4f} "
              f"{r['mean_rank_diff']:>7.2f} {r['plus_phi_spread']:>9.6f} {r['minus_phi_spread']:>9.6f}")

    # Figure: scatter of Shapley vs LOO for each benchmark
    _plot_shapley_vs_loo_scatter(results)
    _plot_phi_spread(results)

    return results


def _plot_shapley_vs_loo_scatter(results: dict):
    """Show that LOO assigns identical value to all d=+1 items,
    while Shapley differentiates them."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.ravel()

    for i, bench in enumerate(BENCHMARKS):
        ax = axes[i]
        R, models, items = load_benchmark(bench)
        contest = top_two(R, models)
        d = contest["d"]
        margin = contest["margin"]

        shap = mc_shapley_flip(d, T=T, seed=SEED)
        phi = shap["phi"]

        active = d != 0
        phi_a = phi[active]
        d_a = d[active].astype(float)

        colors = ["#4C72B0" if v > 0 else "#C44E52" for v in d_a]
        ax.scatter(d_a, phi_a, c=colors, alpha=0.6, s=20, edgecolors="none")
        ax.set_xlabel("d_i (LOO value ∝ this)", fontsize=9)
        ax.set_ylabel("φ_i (flip-Shapley)", fontsize=9)
        ax.set_title(f"{bench.upper()} (margin={margin})", fontsize=10)
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.axvline(0, color="gray", linewidth=0.5)

    plt.suptitle("Flip-Shapley φ vs LOO Value (d_i) — LOO Can't Differentiate Within a Group",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    path = FIGURES_DIR / "e4_shapley_vs_loo.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to {path}")


def _plot_phi_spread(results: dict):
    """Bar chart showing the spread (σ) of φ among d=+1 and d=-1 items."""
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(BENCHMARKS))
    w = 0.35

    plus_spreads = [results[b]["plus_phi_spread"] for b in BENCHMARKS]
    minus_spreads = [results[b]["minus_phi_spread"] for b in BENCHMARKS]

    ax.bar(x - w/2, plus_spreads, w, label="σ(φ) among d=+1 items", color="#4C72B0")
    ax.bar(x + w/2, minus_spreads, w, label="σ(φ) among d=-1 items", color="#C44E52")

    ax.set_xticks(x)
    ax.set_xticklabels([b.upper() for b in BENCHMARKS], fontsize=10)
    ax.set_ylabel("Standard deviation of φ", fontsize=11)
    ax.set_title("Shapley Differentiates Items That LOO Treats Identically", fontsize=12)
    ax.legend(fontsize=9)

    for i, (ps, ms) in enumerate(zip(plus_spreads, minus_spreads)):
        if ps > 0:
            ax.text(i - w/2, ps + max(plus_spreads)*0.02, f"{ps:.4f}",
                    ha="center", fontsize=7, rotation=90)

    plt.tight_layout()
    path = FIGURES_DIR / "e4_phi_spread.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Figure saved to {path}")


if __name__ == "__main__":
    run_e4()
