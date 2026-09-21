"""E3 — Item-Shapley distributions + taxonomy.

Per benchmark, compute φ_i, classify items, report counts and the decisive set.
Figures: Shapley concentration (Lorenz curve / sorted φ).
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
from taxonomy import classify_items
from concentration import concentration_metrics, gini

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"

T = 20_000
SEED = 42


def run_e3():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {}
    all_phi = {}

    for bench in BENCHMARKS:
        print(f"\n{'='*60}")
        print(f"  {bench.upper()}")
        print(f"{'='*60}")

        R, models, items = load_benchmark(bench)
        contest = top_two(R, models)
        d = contest["d"]

        print(f"  #1: {contest['m1_id']}")
        print(f"  #2: {contest['m2_id']}")
        print(f"  Margin: {contest['margin']}, n_active: {(d!=0).sum()}")

        # Compute Shapley
        print(f"  Computing MC Shapley (T={T})...")
        shap = mc_shapley_flip(d, T=T, seed=SEED)
        phi = shap["phi"]

        print(f"  Efficiency: Σφ = {shap['efficiency_sum']:.6f} (expect 0.5)")
        print(f"  Max SE: {shap['max_se']:.6f}")

        # Taxonomy
        tax = classify_items(phi, d)
        print(f"  Taxonomy: {tax['n_decisive']} decisive, "
              f"{tax['n_counter']} counter-evidential, "
              f"{tax['n_redundant']} redundant")

        # Concentration
        conc = concentration_metrics(phi)
        print(f"  Gini (positive φ): {conc['gini_positive']:.4f}")
        print(f"  Top-k for 50% mass: {conc['top_k_50']}")
        print(f"  Top-k for 80% mass: {conc['top_k_80']}")
        print(f"  Top-k for 90% mass: {conc['top_k_90']}")
        print(f"  Max φ: {conc['max_phi']:.6f}, Min φ: {conc['min_phi']:.6f}")

        all_phi[bench] = phi

        all_results[bench] = {
            "m1": contest["m1_id"],
            "m2": contest["m2_id"],
            "margin": contest["margin"],
            "n_items": len(d),
            "n_active": int((d != 0).sum()),
            "n_plus": contest["n_plus"],
            "n_minus": contest["n_minus"],
            "efficiency_sum": shap["efficiency_sum"],
            "efficiency_error": shap["efficiency_error"],
            "max_se": shap["max_se"],
            "T": T,
            "n_decisive": tax["n_decisive"],
            "n_counter": tax["n_counter"],
            "n_redundant": tax["n_redundant"],
            "decisive_phi_sum": tax["decisive_phi_sum"],
            "counter_phi_sum": tax["counter_phi_sum"],
            "gini_positive": conc["gini_positive"],
            "top_k_50": conc["top_k_50"],
            "top_k_80": conc["top_k_80"],
            "top_k_90": conc["top_k_90"],
            "n_positive_phi": conc["n_positive"],
            "n_negative_phi": conc["n_negative"],
            "max_phi": conc["max_phi"],
            "min_phi": conc["min_phi"],
        }

    # Save results
    report_path = RESULTS_DIR / "e3_shapley_taxonomy.json"
    with open(report_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {report_path}")

    # Save phi arrays
    phi_path = RESULTS_DIR / "e3_phi_arrays.npz"
    np.savez_compressed(phi_path, **{b: all_phi[b] for b in BENCHMARKS})
    print(f"Phi arrays saved to {phi_path}")

    # Summary table
    _print_summary(all_results)

    # Figures
    _plot_lorenz_curves(all_phi)
    _plot_sorted_phi(all_phi)
    _plot_taxonomy_bars(all_results)

    return all_results


def _print_summary(results: dict):
    print(f"\n{'='*110}")
    print(f"{'Bench':<12} {'Margin':>7} {'Active':>7} {'Decis':>6} {'Countr':>7} {'Redund':>7} "
          f"{'Gini':>6} {'k50':>5} {'k80':>5} {'k90':>5} {'MaxSE':>8}")
    print(f"{'='*110}")
    for bench in BENCHMARKS:
        r = results[bench]
        print(f"{bench:<12} {r['margin']:>7} {r['n_active']:>7} {r['n_decisive']:>6} "
              f"{r['n_counter']:>7} {r['n_redundant']:>7} "
              f"{r['gini_positive']:>6.3f} {r['top_k_50']:>5} {r['top_k_80']:>5} {r['top_k_90']:>5} "
              f"{r['max_se']:>8.5f}")


def _plot_lorenz_curves(all_phi: dict):
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = plt.cm.Set2(np.linspace(0, 1, len(all_phi)))

    for (bench, phi), color in zip(all_phi.items(), colors):
        pos = np.sort(phi[phi > 0])
        if len(pos) == 0:
            continue
        cumshare = np.cumsum(pos) / pos.sum()
        x = np.linspace(0, 1, len(cumshare))
        g = gini(pos)
        ax.plot(x, cumshare, label=f"{bench.upper()} (Gini={g:.3f})", color=color, linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5, label="Perfect equality")
    ax.set_xlabel("Fraction of positive-φ items (sorted ascending)", fontsize=11)
    ax.set_ylabel("Cumulative share of positive Shapley mass", fontsize=11)
    ax.set_title("Lorenz Curves of Shapley Concentration", fontsize=12)
    ax.legend(fontsize=9, loc="upper left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    path = FIGURES_DIR / "e3_lorenz_curves.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Figure saved to {path}")


def _plot_sorted_phi(all_phi: dict):
    n_bench = len(all_phi)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.ravel()

    for i, (bench, phi) in enumerate(all_phi.items()):
        ax = axes[i]
        active_phi = phi[phi != 0]
        sorted_phi = np.sort(active_phi)[::-1]

        colors = ["#4C72B0" if v > 0 else "#C44E52" for v in sorted_phi]
        ax.bar(range(len(sorted_phi)), sorted_phi, color=colors, width=1.0, edgecolor="none")
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.set_title(f"{bench.upper()} (n_active={len(active_phi)})", fontsize=10)
        ax.set_xlabel("Item rank", fontsize=9)
        ax.set_ylabel("φ_i", fontsize=9)

    plt.suptitle("Sorted Shapley Values (active items only)", fontsize=13, y=1.01)
    plt.tight_layout()
    path = FIGURES_DIR / "e3_sorted_phi.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to {path}")


def _plot_taxonomy_bars(results: dict):
    fig, ax = plt.subplots(figsize=(8, 4))
    benchmarks = list(results.keys())
    x = np.arange(len(benchmarks))
    w = 0.6

    decisive = [results[b]["n_decisive"] for b in benchmarks]
    counter = [results[b]["n_counter"] for b in benchmarks]
    redundant_active = [results[b]["n_active"] - results[b]["n_decisive"] - results[b]["n_counter"]
                        for b in benchmarks]

    ax.bar(x, decisive, w, label="Decisive (φ > 0)", color="#4C72B0")
    ax.bar(x, redundant_active, w, bottom=decisive, label="Redundant (φ ≈ 0, active)", color="#CCCCCC")
    ax.bar(x, counter, w, bottom=[d + r for d, r in zip(decisive, redundant_active)],
           label="Counter-evidential (φ < 0)", color="#C44E52")

    ax.set_xticks(x)
    ax.set_xticklabels([b.upper() for b in benchmarks], fontsize=10)
    ax.set_ylabel("Number of active items", fontsize=11)
    ax.set_title("Item Taxonomy by Shapley Attribution", fontsize=12)
    ax.legend(fontsize=9)
    plt.tight_layout()
    path = FIGURES_DIR / "e3_taxonomy_bars.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Figure saved to {path}")


if __name__ == "__main__":
    run_e3()
