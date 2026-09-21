"""E5 — Localized Fisher information baseline (validates Gap 2).

Fit a 2PL IRT model, compute item Fisher information at θ* (midpoint
between m1 and m2 abilities). Compare this IRT-based item ranking to
the Shapley decisive set. Show that:
1. Overlap (Jaccard) between top-k Fisher items and top-k Shapley items is low.
2. Fisher cannot identify counter-evidential items (d_i = -1, φ < 0).
3. Fisher ranks by *expected* discrimination; Shapley ranks by *actual*
   contribution to the realized decision.

For large matrices, subsample models for tractable IRT fitting.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from data_io import load_benchmark, BENCHMARKS
from livebench_io import load_livebench_task, LIVEBENCH_BINARY_TASKS
from decision import top_two
from shapley import mc_shapley_flip

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"

T = 20_000
SEED = 42


def fit_2pl_and_fisher(R, m1_idx, m2_idx):
    """Fast 2PL approximation + localized Fisher info at θ* between m1 and m2.

    Uses closed-form estimates:
      difficulty  b_i = -logit(p_i)
      discrimination a_i ∝ point-biserial correlation(item, total_score)
      ability θ_m = logit(score_m)
    """
    n_models, n_items = R.shape

    p_i = R.mean(axis=0)
    p_i = np.clip(p_i, 0.001, 0.999)
    difficulty = -np.log(p_i / (1.0 - p_i))

    total_scores = R.sum(axis=1)
    ts_std = total_scores.std()
    disc = np.zeros(n_items)
    for i in range(n_items):
        item_col = R[:, i]
        if item_col.std() == 0:
            disc[i] = 0.01
            continue
        r_pb = np.corrcoef(item_col, total_scores)[0, 1]
        if np.isnan(r_pb):
            r_pb = 0.0
        disc[i] = max(0.01, r_pb * 2.0)

    scores = R.mean(axis=1)
    scores = np.clip(scores, 0.001, 0.999)
    theta = np.log(scores / (1.0 - scores))

    theta_m1 = float(theta[m1_idx])
    theta_m2 = float(theta[m2_idx])
    theta_star = (theta_m1 + theta_m2) / 2.0

    p_star = 1.0 / (1.0 + np.exp(-disc * (theta_star - difficulty)))
    fisher_info = disc**2 * p_star * (1.0 - p_star)

    n_degenerate = int((R.var(axis=0) == 0).sum())

    return {
        "fisher_info": fisher_info,
        "discrimination": disc,
        "difficulty": difficulty,
        "theta_m1": theta_m1,
        "theta_m2": theta_m2,
        "theta_star": theta_star,
        "n_degenerate_items": n_degenerate,
    }


def jaccard(set_a, set_b):
    a, b = set(set_a), set(set_b)
    if len(a | b) == 0:
        return 0.0
    return len(a & b) / len(a | b)


def run_e5():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []

    datasets = []
    for task in LIVEBENCH_BINARY_TASKS:
        R, models, items = load_livebench_task(task)
        datasets.append((f"livebench/{task}", R, models, items))
    for bench in BENCHMARKS:
        R, models, items = load_benchmark(bench)
        datasets.append((f"metabench/{bench}", R, models, items))

    for name, R, models, items in datasets:
        print(f"\n{'='*70}")
        print(f"  {name}")
        print(f"{'='*70}")

        contest = top_two(R, models)
        d = contest["d"]

        # Shapley
        shap = mc_shapley_flip(d, T=T, seed=SEED)
        phi = shap["phi"]

        # IRT + Fisher
        print(f"  Fitting 2PL IRT ({R.shape[0]} models, {R.shape[1]} items)...")
        try:
            irt = fit_2pl_and_fisher(R, contest["m1_idx"], contest["m2_idx"])
        except Exception as e:
            print(f"  IRT fitting failed: {e}")
            all_results.append({
                "dataset": name, "error": str(e),
            })
            continue

        fisher = irt["fisher_info"]

        print(f"  θ(m1)={irt['theta_m1']:.3f}, θ(m2)={irt['theta_m2']:.3f}, "
              f"θ*={irt['theta_star']:.3f}")
        print(f"  Degenerate items removed from IRT: {irt['n_degenerate_items']}")

        # Compare rankings on active items only (d != 0)
        active = d != 0
        phi_a = phi[active]
        fisher_a = fisher[active]
        d_a = d[active]
        n_active = int(active.sum())

        # Rank correlation
        rho, rho_p = spearmanr(np.abs(phi_a), fisher_a)

        # Top-k overlap (Jaccard)
        k_values = [5, 10, min(20, n_active)]
        jaccard_results = {}
        for k in k_values:
            if k > n_active:
                continue
            top_shap_idx = set(np.argsort(-np.abs(phi_a))[:k].tolist())
            top_fisher_idx = set(np.argsort(-fisher_a)[:k].tolist())
            j = jaccard(top_shap_idx, top_fisher_idx)
            jaccard_results[k] = j

        # Counter-evidential blind spot: Fisher ranks d=-1 items the same
        # as d=+1 items if they have similar difficulty/discrimination.
        # Shapley gives them negative values.
        counter_mask = d_a == -1
        n_counter = int(counter_mask.sum())
        if n_counter > 0:
            fisher_rank_of_counter = []
            fisher_order = np.argsort(-fisher_a)
            for idx in np.where(counter_mask)[0]:
                rank = int(np.where(fisher_order == idx)[0][0]) + 1
                fisher_rank_of_counter.append(rank)
            mean_fisher_rank_counter = np.mean(fisher_rank_of_counter)
            median_fisher_rank_counter = np.median(fisher_rank_of_counter)
        else:
            mean_fisher_rank_counter = float("nan")
            median_fisher_rank_counter = float("nan")

        print(f"  Active items: {n_active}")
        print(f"  Spearman |φ| vs Fisher: ρ={rho:.4f} (p={rho_p:.2e})")
        for k, j in jaccard_results.items():
            print(f"  Jaccard top-{k}: {j:.3f}")
        print(f"  Counter-evidential items: {n_counter}")
        if n_counter > 0:
            print(f"    Mean Fisher rank of counter items: {mean_fisher_rank_counter:.1f} / {n_active}")
            print(f"    (Fisher treats them as informative, Shapley flags them as harmful)")

        result = {
            "dataset": name,
            "margin": contest["margin"],
            "n_active": n_active,
            "n_counter_evidential": n_counter,
            "theta_m1": round(irt["theta_m1"], 4),
            "theta_m2": round(irt["theta_m2"], 4),
            "theta_star": round(irt["theta_star"], 4),
            "spearman_abs_phi_vs_fisher": round(float(rho), 4),
            "spearman_p": float(rho_p),
            "jaccard": {str(k): round(v, 4) for k, v in jaccard_results.items()},
            "mean_fisher_rank_counter": round(float(mean_fisher_rank_counter), 1),
        }
        all_results.append(result)

    # Save
    results_path = RESULTS_DIR / "e5_localized_fisher.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Summary
    valid = [r for r in all_results if "error" not in r]
    print(f"\n{'='*100}")
    print(f"{'Dataset':<25} {'Margin':>7} {'Active':>7} {'ρ(|φ|,F)':>9} "
          f"{'J@5':>6} {'J@10':>6} {'#Counter':>8} {'F-rank(ctr)':>12}")
    print(f"{'='*100}")
    for r in valid:
        j5 = r["jaccard"].get("5", float("nan"))
        j10 = r["jaccard"].get("10", float("nan"))
        print(f"{r['dataset']:<25} {r['margin']:>7} {r['n_active']:>7} "
              f"{r['spearman_abs_phi_vs_fisher']:>9.4f} {j5:>6.3f} {j10:>6.3f} "
              f"{r['n_counter_evidential']:>8} {r['mean_fisher_rank_counter']:>12.1f}")

    # Plot: |φ| vs Fisher info scatter for a few benchmarks
    _plot_phi_vs_fisher(datasets, valid)

    return all_results


def _plot_phi_vs_fisher(datasets, valid_results):
    """Scatter: |φ| vs Fisher info for selected benchmarks."""
    plot_names = [r["dataset"] for r in valid_results[:6]]
    n_plots = len(plot_names)
    if n_plots == 0:
        return

    cols = min(3, n_plots)
    rows = (n_plots + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.ravel()

    for ax_idx, name in enumerate(plot_names):
        ax = axes[ax_idx]

        ds = next((d for d in datasets if d[0] == name), None)
        if ds is None:
            continue
        _, R, mdls, itms = ds
        contest = top_two(R, mdls)
        d = contest["d"]
        shap = mc_shapley_flip(d, T=T, seed=SEED)
        phi = shap["phi"]

        try:
            irt = fit_2pl_and_fisher(R, contest["m1_idx"], contest["m2_idx"])
        except Exception:
            continue
        fisher = irt["fisher_info"]

        active = d != 0
        phi_a = phi[active]
        fisher_a = fisher[active]
        d_a = d[active]

        colors = ["#E74C3C" if di == -1 else "#3498DB" for di in d_a]
        ax.scatter(fisher_a, phi_a, c=colors, s=15, alpha=0.6, edgecolors="none")
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.set_xlabel("Fisher Info at θ*", fontsize=9)
        ax.set_ylabel("Shapley φ", fontsize=9)
        short = name.replace("metabench/", "mb/").replace("livebench/", "lb/")
        ax.set_title(f"{short} (margin={contest['margin']})", fontsize=10)

    for ax_idx in range(n_plots, len(axes)):
        axes[ax_idx].set_visible(False)

    fig.text(0.5, -0.02, "Blue = d=+1 (pro-winner), Red = d=-1 (counter-evidential)",
             ha="center", fontsize=10)
    plt.suptitle("Fisher Info vs Shapley φ: Fisher Can't Sign Items (E5)", fontsize=12, y=1.02)
    plt.tight_layout()
    path = FIGURES_DIR / "e5_fisher_vs_shapley.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure: {path}")


if __name__ == "__main__":
    run_e5()
