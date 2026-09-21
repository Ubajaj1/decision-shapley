"""E9 — Multi-rank fragility analysis across LiveBench + metabench.

For each dataset/task, analyze adjacent rank pairs (#1v2, #2v3, ..., #9v10)
to show how fragility varies across the leaderboard. Produces a combined
scatter plot of concentration vs flip probability (E6 from the roadmap).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from data_io import load_benchmark, BENCHMARKS
from livebench_io import load_livebench_task, LIVEBENCH_BINARY_TASKS
from decision import top_two
from bootstrap import flip_probability_vectorized
from shapley import mc_shapley_flip
from taxonomy import classify_items
from concentration import concentration_metrics
from loo import loo_flips, decisive_item_count

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"

B = 50_000
T = 20_000
SEED = 42
MAX_RANK = 10


def analyze_contest(R, models, items, dataset_name, rank1):
    """Run full analysis on one rank boundary. Returns dict or None if no valid contest."""
    try:
        contest = top_two(R, models, rank1=rank1)
    except ValueError:
        return None

    d = contest["d"]
    bp = flip_probability_vectorized(d, B=B, seed=SEED)
    shap = mc_shapley_flip(d, T=T, seed=SEED)
    phi = shap["phi"]
    tax = classify_items(phi, d)
    conc = concentration_metrics(phi)
    loo = loo_flips(d)
    dec_count = decisive_item_count(d)

    return {
        "dataset": dataset_name,
        "rank1": contest["rank1"],
        "rank2": contest["rank2"],
        "m1": contest["m1_id"],
        "m2": contest["m2_id"],
        "m1_score": round(contest["m1_score"], 6),
        "m2_score": round(contest["m2_score"], 6),
        "margin": contest["margin"],
        "n_plus": contest["n_plus"],
        "n_minus": contest["n_minus"],
        "n_zero": contest["n_zero"],
        "n_items": len(d),
        "flip_prob": round(bp["flip_or_tie_prob"], 6),
        "flip_ci": [round(bp["ci_lo"], 6), round(bp["ci_hi"], 6)],
        "loo_vulnerable": loo["loo_vulnerable"],
        "min_removals": dec_count,
        "n_decisive": tax["n_decisive"],
        "n_counter": tax["n_counter"],
        "n_redundant": tax["n_redundant"],
        "gini_positive": round(conc["gini_positive"], 6),
        "top_k_50": conc["top_k_50"],
        "efficiency_sum": round(shap["efficiency_sum"], 6),
    }


def run_e9():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []

    def run_dataset(label, R, models, items):
        seen_ranks = set()
        count = 0
        for rank1 in range(1, R.shape[0]):
            if count >= MAX_RANK:
                break
            result = analyze_contest(R, models, items, label, rank1)
            if result is None:
                break
            rank_key = (result["rank1"], result["rank2"])
            if rank_key in seen_ranks:
                continue
            seen_ranks.add(rank_key)
            all_results.append(result)
            count += 1
            m1_short = result["m1"][:20]
            m2_short = result["m2"][:20]
            print(f"  #{result['rank1']:>2} vs #{result['rank2']:<2}  "
                  f"{m1_short:20s} vs {m2_short:20s}  "
                  f"margin={result['margin']:>4}  flip={result['flip_prob']:.1%}  "
                  f"dec={result['n_decisive']:>3}  gini={result['gini_positive']:.4f}")

    # LiveBench tasks
    for task in LIVEBENCH_BINARY_TASKS:
        print(f"\n{'='*70}")
        print(f"  LiveBench: {task}")
        print(f"{'='*70}")
        R, models, items = load_livebench_task(task)
        print(f"  Matrix: {R.shape[0]} models x {R.shape[1]} items")
        run_dataset(f"livebench/{task}", R, models, items)

    # Metabench benchmarks
    for bench in BENCHMARKS:
        print(f"\n{'='*70}")
        print(f"  Metabench: {bench}")
        print(f"{'='*70}")
        R, models, items = load_benchmark(bench)
        print(f"  Matrix: {R.shape[0]} models x {R.shape[1]} items")
        run_dataset(f"metabench/{bench}", R, models, items)

    # Save all results
    results_path = RESULTS_DIR / "e9_multirank.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n{len(all_results)} contests saved to {results_path}")

    # Summary table
    print(f"\n{'='*120}")
    print(f"{'Dataset':<25} {'Contest':>30} {'Margin':>7} {'Flip%':>7} "
          f"{'Dec':>5} {'Cntr':>5} {'Gini':>7} {'LOO':>5}")
    print(f"{'='*120}")
    for r in all_results:
        label = f"#{r['rank1']}v{r['rank2']}"
        contest = f"{r['m1'][:13]} vs {r['m2'][:13]}"
        print(f"{r['dataset']:<25} {contest:>30} {r['margin']:>7} "
              f"{r['flip_prob']:>6.1%} {r['n_decisive']:>5} "
              f"{r['n_counter']:>5} {r['gini_positive']:>7.4f} "
              f"{'Y' if r['loo_vulnerable'] else 'N':>5}")

    # Plots
    _plot_flip_vs_margin(all_results)
    _plot_flip_vs_concentration(all_results)
    _plot_flip_by_rank(all_results)

    return all_results


def _plot_flip_vs_margin(results):
    """Publication-sized, color-blind-safe scatter of fragility vs margin."""
    fig, ax = plt.subplots(figsize=(3.5, 2.35))

    lb = [r for r in results if r["dataset"].startswith("livebench")]
    mb = [r for r in results if r["dataset"].startswith("metabench")]

    if lb:
        ax.scatter([r["margin"] for r in lb], [r["flip_prob"] for r in lb],
                   c="#D55E00", marker="o", s=34, label="LiveBench", zorder=3,
                   edgecolors="white", linewidth=0.45)
    if mb:
        ax.scatter([r["margin"] for r in mb], [r["flip_prob"] for r in mb],
                   c="#0072B2", marker="^", s=28, alpha=0.75, label="Open LLM", zorder=2,
                   edgecolors="white", linewidth=0.4)

    ax.set_xlabel("Net item margin", fontsize=8.5)
    ax.set_ylabel("Bootstrap flip probability", fontsize=8.5)
    ax.tick_params(axis="both", labelsize=7.5)
    ax.set_ylim(-0.02, max(r["flip_prob"] for r in results) * 1.1 + 0.02)
    ax.axhline(0.05, color="#4D4D4D", linestyle="--", linewidth=0.8,
               alpha=0.8, label="5% reference")
    ax.legend(fontsize=7, loc="upper right", frameon=True, borderpad=0.35,
              handletextpad=0.45, labelspacing=0.25)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.45, alpha=0.7)
    plt.tight_layout(pad=0.35)
    path = FIGURES_DIR / "e9_flip_vs_margin.png"
    fig.savefig(path, dpi=400)
    plt.close(fig)
    print(f"Figure: {path}")


def _plot_flip_vs_concentration(results):
    """E6 scatter: flip probability vs Gini concentration of Shapley values."""
    fig, ax = plt.subplots(figsize=(8, 5))

    lb = [r for r in results if r["dataset"].startswith("livebench")]
    mb = [r for r in results if r["dataset"].startswith("metabench")]

    if lb:
        ax.scatter([r["gini_positive"] for r in lb], [r["flip_prob"] for r in lb],
                   c="#E74C3C", s=60, label="LiveBench", zorder=3, edgecolors="white", linewidth=0.5)
    if mb:
        ax.scatter([r["gini_positive"] for r in mb], [r["flip_prob"] for r in mb],
                   c="#3498DB", s=40, alpha=0.7, label="Metabench", zorder=2, edgecolors="white", linewidth=0.5)

    ax.set_xlabel("Gini Coefficient of Positive Shapley Values", fontsize=11)
    ax.set_ylabel("Bootstrap Flip Probability", fontsize=11)
    ax.set_title("Decision Fragility vs Shapley Concentration (E6)", fontsize=13)
    ax.legend(fontsize=10)
    plt.tight_layout()
    path = FIGURES_DIR / "e6_concentration_vs_flip.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Figure: {path}")


def _plot_flip_by_rank(results):
    """Line plot: flip probability by rank position, one line per dataset."""
    fig, ax = plt.subplots(figsize=(10, 5))

    datasets = sorted(set(r["dataset"] for r in results))
    colors = plt.cm.tab10(np.linspace(0, 1, len(datasets)))

    for ds, color in zip(datasets, colors):
        subset = [r for r in results if r["dataset"] == ds]
        subset.sort(key=lambda r: r["rank1"])
        ranks = [r["rank1"] for r in subset]
        flips = [r["flip_prob"] for r in subset]
        label = ds.replace("metabench/", "mb/").replace("livebench/", "lb/")
        marker = "o" if ds.startswith("livebench") else "s"
        ax.plot(ranks, flips, marker=marker, label=label, color=color,
                linewidth=1.5, markersize=5)

    ax.set_xlabel("Rank Position (#k vs #k+1)", fontsize=11)
    ax.set_ylabel("Bootstrap Flip Probability", fontsize=11)
    ax.set_title("Decision Fragility Across Rank Boundaries", fontsize=13)
    ax.set_xticks(range(1, MAX_RANK + 1))
    ax.set_xticklabels([f"#{i}v{i+1}" for i in range(1, MAX_RANK + 1)], fontsize=9)
    ax.axhline(0.05, color="orange", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.legend(fontsize=7, ncol=3, loc="upper right")
    plt.tight_layout()
    path = FIGURES_DIR / "e9_flip_by_rank.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Figure: {path}")


if __name__ == "__main__":
    run_e9()
