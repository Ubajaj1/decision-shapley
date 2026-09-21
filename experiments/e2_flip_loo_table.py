"""E2 — Flip probability & LOO across all benchmarks (first headline number).

For each benchmark's #1-vs-#2 contest: flip_prob, LOO flips, decisive-item count.
Produces the "stable vs coin-flip" table.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import json
import numpy as np
from data_io import load_benchmark, BENCHMARKS
from decision import top_two
from bootstrap import flip_probability_vectorized
from loo import loo_flips, decisive_item_count

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"

B = 50_000
SEED = 42


def run_e2():
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

        print(f"  #1: {contest['m1_id']}")
        print(f"  #2: {contest['m2_id']}")
        print(f"  Margin: {contest['margin']}")

        # Bootstrap flip probability
        print(f"  Running bootstrap (B={B})...")
        bp = flip_probability_vectorized(d, B=B, seed=SEED)
        print(f"  Flip prob:     {bp['flip_prob']:.4f}")
        print(f"  Tie prob:      {bp['tie_prob']:.4f}")
        print(f"  Flip-or-tie:   {bp['flip_or_tie_prob']:.4f}  "
              f"[{bp['ci_lo']:.4f}, {bp['ci_hi']:.4f}]")

        # LOO analysis
        loo = loo_flips(d)
        print(f"  LOO vulnerable: {loo['loo_vulnerable']} (margin={loo['margin']})")
        print(f"  LOO → tie:  {loo['n_loo_tie']} items")
        print(f"  LOO → loss: {loo['n_loo_loss']} items")

        # Decisive item count
        dec_count = decisive_item_count(d)
        print(f"  Decisive-item count (min removals to flip): {dec_count}")

        results[bench] = {
            "m1": contest["m1_id"],
            "m2": contest["m2_id"],
            "m1_score": contest["m1_score"],
            "m2_score": contest["m2_score"],
            "margin": contest["margin"],
            "n_items": len(d),
            "n_plus": contest["n_plus"],
            "n_minus": contest["n_minus"],
            "n_zero": contest["n_zero"],
            "rank1": contest["rank1"],
            "rank2": contest["rank2"],
            "flip_prob": bp["flip_prob"],
            "tie_prob": bp["tie_prob"],
            "flip_or_tie_prob": bp["flip_or_tie_prob"],
            "ci_lo": bp["ci_lo"],
            "ci_hi": bp["ci_hi"],
            "B": bp["B"],
            "loo_vulnerable": loo["loo_vulnerable"],
            "n_loo_tie": loo["n_loo_tie"],
            "n_loo_loss": loo["n_loo_loss"],
            "decisive_item_count": dec_count,
        }

    # Save JSON
    report_path = RESULTS_DIR / "e2_flip_loo.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {report_path}")

    # Print summary table
    print(f"\n{'='*100}")
    print(f"{'Bench':<12} {'Margin':>7} {'Items':>6} {'d+':>5} {'d-':>5} "
          f"{'Flip%':>7} {'Tie%':>7} {'CI':>15} "
          f"{'LOO':>5} {'Dec#':>5}")
    print(f"{'='*100}")
    for bench in BENCHMARKS:
        r = results[bench]
        ci_str = f"[{r['ci_lo']:.3f},{r['ci_hi']:.3f}]"
        loo_str = f"{r['n_loo_tie']}" if r["loo_vulnerable"] else "—"
        print(f"{bench:<12} {r['margin']:>7} {r['n_items']:>6} {r['n_plus']:>5} {r['n_minus']:>5} "
              f"{r['flip_prob']:>7.4f} {r['tie_prob']:>7.4f} {ci_str:>15} "
              f"{loo_str:>5} {r['decisive_item_count']:>5}")

    # Generate the core figure: flip probability bar chart
    _plot_flip_probabilities(results)

    return results


def _plot_flip_probabilities(results: dict):
    import matplotlib.pyplot as plt

    benchmarks = list(results.keys())
    flip_or_tie = [results[b]["flip_or_tie_prob"] for b in benchmarks]
    flip_only = [results[b]["flip_prob"] for b in benchmarks]
    ci_lo = [results[b]["ci_lo"] for b in benchmarks]
    ci_hi = [results[b]["ci_hi"] for b in benchmarks]

    # Sort by flip probability descending
    order = np.argsort(flip_or_tie)[::-1]
    benchmarks = [benchmarks[i] for i in order]
    flip_or_tie = [flip_or_tie[i] for i in order]
    flip_only = [flip_only[i] for i in order]
    ci_lo = [ci_lo[i] for i in order]
    ci_hi = [ci_hi[i] for i in order]

    err_lo = [f - c for f, c in zip(flip_or_tie, ci_lo)]
    err_hi = [c - f for f, c in zip(flip_or_tie, ci_hi)]

    fig, ax = plt.subplots(figsize=(8, 4))

    x = np.arange(len(benchmarks))
    bars = ax.bar(x, flip_or_tie, color="#4C72B0", edgecolor="white", linewidth=0.5)
    ax.bar(x, flip_only, color="#DD8452", edgecolor="white", linewidth=0.5, label="Flip (strict)")
    ax.errorbar(x, flip_or_tie, yerr=[err_lo, err_hi],
                fmt="none", color="black", capsize=4, linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels([b.upper() for b in benchmarks], fontsize=10)
    ax.set_ylabel("Probability", fontsize=11)
    ax.set_title("Bootstrap Selection-Flip Probability (#1 vs #2)", fontsize=12)
    ax.axhline(0.05, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.text(len(benchmarks) - 0.5, 0.06, "α = 0.05", fontsize=8, color="gray", ha="right")
    ax.legend(["Flip or Tie", "Flip (strict)"], fontsize=9, loc="upper right")
    ax.set_ylim(0, max(flip_or_tie) * 1.2 + 0.02)

    for i, v in enumerate(flip_or_tie):
        ax.text(i, v + max(flip_or_tie) * 0.03, f"{v:.3f}", ha="center", fontsize=8)

    plt.tight_layout()
    fig_path = FIGURES_DIR / "e2_flip_probability.png"
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)
    print(f"\nFigure saved to {fig_path}")


if __name__ == "__main__":
    run_e2()
