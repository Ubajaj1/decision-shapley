"""E4b — Alternative perturbation model (validates Gap 4).

Instead of bootstrap resampling (item superpopulation assumption), use
answer-flip noise: each binary answer flips with probability epsilon.
Compare the fragility ranking across benchmarks under both perturbation
models. High rank correlation → conclusions are robust to perturbation choice.
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
from bootstrap import flip_probability_vectorized

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"

B = 50_000
SEED = 42
EPSILONS = [0.01, 0.03, 0.05, 0.10]


def answer_flip_probability(d: np.ndarray, R: np.ndarray,
                            m1_idx: int, m2_idx: int,
                            epsilon: float = 0.05,
                            B: int = 50_000, seed: int = 42) -> dict:
    """Flip each answer with probability epsilon, recompute decision.

    For each trial: flip a_i with prob epsilon, flip b_i with prob epsilon,
    recompute d_i' = a_i' - b_i', check if sum(d_i') <= 0.
    """
    rng = np.random.default_rng(seed)
    a = R[m1_idx].astype(np.int8)
    b = R[m2_idx].astype(np.int8)
    n = len(a)

    flips = 0
    ties = 0

    flip_a = rng.random((B, n)) < epsilon
    flip_b = rng.random((B, n)) < epsilon

    a_perturbed = np.where(flip_a, 1 - a, a)
    b_perturbed = np.where(flip_b, 1 - b, b)
    d_perturbed = a_perturbed - b_perturbed
    margins = d_perturbed.sum(axis=1)

    flips = int((margins < 0).sum())
    ties = int((margins == 0).sum())

    return {
        "flip_prob": flips / B,
        "tie_prob": ties / B,
        "flip_or_tie_prob": (flips + ties) / B,
        "epsilon": epsilon,
        "B": B,
    }


def run_e4b():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []

    datasets = []
    # LiveBench tasks
    for task in LIVEBENCH_BINARY_TASKS:
        R, models, items = load_livebench_task(task)
        datasets.append((f"livebench/{task}", R, models, items))

    # Metabench benchmarks
    for bench in BENCHMARKS:
        R, models, items = load_benchmark(bench)
        datasets.append((f"metabench/{bench}", R, models, items))

    for name, R, models, items in datasets:
        print(f"\n{'='*70}")
        print(f"  {name}")
        print(f"{'='*70}")

        contest = top_two(R, models)
        d = contest["d"]
        margin = contest["margin"]

        # Bootstrap flip probability
        bp = flip_probability_vectorized(d, B=B, seed=SEED)

        # Answer-flip at multiple epsilon values
        eps_results = {}
        for eps in EPSILONS:
            af = answer_flip_probability(
                d, R, contest["m1_idx"], contest["m2_idx"],
                epsilon=eps, B=B, seed=SEED
            )
            eps_results[eps] = af

        print(f"  #1: {contest['m1_id'][:30]}")
        print(f"  #2: {contest['m2_id'][:30]}")
        print(f"  Margin: {margin}")
        print(f"  Bootstrap flip:    {bp['flip_or_tie_prob']:.4f}")
        for eps in EPSILONS:
            af = eps_results[eps]
            print(f"  Answer-flip ε={eps:.2f}: {af['flip_or_tie_prob']:.4f}")

        result = {
            "dataset": name,
            "m1": contest["m1_id"],
            "m2": contest["m2_id"],
            "margin": margin,
            "n_items": len(d),
            "bootstrap_flip": round(bp["flip_or_tie_prob"], 6),
        }
        for eps in EPSILONS:
            result[f"answerflip_{eps}"] = round(eps_results[eps]["flip_or_tie_prob"], 6)
        all_results.append(result)

    # Save results
    results_path = RESULTS_DIR / "e4b_alt_perturbation.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Rank correlation: bootstrap vs answer-flip
    print(f"\n{'='*70}")
    print("  Rank Correlations: Bootstrap vs Answer-Flip")
    print(f"{'='*70}")

    bootstrap_flips = [r["bootstrap_flip"] for r in all_results]
    for eps in EPSILONS:
        af_flips = [r[f"answerflip_{eps}"] for r in all_results]
        rho, p = spearmanr(bootstrap_flips, af_flips)
        print(f"  ε={eps:.2f}: Spearman ρ = {rho:.4f} (p={p:.2e})")

    # Plot: bootstrap vs answer-flip scatter for each epsilon
    fig, axes = plt.subplots(1, len(EPSILONS), figsize=(4 * len(EPSILONS), 4))
    for ax, eps in zip(axes, EPSILONS):
        af_flips = [r[f"answerflip_{eps}"] for r in all_results]
        lb_mask = [r["dataset"].startswith("livebench") for r in all_results]
        mb_mask = [not m for m in lb_mask]

        lb_x = [b for b, m in zip(bootstrap_flips, lb_mask) if m]
        lb_y = [a for a, m in zip(af_flips, lb_mask) if m]
        mb_x = [b for b, m in zip(bootstrap_flips, mb_mask) if m]
        mb_y = [a for a, m in zip(af_flips, mb_mask) if m]

        ax.scatter(mb_x, mb_y, c="#3498DB", s=30, alpha=0.7, label="Metabench")
        ax.scatter(lb_x, lb_y, c="#E74C3C", s=50, label="LiveBench", zorder=3)

        lim = max(max(bootstrap_flips), max(af_flips)) * 1.05 + 0.01
        ax.plot([0, lim], [0, lim], "k--", alpha=0.3, linewidth=0.8)
        ax.set_xlim(-0.02, lim)
        ax.set_ylim(-0.02, lim)
        ax.set_xlabel("Bootstrap Flip Prob", fontsize=9)
        ax.set_ylabel("Answer-Flip Prob", fontsize=9)

        rho, _ = spearmanr(bootstrap_flips, af_flips)
        ax.set_title(f"ε={eps:.2f} (ρ={rho:.3f})", fontsize=10)
        if eps == EPSILONS[0]:
            ax.legend(fontsize=8)

    plt.suptitle("Bootstrap vs Answer-Flip Perturbation (E4b)", fontsize=12, y=1.02)
    plt.tight_layout()
    path = FIGURES_DIR / "e4b_bootstrap_vs_answerflip.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure: {path}")

    # Summary table
    print(f"\n{'='*110}")
    print(f"{'Dataset':<25} {'Margin':>7} {'Bootstrap':>10} " +
          " ".join(f"{'ε='+str(e):>10}" for e in EPSILONS))
    print(f"{'='*110}")
    for r in all_results:
        eps_vals = " ".join(f"{r[f'answerflip_{e}']:>10.4f}" for e in EPSILONS)
        print(f"{r['dataset']:<25} {r['margin']:>7} {r['bootstrap_flip']:>10.4f} {eps_vals}")

    return all_results


if __name__ == "__main__":
    run_e4b()
