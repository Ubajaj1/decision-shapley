"""E0 — Data integrity checks.

Load all matrices; verify shapes, value ranges, and that aggregate rankings
are internally consistent. Report the #1-vs-#2 contest for each benchmark.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import json
import numpy as np
from data_io import load_benchmark, BENCHMARKS
from decision import top_two, model_scores

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def run_integrity_checks():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report = {}

    for bench in BENCHMARKS:
        print(f"\n{'='*60}")
        print(f"  {bench.upper()}")
        print(f"{'='*60}")

        R, models, items = load_benchmark(bench, cache=True)
        n_models, n_items = R.shape

        # Basic checks
        assert R.shape == (len(models), len(items)), "Shape mismatch"
        unique_vals = set(np.unique(R).tolist())
        assert unique_vals <= {0.0, 1.0}, f"Non-binary values: {unique_vals}"
        assert not np.isnan(R).any(), "NaN values present"
        print(f"  Shape: {n_models} models × {n_items} items  ✓")
        print(f"  Values: binary {{0, 1}}  ✓")

        # Score distribution
        scores = model_scores(R)
        print(f"  Score range: [{scores.min():.4f}, {scores.max():.4f}]")
        print(f"  Score mean:  {scores.mean():.4f}")

        # Ranking is consistent with raw sum
        sum_ranking = np.argsort(-R.sum(axis=1))
        mean_ranking = np.argsort(-scores)
        assert np.array_equal(sum_ranking, mean_ranking), "Sum vs mean ranking mismatch"
        print(f"  Sum-vs-mean ranking: consistent  ✓")

        # Top-2 contest
        try:
            contest = top_two(R, models)
            print(f"\n  #1-vs-#2 contest (ranks {contest['rank1']} vs {contest['rank2']}):")
            print(f"    #1: {contest['m1_id']}")
            print(f"        score = {contest['m1_score']:.6f}")
            print(f"    #2: {contest['m2_id']}")
            print(f"        score = {contest['m2_score']:.6f}")
            print(f"    Margin: {contest['margin']} items")
            print(f"    d breakdown: +1={contest['n_plus']}, 0={contest['n_zero']}, -1={contest['n_minus']}")
            if contest["n_tied_at_top"] > 0:
                print(f"    ⚠ Skipped {contest['n_tied_at_top']} tied models at top")

            report[bench] = {
                "n_models": n_models,
                "n_items": n_items,
                "m1": contest["m1_id"],
                "m2": contest["m2_id"],
                "m1_score": contest["m1_score"],
                "m2_score": contest["m2_score"],
                "margin": contest["margin"],
                "n_plus": contest["n_plus"],
                "n_minus": contest["n_minus"],
                "n_zero": contest["n_zero"],
                "rank1": contest["rank1"],
                "rank2": contest["rank2"],
                "n_tied_at_top": contest["n_tied_at_top"],
            }
        except ValueError as e:
            print(f"    ✗ {e}")
            report[bench] = {"n_models": n_models, "n_items": n_items, "error": str(e)}

    # Save report
    report_path = RESULTS_DIR / "e0_integrity.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n\nReport saved to {report_path}")

    # Summary table
    print(f"\n{'='*80}")
    print(f"{'Benchmark':<12} {'Models':>7} {'Items':>7} {'#1 Score':>10} {'#2 Score':>10} {'Margin':>8} {'Tied@Top':>10}")
    print(f"{'='*80}")
    for bench in BENCHMARKS:
        r = report[bench]
        if "error" in r:
            print(f"{bench:<12} {r['n_models']:>7} {r['n_items']:>7} {'ERROR':>10}")
        else:
            print(f"{bench:<12} {r['n_models']:>7} {r['n_items']:>7} {r['m1_score']:>10.6f} {r['m2_score']:>10.6f} {r['margin']:>8} {r['n_tied_at_top']:>10}")

    return report


if __name__ == "__main__":
    run_integrity_checks()
