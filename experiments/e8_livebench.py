"""E8 — Full decision-robustness analysis on LiveBench binary tasks.

Runs bootstrap flip probability, Shapley attribution, taxonomy, and
report card on LiveBench's LCB_generation, coding_completion, and typos
tasks — featuring frontier models (Gemini 2.5 Pro, o3-mini, GPT-4.5,
Claude 3.7 Sonnet, DeepSeek R1, Grok 3).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import json
import numpy as np
from livebench_io import load_livebench_task, LIVEBENCH_BINARY_TASKS
from decision import top_two
from bootstrap import flip_probability_vectorized
from shapley import mc_shapley_flip
from taxonomy import classify_items
from concentration import concentration_metrics
from loo import loo_flips, decisive_item_count
from reportcard import generate_report, report_to_json, report_to_html

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

B = 50_000
T = 20_000
SEED = 42


def run_e8():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    summary = {}

    for task in LIVEBENCH_BINARY_TASKS:
        print(f"\n{'='*70}")
        print(f"  LiveBench Task: {task}")
        print(f"{'='*70}")

        R, models, items = load_livebench_task(task)
        n_models, n_items = R.shape
        print(f"  Matrix: {n_models} models x {n_items} items")

        contest = top_two(R, models)
        d = contest["d"]
        margin = contest["margin"]

        print(f"\n  Contest: #1 vs #2")
        print(f"    #1: {contest['m1_id']} (score: {contest['m1_score']:.4f})")
        print(f"    #2: {contest['m2_id']} (score: {contest['m2_score']:.4f})")
        print(f"    Margin: {margin}, d+: {contest['n_plus']}, "
              f"d-: {contest['n_minus']}, d=0: {contest['n_zero']}")

        # Bootstrap
        bp = flip_probability_vectorized(d, B=B, seed=SEED)
        print(f"\n  Bootstrap (B={B}):")
        print(f"    Flip probability: {bp['flip_or_tie_prob']:.4f} "
              f"[{bp['ci_lo']:.4f}, {bp['ci_hi']:.4f}]")

        # LOO
        loo = loo_flips(d)
        dec_count = decisive_item_count(d)
        print(f"    LOO vulnerable: {loo['loo_vulnerable']}")
        print(f"    Min removals to flip: {dec_count}")

        # Shapley
        shap = mc_shapley_flip(d, T=T, seed=SEED)
        phi = shap["phi"]
        print(f"\n  Shapley (T={T}):")
        print(f"    Efficiency sum: {shap['efficiency_sum']:.6f} (target: 0.5)")
        print(f"    Max SE: {shap['max_se']:.6f}")

        # Taxonomy
        tax = classify_items(phi, d)
        print(f"\n  Taxonomy:")
        print(f"    Decisive: {tax['n_decisive']}")
        print(f"    Counter-evidential: {tax['n_counter']}")
        print(f"    Redundant: {tax['n_redundant']}")

        # Concentration
        conc = concentration_metrics(phi)
        print(f"\n  Concentration:")
        print(f"    Gini (positive phi): {conc['gini_positive']:.4f}")
        print(f"    Top-k for 50% mass: {conc['top_k_50']}")

        # Top decisive items
        order = np.argsort(-phi)
        print(f"\n  Top decisive items:")
        for rank, idx in enumerate(order[:5]):
            if phi[idx] <= 0:
                break
            print(f"    {items[idx][:40]:40s}  d={int(d[idx]):+d}  "
                  f"phi={phi[idx]:.6f} +/- {shap['se'][idx]:.6f}")

        # Counter-evidential items
        counter_order = np.argsort(phi)
        print(f"\n  Top counter-evidential items:")
        for rank, idx in enumerate(counter_order[:5]):
            if phi[idx] >= 0:
                break
            print(f"    {items[idx][:40]:40s}  d={int(d[idx]):+d}  "
                  f"phi={phi[idx]:.6f}")

        # Generate report card
        report = generate_report(R, models, item_ids=items,
                                 benchmark_name=f"livebench/{task}",
                                 B=B, T=T, seed=SEED)
        json_path = RESULTS_DIR / f"livebench_{task}.json"
        html_path = RESULTS_DIR / f"livebench_{task}.html"
        report_to_json(report, json_path)
        report_to_html(report, html_path)
        print(f"\n  Report: {json_path}")
        print(f"  Report: {html_path}")

        summary[task] = {
            "n_models": n_models,
            "n_items": n_items,
            "m1": contest["m1_id"],
            "m2": contest["m2_id"],
            "m1_score": round(contest["m1_score"], 4),
            "m2_score": round(contest["m2_score"], 4),
            "margin": margin,
            "n_plus": contest["n_plus"],
            "n_minus": contest["n_minus"],
            "flip_prob": round(bp["flip_or_tie_prob"], 4),
            "flip_ci": [round(bp["ci_lo"], 4), round(bp["ci_hi"], 4)],
            "loo_vulnerable": loo["loo_vulnerable"],
            "min_removals": dec_count,
            "n_decisive": tax["n_decisive"],
            "n_counter": tax["n_counter"],
            "gini_positive": round(conc["gini_positive"], 4),
        }

    # Summary table
    print(f"\n{'='*100}")
    print(f"{'Task':<22} {'#1 vs #2':>30} {'Margin':>7} {'Flip%':>7} "
          f"{'LOO':>5} {'Dec':>5} {'Cntr':>5} {'Gini':>6}")
    print(f"{'='*100}")
    for task in LIVEBENCH_BINARY_TASKS:
        s = summary[task]
        contest_str = f"{s['m1'][:14]} vs {s['m2'][:14]}"
        print(f"{task:<22} {contest_str:>30} {s['margin']:>7} "
              f"{s['flip_prob']:>6.1%} {s['loo_vulnerable']:>5} "
              f"{s['n_decisive']:>5} {s['n_counter']:>5} {s['gini_positive']:>6.3f}")

    # Save summary
    summary_path = RESULTS_DIR / "e8_livebench_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    run_e8()
