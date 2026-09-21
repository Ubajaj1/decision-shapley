"""E7 — Report-card artifact end-to-end.

Run the tool on a held-out matrix (winogrande — smallest active set, most
fragile) and produce JSON + HTML reports. Success: runs in one command on
arbitrary input.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data_io import load_benchmark
from reportcard import generate_report, report_to_json, report_to_html

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def run_e7():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Run on two benchmarks to demonstrate generality:
    # winogrande (fragile, few active items) and gsm8k (borderline)
    for bench in ["winogrande", "gsm8k", "arc"]:
        print(f"\n{'='*60}")
        print(f"  Generating report card: {bench.upper()}")
        print(f"{'='*60}")

        R, models, items = load_benchmark(bench)
        report = generate_report(
            R, models,
            item_ids=items,
            benchmark_name=bench,
            B=10_000, T=10_000, seed=42,
        )

        # JSON
        json_path = RESULTS_DIR / f"reportcard_{bench}.json"
        report_to_json(report, json_path)
        print(f"  JSON: {json_path}")

        # HTML
        html_path = RESULTS_DIR / f"reportcard_{bench}.html"
        report_to_html(report, html_path)
        print(f"  HTML: {html_path}")

        # Summary
        fp = report["flip_probability"]
        c = report["contest"]
        print(f"\n  #1: {c['m1']} ({c['m1_score']:.4f})")
        print(f"  #2: {c['m2']} ({c['m2_score']:.4f})")
        print(f"  Margin: {c['margin']}")
        print(f"  Flip probability: {fp['flip_or_tie']:.4f} [{fp['ci_95'][0]:.4f}, {fp['ci_95'][1]:.4f}]")
        print(f"  Decisive items: {report['taxonomy']['n_decisive']}")
        print(f"  Counter-evidential: {report['taxonomy']['n_counter_evidential']}")

    print("\n✓ Report card artifact runs end-to-end on arbitrary matrices.")


if __name__ == "__main__":
    run_e7()
