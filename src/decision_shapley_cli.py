"""Command-line interface for Decision-Shapley report cards."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from reportcard import generate_report, report_to_html, report_to_json


def load_matrix(path: Path, model_column: str | None = None):
    """Load a binary model-by-item matrix from CSV or NPZ.

    CSV files must contain a model identifier column (the first column by
    default) followed by one 0/1 column per item. NPZ files must contain ``R``
    and may contain ``model_ids`` and ``item_ids``.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        frame = pd.read_csv(path)
        if frame.shape[1] < 2:
            raise ValueError("CSV input needs a model column and at least one item column.")
        id_column = model_column or str(frame.columns[0])
        if id_column not in frame.columns:
            raise ValueError(f"Model column {id_column!r} was not found in the CSV.")
        model_ids = frame[id_column].astype(str).tolist()
        item_frame = frame.drop(columns=[id_column])
        item_ids = [str(column) for column in item_frame.columns]
        R = item_frame.to_numpy(dtype=float)
    elif suffix == ".npz":
        archive = np.load(path, allow_pickle=False)
        if "R" not in archive:
            raise ValueError("NPZ input must contain an array named 'R'.")
        R = archive["R"]
        model_ids = (
            archive["model_ids"].astype(str).tolist()
            if "model_ids" in archive
            else [f"model_{index}" for index in range(R.shape[0])]
        )
        item_ids = (
            archive["item_ids"].astype(str).tolist()
            if "item_ids" in archive
            else [f"item_{index}" for index in range(R.shape[1])]
        )
    else:
        raise ValueError("Input must be a .csv or .npz file.")

    if np.isnan(np.asarray(R, dtype=float)).any() or not np.isin(R, [0, 1]).all():
        raise ValueError("The response matrix must contain only binary 0/1 values.")
    return np.asarray(R, dtype=np.int8), model_ids, item_ids


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="decision-shapley",
        description="Generate a decision-robustness report from a binary response matrix.",
    )
    parser.add_argument("input", type=Path, help="CSV or NPZ model-by-item matrix")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results"),
        help="directory for JSON and HTML reports (default: results)",
    )
    parser.add_argument("--benchmark", help="benchmark label shown in the report")
    parser.add_argument(
        "--model-column",
        help="CSV model identifier column (default: first column)",
    )
    parser.add_argument(
        "--rank-boundary", type=int, default=1,
        help="adjacent rank boundary to analyze (default: 1 for #1 vs #2)",
    )
    parser.add_argument(
        "--value-function", choices=("confidence", "information"),
        default="confidence", help="primary attribution shown in rankings",
    )
    parser.add_argument(
        "--include-alternative", action="store_true",
        help="also compute the other value function (slower on large active sets)",
    )
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--permutations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    R, model_ids, item_ids = load_matrix(args.input, args.model_column)
    benchmark = args.benchmark or args.input.stem

    report = generate_report(
        R,
        model_ids,
        item_ids=item_ids,
        benchmark_name=benchmark,
        B=args.bootstrap,
        T=args.permutations,
        seed=args.seed,
        rank1=args.rank_boundary,
        value_function=args.value_function,
        include_alternative=args.include_alternative,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.input.stem}_report"
    json_path = args.output_dir / f"{stem}.json"
    html_path = args.output_dir / f"{stem}.html"
    report_to_json(report, json_path)
    report_to_html(report, html_path)

    contest = report["contest"]
    flip = report["flip_probability"]
    print(f"Contest: {contest['m1']} vs {contest['m2']}")
    print(f"Item margin: {contest['margin']}")
    print(f"Flip-or-tie probability: {flip['flip_or_tie']:.3%}")
    print(f"JSON: {json_path}")
    print(f"HTML: {html_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
