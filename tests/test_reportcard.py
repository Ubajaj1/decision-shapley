"""Public report-card and CLI smoke tests."""

import json

import numpy as np
import pandas as pd

from decision_shapley_cli import main
from reportcard import generate_report


def _matrix():
    return np.array([
        [1, 1, 1, 1, 1, 0, 1, 0],
        [1, 1, 0, 1, 0, 1, 1, 0],
        [1, 0, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
    ], dtype=np.int8)


def test_reportcard_computes_both_non_degenerate_attributions():
    report = generate_report(
        _matrix(),
        ["alpha", "beta", "gamma", "delta"],
        B=200,
        T=200,
        value_function="information",
        include_alternative=True,
    )

    assert report["shapley"]["primary_value_function"] == "information"
    assert report["shapley"]["computed_value_functions"] == ["information", "confidence"]
    assert report["shapley"]["confidence"] is not None
    assert report["shapley"]["information"] is not None
    item = report["decisive_items"][0]
    assert set(item) >= {"phi_confidence", "phi_information"}


def test_cli_writes_json_and_html(tmp_path):
    frame = pd.DataFrame(_matrix(), columns=[f"item_{i}" for i in range(8)])
    frame.insert(0, "model_id", ["alpha", "beta", "gamma", "delta"])
    input_path = tmp_path / "matrix.csv"
    output_dir = tmp_path / "reports"
    frame.to_csv(input_path, index=False)

    exit_code = main([
        str(input_path),
        "--output-dir", str(output_dir),
        "--bootstrap", "100",
        "--permutations", "100",
    ])

    assert exit_code == 0
    json_path = output_dir / "matrix_report.json"
    html_path = output_dir / "matrix_report.html"
    assert json_path.exists()
    assert html_path.exists()
    payload = json.loads(json_path.read_text())
    assert payload["shapley"]["primary_value_function"] == "confidence"
    assert "Flip probability is sensitivity" in html_path.read_text()
