# Decision-Shapley

Decision-Shapley audits a specific adjacent-model leaderboard decision. Given a
binary model-by-item response matrix, it reports:

- item-bootstrap flip probability for the selected rank boundary;
- decisive, counter-evidential, and decision-redundant items;
- correlation-aware confidence Shapley values;
- Fisher-information-weighted Shapley values; and
- machine-readable JSON plus a standalone HTML report card.

The method accompanies the ICTAI 2026 paper **“Won by a Single Question:
Item-Level Attribution and the Robustness of LLM Leaderboard Decisions”** by
Utkarsh Bajaj and Madhur Mehta.

## Installation

Decision-Shapley requires Python 3.11 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

## Quick start

The input is a CSV whose first column contains model identifiers and whose
remaining columns contain binary item results:

```text
model_id,item_1,item_2,item_3
model_a,1,1,0
model_b,1,0,1
```

Generate a report from the included example:

```bash
decision-shapley examples/example_matrix.csv \
  --benchmark example \
  --output-dir results/example
```

The command writes `example_matrix_report.json` and
`example_matrix_report.html`. Use `--value-function confidence` for the
correlation-aware attribution or `--value-function information` for the
Fisher-weighted attribution. Add `--include-alternative` to compute and report
both; this comparison is slower when a contest has many active items.

NPZ inputs are also supported. They must contain an array named `R` and may
contain `model_ids` and `item_ids` arrays.

## Interpretation

Flip probability is sensitivity under the item-resampling model. It is **not**
the posterior probability that the reported winner is wrong.

Item attribution is a triage signal for review. A high-magnitude attribution
does not establish that an item is good, bad, representative, or mislabeled,
and items should not be removed or reweighted solely because they decide one
model pair.

## Reproducing the experiments

Run the test suite:

```bash
python -m pytest -q
```

The scripts in `experiments/` reproduce the analyses and write outputs to
`results/`. LiveBench data can be loaded directly through
`src/livebench_io.py`, which uses the Hugging Face `livebench/model_judgment`
dataset.

Raw Open LLM Leaderboard matrices are not redistributed in this repository.
Experiments using them expect long-format CSV files under
`data/raw/benchmark-data/` with columns `source`, `item`, and `correct`. The
`data/raw/` and `data/processed/` directories are intentionally excluded from
version control.

## Repository layout

- `src/`: report generation, attribution, bootstrap, and validation code
- `experiments/`: paper experiment entry points
- `tests/`: confidence, information, calibration, and validation tests
- `results/`: compact generated paper results and example reports
- `figures/`: generated paper figures
- `paper/`: camera-ready LaTeX source and PDF

## Scope

The current implementation assumes binary item scores and analyzes adjacent
rank boundaries. Partial-credit values and all-pairs attribution are future
work. The software is a research prototype and should not be treated as an
automated benchmark-quality verdict.

## Citation

Please use the metadata in `CITATION.cff`. A proceedings DOI will be added after
publication.

## License

The software in this repository is released under the MIT License. The license
applies to this repository's code, not to third-party benchmark datasets; users
must follow the terms of the original data providers.
