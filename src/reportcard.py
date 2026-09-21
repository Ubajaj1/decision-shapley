"""Generate decision-robustness report cards from binary response matrices."""

import json
from pathlib import Path

import numpy as np

from decision import top_two, model_scores
from bootstrap import flip_probability_vectorized
from shapley import mc_shapley_confidence, mc_shapley_information
from taxonomy import classify_items
from concentration import concentration_metrics
from loo import loo_flips, decisive_item_count


def generate_report(R: np.ndarray, model_ids: list[str],
                    item_ids: list = None,
                    benchmark_name: str = "unknown",
                    B: int = 10_000, T: int = 10_000,
                    seed: int = 42, rank1: int = 1,
                    value_function: str = "confidence",
                    include_alternative: bool = False) -> dict:
    """Generate a full decision-robustness report for a result matrix.

    Args:
        R: binary response matrix [n_models, n_items]
        model_ids: list of model identifiers
        item_ids: optional list of item identifiers
        benchmark_name: name for the report
        B: bootstrap replicates
        T: Shapley permutations
        seed: RNG seed
        rank1: which rank boundary to analyze (1 = #1 vs #2)
        value_function: primary attribution, either ``confidence`` or
            ``information``.
        include_alternative: also compute the other value function. This is
            useful for comparison but can be slow for large active item sets.
    """
    R = np.asarray(R)
    if R.ndim != 2:
        raise ValueError("R must be a two-dimensional [models, items] matrix.")
    n_models, n_items = R.shape
    if n_models < 2 or n_items < 1:
        raise ValueError("R must contain at least two models and one item.")
    if len(model_ids) != n_models:
        raise ValueError("model_ids length must match the number of rows in R.")
    if np.isnan(R.astype(float)).any() or not np.isin(R, [0, 1]).all():
        raise ValueError("R must contain only binary 0/1 values and no missing data.")
    if value_function not in {"confidence", "information"}:
        raise ValueError("value_function must be 'confidence' or 'information'.")

    if item_ids is None:
        item_ids = list(range(n_items))
    elif len(item_ids) != n_items:
        raise ValueError("item_ids length must match the number of columns in R.")

    contest = top_two(R, model_ids, rank1=rank1)
    d = contest["d"]

    bp = flip_probability_vectorized(d, B=B, seed=seed)
    loo = loo_flips(d)
    dec_count = decisive_item_count(d)
    attribution_functions = {
        "confidence": mc_shapley_confidence,
        "information": mc_shapley_information,
    }
    requested = [value_function]
    if include_alternative:
        requested.append("information" if value_function == "confidence" else "confidence")
    attributions = {
        name: attribution_functions[name](
            d, R, contest["m1_idx"], contest["m2_idx"], T=T, seed=seed,
        )
        for name in requested
    }
    primary = attributions[value_function]
    confidence = attributions.get("confidence")
    information = attributions.get("information")
    phi = primary["phi"]
    tax = classify_items(phi, d)
    conc = concentration_metrics(phi)

    # Build decisive item list with details
    decisive_items = []
    order = np.argsort(-phi)
    for idx in order:
        if phi[idx] <= 0:
            break
        decisive_items.append({
            "item_id": item_ids[idx] if idx < len(item_ids) else int(idx),
            "phi": round(float(phi[idx]), 6),
            "d_i": int(d[idx]),
            "se": round(float(primary["se"][idx]), 6),
        })
        if confidence is not None:
            decisive_items[-1]["phi_confidence"] = round(float(confidence["phi"][idx]), 6)
        if information is not None:
            decisive_items[-1].update({
                "phi_information": round(float(information["phi"][idx]), 6),
                "discrimination": round(float(information["discrimination"][idx]), 6),
                "fisher_information": round(float(information["J"][idx]), 6),
            })

    counter_items = []
    for idx in np.argsort(phi):
        if phi[idx] >= 0:
            break
        counter_items.append({
            "item_id": item_ids[idx] if idx < len(item_ids) else int(idx),
            "phi": round(float(phi[idx]), 6),
            "d_i": int(d[idx]),
        })
        if confidence is not None:
            counter_items[-1]["phi_confidence"] = round(float(confidence["phi"][idx]), 6)
        if information is not None:
            counter_items[-1]["phi_information"] = round(float(information["phi"][idx]), 6)

    report = {
        "benchmark": benchmark_name,
        "n_models": n_models,
        "n_items": n_items,
        "contest": {
            "m1": contest["m1_id"],
            "m2": contest["m2_id"],
            "m1_score": round(contest["m1_score"], 6),
            "m2_score": round(contest["m2_score"], 6),
            "margin": contest["margin"],
            "n_plus": contest["n_plus"],
            "n_minus": contest["n_minus"],
            "n_zero": contest["n_zero"],
        },
        "flip_probability": {
            "flip_or_tie": round(bp["flip_or_tie_prob"], 6),
            "flip_strict": round(bp["flip_prob"], 6),
            "tie": round(bp["tie_prob"], 6),
            "ci_95": [round(bp["ci_lo"], 6), round(bp["ci_hi"], 6)],
            "B": B,
        },
        "loo": {
            "vulnerable": loo["loo_vulnerable"],
            "n_loo_flip_items": loo["n_loo_any"],
        },
        "shapley": {
            "primary_value_function": value_function,
            "computed_value_functions": requested,
            "confidence": None,
            "information": None,
        },
        "taxonomy": {
            "n_decisive": tax["n_decisive"],
            "n_counter_evidential": tax["n_counter"],
            "n_redundant": tax["n_redundant"],
        },
        "concentration": {
            "gini_positive": round(conc["gini_positive"], 4),
            "top_k_50_pct": conc["top_k_50"],
            "top_k_80_pct": conc["top_k_80"],
            "decisive_item_count_min_removals": dec_count,
        },
        "decisive_items": decisive_items[:20],
        "counter_evidential_items": counter_items[:10],
        "seed": seed,
    }

    if confidence is not None:
        report["shapley"]["confidence"] = {
            "T": T,
            "v_all": round(confidence["v_all"], 6),
            "efficiency_sum": round(confidence["efficiency_sum"], 6),
            "max_se": round(confidence["max_se"], 6),
        }
    if information is not None:
        report["shapley"]["information"] = {
            "T": T,
            "v_all": round(information["v_all"], 6),
            "efficiency_sum": round(information["efficiency_sum"], 6),
            "max_se": round(information["max_se"], 6),
            "correlation_shrinkage": round(information["delta"], 6),
        }

    return report


def report_to_json(report: dict, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(report, f, indent=2)


def report_to_html(report: dict, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    c = report["contest"]
    fp = report["flip_probability"]
    tax = report["taxonomy"]
    conc = report["concentration"]
    computed = report["shapley"]["computed_value_functions"]

    attribution_headers = "".join(
        f"<th>&phi; {name}</th>" for name in computed
    )

    def attribution_cells(item):
        return "".join(
            f"<td>{item[f'phi_{name}']:.6f}</td>" for name in computed
        )

    decisive_rows = ""
    for item in report["decisive_items"]:
        decisive_rows += (
            f"<tr><td>{item['item_id']}</td><td>{item['d_i']:+d}</td>"
            f"{attribution_cells(item)}</tr>\n"
        )

    counter_rows = ""
    for item in report["counter_evidential_items"]:
        counter_rows += (
            f"<tr><td>{item['item_id']}</td><td>{item['d_i']:+d}</td>"
            f"{attribution_cells(item)}</tr>\n"
        )

    stability_class = "stable"
    if fp["flip_or_tie"] >= 0.05:
        stability_class = "fragile"
    elif fp["flip_or_tie"] >= 0.01:
        stability_class = "borderline"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Decision Robustness Report: {report['benchmark']}</title>
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 800px; margin: 2em auto; padding: 0 1em; }}
  h1 {{ border-bottom: 2px solid #333; padding-bottom: 0.3em; }}
  .metric {{ display: inline-block; background: #f5f5f5; border-radius: 8px;
             padding: 1em; margin: 0.5em; min-width: 140px; text-align: center; }}
  .metric .value {{ font-size: 1.8em; font-weight: bold; }}
  .metric .label {{ font-size: 0.85em; color: #666; }}
  .stable {{ color: #2e7d32; }}
  .borderline {{ color: #f57c00; }}
  .fragile {{ color: #c62828; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1em 0; }}
  th, td {{ border: 1px solid #ddd; padding: 6px 10px; text-align: right; }}
  th {{ background: #f5f5f5; text-align: left; }}
  .section {{ margin: 2em 0; }}
</style>
</head>
<body>
<h1>Decision Robustness Report Card</h1>
<p><strong>Benchmark:</strong> {report['benchmark']} &nbsp;|&nbsp;
   <strong>Models:</strong> {report['n_models']} &nbsp;|&nbsp;
   <strong>Items:</strong> {report['n_items']} &nbsp;|&nbsp;
   <strong>Primary attribution:</strong> {report['shapley']['primary_value_function']}</p>

<div class="section">
<h2>Contest: #1 vs #2</h2>
<p><strong>#1:</strong> {c['m1']} (score: {c['m1_score']:.4f})<br>
   <strong>#2:</strong> {c['m2']} (score: {c['m2_score']:.4f})</p>
<div>
  <div class="metric">
    <div class="value">{c['margin']}</div>
    <div class="label">Item margin</div>
  </div>
  <div class="metric">
    <div class="value {stability_class}">{fp['flip_or_tie']:.1%}</div>
    <div class="label">Flip probability</div>
  </div>
  <div class="metric">
    <div class="value">{conc['decisive_item_count_min_removals']}</div>
    <div class="label">Min removals to flip</div>
  </div>
  <div class="metric">
    <div class="value">{conc['gini_positive']:.3f}</div>
    <div class="label">Gini (positive &phi;)</div>
  </div>
</div>
<p>95% CI for flip-or-tie: [{fp['ci_95'][0]:.4f}, {fp['ci_95'][1]:.4f}]</p>
</div>

<div class="section">
<h2>Item Taxonomy</h2>
<p><strong>{tax['n_decisive']}</strong> decisive &nbsp;|&nbsp;
   <strong>{tax['n_counter_evidential']}</strong> counter-evidential &nbsp;|&nbsp;
   <strong>{tax['n_redundant']}</strong> redundant</p>
</div>

<div class="section">
<h2>Top Decisive Items</h2>
<table>
<tr><th>Item ID</th><th>d_i</th>{attribution_headers}</tr>
{decisive_rows}
</table>
</div>

{"<div class='section'><h2>Counter-Evidential Items</h2><table><tr><th>Item ID</th><th>d_i</th>" + attribution_headers + "</tr>" + counter_rows + "</table></div>" if counter_rows else ""}

<div class="section">
<h2>Interpretation</h2>
<p>Flip probability is sensitivity under the item-resampling model, not the
probability that the reported winner is wrong. Item attribution is a review
priority signal, not evidence that an item is invalid or mislabeled.</p>
</div>

<div class="section" style="font-size: 0.85em; color: #888;">
<p>Generated with decision-robustness report card tool.
   B={fp['B']}, T={report['shapley'][report['shapley']['primary_value_function']]['T']}, seed={report['seed']}.</p>
</div>
</body>
</html>"""

    path.write_text(html)
