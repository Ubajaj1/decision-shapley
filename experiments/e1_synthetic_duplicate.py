"""E1 — Synthetic duplicate demo (validates Gap 1).

Construct a tiny matrix with a planted duplicate decisive item.
Compute LOO, margin-Shapley, flip-Shapley. Show:
  - LOO and margin-Shapley mark duplicates as "unimportant" (same as non-duplicated)
  - Flip-Shapley splits credit across the duplicate pair

This is a headline figure validating the core methodological choice.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import json
import numpy as np
import matplotlib.pyplot as plt
from shapley import mc_shapley_flip, mc_shapley_margin

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"

T = 100_000
SEED = 42


def make_synthetic_d(n_unique_plus: int = 5, n_minus: int = 3,
                     n_zero: int = 10, dup_index: int = 0,
                     n_duplicates: int = 3) -> tuple[np.ndarray, dict]:
    """Build a synthetic d vector with planted duplicates.

    Args:
        n_unique_plus: number of unique d=+1 items
        n_minus: number of d=-1 items
        n_zero: number of d=0 items
        dup_index: which +1 item to duplicate
        n_duplicates: how many copies of the duplicated item (total, including original)

    Returns (d, metadata).
    """
    parts = []
    labels = []

    # Unique +1 items
    for i in range(n_unique_plus):
        if i == dup_index:
            # Plant duplicates
            for k in range(n_duplicates):
                parts.append(1)
                labels.append(f"dup_{k}" if k > 0 else "original")
        else:
            parts.append(1)
            labels.append(f"unique_plus_{i}")

    # -1 items
    for i in range(n_minus):
        parts.append(-1)
        labels.append(f"minus_{i}")

    # 0 items
    for i in range(n_zero):
        parts.append(0)
        labels.append(f"zero_{i}")

    d = np.array(parts, dtype=np.int64)
    margin = int(d.sum())

    meta = {
        "n_total": len(d),
        "n_unique_plus": n_unique_plus,
        "n_minus": n_minus,
        "n_zero": n_zero,
        "n_duplicates": n_duplicates,
        "dup_index": dup_index,
        "margin": margin,
        "labels": labels,
    }
    return d, meta


def compute_loo_values(d: np.ndarray) -> np.ndarray:
    """LOO 'importance': does removing item i change the decision?

    For binary decision, LOO value = 1 if removing i causes a flip/tie, else 0.
    Also compute LOO margin contribution = d_i (the item's marginal effect).
    """
    margin = d.sum()
    loo_flips = np.zeros(len(d), dtype=np.float64)
    for i in range(len(d)):
        new_margin = margin - d[i]
        if new_margin <= 0 and margin > 0:
            loo_flips[i] = 1.0
    return loo_flips


def run_e1():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Scenario: 5 unique +1, 3 -1, 10 zeros. Margin = 5-3 = 2.
    # Duplicate item 0 three times → now have 7 +1 items, 3 -1, margin = 4.
    # But the "information" hasn't changed — one item was just copied.
    d_no_dup, meta_no_dup = make_synthetic_d(
        n_unique_plus=5, n_minus=3, n_zero=10, dup_index=0, n_duplicates=1
    )
    d_with_dup, meta_with_dup = make_synthetic_d(
        n_unique_plus=5, n_minus=3, n_zero=10, dup_index=0, n_duplicates=3
    )

    print("="*60)
    print("  Scenario WITHOUT duplicates")
    print("="*60)
    print(f"  d = {d_no_dup.tolist()}")
    print(f"  labels = {meta_no_dup['labels']}")
    print(f"  margin = {meta_no_dup['margin']}")
    _analyze(d_no_dup, meta_no_dup, "no_dup")

    print("\n" + "="*60)
    print("  Scenario WITH duplicates (item 0 copied 3×)")
    print("="*60)
    print(f"  d = {d_with_dup.tolist()}")
    print(f"  labels = {meta_with_dup['labels']}")
    print(f"  margin = {meta_with_dup['margin']}")
    results_dup = _analyze(d_with_dup, meta_with_dup, "with_dup")

    # Also run a tight-margin case: margin=1 to show flip-Shapley peaks
    print("\n" + "="*60)
    print("  Tight-margin scenario (margin=1, with 3× duplicate)")
    print("="*60)
    d_tight, meta_tight = make_synthetic_d(
        n_unique_plus=3, n_minus=2, n_zero=5, dup_index=0, n_duplicates=3
    )
    # margin = 3+2(dups) - 2 = 3, need tighter: use 2 unique + 3 minus
    d_tight, meta_tight = make_synthetic_d(
        n_unique_plus=2, n_minus=3, n_zero=5, dup_index=0, n_duplicates=3
    )
    # d: [1, 1, 1, 1, -1, -1, -1, 0, 0, 0, 0, 0] → margin = 4-3 = 1
    print(f"  d = {d_tight.tolist()}")
    print(f"  labels = {meta_tight['labels']}")
    print(f"  margin = {meta_tight['margin']}")
    _analyze(d_tight, meta_tight, "tight")

    # Generate the headline comparison figure
    _plot_comparison(d_with_dup, meta_with_dup, d_tight, meta_tight)

    return results_dup


def _analyze(d: np.ndarray, meta: dict, tag: str) -> dict:
    labels = meta["labels"]

    # LOO
    loo_vals = compute_loo_values(d)

    # Margin-Shapley
    margin_shap = mc_shapley_margin(d, T=T, seed=SEED)

    # Flip-Shapley
    flip_shap = mc_shapley_flip(d, T=T, seed=SEED)

    print(f"\n  {'Item':<20s} {'d_i':>4} {'LOO':>6} {'φ_margin':>10} {'φ_flip':>10}")
    print(f"  {'-'*52}")
    for i, label in enumerate(labels):
        if d[i] == 0 and i > 0 and labels[i-1].startswith("zero"):
            continue  # skip boring zeros after first
        print(f"  {label:<20s} {d[i]:>4} {loo_vals[i]:>6.3f} "
              f"{margin_shap['phi'][i]:>10.6f} {flip_shap['phi'][i]:>10.6f}")
    if meta["n_zero"] > 1:
        print(f"  {'(... zeros omitted)':<20s}")

    print(f"\n  Efficiency check:")
    print(f"    Margin-Shapley Σφ = {margin_shap['efficiency_sum']:.6f} (expect {meta['margin']})")
    print(f"    Flip-Shapley   Σφ = {flip_shap['efficiency_sum']:.6f} (expect 0.5)")

    result = {
        "tag": tag,
        "d": d.tolist(),
        "labels": labels,
        "margin": meta["margin"],
        "loo": loo_vals.tolist(),
        "phi_margin": margin_shap["phi"].tolist(),
        "phi_flip": flip_shap["phi"].tolist(),
    }

    report_path = RESULTS_DIR / f"e1_{tag}.json"
    with open(report_path, "w") as f:
        json.dump(result, f, indent=2)

    return result


def _plot_comparison(d_dup, meta_dup, d_tight, meta_tight):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, d, meta, title in [
        (axes[0], d_dup, meta_dup, f"Margin={int(d_dup.sum())}: Duplicated item"),
        (axes[1], d_tight, meta_tight, f"Margin={int(d_tight.sum())}: Tight + duplicated"),
    ]:
        labels = meta["labels"]
        active = [i for i in range(len(d)) if d[i] != 0]
        active_labels = [labels[i] for i in active]

        loo = compute_loo_values(d)
        margin_shap = mc_shapley_margin(d, T=T, seed=SEED)
        flip_shap = mc_shapley_flip(d, T=T, seed=SEED)

        x = np.arange(len(active))
        w = 0.25

        # Normalize margin-Shapley to [0, max_flip] for visual comparison
        phi_m = np.array([margin_shap["phi"][i] for i in active])
        phi_f = np.array([flip_shap["phi"][i] for i in active])

        ax.bar(x - w, [d[i] for i in active], w, label="d_i", color="#AAAAAA", alpha=0.5)
        ax.bar(x, phi_m, w, label="φ margin (= d_i)", color="#DD8452")
        ax.bar(x + w, phi_f, w, label="φ flip", color="#4C72B0")

        ax.set_xticks(x)
        ax.set_xticklabels(active_labels, rotation=45, ha="right", fontsize=8)
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.set_ylabel("Value", fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=8, loc="upper right")

        # Highlight duplicates
        for i, idx in enumerate(active):
            if "dup" in labels[idx] or labels[idx] == "original":
                ax.axvspan(i - 0.4, i + 0.4, alpha=0.1, color="yellow")

    plt.suptitle("E1: LOO/Margin-Shapley vs Flip-Shapley on Duplicated Items", fontsize=13)
    plt.tight_layout()
    path = FIGURES_DIR / "e1_duplicate_comparison.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved to {path}")


if __name__ == "__main__":
    run_e1()
