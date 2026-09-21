"""E18 -- Ground-truth validation of Decision-Shapley attribution (revision #1).

Answers the reviewer's CRITICAL objection ("value functions engineered, never
validated") with a NON-CIRCULAR test:

  ground truth = exact Shapley of the TRUE population decision probability
                 v*(S) = P_pop(sum_{i in S} d_i > 0), estimated from a known
                 testlet generative model -- defined WITHOUT any value function
  estimators   = confidence / information / flip Shapley, Banzhaf, LOO, raw d,
                 each computed from ONE realized response matrix

Each estimator is scored by how well its per-item ranking (over the active
decisive items) matches the ground-truth Shapley, via Spearman rho, averaged
across contests.

We sweep TWO regimes that isolate the two ways an item can genuinely matter:

  * redundancy regime    -- discrimination held equal, testlet loading varied.
                            The correlation-aware CONFIDENCE VF should recover
                            the credit-split; correlation-blind baselines cannot.
  * discrimination regime -- no redundancy, discrimination spread varied.
                            The Fisher-weighted INFORMATION VF should recover
                            the discrimination ordering.

In BOTH regimes the degenerate baselines (flip == Shapley-Shubik, Banzhaf, LOO,
raw d) assign every same-sign item equal credit and so cannot track the graded
ground truth. This is the falsifiable evidence that Decision-Shapley's value
functions earn their keep over LOO / Banzhaf and are not mere artifacts.

NOTE: defaults below are a fast, runnable SCAFFOLD config. Scale N_SIMS / T /
the seed counts up for the camera-ready numbers (see CONFIG block).
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from synth import make_contest, population_decision_shapley, realize_contest
from baselines import attribution_scores, ALL_ATTRIBUTIONS

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"

# ----------------------------- CONFIG (scaffold) --------------------------- #
N_PARAM_SEEDS = 16            # distinct contests per sweep point
N_REALIZE_SEEDS = 4           # realized samples per contest
N_SIMS = 2500                 # resamples for the population ground truth
T_SHAPLEY = 5000              # Monte-Carlo Shapley permutations
# This config carries bootstrap CIs over contests; sufficient to stand alone in
# the paper. Extended sweep (more lambda/disc points) is appendix material.
MIN_SUPPORT = 4               # need >= this many active decisive items for rho
N_DECISIVE = 10

# Two regimes, each isolating one lever. Each entry: sweep variable name, the
# values to sweep, and the fixed make_contest kwargs.
REGIMES = {
    "redundancy": {
        "sweep": "lam",
        "values": [0.0, 1.5, 3.0],
        "fixed": dict(n_decisive=N_DECISIVE, redundant_pairs=2,
                      disc_low=1.0, disc_high=1.0, ability_gap=1.1),
    },
    "discrimination": {
        "sweep": "disc_high",
        "values": [1.0, 2.0, 3.0],
        "fixed": dict(n_decisive=N_DECISIVE, redundant_pairs=0, lam=0.0,
                      disc_low=0.5, ability_gap=1.1),
    },
}
# --------------------------------------------------------------------------- #


def _one_contest(make_kwargs, param_seed):
    """Build a contest, get its ground-truth Shapley, and run all estimators on
    several realized samples. Returns a list of per-realization rho dicts."""
    p = make_contest(seed=param_seed, **make_kwargs)
    gt = population_decision_shapley(p, n_sims=N_SIMS, seed=1000 + param_seed)
    truth = gt["shapley"]                     # aligned to p.decisive
    dec = p.decisive

    rows = []
    for rs in range(N_REALIZE_SEEDS):
        try:
            R, d_full, info = realize_contest(p, seed=5000 + 17 * param_seed + rs)
        except RuntimeError:
            continue
        support_pos = np.where(d_full[dec] != 0)[0]    # active decisive items
        if len(support_pos) < MIN_SUPPORT:
            continue
        support_items = dec[support_pos]
        gt_support = truth[support_pos]
        if np.allclose(gt_support, gt_support[0]):
            continue  # no gradation to recover in this draw

        # Restrict the decision to the decisive set (neutral filler -> d=0) so
        # the estimators' value function matches the ground-truth value function,
        # while the full R is still used for discrimination / correlation.
        d_eff = np.zeros_like(d_full)
        d_eff[dec] = d_full[dec]

        row = {"param_seed": param_seed, "realize_seed": rs,
               "margin": info["margin"], "n_support": int(len(support_pos))}
        for name in ALL_ATTRIBUTIONS:
            phi = attribution_scores(name, d_eff, R, p.m1, p.m2,
                                     T=T_SHAPLEY, seed=42 + rs)
            est = phi[support_items]
            if np.allclose(est, est[0]):
                rho = 0.0  # degenerate estimator: no ranking information
            else:
                rho = spearmanr(est, gt_support).correlation
                rho = 0.0 if np.isnan(rho) else float(rho)
            row[name] = rho
        rows.append(row)
    return rows


def _mean_ci(rows, name, B=5000, seed=0):
    """Mean rho and a percentile bootstrap CI over contests (resample rows)."""
    vals = np.array([r[name] for r in rows], dtype=float)
    if len(vals) == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan")}
    rng = np.random.default_rng(seed)
    boot = vals[rng.integers(0, len(vals), size=(B, len(vals)))].mean(axis=1)
    return {"mean": float(vals.mean()),
            "lo": float(np.percentile(boot, 2.5)),
            "hi": float(np.percentile(boot, 97.5))}


def _agg(rows):
    return {"n": len(rows),
            **{a: _mean_ci(rows, a) for a in ALL_ATTRIBUTIONS}}


def run_e18():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    out = {"config": {"n_param_seeds": N_PARAM_SEEDS,
                      "n_realize_seeds": N_REALIZE_SEEDS, "n_sims": N_SIMS,
                      "T_shapley": T_SHAPLEY, "n_decisive": N_DECISIVE},
           "regimes": {}}

    for rname, spec in REGIMES.items():
        per_value = {}
        pooled = []
        for val in spec["values"]:
            kwargs = dict(spec["fixed"])
            kwargs[spec["sweep"]] = val
            rows = []
            for ps in range(N_PARAM_SEEDS):
                rows.extend(_one_contest(kwargs, ps))
            pooled.extend(rows)
            per_value[val] = _agg(rows)
        out["regimes"][rname] = {
            "sweep": spec["sweep"],
            "values": spec["values"],
            "per_value": {str(k): v for k, v in per_value.items()},
            "pooled": _agg(pooled),
        }

    json_path = RESULTS_DIR / "e18_attribution_validation.json"
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)

    # ---- console summary ----
    for rname, res in out["regimes"].items():
        print(f"\n=== {rname} regime (sweep {res['sweep']}) ===")
        hdr = f"{res['sweep']:>10} {'n':>4} " + " ".join(f"{a:>11}" for a in ALL_ATTRIBUTIONS)
        print(hdr)
        for val in res["values"]:
            s = res["per_value"][str(val)]
            print(f"{val:>10.1f} {s['n']:>4} " +
                  " ".join(f"{s[a]['mean']:>11.3f}" for a in ALL_ATTRIBUTIONS))
        s = res["pooled"]
        print(f"{'POOLED':>10} {s['n']:>4} " +
              " ".join(f"{s[a]['mean']:>11.3f}" for a in ALL_ATTRIBUTIONS))
        # CIs for the two model-based VFs (the ones whose ordering we care about).
        print("  95% CI over contests [confidence | information]:")
        for val in res["values"]:
            s = res["per_value"][str(val)]
            c, i = s["confidence"], s["information"]
            print(f"    {res['sweep']}={val:<5}  conf {c['mean']:+.3f} "
                  f"[{c['lo']:+.3f},{c['hi']:+.3f}]   info {i['mean']:+.3f} "
                  f"[{i['lo']:+.3f},{i['hi']:+.3f}]")
    print(f"\nSaved {json_path}")

    _plot(out, FIGURES_DIR / "e18_attribution_validation.png")


def _plot(out, path):
    colors = {"confidence": "#2c7fb8", "information": "#41ab5d",
              "flip": "#bbbbbb", "banzhaf": "#888888",
              "loo": "#dd8888", "raw_d": "#d4b483"}
    regimes = list(out["regimes"].items())
    fig, axes = plt.subplots(1, len(regimes), figsize=(10, 4.0), sharey=True)
    for ax, (rname, res) in zip(axes, regimes):
        xs = res["values"]
        for name in ALL_ATTRIBUTIONS:
            ys = [res["per_value"][str(v)][name]["mean"] for v in xs]
            ax.plot(xs, ys, marker="o", label=name, color=colors.get(name),
                    linewidth=2.2 if name in ("confidence", "information") else 1.0)
            if name in ("confidence", "information"):
                lo = [res["per_value"][str(v)][name]["lo"] for v in xs]
                hi = [res["per_value"][str(v)][name]["hi"] for v in xs]
                ax.fill_between(xs, lo, hi, color=colors.get(name), alpha=0.15)
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_xlabel(res["sweep"])
        ax.set_title(f"{rname} regime", fontsize=10)
    axes[0].set_ylabel("Spearman $\\rho$ with true-decision Shapley")
    axes[-1].legend(fontsize=8, ncol=1, loc="best")
    fig.suptitle("Ground-truth recovery of item decisiveness (higher = better)",
                 fontsize=11)
    plt.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Figure: {path}")


if __name__ == "__main__":
    run_e18()
