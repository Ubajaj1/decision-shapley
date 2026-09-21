"""Attribution baselines for the ground-truth validation (revision Step #1).

Collects the competing per-item attributions so experiments/e18 can score each
against the value-function-free population pivotality from src/synth.py:

  * confidence  -- correlation-aware decision-probability Shapley   (our primary)
  * information -- Fisher-weighted Shapley                          (our diagnostic)
  * flip        -- flip-indicator Shapley == Shapley-Shubik index   (degenerate)
  * banzhaf     -- Banzhaf swing frequency on the flip game         (this module)
  * loo         -- leave-one-out flip indicator                     (correlation-blind)
  * raw_d       -- the raw per-item differential d_i                (the trivial baseline)

All functions return a full-length (N,) array aligned to item index, with zeros
on inactive (d_i = 0) items, matching the convention of shapley.mc_shapley_*.
The correlation-blind baselines (flip/banzhaf/loo/raw_d) assign every same-sign
item the same score and so cannot recover graded redundancy structure; that is
exactly the gap the validation is built to expose.
"""

from itertools import combinations

import numpy as np

from shapley import (
    mc_shapley_flip,
    mc_shapley_confidence,
    mc_shapley_information,
)
from loo import loo_flips


def _flip_value(s: int) -> float:
    """Flip-indicator value of a coalition with differential sum s."""
    if s > 0:
        return 1.0
    if s < 0:
        return 0.0
    return 0.5


def banzhaf_flip(d: np.ndarray, T: int = 20_000, seed: int = 42,
                 exact_max: int = 18) -> dict:
    """Banzhaf swing frequency for the flip game, per active item.

    Item i's raw Banzhaf score is the probability that, for a uniformly random
    coalition S of the OTHER active items, adding i changes the flip value
    v(S u {i}) != v(S). Exact enumeration is used when the active count is small
    (<= exact_max), otherwise Monte-Carlo with T samples.

    Returns dict with full-length ``phi`` (the swing frequency on active items,
    0 elsewhere), ``n_active``, and ``mode`` ("exact" or "mc").
    """
    d = np.asarray(d)
    n = len(d)
    active = np.where(d != 0)[0]
    da = d[active].astype(int)
    k = len(active)
    swing = np.zeros(k, dtype=np.float64)

    if k == 0:
        return {"phi": np.zeros(n), "n_active": 0, "mode": "empty"}

    if k - 1 <= exact_max:
        mode = "exact"
        for idx in range(k):
            others = np.delete(da, idx)
            di = da[idx]
            total = 0
            count = 0
            for r in range(len(others) + 1):
                for combo in combinations(range(len(others)), r):
                    s = int(others[list(combo)].sum()) if combo else 0
                    if _flip_value(s + di) != _flip_value(s):
                        count += 1
                    total += 1
            swing[idx] = count / total
    else:
        mode = "mc"
        rng = np.random.default_rng(seed)
        for idx in range(k):
            others = np.delete(da, idx)
            di = da[idx]
            # Each other item included independently w.p. 0.5.
            incl = rng.random((T, len(others))) < 0.5
            sums = (incl * others[None, :]).sum(axis=1)
            v_with = np.where(sums + di > 0, 1.0, np.where(sums + di < 0, 0.0, 0.5))
            v_without = np.where(sums > 0, 1.0, np.where(sums < 0, 0.0, 0.5))
            swing[idx] = float(np.mean(v_with != v_without))

    phi = np.zeros(n, dtype=np.float64)
    phi[active] = swing
    return {"phi": phi, "n_active": k, "mode": mode}


def loo_indicator(d: np.ndarray) -> dict:
    """LOO attribution: 1 if removing the item flips or ties the decision.

    Built from loo.loo_flips. Correlation-blind and, at margin 1, degenerate
    (every d=+1 item ties -> all get score 1), which is precisely why it cannot
    rank decisive items by redundancy.
    """
    d = np.asarray(d)
    n = len(d)
    res = loo_flips(d)
    phi = np.zeros(n, dtype=np.float64)
    for i in res["loo_tie_indices"]:
        phi[i] = 1.0
    for i in res["loo_loss_indices"]:
        phi[i] = 1.0
    return {"phi": phi, "n_active": int((d != 0).sum())}


def raw_differential(d: np.ndarray) -> dict:
    """The trivial baseline: attribution = the raw differential d_i."""
    d = np.asarray(d, dtype=np.float64)
    return {"phi": d.copy(), "n_active": int((d != 0).sum())}


def attribution_scores(name: str, d: np.ndarray, R: np.ndarray,
                       m1: int, m2: int, T: int = 20_000, seed: int = 42) -> np.ndarray:
    """Dispatch to a named attribution and return its full-length phi array.

    name in {confidence, information, flip, banzhaf, loo, raw_d}.
    """
    if name == "confidence":
        return mc_shapley_confidence(d, R, m1, m2, T=T, seed=seed)["phi"]
    if name == "information":
        return mc_shapley_information(d, R, m1, m2, T=T, seed=seed)["phi"]
    if name == "flip":
        return mc_shapley_flip(d, T=T, seed=seed)["phi"]
    if name == "banzhaf":
        return banzhaf_flip(d, T=T, seed=seed)["phi"]
    if name == "loo":
        return loo_indicator(d)["phi"]
    if name == "raw_d":
        return raw_differential(d)["phi"]
    raise ValueError(f"unknown attribution: {name!r}")


ALL_ATTRIBUTIONS = ["confidence", "information", "flip", "banzhaf", "loo", "raw_d"]
