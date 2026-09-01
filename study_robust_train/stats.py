"""Uncertainty tools that respect the nesting of the design (ACML R2.3, R3).

The submitted paper bootstrapped over flat ``(train_seed, calibration_split)`` cells, which treats
splits as exchangeable with seeds. They are not: splits are *nested within* a trained model, so the
flat bootstrap understates between-seed variance whenever seed-to-seed spread exceeds split-to-split
spread. R2.3 asked for a hierarchical analysis; R3 asked for equivalence tests behind the
"Mondrian is flat across training methods" claim and for uncertainty on the reported correlations.

Everything here is pure numpy over already-computed records — no retraining, no GPU.

  ``cluster_bootstrap_ci``   two-stage bootstrap: resample seeds with replacement, then splits
                             within each drawn seed. Falls back gracefully to the flat bootstrap
                             when only one seed is present (and says so).
  ``paired_cluster_diff_ci`` the same, for a paired difference between two conditions measured on
                             the same (seed, split) cells — the right interval for "does A beat B".
  ``tost_equivalence``       two one-sided tests: is a spread/difference statistically *within* a
                             margin? Turns "the spread is small" into a claim that can be rejected.
  ``correlation_ci``         bootstrap CI for Pearson/Spearman correlation, with an explicit
                             small-n warning (our correlations are over ~4-5 training methods).
"""
from __future__ import annotations

import numpy as np

DEFAULT_B = 2000


def _as_cells(values, seeds, splits=None):
    v = np.asarray(values, dtype=np.float64)
    s = np.asarray(seeds)
    if v.shape[0] != s.shape[0]:
        raise ValueError(f"values ({v.shape[0]}) and seeds ({s.shape[0]}) must align")
    sp = np.arange(v.shape[0]) if splits is None else np.asarray(splits)
    return v, s, sp


def cluster_bootstrap_ci(values, seeds, splits=None, *, stat=np.mean, B: int = DEFAULT_B,
                         alpha: float = 0.05, seed: int = 0) -> dict:
    """Two-stage (hierarchical) bootstrap CI for ``stat`` over seed-clustered observations.

    Stage 1 resamples training seeds with replacement; stage 2 resamples the calibration splits
    within each drawn seed. This propagates model-to-model variance, which the flat bootstrap over
    (seed, split) cells does not.

    Returns ``{point, lo, hi, n_seeds, n_obs, method}``. ``method`` is ``"cluster"``, or
    ``"flat (1 seed)"`` when clustering is impossible — reported, never silently substituted.
    """
    v, s, _ = _as_cells(values, seeds, splits)
    if v.size == 0:
        return {"point": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "n_seeds": 0, "n_obs": 0, "method": "empty"}
    uniq = np.unique(s)
    by_seed = [np.flatnonzero(s == u) for u in uniq]
    rng = np.random.default_rng(seed)
    point = float(stat(v))

    if len(uniq) < 2:                       # cannot resample clusters -> flat, and say so
        draws = [float(stat(v[rng.integers(0, v.size, v.size)])) for _ in range(B)]
        lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        return {"point": point, "lo": float(lo), "hi": float(hi), "n_seeds": int(len(uniq)),
                "n_obs": int(v.size), "method": "flat (1 seed)"}

    draws = np.empty(B)
    for b in range(B):
        pick = rng.integers(0, len(by_seed), len(by_seed))       # stage 1: seeds
        idx = []
        for j in pick:
            grp = by_seed[j]
            idx.append(grp[rng.integers(0, grp.size, grp.size)])  # stage 2: splits within seed
        draws[b] = stat(v[np.concatenate(idx)])
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"point": point, "lo": float(lo), "hi": float(hi), "n_seeds": int(len(uniq)),
            "n_obs": int(v.size), "method": "cluster"}


def paired_cluster_diff_ci(a_values, b_values, seeds, *, B: int = DEFAULT_B, alpha: float = 0.05,
                           seed: int = 0) -> dict:
    """Cluster-bootstrap CI for mean(a) - mean(b) when a and b are measured on the SAME cells.

    Pairing removes split-level noise common to both conditions, which is exactly the situation for
    "same model and split, two calibration policies" comparisons.
    """
    a = np.asarray(a_values, dtype=np.float64)
    b = np.asarray(b_values, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"paired arrays must align: {a.shape} vs {b.shape}")
    out = cluster_bootstrap_ci(a - b, seeds, stat=np.mean, B=B, alpha=alpha, seed=seed)
    out["excludes_zero"] = bool(out["lo"] > 0 or out["hi"] < 0)
    return out


def tost_equivalence(values, seeds, *, margin: float, splits=None, B: int = DEFAULT_B,
                     alpha: float = 0.05, seed: int = 0, center: float = 0.0) -> dict:
    """Two one-sided tests: is the effect statistically *equivalent to* ``center`` within ``margin``?

    A conventional CI that includes 0 only says "we failed to detect a difference". Equivalence is
    the claim we actually want for "Mondrian is flat across training methods", so we test it
    directly: equivalence holds when the whole (1-2*alpha) interval lies inside
    ``[center-margin, center+margin]``.

    Note the interval is the 90% (1-2*alpha) interval by TOST convention, not the 95% one.
    """
    ci = cluster_bootstrap_ci(values, seeds, splits, stat=np.mean, B=B, alpha=2 * alpha, seed=seed)
    lo, hi = ci["lo"], ci["hi"]
    equivalent = bool(lo > center - margin and hi < center + margin)
    return {**ci, "margin": float(margin), "center": float(center), "equivalent": equivalent,
            "conf_level": 1 - 2 * alpha,
            "verdict": "EQUIVALENT" if equivalent else "not equivalent (interval escapes margin)"}


def spread_equivalence(by_method: dict, *, margin: float, B: int = DEFAULT_B, alpha: float = 0.05,
                       seed: int = 0) -> dict:
    """Equivalence test for the cross-method *spread* (max-min) of a per-method quantity.

    ``by_method`` maps method -> (values, seeds). We bootstrap the spread of the per-method means
    (resampling seeds within each method) and declare flatness only if the upper bound of the
    spread's one-sided interval sits below ``margin``. This is the statistical version of the
    paper's "cross-training spread <= 0.024" sentence, which R3 asked to be backed by a test.
    """
    names = sorted(by_method)
    if len(names) < 2:
        return {"spread": float("nan"), "hi": float("nan"), "margin": float(margin),
                "equivalent": False, "verdict": "need >=2 methods", "methods": names}
    prepared = []
    for m in names:
        v, s = by_method[m]
        v = np.asarray(v, dtype=np.float64); s = np.asarray(s)
        prepared.append((v, s, [np.flatnonzero(s == u) for u in np.unique(s)]))
    point = float(np.ptp([v.mean() for v, _, _ in prepared]))

    rng = np.random.default_rng(seed)
    draws = np.empty(B)
    for b in range(B):
        means = []
        for v, _, by_seed in prepared:
            if len(by_seed) < 2:
                means.append(v[rng.integers(0, v.size, v.size)].mean())
                continue
            pick = rng.integers(0, len(by_seed), len(by_seed))
            idx = [by_seed[j][rng.integers(0, by_seed[j].size, by_seed[j].size)] for j in pick]
            means.append(v[np.concatenate(idx)].mean())
        draws[b] = np.ptp(means)
    hi = float(np.percentile(draws, 100 * (1 - alpha)))          # one-sided: spread is >= 0
    return {"spread": point, "hi": hi, "margin": float(margin), "equivalent": bool(hi < margin),
            "conf_level": 1 - alpha, "methods": names,
            "verdict": ("EQUIVALENT (spread bounded below margin)" if hi < margin
                        else "not equivalent (spread upper bound exceeds margin)")}


def correlation_ci(x, y, *, kind: str = "pearson", B: int = DEFAULT_B, alpha: float = 0.05,
                   seed: int = 0) -> dict:
    """Bootstrap CI for a correlation, resampling the (x, y) pairs.

    Our C2 correlations are over the 4-5 *training methods* in a cell, so n is tiny and the interval
    is correspondingly wide. ``small_n`` flags this rather than letting a point estimate of -1.00
    read as precise: with n=4 a Pearson r of -1.00 is entirely compatible with a weak relationship.
    """
    x = np.asarray(x, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"x and y must align: {x.shape} vs {y.shape}")
    n = x.size

    def _r(xa, ya):
        if xa.size < 3 or np.std(xa) == 0 or np.std(ya) == 0:
            return float("nan")
        if kind == "spearman":
            xa = np.argsort(np.argsort(xa)).astype(float)
            ya = np.argsort(np.argsort(ya)).astype(float)
        return float(np.corrcoef(xa, ya)[0, 1])

    point = _r(x, y)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        v = _r(x[idx], y[idx])
        if not np.isnan(v):
            draws.append(v)
    if len(draws) < B // 10:                       # degenerate resamples dominate -> no usable CI
        return {"r": point, "lo": float("nan"), "hi": float("nan"), "n": int(n), "kind": kind,
                "small_n": True, "excludes_zero": False,
                "note": "CI unavailable: too many degenerate resamples at this n"}
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"r": point, "lo": float(lo), "hi": float(hi), "n": int(n), "kind": kind,
            "small_n": bool(n < 8), "excludes_zero": bool(lo > 0 or hi < 0),
            "note": ("n is small; treat as a tendency, not a precise estimate" if n < 8 else "")}


def min_of_k_shortfall(k: int, sd: float) -> float:
    """Expected downward bias of ``min`` over ``k`` noisy group coverages, each with SD ``sd``.

    Why the paper's Mondrian coverage sits at 0.85-0.89 rather than 0.90 (ACML R1.1, R3): per-group
    split conformal is valid in expectation, but the reported statistic is a *minimum* over groups,
    and the minimum of k noisy estimates is biased low. For Gaussian group coverages the bias is
    ``E[min] = mu - c_k * sd`` with c_k the expected-minimum constant (c_4 ~ 1.029).
    Use ``simulate_min_coverage`` for the exact conformal law; this is the closed-form sanity check.
    """
    if k < 2:
        return 0.0
    rng = np.random.default_rng(0)
    return float(-np.mean(np.min(rng.standard_normal((200000, k)), axis=1)) * sd)


def simulate_min_coverage(n_cal, n_test, *, k_groups: int = 4, alpha: float = 0.1,
                          policy: str = "mondrian", n_draws: int = 4000, seed: int = 0) -> dict:
    """Exact-law simulation of worst-group coverage under *perfect* exchangeability.

    No model, no data, no representation: only the conformal quantile's finite-sample law. Any
    shortfall this returns is attributable to the calibration counts and the min-over-groups
    statistic alone, which is what makes it the right null for the paper's sub-target level.

    ``n_cal``/``n_test`` accept either a scalar (all groups equal) or a per-group sequence. Passing
    the *realised* per-group counts matters: at rho=0.95 the minority groups hold a few percent of
    the calibration set, and two distinct effects then push the reported minimum below target.

      1. ``min`` over k noisy per-group coverages is biased low (affects any policy);
      2. under ``mondrian`` each group's threshold is estimated from that group's own -- possibly
         very small -- calibration sample, so the threshold is itself noisy (Mondrian-specific).

    Effect 2 is why Mondrian can sit *below* marginal when minority calibration counts are tiny,
    and why the shortfall shrinks as those counts grow. ``policy="marginal"`` pools the calibration
    scores into one global threshold, isolating effect 1.
    """
    import math
    rng = np.random.default_rng(seed)
    ncal = np.full(k_groups, int(n_cal)) if np.isscalar(n_cal) else np.asarray(n_cal, dtype=int)
    ntest = np.full(k_groups, int(n_test)) if np.isscalar(n_test) else np.asarray(n_test, dtype=int)
    if ncal.size != k_groups or ntest.size != k_groups:
        raise ValueError(f"per-group counts must have length k_groups={k_groups}")
    if policy not in ("mondrian", "marginal"):
        raise ValueError(f"unknown policy {policy!r}")

    def q(s):
        n = s.size
        kk = math.ceil((n + 1) * (1 - alpha))
        return np.inf if kk > n else np.partition(s, kk - 1)[kk - 1]

    mins, means, per_group = [], [], []
    for _ in range(n_draws):
        cal = [rng.random(int(n)) for n in ncal]
        qs = [q(c) for c in cal] if policy == "mondrian" else [q(np.concatenate(cal))] * k_groups
        covs = [float((rng.random(int(ntest[j])) <= qs[j]).mean()) for j in range(k_groups)]
        mins.append(min(covs)); means.append(float(np.mean(covs))); per_group.extend(covs)
    mins = np.asarray(mins)
    sd = float(np.std(per_group))
    return {"policy": policy, "n_cal_per_group": ncal.tolist(), "n_test_per_group": ntest.tolist(),
            "k_groups": int(k_groups), "target": 1 - alpha,
            "mean_over_groups": float(np.mean(means)), "expected_min": float(mins.mean()),
            "min_lo": float(np.percentile(mins, 2.5)), "min_hi": float(np.percentile(mins, 97.5)),
            "shortfall": float((1 - alpha) - mins.mean()), "per_group_sd": sd,
            "shortfall_over_sd": float(((1 - alpha) - mins.mean()) / sd) if sd > 0 else float("nan")}


def group_counts_at_rho(n_total: int, rho: float, k_groups: int = 4) -> np.ndarray:
    """Expected per-group calibration counts for a 4-group (y, a) pool at correlation strength rho.

    Groups are ordered ``2*y + a``; the majority pattern (a aligned with y) holds probability rho.
    With balanced classes each majority group gets ``rho/2`` and each minority group ``(1-rho)/2``
    of the pool -- at rho=0.95 that is 47.5% vs 2.5%, a 19x imbalance in threshold precision.
    """
    if k_groups != 4:
        raise ValueError("group_counts_at_rho is defined for the 4-group (y, a) layout")
    maj, mino = rho / 2.0, (1.0 - rho) / 2.0
    return np.round(np.array([maj, mino, mino, maj]) * n_total).astype(int)
