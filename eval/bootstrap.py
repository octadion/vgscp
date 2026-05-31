"""Bootstrap confidence intervals + paired significance tests (Sections 9, 15).

All headline numbers are reported with 95% bootstrap CIs (>=1000 resamples). Comparisons vs
V_full use a PAIRED bootstrap (resample sample indices once, recompute both metrics on the same
resample) and a Holm correction across the family of baselines.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


@dataclass
class CI:
    estimate: float
    lo: float
    hi: float
    se: float

    def as_tuple(self):
        return (self.estimate, self.lo, self.hi)


def bootstrap_ci(
    metric_fn: Callable[..., float],
    *arrays: np.ndarray,
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
) -> CI:
    """Percentile bootstrap CI for a metric computed on aligned per-sample arrays."""
    rng = np.random.default_rng(seed)
    n = len(arrays[0])
    point = metric_fn(*arrays)
    stats = np.empty(n_resamples, dtype=np.float64)
    for b in range(n_resamples):
        idx = rng.integers(0, n, n)
        stats[b] = metric_fn(*[a[idx] for a in arrays])
    stats = stats[np.isfinite(stats)]
    alpha = 1.0 - ci
    lo = float(np.quantile(stats, alpha / 2)) if len(stats) else float("nan")
    hi = float(np.quantile(stats, 1 - alpha / 2)) if len(stats) else float("nan")
    se = float(stats.std(ddof=1)) if len(stats) > 1 else float("nan")
    return CI(estimate=float(point), lo=lo, hi=hi, se=se)


@dataclass
class PairedTest:
    name_a: str
    name_b: str
    delta: float          # metric(a) - metric(b), point estimate
    delta_lo: float
    delta_hi: float
    p_value: float        # two-sided bootstrap p for delta != 0
    favored: str          # which side the CI favors, or 'tie'


def paired_bootstrap_test(
    metric_fn: Callable[..., float],
    arrays_a: tuple,
    arrays_b: tuple,
    name_a: str,
    name_b: str,
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
) -> PairedTest:
    """Paired bootstrap of delta = metric(a) - metric(b) on the SAME resampled indices.

    arrays_a / arrays_b are the per-sample argument tuples for each method's metric. They must
    share the same length and sample alignment.
    """
    rng = np.random.default_rng(seed)
    n = len(arrays_a[0])
    point = metric_fn(*arrays_a) - metric_fn(*arrays_b)
    deltas = np.empty(n_resamples, dtype=np.float64)
    for b in range(n_resamples):
        idx = rng.integers(0, n, n)
        da = metric_fn(*[a[idx] for a in arrays_a])
        db = metric_fn(*[a[idx] for a in arrays_b])
        deltas[b] = da - db
    deltas = deltas[np.isfinite(deltas)]
    alpha = 1.0 - ci
    lo = float(np.quantile(deltas, alpha / 2))
    hi = float(np.quantile(deltas, 1 - alpha / 2))
    # two-sided p: fraction of resamples on the opposite side of 0, doubled
    p = 2.0 * min((deltas <= 0).mean(), (deltas >= 0).mean())
    p = float(min(p, 1.0))
    favored = name_a if lo > 0 else name_b if hi < 0 else "tie"
    return PairedTest(name_a, name_b, float(point), lo, hi, p, favored)


def holm_correction(pvalues: dict[str, float], alpha: float = 0.05) -> dict[str, bool]:
    """Holm-Bonferroni step-down. Returns {key: reject_null} controlling FWER at alpha."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    reject = {}
    prev_reject = True
    for rank, (key, p) in enumerate(items):
        thresh = alpha / (m - rank)
        decision = prev_reject and (p <= thresh)
        reject[key] = decision
        prev_reject = decision
    return reject
