"""Group-conditional (Mondrian) and distribution-robust split-conformal variants.

Both build on ``conformal.split_conformal.conformal_quantile`` (the finite-sample-corrected
k-th-order-statistic threshold) and differ only in HOW the threshold(s) are chosen:

  - Mondrian / group-conditional CP (Vovk 2003): a SEPARATE conformal quantile per group, so
    coverage holds within each group by construction. A test point uses the quantile of its own
    group; groups unseen in calibration fall back to +inf (full set — still valid).

  - TV-robust CP (Cauchois et al. 2020, "Robust Validation: Confident Predictions Even When
    Distributions Shift"; same bound in Barber et al. 2023, "Conformal prediction beyond
    exchangeability"): coverage under a shifted test distribution P_test drops by at most the
    total-variation distance to the calibration distribution,
        cov_{P_test} >= cov_{P_cal} - TV(P_cal, P_test).
    So to retain >= 1-alpha at test within a TV ball of radius eps, calibrate at the inflated
    level 1-alpha+eps, i.e. use miscoverage alpha' = max(alpha - eps, 0). We MATCH eps to the
    OBSERVED shift: the empirical TV distance between the calibration and test true-label score
    distributions (shared-bin histogram). This uses test scores to size the ball, making it a
    deliberately STRONG (oracle-ish, conservative) robust baseline for the de-risk to beat.
"""
from __future__ import annotations

import numpy as np

from .split_conformal import conformal_quantile


def mondrian_quantiles(cal_scores_true: np.ndarray, cal_group: np.ndarray, alpha: float) -> dict:
    """Per-group conformal quantile {group: qhat}. Groups with no cal points are omitted."""
    out = {}
    for g in np.unique(cal_group):
        s = cal_scores_true[cal_group == g]
        out[int(g)] = conformal_quantile(s, alpha)
    return out


def mondrian_build_sets(test_scores_all: np.ndarray, test_group: np.ndarray,
                        group_q: dict) -> np.ndarray:
    """(N, C) membership using each test point's OWN-group quantile (+inf -> full set fallback)."""
    qvec = np.array([group_q.get(int(g), float("inf")) for g in test_group])
    return test_scores_all <= qvec[:, None]


def score_tv_distance(cal_scores: np.ndarray, test_scores: np.ndarray, n_bins: int = 50) -> float:
    """Empirical total-variation distance between two 1-D score distributions (shared bins)."""
    cal_scores = np.asarray(cal_scores, dtype=np.float64)
    test_scores = np.asarray(test_scores, dtype=np.float64)
    if cal_scores.size == 0 or test_scores.size == 0:
        return 0.0
    lo = min(cal_scores.min(), test_scores.min())
    hi = max(cal_scores.max(), test_scores.max())
    if hi <= lo:
        return 0.0
    edges = np.linspace(lo, hi, n_bins + 1)
    pc, _ = np.histogram(cal_scores, bins=edges, density=False)
    pt, _ = np.histogram(test_scores, bins=edges, density=False)
    pc = pc / pc.sum()
    pt = pt / pt.sum()
    return float(0.5 * np.abs(pc - pt).sum())


def robust_quantile(cal_scores_true: np.ndarray, alpha: float, eps: float) -> tuple[float, float]:
    """TV-robust threshold: inflate the level by eps. Returns (qhat, alpha_robust)."""
    alpha_robust = max(alpha - eps, 0.0)
    return conformal_quantile(cal_scores_true, alpha_robust), alpha_robust
