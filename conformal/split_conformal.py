"""Split (inductive) conformal prediction.

Given calibration true-label scores, compute the conformal quantile

    qhat = the ceil((n+1)(1-alpha)) / n - th smallest of the calibration scores
         = np.quantile(cal_scores, ceil((n+1)(1-alpha))/n, method='higher')

and build prediction sets on test as ``{y : s(x,y) <= qhat}``. This guarantees marginal
coverage >= 1 - alpha when (cal, test) are exchangeable.
"""
from __future__ import annotations

import math

import numpy as np


def conformal_quantile(cal_scores: np.ndarray, alpha: float) -> float:
    """Finite-sample-corrected split-conformal quantile of calibration scores.

    qhat = the k-th smallest calibration score with k = ceil((n+1)(1-alpha)). This is the exact
    inductive-conformal threshold guaranteeing marginal coverage >= 1-alpha under exchangeability.
    Returns +inf when k > n (n too small for the level => full sets, still valid).
    """
    n = cal_scores.shape[0]
    if n == 0:
        return float("inf")
    k = math.ceil((n + 1) * (1.0 - alpha))
    if k > n:
        return float("inf")  # cannot achieve the level with n points -> full sets
    # k-th smallest (1-indexed) == 0-indexed (k-1) order statistic.
    return float(np.partition(cal_scores, k - 1)[k - 1])


def build_sets(test_scores_all: np.ndarray, qhat: float) -> np.ndarray:
    """Boolean membership matrix (N, C): True where s(x,y) <= qhat."""
    return test_scores_all <= qhat


def set_sizes(membership: np.ndarray) -> np.ndarray:
    return membership.sum(axis=1)


def covered(membership: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """1 if y_true is in the predicted set."""
    return membership[np.arange(membership.shape[0]), y_true].astype(np.int64)


def marginal_coverage(membership: np.ndarray, y_true: np.ndarray) -> float:
    return float(covered(membership, y_true).mean())


def empirical_coverage_by_group(
    membership: np.ndarray, y_true: np.ndarray, group_id: np.ndarray
) -> dict:
    cov = covered(membership, y_true)
    return {int(g): float(cov[group_id == g].mean()) for g in np.unique(group_id)}
