"""Verifiability-gated selective conformal prediction (Section 2.2 — PRIMARY method).

Given a reliability signal g(x) (higher = more reliable) and an abstention budget b:
  1. tau = quantile_b(g over D_learn)            # threshold from D_learn, x-measurable
  2. retain x iff g(x) >= tau                     # gate uses x only, NOT calibration labels
  3. on retained CALIBRATION points compute qhat for base score s
  4. test set on retained test: C(x) = {y : s(x,y) <= qhat}

Validity (P1): because the gate g is a fixed function of x (no calibration labels) and the
same tau, g, s are applied to retained-cal and retained-test, marginal coverage is valid on
the retained distribution. ``conformal/validity.py`` asserts the protocol.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .split_conformal import build_sets, conformal_quantile, covered, set_sizes


def gate_threshold(g_learn: np.ndarray, budget: float) -> float:
    """tau = the budget-quantile of g over D_learn. Retain the top (1-budget) fraction.

    budget=0 -> retain all (tau = -inf). budget in (0,1) -> abstain the lowest-g fraction.
    """
    if budget <= 0.0:
        return float("-inf")
    if budget >= 1.0:
        return float("inf")
    return float(np.quantile(g_learn, budget, method="lower"))


def retained_mask(g: np.ndarray, tau: float) -> np.ndarray:
    """Retain x iff g(x) >= tau (x-measurable gate)."""
    return g >= tau


@dataclass
class SelectiveConformalResult:
    budget: float
    tau: float
    qhat: float
    n_retained_cal: int
    n_retained_test: int
    retained_test_mask: np.ndarray
    membership: np.ndarray          # (n_retained_test, C)
    coverage: float
    avg_set_size: float
    abstention_rate: float


def selective_conformal(
    g_learn: np.ndarray,
    g_cal: np.ndarray,
    g_test: np.ndarray,
    cal_scores_all: np.ndarray,
    cal_y_true: np.ndarray,
    test_scores_all: np.ndarray,
    test_y_true: np.ndarray,
    alpha: float,
    budget: float,
) -> SelectiveConformalResult:
    """Run the full gated selective-conformal pipeline at one budget."""
    tau = gate_threshold(g_learn, budget)

    cal_keep = retained_mask(g_cal, tau)
    test_keep = retained_mask(g_test, tau)

    cal_true_scores = cal_scores_all[np.arange(cal_scores_all.shape[0]), cal_y_true][cal_keep]
    qhat = conformal_quantile(cal_true_scores, alpha)

    test_scores_kept = test_scores_all[test_keep]
    test_y_kept = test_y_true[test_keep]
    membership = build_sets(test_scores_kept, qhat)

    cov = float(covered(membership, test_y_kept).mean()) if test_keep.any() else float("nan")
    avg_size = float(set_sizes(membership).mean()) if test_keep.any() else float("nan")
    abst = 1.0 - float(test_keep.mean())

    return SelectiveConformalResult(
        budget=budget,
        tau=tau,
        qhat=qhat,
        n_retained_cal=int(cal_keep.sum()),
        n_retained_test=int(test_keep.sum()),
        retained_test_mask=test_keep,
        membership=membership,
        coverage=cov,
        avg_set_size=avg_size,
        abstention_rate=abst,
    )
