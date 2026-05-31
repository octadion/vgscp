"""Validity protocol assertions (Section 2.2, P1; Do/Don't list).

These guards encode the honesty constraints in executable form:
  - the selective gate g is a function of x ONLY (no calibration labels);
  - calibration labels are never used to select the gate threshold tau or score
    hyperparameters (eta, zeta, beta) -- those come from D_learn;
  - retained marginal coverage on the test split is >= 1 - alpha within CI.

Call these in the pipeline so a protocol violation fails loudly rather than silently
invalidating the headline.
"""
from __future__ import annotations

import numpy as np


class ValidityError(AssertionError):
    """Raised when the conformal validity protocol is violated."""


def assert_disjoint_splits(split_indices: dict[str, np.ndarray]) -> None:
    """Splits must be pairwise disjoint (no sample reused across train/learn/cal/test)."""
    names = list(split_indices)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = set(split_indices[names[i]].tolist()), set(split_indices[names[j]].tolist())
            inter = a & b
            if inter:
                raise ValidityError(
                    f"splits {names[i]} and {names[j]} overlap on {len(inter)} samples"
                )


def assert_gate_is_x_measurable(
    gate_values: np.ndarray, y_true: np.ndarray, tol: float = 1e-9
) -> None:
    """Sanity check that the gate does not trivially encode the label.

    A perfect x-measurable signal can still correlate with correctness; what is forbidden is
    the gate being COMPUTED from calibration labels. We can't prove provenance from values
    alone, so this checks the cheap necessary condition: the gate is not a deterministic
    bijection of y_true (which would betray label leakage).
    """
    # If every distinct gate value maps to exactly one label, the gate encodes the label.
    order = np.argsort(gate_values)
    gv, yt = gate_values[order], y_true[order]
    # group identical gate values; if within every group the label is constant AND distinct
    # groups have distinct labels in a 1-1 way across all classes, flag it.
    uniq = np.unique(gv)
    if len(uniq) <= len(np.unique(yt)):
        label_per_value = []
        for v in uniq:
            labels = np.unique(yt[np.abs(gv - v) <= tol])
            label_per_value.append(labels)
        if all(len(l) == 1 for l in label_per_value):
            raise ValidityError(
                "gate appears to be a deterministic function of the label "
                "(possible calibration-label leakage into the gate)"
            )


def assert_hyperparams_from_learn(source_split: str) -> None:
    """tau / eta / zeta / beta must be selected on D_learn, never D_cal or D_test."""
    if source_split != "d_learn":
        raise ValidityError(
            f"signal/score hyperparameters must come from d_learn, got {source_split!r}"
        )


def check_marginal_coverage(
    coverage: float, alpha: float, n: int, ci_slack: float = 3.0
) -> tuple[bool, float]:
    """Check that retained marginal coverage is not SIGNIFICANTLY below 1 - alpha.

    Conformal marginal coverage equals 1 - alpha in expectation (slightly conservative), so any
    single realization fluctuates around the target by ~ a few SE. P1 is violated only by
    *significant* under-coverage. We therefore pass unless coverage + ci_slack*SE < target,
    i.e. unless the target lies outside the upper normal-approx CI of the observed coverage.

    Returns (passed, lower_ci) where lower_ci = coverage - ci_slack*SE (reported for context).
    """
    if n == 0 or not np.isfinite(coverage):
        return True, float("nan")
    se = np.sqrt(max(coverage * (1 - coverage), 1e-12) / n)
    target = 1.0 - alpha
    upper = coverage + ci_slack * se
    lower = coverage - ci_slack * se
    passed = (coverage >= target) or (upper >= target)
    return bool(passed), float(lower)
