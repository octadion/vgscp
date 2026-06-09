"""E4 — the LOCKED scacp applicability gate (committed; never tuned to pass).

scacp / score-conditional-attribute-CP is a dead method. The gate decides, PER attribute, whether
the method would even be worth attempting -- it must be simultaneously (i) needed (the predicted-
attribute noise is DIFFERENTIAL, the only regime where a score-conditional correction matters and,
per the KS-1b kill-check, where the simple version breaks), (ii) consequential (a real naive->oracle
minority coverage gap to close), and (iii) estimable (enough minority support). The locked
thresholds are:

    differential-noise diagnostic AUROC >= 0.70   AND
    naive(Â)->oracle(A_true) minority coverage gap >= 0.03   AND
    minority support >= 100.

Expected result of scanning all 312 CUB attributes: 0/312 pass (median diagnostic AUROC ~0.48),
i.e. there is no attribute/setting where the method is both needed and viable. We DO NOT search for
an attribute that passes -- we report the count.

All metrics reuse the repo's conformal primitives (Mondrian quantiles + true-attribute-conditional
coverage) and AUROC. Pure numpy.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from conformal.split_conformal import conformal_quantile
from eval.metrics import auroc

DIFF_AUROC_MIN = 0.70
GAP_MIN = 0.03
SUPPORT_MIN = 100


@dataclass
class AttributeGateResult:
    attr_id: int
    attr_name: str
    minority_support: int
    diff_noise_auroc: float       # AUROC(nonconformity score -> probe error) on the minority
    naive_minority_cov: float     # true-minority coverage under Mondrian-on-Â (naive)
    oracle_minority_cov: float    # true-minority coverage under Mondrian-on-A_true (oracle)
    naive_to_oracle_gap: float    # oracle - naive (room a perfect correction could recover)
    pass_diff: bool
    pass_gap: bool
    pass_support: bool
    passed: bool                  # all three (the locked gate)


def _minority_value(A: np.ndarray) -> int:
    """The scarcer attribute value = the minority group."""
    vals, counts = np.unique(A, return_counts=True)
    return int(vals[np.argmin(counts)])


def gate_attribute(
    s_true_cal: np.ndarray, A_true_cal: np.ndarray, Ahat_cal: np.ndarray,
    s_true_test: np.ndarray, A_true_test: np.ndarray, Ahat_test: np.ndarray,
    alpha: float, attr_id: int = -1, attr_name: str = "",
) -> AttributeGateResult:
    """Compute the three locked-gate metrics for ONE binary attribute and the pass/fail decision.

    Inputs are the TRUE-label nonconformity scores (lower = more conforming) on calibration and test
    splits, plus the true attribute ``A_true`` and the noisy predicted attribute ``Ahat`` on each.
    Coverage is ALWAYS measured on the TRUE-minority subpopulation (true-attribute-conditional).
    """
    minority = _minority_value(A_true_test)
    support = int((A_true_test == minority).sum())

    # ---- naive (Mondrian-on-Â) vs oracle (Mondrian-on-A_true) per-group quantiles from cal ----
    def per_group_q(group_labels):
        return {int(a): conformal_quantile(s_true_cal[group_labels == a], alpha)
                for a in np.unique(group_labels)}
    q_oracle = per_group_q(A_true_cal)     # keyed by true a
    q_naive = per_group_q(Ahat_cal)        # keyed by predicted a

    # quantile applied to each TEST point: oracle uses its true a; naive uses its predicted a.
    q_or_test = np.array([q_oracle.get(int(a), np.inf) for a in A_true_test])
    q_na_test = np.array([q_naive.get(int(a), np.inf) for a in Ahat_test])
    covered_oracle = (s_true_test <= q_or_test).astype(int)
    covered_naive = (s_true_test <= q_na_test).astype(int)

    m = A_true_test == minority
    oracle_cov = float(covered_oracle[m].mean()) if m.any() else float("nan")
    naive_cov = float(covered_naive[m].mean()) if m.any() else float("nan")
    gap = oracle_cov - naive_cov

    # ---- differential-noise diagnostic: DIRECTIONAL AUROC(nonconformity score -> probe error) on
    # the minority. Differential noise means HARD (high-score) samples are the ones the probe
    # mislabels, so AUROC(score -> error) > 0.5; non-differential noise gives ~0.5. We use the
    # directional AUROC (NOT max(a,1-a)) -- only positively-correlated (genuinely differential)
    # noise should fire the gate; anti-correlation is not the differential regime the method targets.
    mc = A_true_cal == minority
    probe_err = (Ahat_cal != A_true_cal).astype(int)
    if mc.sum() > 0 and len(np.unique(probe_err[mc])) > 1:
        diff_auroc = float(auroc(s_true_cal[mc], probe_err[mc]))
    else:
        diff_auroc = float("nan")

    pass_diff = bool(np.isfinite(diff_auroc) and diff_auroc >= DIFF_AUROC_MIN)
    pass_gap = bool(np.isfinite(gap) and gap >= GAP_MIN)
    pass_support = bool(support >= SUPPORT_MIN)
    return AttributeGateResult(
        attr_id=attr_id, attr_name=attr_name, minority_support=support,
        diff_noise_auroc=diff_auroc, naive_minority_cov=naive_cov,
        oracle_minority_cov=oracle_cov, naive_to_oracle_gap=float(gap),
        pass_diff=pass_diff, pass_gap=pass_gap, pass_support=pass_support,
        passed=bool(pass_diff and pass_gap and pass_support))
