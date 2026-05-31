"""Core metrics (Section 9). Pure numpy so the theory testbed and unit tests need no torch.

Conventions:
  - reliability signals are "higher = more reliable" (more likely correct);
  - correctness y in {0,1} with 1 = correct;
  - selective prediction RETAINS the highest-signal fraction (abstains the lowest).
"""
from __future__ import annotations

from typing import Optional

import numpy as np


# --------------------------------------------------------------------------------------
# AUROC (rank-based / Mann-Whitney; handles ties via average ranks)
# --------------------------------------------------------------------------------------
def auroc(score: np.ndarray, label: np.ndarray) -> float:
    """AUROC of ``score`` predicting binary ``label`` (1 = positive). NaN if degenerate."""
    score = np.asarray(score, dtype=np.float64)
    label = np.asarray(label).astype(int)
    pos = label == 1
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=np.float64)
    s_sorted = score[order]
    # average ranks for ties
    ranks_sorted = np.arange(1, len(score) + 1, dtype=np.float64)
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        avg = ranks_sorted[i : j + 1].mean()
        ranks_sorted[i : j + 1] = avg
        i = j + 1
    ranks[order] = ranks_sorted
    sum_pos = ranks[pos].sum()
    return float((sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def error_detection_auroc(signal: np.ndarray, correct: np.ndarray) -> float:
    """AUROC of a reliability signal for detecting CORRECT predictions (higher=more reliable).

    Returned so that a good error detector scores > 0.5. (Equivalent to AUROC for detecting
    errors using the negated signal.)
    """
    return auroc(signal, correct)


# --------------------------------------------------------------------------------------
# Selective risk / risk-coverage / AURC
# --------------------------------------------------------------------------------------
def risk_coverage_curve(
    signal: np.ndarray, correct: np.ndarray, coverages: Optional[np.ndarray] = None
) -> tuple[np.ndarray, np.ndarray]:
    """Risk (error rate) among the top-``c`` fraction by signal, for each coverage c.

    Returns (coverages, risks). Lower risk at a given coverage = better. The risk is
    error = 1 - mean(correct) over the retained (highest-signal) samples.
    """
    n = len(signal)
    err = 1 - np.asarray(correct).astype(float)
    order = np.argsort(-np.asarray(signal, dtype=np.float64), kind="mergesort")  # most reliable first
    err_sorted = err[order]
    cum_err = np.cumsum(err_sorted)
    if coverages is None:
        ks = np.arange(1, n + 1)
        cov = ks / n
        risk = cum_err[ks - 1] / ks
        return cov, risk
    cov = np.asarray(coverages, dtype=np.float64)
    ks = np.clip(np.round(cov * n).astype(int), 1, n)
    risk = cum_err[ks - 1] / ks
    return cov, risk


def aurc(signal: np.ndarray, correct: np.ndarray) -> float:
    """Area Under the Risk-Coverage curve (lower = better)."""
    cov, risk = risk_coverage_curve(signal, correct)
    # trapezoidal integration over coverage in [1/n, 1]
    return float(np.trapz(risk, cov))


def selective_risk_at_budget(signal: np.ndarray, correct: np.ndarray, budget: float) -> float:
    """Error among retained when abstaining the lowest-signal ``budget`` fraction."""
    coverage = 1.0 - budget
    _, risk = risk_coverage_curve(signal, correct, np.array([coverage]))
    return float(risk[0])


# --------------------------------------------------------------------------------------
# Confident-but-wrong capture rate (the key diagnostic)
# --------------------------------------------------------------------------------------
def confident_but_wrong_capture_rate(
    signal: np.ndarray,
    correct: np.ndarray,
    conf: np.ndarray,
    budget: float,
) -> float:
    """Among confident errors {yhat!=y AND conf>=median(conf)}, fraction abstained by ``signal``.

    Abstention at budget b removes the lowest-signal b fraction. By construction confidence
    captures ~0% of these (they are confident). A good signal captures many.
    """
    correct = np.asarray(correct).astype(int)
    cbw = (correct == 0) & (conf >= np.median(conf))
    if cbw.sum() == 0:
        return float("nan")
    n = len(signal)
    k_abstain = int(round(budget * n))
    if k_abstain == 0:
        return 0.0
    abstained = np.zeros(n, dtype=bool)
    order = np.argsort(np.asarray(signal, dtype=np.float64), kind="mergesort")  # lowest first
    abstained[order[:k_abstain]] = True
    return float((abstained & cbw).sum() / cbw.sum())


# --------------------------------------------------------------------------------------
# Contamination metrics (P2)
# --------------------------------------------------------------------------------------
def contamination_auroc(
    signal: np.ndarray,
    spurious_attr: np.ndarray,
    y_true: Optional[np.ndarray] = None,
) -> float:
    """AUROC of a signal predicting the (binary) spurious attribute (P2 contamination).

    HIGH => the signal is contaminated by the shortcut (dependent on the spurious attr). LOW
    (~0.5) => the signal is approximately independent of the spurious attribute. Reported as
    max(auroc, 1-auroc) so direction-agnostic dependence shows as > 0.5.

    When ``y_true`` is given, the AUROC is computed CLASS-CONDITIONALLY (within each true
    class, then sample-weighted averaged). This is the faithful contamination notion: does the
    signal vary with the nuisance attribute while the label is held fixed? In a symmetric
    multi-class model the marginal dependence can cancel even when the conditional dependence
    (the real contamination) is large, so the conditional form is preferred for P2.
    """
    vals = np.unique(spurious_attr)
    if len(vals) != 2:
        return float("nan")
    pos = (spurious_attr == vals[1]).astype(int)

    if y_true is None:
        a = auroc(signal, pos)
        return float(max(a, 1.0 - a)) if not np.isnan(a) else a

    num, den = 0.0, 0
    for c in np.unique(y_true):
        m = y_true == c
        if len(np.unique(pos[m])) < 2:
            continue
        a = auroc(signal[m], pos[m])
        if np.isnan(a):
            continue
        num += max(a, 1.0 - a) * m.sum()
        den += m.sum()
    return float(num / den) if den else float("nan")


def mutual_information(
    signal: np.ndarray,
    spurious_attr: np.ndarray,
    n_bins: int = 16,
    y_true: Optional[np.ndarray] = None,
) -> float:
    """Estimated MI I(signal; spurious_attr) in nats via equal-frequency binning of signal.

    With ``y_true`` given, returns the class-conditional MI I(signal; spurious_attr | y),
    sample-weighted averaged — the contamination notion holding the label fixed (see
    ``contamination_auroc``).
    """
    if y_true is not None:
        num, den = 0.0, 0
        for c in np.unique(y_true):
            m = np.asarray(y_true) == c
            if m.sum() < 2:
                continue
            num += mutual_information(signal[m], spurious_attr[m], n_bins) * m.sum()
            den += m.sum()
        return float(num / den) if den else float("nan")
    signal = np.asarray(signal, dtype=np.float64)
    attr = np.asarray(spurious_attr)
    n = len(signal)
    if n == 0:
        return float("nan")
    # equal-frequency bins (quantile edges) for the continuous signal
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(signal, qs))
    if len(edges) < 2:
        return 0.0
    sig_bin = np.clip(np.digitize(signal, edges[1:-1]), 0, len(edges) - 2)
    attr_vals, attr_idx = np.unique(attr, return_inverse=True)
    n_sb = len(edges) - 1
    n_ab = len(attr_vals)
    joint = np.zeros((n_sb, n_ab), dtype=np.float64)
    for sb, ab in zip(sig_bin, attr_idx):
        joint[sb, ab] += 1
    joint /= n
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    nz = joint > 0
    mi = (joint[nz] * np.log(joint[nz] / (px @ py)[nz])).sum()
    return float(max(mi, 0.0))
