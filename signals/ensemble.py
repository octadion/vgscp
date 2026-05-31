"""Deep-ensemble disagreement signal.

Given M independently-seeded ERM models' cached probabilities, the reliability signal is the
NEGATIVE mean pairwise prediction disagreement (or negative predictive variance). Higher =
more reliable (i.e., the ensemble agrees).

Input ``member_probs``: array (M, N, C) of per-member softmax probabilities.
"""
from __future__ import annotations

import numpy as np


def mean_pairwise_disagreement(member_probs: np.ndarray) -> np.ndarray:
    """Fraction of model pairs that predict different labels, per sample. In [0,1]."""
    preds = member_probs.argmax(axis=2)  # (M, N)
    M, N = preds.shape
    if M < 2:
        return np.zeros(N)
    disagree = np.zeros(N, dtype=np.float64)
    n_pairs = 0
    for i in range(M):
        for j in range(i + 1, M):
            disagree += (preds[i] != preds[j]).astype(np.float64)
            n_pairs += 1
    return disagree / max(n_pairs, 1)


def predictive_variance(member_probs: np.ndarray) -> np.ndarray:
    """Mean across classes of the variance of p(y|x) over ensemble members. Per sample."""
    var = member_probs.var(axis=0)  # (N, C)
    return var.mean(axis=1)


def ensemble_disagreement_signal(member_probs: np.ndarray, mode: str = "disagree") -> np.ndarray:
    """Reliability signal (higher = more reliable).

    mode='disagree' -> -mean_pairwise_disagreement
    mode='variance' -> -predictive_variance
    """
    if mode == "disagree":
        return -mean_pairwise_disagreement(member_probs)
    if mode == "variance":
        return -predictive_variance(member_probs)
    raise ValueError(f"unknown ensemble mode: {mode}")


def ensemble_mean_probs(member_probs: np.ndarray) -> np.ndarray:
    """Mean ensemble probability (N, C) — used if the ensemble is also the base predictor."""
    return member_probs.mean(axis=0)
