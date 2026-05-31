"""Conformal nonconformity scores: THR, APS, RAPS.

Each ``*_scores_all(probs, ...)`` returns an (N, C) array giving the nonconformity score
s(x, y) for EVERY label y. Convention: LOWER score = more conforming, so the prediction set
is ``{y : s(x,y) <= qhat}`` and the calibration score is ``s(x, y_true)``.

Scores:
  THR  (a.k.a. LAC / HPS):  s(x,y) = 1 - p(y|x)
  APS  (Romano et al. 2020): cumulative prob mass of labels at least as likely as y, with
       optional uniform randomization for exact coverage.
  RAPS (Angelopoulos et al. 2021): APS + regularization lam_reg * (rank(y) - k_reg)_+ that
       discourages large sets.

Randomization (u): pass a per-sample uniform array u in [0,1) for exact (non-conservative)
coverage; pass None for the deterministic (slightly conservative) variant. The SAME u must be
used for calibration and test of a given sample-set to preserve validity — callers draw u once
per split with a fixed seed and cache it.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def thr_scores_all(probs: np.ndarray) -> np.ndarray:
    """THR / LAC: s(x,y) = 1 - p(y|x)."""
    return 1.0 - probs


def _aps_like_scores(
    probs: np.ndarray,
    u: Optional[np.ndarray],
    lam_reg: float,
    k_reg: int,
) -> np.ndarray:
    """Shared APS/RAPS machinery. lam_reg=0 -> APS; lam_reg>0 -> RAPS.

    For each sample, sort labels by descending probability. The score of label y is the
    cumulative probability of all labels strictly more probable than y, plus (optionally
    randomized) the probability of y itself, plus the RAPS rank penalty.
    """
    n, c = probs.shape
    order = np.argsort(-probs, axis=1)  # (N, C) descending prob; ties broken by index
    sorted_p = np.take_along_axis(probs, order, axis=1)  # (N, C)
    cum = np.cumsum(sorted_p, axis=1)  # cumulative including current rank
    cum_before = cum - sorted_p  # cumulative strictly before current rank

    # randomization term on the current label's own mass
    if u is None:
        rand_term = sorted_p  # deterministic: include full mass of y
    else:
        rand_term = u[:, None] * sorted_p

    # RAPS regularization: penalty for rank index k (0-based) = lam * max(k+1 - k_reg, 0)
    ranks = np.arange(1, c + 1)[None, :]  # 1-based rank
    penalty = lam_reg * np.maximum(ranks - k_reg, 0)

    scores_sorted = cum_before + rand_term + penalty  # (N, C) in sorted order

    # scatter back to label order
    scores = np.empty_like(scores_sorted)
    np.put_along_axis(scores, order, scores_sorted, axis=1)
    return scores


def aps_scores_all(probs: np.ndarray, u: Optional[np.ndarray] = None) -> np.ndarray:
    """APS nonconformity score for every label."""
    return _aps_like_scores(probs, u, lam_reg=0.0, k_reg=1)


def raps_scores_all(
    probs: np.ndarray,
    u: Optional[np.ndarray] = None,
    lam_reg: float = 0.01,
    k_reg: int = 1,
) -> np.ndarray:
    """RAPS nonconformity score for every label."""
    return _aps_like_scores(probs, u, lam_reg=lam_reg, k_reg=k_reg)


SCORE_FNS = {
    "THR": lambda probs, u=None, **kw: thr_scores_all(probs),
    "APS": lambda probs, u=None, **kw: aps_scores_all(probs, u),
    "RAPS": lambda probs, u=None, lam_reg=0.01, k_reg=1, **kw: raps_scores_all(
        probs, u, lam_reg=lam_reg, k_reg=k_reg
    ),
}


def scores_all(name: str, probs: np.ndarray, u: Optional[np.ndarray] = None, **kw) -> np.ndarray:
    """Dispatch to a named score family."""
    if name not in SCORE_FNS:
        raise ValueError(f"unknown score {name!r}; choose from {list(SCORE_FNS)}")
    return SCORE_FNS[name](probs, u=u, **kw)


def true_label_scores(scores_all_arr: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """Pick s(x, y_true) from an (N, C) score array."""
    return scores_all_arr[np.arange(scores_all_arr.shape[0]), y_true]


def draw_randomization(n: int, seed: int) -> np.ndarray:
    """Per-sample uniform randomization, drawn once with a fixed seed and cached."""
    rng = np.random.default_rng(seed)
    return rng.random(n)
