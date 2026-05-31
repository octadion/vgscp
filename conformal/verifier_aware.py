"""Verifier-aware nonconformity score (Section 2.3 — SECONDARY method / ablation).

    s_V(x, y) = s_base(x, y) + eta * (1 - p_A(y | S_M)) + zeta * R_adv(x, y)

with eta, zeta fixed on D_learn; qhat computed on D_cal with s_V. The claim here is *set
trustworthiness* (in-set labels are concept-supported), NOT a new coverage guarantee — but
marginal coverage must still be ~ 1 - alpha, which the validity check verifies.

p_A(y|S_M) is Arthur's per-label competence given Merlin's support set; R_adv(x,y) is the
per-label adversarial vulnerability. Both are cached (N, C) arrays from the NCV adapter.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def verifier_aware_scores_all(
    s_base_all: np.ndarray,
    pA_given_SM_all: np.ndarray,
    r_adv_all: Optional[np.ndarray],
    eta: float,
    zeta: float,
) -> np.ndarray:
    """Per-label verifier-aware score (N, C). Higher term -> less conforming label."""
    out = s_base_all + eta * (1.0 - pA_given_SM_all)
    if r_adv_all is not None and zeta != 0.0:
        out = out + zeta * r_adv_all
    return out


def tune_eta_zeta(
    s_base_all: np.ndarray,
    pA_given_SM_all: np.ndarray,
    r_adv_all: Optional[np.ndarray],
    y_true: np.ndarray,
    eta_grid,
    zeta_grid,
    objective_fn,
) -> tuple[float, float, float]:
    """Pick (eta, zeta) on D_learn maximizing objective_fn over the true-label scores.

    objective_fn(true_label_scores) -> float to MAXIMIZE (e.g. negative mean set size at fixed
    coverage, or in-set trustworthiness on D_learn). Uses D_learn only.
    """
    rows = np.arange(s_base_all.shape[0])
    best = (eta_grid[0], zeta_grid[0], -np.inf)
    for eta in eta_grid:
        for zeta in zeta_grid:
            sv = verifier_aware_scores_all(s_base_all, pA_given_SM_all, r_adv_all, eta, zeta)
            score = objective_fn(sv[rows, y_true])
            if score > best[2]:
                best = (eta, zeta, score)
    return best
