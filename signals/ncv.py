"""NCV verifiability signals (ours) — Section 2.1.

Computed from cached Prover-Verifier-Game outputs (Merlin/Morgana/Arthur over concept
encodings). All inputs are produced once by ``models/verifier_adapter.py`` and cached, so this
module is pure vectorized numpy.

Definitions (higher = more reliable):
  V_comp(x)  = p_A(yhat | S_M)                          # core-concept support for f's label
  R_adv(x)   = max_{y'!=yhat} p_A(y'|S_A) - p_A(reject|S_A)   # adversarial vulnerability
  V_sound(x) = 1 - clip(R_adv, 0, 1)
  V_full(x)  = beta * V_comp + (1-beta) * V_sound       # beta tuned on D_learn
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def v_comp(pA_given_SM: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """p_A(yhat | S_M). pA_given_SM: (N, C). Higher = more reliable."""
    return pA_given_SM[np.arange(pA_given_SM.shape[0]), y_pred]


def r_adv(
    pA_given_SA: np.ndarray,
    y_pred: np.ndarray,
    reject_prob: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Adversarial vulnerability under Morgana's misleading set S_A.

    max_{y' != yhat} p_A(y'|S_A) - p_A(reject|S_A). If no reject class, the reject term is 0.
    """
    n = pA_given_SA.shape[0]
    rows = np.arange(n)
    masked = pA_given_SA.copy()
    masked[rows, y_pred] = -np.inf
    max_other = masked.max(axis=1)
    if reject_prob is not None:
        return max_other - reject_prob
    return max_other


def v_sound(
    pA_given_SA: np.ndarray,
    y_pred: np.ndarray,
    reject_prob: Optional[np.ndarray] = None,
) -> np.ndarray:
    """1 - clip(R_adv, 0, 1). Higher = more reliable (less adversarially vulnerable)."""
    return 1.0 - np.clip(r_adv(pA_given_SA, y_pred, reject_prob), 0.0, 1.0)


def v_full(
    vc: np.ndarray,
    vs: np.ndarray,
    beta: float,
) -> np.ndarray:
    """beta * V_comp + (1 - beta) * V_sound."""
    return beta * vc + (1.0 - beta) * vs


def tune_beta(
    vc: np.ndarray,
    vs: np.ndarray,
    correctness: np.ndarray,
    beta_grid,
    objective_fn,
    minority_mask: Optional[np.ndarray] = None,
) -> tuple[float, float]:
    """Pick beta on D_learn maximizing ``objective_fn`` (e.g. minority error-detection AUROC).

    Uses correctness labels from D_learn ONLY (signal-hyperparameter selection, never the
    conformal calibration split). Returns (best_beta, best_score).

    objective_fn(signal, correctness) -> float to MAXIMIZE.
    """
    mask = np.ones_like(correctness, dtype=bool) if minority_mask is None else minority_mask
    best_beta, best_score = beta_grid[0], -np.inf
    for b in beta_grid:
        sig = v_full(vc, vs, b)
        score = objective_fn(sig[mask], correctness[mask])
        if score > best_score:
            best_score, best_beta = score, b
    return best_beta, best_score


def compute_ncv_signals(
    pA_given_SM: np.ndarray,
    pA_given_SA: np.ndarray,
    y_pred: np.ndarray,
    beta: float,
    reject_prob: Optional[np.ndarray] = None,
) -> dict:
    """Convenience: compute V_comp, R_adv, V_sound, V_full together."""
    vc = v_comp(pA_given_SM, y_pred)
    ra = r_adv(pA_given_SA, y_pred, reject_prob)
    vs = 1.0 - np.clip(ra, 0.0, 1.0)
    vf = v_full(vc, vs, beta)
    return {"V_comp": vc, "R_adv": ra, "V_sound": vs, "V_full": vf}
