"""Signal registry — assemble all reliability signals from cached arrays into one dict.

Every signal here is "higher = more reliable", so the same selective-prediction / gate code
applies uniformly. This is the single place that defines the canonical signal set compared in
Phase 1.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from . import confidence, ensemble, mcdropout, ncv, trust

# Canonical signal names compared in Phase 1. The kill-switch focuses on V_full vs trust and
# ensemble_disagree (NOT just conf_msp).
CANONICAL_SIGNALS = [
    "conf_msp",
    "trust",
    "ensemble_disagree",
    "mcdropout",
    "V_comp",
    "V_sound",
    "V_full",
]


def build_signals(
    probs: np.ndarray,
    y_pred: np.ndarray,
    *,
    train_feats: Optional[np.ndarray] = None,
    train_labels: Optional[np.ndarray] = None,
    query_feats: Optional[np.ndarray] = None,
    member_probs: Optional[np.ndarray] = None,
    mc_pass_probs: Optional[np.ndarray] = None,
    pA_given_SM: Optional[np.ndarray] = None,
    pA_given_SA: Optional[np.ndarray] = None,
    reject_prob: Optional[np.ndarray] = None,
    n_classes: Optional[int] = None,
    beta: float = 0.5,
    knn_chunk: int = 4096,
) -> dict:
    """Return {signal_name: np.ndarray(N,)} for all available signals (higher = more reliable).

    Missing inputs simply skip the corresponding signal, so the same function works for the
    synthetic testbed (no NCV concepts) and the full CLEVR-Hans pipeline.
    """
    out: dict[str, np.ndarray] = {}
    out["conf_msp"] = confidence.msp(probs)

    if train_feats is not None and query_feats is not None:
        nc = n_classes or int(train_labels.max() + 1)
        out["trust"] = trust.trust_score(
            train_feats, train_labels, query_feats, y_pred, nc, chunk=knn_chunk
        )
    if member_probs is not None:
        out["ensemble_disagree"] = ensemble.ensemble_disagreement_signal(member_probs)
    if mc_pass_probs is not None:
        out["mcdropout"] = mcdropout.mcdropout_signal(mc_pass_probs)

    if pA_given_SM is not None and pA_given_SA is not None:
        nv = ncv.compute_ncv_signals(pA_given_SM, pA_given_SA, y_pred, beta, reject_prob)
        out.update({k: v for k, v in nv.items() if k != "R_adv"})
        out["_R_adv"] = nv["R_adv"]  # underscore = not a reliability signal, kept for logging
    return out
