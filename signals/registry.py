"""Signal registry — assemble all reliability signals from cached arrays into one dict.

Every signal here is "higher = more reliable", so the same selective-prediction / gate code
applies uniformly. This is the single place that defines the canonical signal set compared in
Phase 1.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from . import concept_probe, confidence, ensemble, mcdropout, ncv, trust

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

# Premise-2 concept-space controls (same concept space as V): V must beat these to have any
# contribution beyond "use a concept bottleneck".
CONCEPT_CONTROL_SIGNALS = ["probe_concept", "trust_concept"]


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
    train_concepts: Optional[np.ndarray] = None,
    query_concepts: Optional[np.ndarray] = None,
    concept_probe_kind: str = "logistic",
    verifier=None,
    clean_mask: Optional[np.ndarray] = None,
    gap_lam: float = 0.0,
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

    # premise-2 concept-space controls (same concept space the verifier uses)
    if train_concepts is not None and query_concepts is not None and train_labels is not None:
        nc = n_classes or int(train_labels.max() + 1)
        out["probe_concept"] = concept_probe.probe_concept_confidence(
            train_concepts, train_labels, query_concepts, kind=concept_probe_kind
        )
        out["trust_concept"] = concept_probe.trust_concept_score(
            train_concepts, train_labels, query_concepts, y_pred, nc, chunk=knn_chunk
        )

    if pA_given_SM is not None and pA_given_SA is not None:
        nv = ncv.compute_ncv_signals(pA_given_SM, pA_given_SA, y_pred, beta, reject_prob)
        out.update({k: v for k, v in nv.items() if k != "R_adv"})
        out["_R_adv"] = nv["R_adv"]  # underscore = not a reliability signal, kept for logging

    # counterfactual support-gap signals (kill-switch for Arah 2). Needs a trained verifier + the
    # rho-derived clean_mask + the query concepts. support_full reuses the full-Merlin V_comp when
    # already computed above. Does NOT touch any existing signal.
    if verifier is not None and clean_mask is not None and query_concepts is not None:
        from . import spurious_gap

        support_full = out.get("V_comp")  # full-bank Merlin completeness, if available
        gs = spurious_gap.gap_signals(verifier, query_concepts, y_pred, clean_mask,
                                      lam=gap_lam, support_full=support_full)
        out["support_clean"] = gs["support_clean"]
        out["V_gap"] = gs["V_gap"]
        out["V_gap_pure"] = gs["V_gap_pure"]
        out["_support_full"] = gs["_support_full"]
        out["_gap"] = gs["_gap"]
    return out
