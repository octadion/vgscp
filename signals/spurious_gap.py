"""Counterfactual support-gap signals (KILL-SWITCH for Arah 2).

The mechanism under test: a prediction is unreliable if the verifier's support for it DROPS when
the spurious concepts are removed — i.e. f's prediction is "propped up" by a shortcut. A genuinely
supported prediction keeps its support without the spurious concepts.

Three pieces:
  1. ``concept_spuriousness``  -> per-concept spuriousness score rho_j (1-D AUROC of concept_j
     predicting the known spurious attribute a, train-only). High rho = spurious-correlated.
  2. ``clean_mask_from_rho``   -> data-driven clean/spurious split of the concept dims (2-cluster
     1-D k-means midpoint on rho, fallback tau=0.5). clean_mask = (rho <= tau).
  3. ``gap_signals``           -> through ONE trained verifier (Arthur), compute Merlin's support
     for f's predicted label under (a) ALL concepts and (b) clean concepts only, and the gap.

Reliability convention everywhere: higher = more reliable.
  support_clean = p_A(y_pred | S_M selected from clean concepts only)   (higher = reliable)
  gap           = support_full - support_clean   (>= ~0; large = shortcut-propped = unreliable)
  V_gap         = support_clean - lam * gap       (THE MECHANISM)
  V_gap_pure    = -gap                             (pure gap, lam-free)

HONESTY: rho_j, tau and clean_mask are fit on TRAIN concepts + the TRAIN/CAL spurious attribute a
ONLY (never test labels of any kind). The method uses the rho-derived mask, NOT the by-construction
CUB-vs-CLIP identity.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from eval.metrics import auroc


def concept_spuriousness(train_concepts: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Per-concept spuriousness rho_j = 2 * |AUROC(concept_j -> a) - 0.5| on TRAIN only.

    ``train_concepts`` : (N_train, D) train concept matrix (already standardized upstream).
    ``a``              : (N_train,) binary spurious attribute label (Waterbirds place), TRAIN only.
    Returns rho (D,) in [0, 1]; high = strongly correlated (in either direction) with a.
    """
    train_concepts = np.asarray(train_concepts, dtype=np.float64)
    a = np.asarray(a)
    vals = np.unique(a)
    if len(vals) != 2:
        raise ValueError(f"concept_spuriousness needs a BINARY spurious attribute; got {vals}")
    pos = (a == vals[1]).astype(int)
    d = train_concepts.shape[1]
    rho = np.empty(d, dtype=np.float64)
    for j in range(d):
        au = auroc(train_concepts[:, j], pos)
        rho[j] = 0.0 if np.isnan(au) else 2.0 * abs(au - 0.5)
    return rho


def clean_mask_from_rho(rho: np.ndarray, tau: Optional[float] = None) -> tuple[np.ndarray, float]:
    """Split concept dims into clean (low rho) vs spurious (high rho).

    If ``tau`` is None, choose it data-drivenly as the midpoint of a 2-cluster 1-D k-means on
    ``rho`` (fallback tau=0.5 if k-means is unavailable or rho has < 2 distinct values).
    Returns (clean_mask (D,) bool where rho <= tau, tau).
    """
    rho = np.asarray(rho, dtype=np.float64)
    if tau is None:
        tau = _kmeans_midpoint(rho)
    clean_mask = rho <= tau
    return clean_mask, float(tau)


def _kmeans_midpoint(rho: np.ndarray, fallback: float = 0.5) -> float:
    """Midpoint between the two cluster centers of a 2-cluster 1-D k-means on rho."""
    rho = np.asarray(rho, dtype=np.float64).reshape(-1, 1)
    if len(np.unique(rho)) < 2:
        return fallback
    try:
        from sklearn.cluster import KMeans

        km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(rho)
        centers = np.sort(km.cluster_centers_.ravel())
        return float((centers[0] + centers[1]) / 2.0)
    except Exception:
        return fallback


def gap_signals(
    verifier,
    query_concepts: np.ndarray,
    y_pred: np.ndarray,
    clean_mask: np.ndarray,
    lam: float = 0.0,
    support_full: Optional[np.ndarray] = None,
) -> dict:
    """Counterfactual support-gap signals computed through ONE trained verifier.

    ``support_full`` (= existing V_comp = p_A(y_pred | S_M over ALL concepts)) may be passed in to
    avoid recomputing the full-Merlin pass; otherwise it is computed here. ``support_clean`` is
    always computed by restricting Merlin to ``clean_mask``.

    Returns (higher = more reliable, plus underscore-prefixed internals for logging):
      support_clean, V_gap, V_gap_pure, _support_full, _gap
    """
    clean_mask = np.asarray(clean_mask, dtype=bool)
    if support_full is None:
        support_full = verifier.merlin_completeness(query_concepts, y_pred, allowed_mask=None)
    support_full = np.asarray(support_full, dtype=np.float64)
    support_clean = np.asarray(
        verifier.merlin_completeness(query_concepts, y_pred, allowed_mask=clean_mask),
        dtype=np.float64,
    )
    gap = support_full - support_clean
    return {
        "support_clean": support_clean,
        "V_gap": support_clean - float(lam) * gap,
        "V_gap_pure": -gap,
        "_support_full": support_full,
        "_gap": gap,
    }


def rho_recovery_summary(rho: np.ndarray, clean_mask: np.ndarray, n_clean_true: int) -> dict:
    """How well the rho-derived ``clean_mask`` recovers the KNOWN clean/spurious split, where the
    first ``n_clean_true`` dims are the genuinely-clean concepts (CUB attributes) and the rest are
    the genuinely-spurious ones (CLIP scene concepts). Validation of the scorer ONLY — the method
    itself never uses this identity. Returns recall/specificity-style recovery stats."""
    rho = np.asarray(rho, dtype=np.float64)
    clean_mask = np.asarray(clean_mask, dtype=bool)
    d = len(rho)
    true_clean = np.zeros(d, dtype=bool)
    true_clean[:n_clean_true] = True
    true_spur = ~true_clean
    n_spur = int(true_spur.sum())
    # clean-recall: fraction of true-clean dims flagged clean; spurious-recall: fraction of
    # true-spurious dims flagged spurious.
    clean_recall = float(clean_mask[true_clean].mean()) if n_clean_true else float("nan")
    spur_recall = float((~clean_mask[true_spur]).mean()) if n_spur else float("nan")
    return {
        "n_clean_true": int(n_clean_true),
        "n_spurious_true": n_spur,
        "n_flagged_clean": int(clean_mask.sum()),
        "n_flagged_spurious": int((~clean_mask).sum()),
        "clean_recall": clean_recall,            # CUB attrs correctly kept clean
        "spurious_recall": spur_recall,          # CLIP concepts correctly flagged spurious
        "rho_clean_mean": float(rho[true_clean].mean()) if n_clean_true else float("nan"),
        "rho_spurious_mean": float(rho[true_spur].mean()) if n_spur else float("nan"),
    }
