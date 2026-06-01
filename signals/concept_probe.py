"""Concept-space control signals (premise-2 task, Section 2).

These are the KEY CONTROLS for premise 2: they live in the SAME concept space the verifier uses,
so beating them isolates the contribution of the adversarial verifiability mechanism from the
"just use a concept bottleneck" effect.

  - ``probe_concept`` : confidence max(p, 1-p) of a plain logistic/MLP probe trained TRAIN-only on
    the concept space (a canonical Concept Bottleneck Model, Koh et al. 2020). Higher = more
    reliable. This is the control V MUST beat to claim anything beyond "use a concept bottleneck".
  - ``trust_concept`` : the Jiang et al. (2018) trust score computed IN the concept space (class
    reference sets fit on TRAIN concepts), reusing ``signals/trust.py`` unchanged.

Both are "higher = more reliable" so they slot into the same selective-prediction machinery.
Standardization (if any) is applied to the concept vectors with TRAIN-only stats before fitting,
exactly as the verifier does — no leakage.
"""
from __future__ import annotations

import numpy as np

from .trust import trust_score


def _fit_probe(train_concepts: np.ndarray, train_labels: np.ndarray, kind: str, seed: int):
    if kind == "mlp":
        from sklearn.neural_network import MLPClassifier

        clf = MLPClassifier(hidden_layer_sizes=(128,), max_iter=500, random_state=seed)
    else:
        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
    clf.fit(train_concepts, train_labels)
    return clf


def probe_concept_confidence(
    train_concepts: np.ndarray,
    train_labels: np.ndarray,
    query_concepts: np.ndarray,
    kind: str = "logistic",
    seed: int = 0,
) -> np.ndarray:
    """max-softmax confidence of a plain concept->y probe (TRAIN-only fit). Higher = reliable."""
    clf = _fit_probe(train_concepts, train_labels, kind, seed)
    p = clf.predict_proba(query_concepts)
    return p.max(axis=1)


def trust_concept_score(
    train_concepts: np.ndarray,
    train_labels: np.ndarray,
    query_concepts: np.ndarray,
    query_pred: np.ndarray,
    n_classes: int,
    chunk: int = 4096,
) -> np.ndarray:
    """Trust score (Jiang et al. 2018) computed IN the concept space. Higher = reliable."""
    return trust_score(train_concepts, train_labels, query_concepts, query_pred, n_classes,
                       chunk=chunk)
