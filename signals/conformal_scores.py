"""Nonconformity scores for the shift-CP de-risk (go/no-go gate).

Two true-label nonconformity scores, both consumed by the SAME split-conformal machinery in
``conformal.scores`` / ``conformal.split_conformal`` (LOWER score = more conforming, set =
``{y : s(x,y) <= qhat}``):

  - ``f_score_all``      : APS (default) / RAPS / THR on the f-softmax. This is the CONTAMINATED
    baseline — f reads the background shortcut, so its score depends on the spurious attribute.
  - ``concept_score_all``: the PROPOSED shortcut-invariant score. Fit a plain logistic probe
    ``y ~ concepts`` on the CUB attribute vectors (TRAIN only, no leakage), then take a conformal
    score on the probe's class posteriors. Default is THR (``s = 1 - p_probe(y|c)``), exactly the
    "1 - p_probe(y|concepts)" the task specifies; APS on the probe posteriors is also available
    ("APS-style ... classwise") via ``kind="APS"``.

Because both scores end up as (N, C) arrays in the same convention, every method (pooled split CP,
group-conditional Mondrian, robust-CP) treats them identically — the only thing that changes is
which (N, C) score array is fed in.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from conformal import scores as cscores


# --------------------------------------------------------------------------------------
# f-softmax score (contaminated baseline)
# --------------------------------------------------------------------------------------
def f_score_all(
    probs: np.ndarray,
    kind: str = "APS",
    u: Optional[np.ndarray] = None,
    lam_reg: float = 0.01,
    k_reg: int = 1,
) -> np.ndarray:
    """(N, C) nonconformity from the f-softmax. ``kind`` in {APS, RAPS, THR}; reuses conformal.scores."""
    return cscores.scores_all(kind, probs, u=u, lam_reg=lam_reg, k_reg=k_reg)


# --------------------------------------------------------------------------------------
# Concept-space probe score (the proposed shortcut-invariant score)
# --------------------------------------------------------------------------------------
def fit_concept_probe(train_concepts: np.ndarray, train_labels: np.ndarray,
                      kind: str = "logistic", seed: int = 0, C: float = 1.0):
    """Plain probe ``y ~ concepts`` fit on TRAIN concepts only (the CUB attribute space)."""
    if kind == "mlp":
        from sklearn.neural_network import MLPClassifier

        clf = MLPClassifier(hidden_layer_sizes=(128,), max_iter=500, random_state=seed)
    else:
        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression(max_iter=2000, C=C, random_state=seed)
    clf.fit(train_concepts, train_labels)
    return clf


def probe_posteriors(clf, query_concepts: np.ndarray, n_classes: int) -> np.ndarray:
    """(N, C) probe class posteriors, column-aligned to label index 0..n_classes-1.

    ``clf.classes_`` may omit a class absent from TRAIN; we scatter into the full C columns so the
    conformal score array always has one column per label (missing classes get probability 0).
    """
    p = clf.predict_proba(query_concepts)
    classes = np.asarray(clf.classes_).astype(int)
    if classes.shape[0] == n_classes and np.array_equal(classes, np.arange(n_classes)):
        return p
    full = np.zeros((query_concepts.shape[0], n_classes), dtype=np.float64)
    full[:, classes] = p
    return full


def concept_score_all(
    train_concepts: np.ndarray,
    train_labels: np.ndarray,
    query_concepts: np.ndarray,
    n_classes: int,
    score_kind: str = "THR",
    probe_kind: str = "logistic",
    seed: int = 0,
    u: Optional[np.ndarray] = None,
    clf=None,
) -> np.ndarray:
    """(N, C) shortcut-invariant nonconformity from a concept->y probe.

    ``score_kind="THR"`` gives the task's ``s = 1 - p_probe(y|concepts)``; ``"APS"`` gives the
    APS-style classwise score on the probe posteriors. Pass a pre-fit ``clf`` to reuse one probe
    across calibration and test (so the probe is fit ONCE on TRAIN).
    """
    if clf is None:
        clf = fit_concept_probe(train_concepts, train_labels, kind=probe_kind, seed=seed)
    p = probe_posteriors(clf, query_concepts, n_classes)
    return cscores.scores_all(score_kind, p, u=u)
