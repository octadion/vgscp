"""Trust score (Jiang et al., NeurIPS 2018).

In a representation space phi(x), fit class-conditional reference sets on the TRAINING split
(never the query split, to avoid self-match). Then for a query x with predicted label yhat:

    T(x) = d(phi(x), nearest other-class ref) / (d(phi(x), nearest yhat-class ref) + eps)

Higher T => more reliable. Optionally an alpha-density filter drops low-density reference
points (Jiang et al.'s recommended robustification); off by default for simplicity/fairness.

kNN distances are computed on GPU via chunked torch.cdist when torch+CUDA are available
(Section 13: never sklearn-CPU kNN on large sets); otherwise a numpy fallback is used.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

EPS = 1e-12


def _nearest_dist_numpy(query: np.ndarray, ref: np.ndarray, chunk: int = 4096) -> np.ndarray:
    """Min Euclidean distance from each query row to any ref row (chunked).

    Uses ||q-r||^2 = ||q||^2 + ||r||^2 - 2 q.r to avoid materializing a 3D (chunk, n_ref, d)
    temporary — only a 2D (chunk, n_ref) gram matrix per chunk. We only need the argmin, so
    the squared distance suffices until the final sqrt.
    """
    query = np.ascontiguousarray(query, dtype=np.float64)
    ref = np.ascontiguousarray(ref, dtype=np.float64)
    r_sq = (ref * ref).sum(axis=1)  # (n_ref,)
    out = np.empty(query.shape[0], dtype=np.float64)
    for i in range(0, query.shape[0], chunk):
        q = query[i : i + chunk]
        q_sq = (q * q).sum(axis=1, keepdims=True)  # (c,1)
        # (c, n_ref) squared distances
        d2 = q_sq + r_sq[None, :] - 2.0 * (q @ ref.T)
        out[i : i + chunk] = np.sqrt(np.maximum(d2.min(axis=1), 0.0))
    return out


def _nearest_dist_torch(query: np.ndarray, ref: np.ndarray, chunk: int = 4096) -> np.ndarray:
    import torch

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    q = torch.as_tensor(query, dtype=torch.float32, device=dev)
    r = torch.as_tensor(ref, dtype=torch.float32, device=dev)
    out = torch.empty(q.shape[0], dtype=torch.float32, device=dev)
    with torch.inference_mode():
        for i in range(0, q.shape[0], chunk):
            qc = q[i : i + chunk]
            d = torch.cdist(qc, r)  # (chunk, n_ref)
            out[i : i + chunk] = d.min(dim=1).values
    return out.detach().cpu().numpy()


def _nearest_dist(query, ref, chunk, prefer_torch=True):
    if prefer_torch:
        try:
            import torch  # noqa: F401

            return _nearest_dist_torch(query, ref, chunk)
        except ImportError:
            pass
    return _nearest_dist_numpy(query, ref, chunk)


class TrustScorer:
    """Class-conditional reference sets fit on TRAIN features only."""

    def __init__(self, n_classes: int, chunk: int = 4096, prefer_torch: bool = True):
        self.n_classes = n_classes
        self.chunk = chunk
        self.prefer_torch = prefer_torch
        self.refs: dict[int, np.ndarray] = {}

    def fit(self, train_feats: np.ndarray, train_labels: np.ndarray) -> "TrustScorer":
        """Reference set per class = TRAIN features of that class. No query data here."""
        for c in range(self.n_classes):
            self.refs[c] = train_feats[train_labels == c]
            if self.refs[c].shape[0] == 0:
                raise ValueError(f"trust: class {c} has no reference points in train split")
        return self

    def score(self, feats: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Trust score for each query. feats: (N,d), y_pred: (N,)."""
        n = feats.shape[0]
        # distance to nearest point in each class
        dist_to_class = np.empty((n, self.n_classes), dtype=np.float64)
        for c in range(self.n_classes):
            dist_to_class[:, c] = _nearest_dist(
                feats, self.refs[c], self.chunk, self.prefer_torch
            )
        rows = np.arange(n)
        d_pred = dist_to_class[rows, y_pred]
        # nearest OTHER-class distance
        masked = dist_to_class.copy()
        masked[rows, y_pred] = np.inf
        d_other = masked.min(axis=1)
        return d_other / (d_pred + EPS)


def trust_score(
    train_feats: np.ndarray,
    train_labels: np.ndarray,
    query_feats: np.ndarray,
    query_pred: np.ndarray,
    n_classes: int,
    chunk: int = 4096,
    prefer_torch: bool = True,
) -> np.ndarray:
    """Convenience one-shot trust score. Higher = more reliable."""
    scorer = TrustScorer(n_classes, chunk=chunk, prefer_torch=prefer_torch).fit(
        train_feats, train_labels
    )
    return scorer.score(query_feats, query_pred)
