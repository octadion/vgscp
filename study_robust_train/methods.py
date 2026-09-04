"""Last-layer group-robustness methods on FROZEN features (backbone-agnostic).

Every method returns a fitted head exposing ``predict_proba(X) -> (N, C)`` and ``classes_``,
so the conformal evaluation treats them identically. All operate on frozen features (ERM
ResNet-50 OR CLIP ViT-B/32) and re-fit only a linear last layer — the study's primary,
cheap-first regime (spec §3).

Methods (spec §3):
  erm                : standardized multinomial logistic on the train split (reuses heads.fit_erm).
  dfr                : Deep Feature Reweighting — group-balanced last-layer retrain, averaged over
                       subsets, on a HELD-OUT reweighting split (reuses heads.fit_dfr).
  afr                : Automatic Feature Reweighting (Qiu et al. 2023) — group-LABEL-FREE. Fit ERM,
                       then retrain the last layer weighting each reweighting-split example by
                       (1 - p_ERM(y_true))^gamma (upweight what ERM gets wrong).
  groupdro_ll        : last-layer GroupDRO (Sagawa et al. 2020) — softmax head trained to minimize
                       the WORST-group loss via the online exponentiated group-weight update.
  balanced_subsample : one group-balanced subsample (no averaging) + standardized head.

The "CLIP linear probe" baseline in the spec list == (erm x clip_vitb32 backbone); it is not a
separate method here — the backbone axis covers it (documented in grid.py).
"""
from __future__ import annotations

import numpy as np

from .heads import assert_l2_normalized, assert_multinomial_safe, fit_dfr, fit_erm, fit_species_head

__all__ = ["METHODS", "fit_method", "fit_afr", "fit_groupdro_ll", "fit_balanced_subsample",
           "SoftmaxGroupDRO"]


# ----------------------------------------------------------------------------------------
# AFR — Automatic Feature Reweighting (group-label-free)
# ----------------------------------------------------------------------------------------
def fit_afr(X_rw: np.ndarray, y_rw: np.ndarray, *, gamma: float = 2.0, C: float = 1.0,
            max_iter: int = 5000, seed: int = 0):
    """ERM, then a last-layer retrain that upweights examples ERM gets wrong (no group labels).

    Weight w_i ∝ (1 - p_ERM(y_i | x_i))^gamma on the reweighting split, normalized to mean 1, then
    a standardized weighted multinomial logistic. gamma>0 concentrates weight on hard/minority-like
    examples without ever seeing the group label (the AFR premise: the spurious-reliant ERM is
    wrong precisely on the minority).
    """
    assert_multinomial_safe()
    assert_l2_normalized(X_rw, tag="AFR reweight features")
    erm = fit_species_head(X_rw, y_rw, C=C, max_iter=max_iter, seed=seed)
    p = erm.predict_proba(X_rw)
    cls = list(erm.classes_)
    p_true = np.array([p[i, cls.index(y_rw[i])] for i in range(len(y_rw))])
    w = np.power(np.clip(1.0 - p_true, 1e-6, 1.0), gamma)
    w = w / w.mean()
    # standardized weighted logistic (same recipe as fit_species_head, with sample_weight)
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(C=C, max_iter=max_iter, random_state=seed))
    clf.fit(X_rw, y_rw, logisticregression__sample_weight=w)
    return clf


# ----------------------------------------------------------------------------------------
# last-layer GroupDRO — worst-group-loss softmax head (numpy, no torch)
# ----------------------------------------------------------------------------------------
def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class SoftmaxGroupDRO:
    """Standardized softmax-regression head trained with the online GroupDRO objective.

    Minimizes max_g E_{i in g}[CE loss] via Sagawa et al. (2020): maintain group weights q on the
    simplex, update q_g ∝ q_g * exp(eta_q * L_g) each step, and take a gradient step on
    sum_g q_g * L_g (per-sample weight q_{g(i)} / n_{g(i)}). L2-regularized. Pure numpy.
    """

    def __init__(self, mean, std, W, b, classes_):
        self.mean, self.std, self.W, self.b = mean, std, W, b
        self.classes_ = classes_

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # Same one-copy, in-place standardisation as the fitter: `(asarray(X) - m) / s` holds two
        # float64 copies of the eval matrix at once. Elementwise ops and order are unchanged.
        Z = np.asarray(X, dtype=np.float64, copy=True)
        Z -= self.mean
        Z /= self.std
        return _softmax(Z @ self.W + self.b)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


def fit_groupdro_ll(X: np.ndarray, y: np.ndarray, group: np.ndarray, *, lr: float = 0.05,
                    eta_q: float = 0.05, l2: float = 1e-3, steps: int = 2000,
                    seed: int = 0) -> SoftmaxGroupDRO:
    """Train a last-layer softmax head with online GroupDRO. Full-batch gradient descent."""
    assert_l2_normalized(X, tag="GroupDRO features")
    # ONE float64 working copy, standardised in place. `X = asarray(...); Z = (X-m)/s` holds two
    # full copies for the whole optimisation and X is never read again. Measured, this copy is
    # 2.00x the float32 input; combined with the std temporary removed below it took this arm from
    # 4.30x to 2.29x. The elementwise ops and their order are unchanged, so this is bit-identical.
    Z = np.asarray(X, dtype=np.float64, copy=True)
    y = np.asarray(y)
    g = np.asarray(group)
    classes_ = np.array(sorted(np.unique(y)))
    cidx = {c: i for i, c in enumerate(classes_)}
    yk = np.array([cidx[v] for v in y])
    n, d = Z.shape
    C = len(classes_)

    # `Z.std(axis=0)` builds `Z - mean` as a full float64 temporary and squares it in place --
    # another 2x the input, measured, and together with Z itself that is the 4.00x that exhausted
    # RAM. Centring first lets the same sum of squares be streamed by einsum with no temporary.
    # numpy's std IS sqrt(sum((Z-mean)^2)/n), so the summands and their order are unchanged and the
    # result is bit-identical -- verified against Z.std(axis=0) at three shapes.
    mean = Z.mean(axis=0)
    Z -= mean
    std = np.sqrt(np.einsum("ij,ij->j", Z, Z) / n)
    std[std == 0] = 1.0
    Z /= std

    groups = np.array(sorted(np.unique(g)))
    G = len(groups)
    gk = np.array([int(np.where(groups == v)[0][0]) for v in g])
    group_n = np.array([(gk == j).sum() for j in range(G)], dtype=np.float64)

    rng = np.random.default_rng(seed)
    W = rng.standard_normal((d, C)) * 0.01
    b = np.zeros(C)
    q = np.ones(G) / G

    Y = np.zeros((n, C))
    Y[np.arange(n), yk] = 1.0

    for _ in range(steps):
        P = _softmax(Z @ W + b)                       # (n, C)
        ce = -np.log(np.clip(P[np.arange(n), yk], 1e-12, 1.0))   # (n,)
        # per-group mean loss
        L = np.array([ce[gk == j].mean() if group_n[j] > 0 else 0.0 for j in range(G)])
        # exponentiated group-weight update
        q = q * np.exp(eta_q * L)
        q = q / q.sum()
        # per-sample weight = q_{g(i)} / n_{g(i)} (so each group contributes q_g * mean_g)
        sw = q[gk] / group_n[gk]
        grad_logits = (P - Y) * sw[:, None]           # (n, C)
        gW = Z.T @ grad_logits + l2 * W
        gb = grad_logits.sum(axis=0)
        W -= lr * gW
        b -= lr * gb

    return SoftmaxGroupDRO(mean=mean, std=std, W=W, b=b, classes_=classes_)


# ----------------------------------------------------------------------------------------
# balanced subsampling — single group-balanced draw + standardized head
# ----------------------------------------------------------------------------------------
def fit_balanced_subsample(X: np.ndarray, y: np.ndarray, group: np.ndarray, *, C: float = 1.0,
                           max_iter: int = 5000, seed: int = 0):
    """Subsample every group down to the smallest group's count (once), then standardized head."""
    assert_multinomial_safe()
    assert_l2_normalized(X, tag="balanced-subsample features")
    g = np.asarray(group)
    groups = np.unique(g)
    per = int(min((g == grp).sum() for grp in groups))
    rng = np.random.default_rng(seed)
    idx = np.concatenate([rng.choice(np.where(g == grp)[0], size=per, replace=False)
                          for grp in groups])
    rng.shuffle(idx)
    return fit_species_head(np.asarray(X)[idx], np.asarray(y)[idx], C=C, max_iter=max_iter, seed=seed)


# ----------------------------------------------------------------------------------------
# unified registry
# ----------------------------------------------------------------------------------------
def fit_method(name: str, train: tuple, reweight: tuple, *, seed: int = 0, **hp):
    """Fit a method by name. ``train``/``reweight`` are (X, y, group) tuples (same composited
    distribution, §2 in-domain). Methods pick which split they need."""
    Xtr, ytr, gtr = train
    Xrw, yrw, grw = reweight
    if name == "erm":
        return fit_erm(Xtr, ytr, seed=seed, **hp)
    if name == "dfr":
        return fit_dfr(Xrw, yrw, grw, seed=seed, **hp)
    if name == "afr":
        return fit_afr(Xrw, yrw, seed=seed, **hp)
    if name == "groupdro_ll":
        return fit_groupdro_ll(Xtr, ytr, gtr, seed=seed, **hp)
    if name == "balanced_subsample":
        return fit_balanced_subsample(Xtr, ytr, gtr, seed=seed, **hp)
    raise ValueError(f"unknown method {name!r}; choose from {list(METHODS)}")


METHODS = ("erm", "dfr", "afr", "groupdro_ll", "balanced_subsample")
