"""Synthetic f, ensemble, and a concept-space NCV verifier for the theory testbed.

These are deliberately simple (logistic regression) so the testbed runs in seconds on CPU with
no torch. The point is to verify the MECHANISM of P2, not to train deep nets.

  - Base classifier f: logistic regression on the FULL feature vector phi(x). Because the
    spurious feature is strongly separated and agrees with the label on the (majority of)
    training data, f learns to use it and becomes confident-but-wrong on the minority.
  - Ensemble: M logistic models on bootstrap resamples / different seeds. They all latch onto
    the same shortcut, so they AGREE on minority errors => disagreement fails to flag them.
  - NCV (Arthur): a verifier over the CONCEPT space [core | spurious]. Arthur is trained with
    Morgana-style ADVERSARIAL augmentation (the spurious concept is randomized on half the
    training data) so it LEARNS that the spurious concept is unreliable and downweights it.
    Its robustness to the shortcut is therefore earned by adversarial training, not assumed.
      V_comp(x) = p_A(yhat | S_M)  with S_M = honest (cooperative-Merlin) core concepts.
      R_adv(x)  = p_A(non-yhat | S_A) under a bounded concept-space adversary (Morgana).
      V_sound   = 1 - clip(R_adv, 0, 1).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression


def fit_base_classifier(x: np.ndarray, y: np.ndarray, seed: int = 0) -> LogisticRegression:
    """f: logistic regression on the full (shortcut-contaminated) feature vector."""
    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
    clf.fit(x, y)
    return clf


def fit_ensemble(x: np.ndarray, y: np.ndarray, n_members: int, seed: int = 0):
    """M ERM models on bootstrap resamples (different seeds)."""
    rng = np.random.default_rng(seed)
    members = []
    n = x.shape[0]
    for m in range(n_members):
        idx = rng.integers(0, n, n)
        clf = LogisticRegression(max_iter=2000, C=1.0, random_state=seed + m + 1)
        clf.fit(x[idx], y[idx])
        members.append(clf)
    return members


def ensemble_member_probs(members, x: np.ndarray) -> np.ndarray:
    """(M, N, C) stacked member probabilities."""
    return np.stack([m.predict_proba(x) for m in members], axis=0)


@dataclass
class SyntheticNCV:
    """Concept-space verifier (Arthur) with adversarial (Morgana) training.

    Concept vector c = [core | spurious]. ``adv_eps`` is the L2 budget of the Morgana
    concept-space adversary used for the soundness probe.
    """

    core_slice: slice
    spurious_slice: slice
    adv_eps: float = 1.0
    arthur: LogisticRegression = None
    _concept_dim: int = 0

    def _concepts(self, x: np.ndarray) -> np.ndarray:
        core = x[:, self.core_slice]
        spur = x[:, self.spurious_slice]
        return np.concatenate([core, spur], axis=1)

    def fit(self, x: np.ndarray, y: np.ndarray, seed: int = 0) -> "SyntheticNCV":
        c = self._concepts(x)
        self._concept_dim = c.shape[1]
        rng = np.random.default_rng(seed + 777)
        # Morgana-style adversarial augmentation: a copy of the data where the spurious
        # concept dims are randomized (drawn ~ N(0,1)). Arthur thus cannot rely on the
        # spurious concept and downweights it.
        n_core = self.core_slice.stop - self.core_slice.start
        c_aug = c.copy()
        c_aug[:, n_core:] = rng.normal(0, 1, size=c_aug[:, n_core:].shape)
        c_train = np.concatenate([c, c_aug], axis=0)
        y_train = np.concatenate([y, y], axis=0)
        self.arthur = LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
        self.arthur.fit(c_train, y_train)
        return self

    def pA_given_SM(self, x: np.ndarray) -> np.ndarray:
        """Arthur's class probs given Merlin's honest core concepts (spurious neutralized)."""
        c = self._concepts(x).copy()
        n_core = self.core_slice.stop - self.core_slice.start
        c[:, n_core:] = 0.0  # cooperative Merlin presents core concepts; spurious set neutral
        return self.arthur.predict_proba(c)

    def pA_given_SA(self, x: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Arthur's class probs under a bounded Morgana adversary in concept space.

        Binary logistic: Arthur's logit for class 1 is z = w.c + b. The worst-case L2
        perturbation of budget adv_eps that pushes AWAY from y_pred moves c by
        -(2*y_pred-1) * adv_eps * w/||w||. We return Arthur's probs at the perturbed concepts.
        """
        c = self._concepts(x).copy()
        w = self.arthur.coef_[0]
        wn = np.linalg.norm(w) + 1e-12
        direction = (w / wn)[None, :]
        sign = (2 * y_pred - 1)[:, None]  # +1 if predicted class 1
        c_adv = c - sign * self.adv_eps * direction
        return self.arthur.predict_proba(c_adv)

    def concept_weights(self) -> dict:
        """For reporting: how much weight Arthur puts on core vs spurious concepts."""
        n_core = self.core_slice.stop - self.core_slice.start
        w = self.arthur.coef_[0]
        return {
            "core_l2": float(np.linalg.norm(w[:n_core])),
            "spurious_l2": float(np.linalg.norm(w[n_core:])),
        }
