"""Synthetic frozen-feature generator for LOCAL LOGIC VALIDATION ONLY.

Produces a Phase0Data with a controllable spurious correlation so the Phase-0 machinery can be
exercised without torch / CLIP / real images (which are unavailable locally — Colab-only). The
features carry a CORE signal (predicts y) and a SPURIOUS signal (predicts place, correlated with
y at train time). ERM is free to use both; DFR's group balancing removes the place-y coupling.

THIS IS NOT A RESULT GENERATOR. It exists so we can assert shapes / that DFR reweighting runs /
that the divergence computes / that the in-domain split wiring is correct. Any accuracy or
divergence number it yields is an artifact of the toy generator, not the Waterbirds phenomenon.
"""
from __future__ import annotations

import numpy as np

from .phase0 import Phase0Data


def _l2(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n


def _make_split(n: int, rho: float, d: int, rng) -> tuple:
    """Generate one composited split at correlation strength rho = P(place == y)."""
    y = rng.integers(0, 2, size=n)
    # place == y with prob rho (concordant), else flipped.
    concord = rng.random(n) < rho
    place = np.where(concord, y, 1 - y).astype(int)

    X = rng.standard_normal((n, d)).astype(np.float64) * 0.5
    # core signal: dims [0:2] separate the classes
    X[:, 0] += np.where(y == 1, 1.5, -1.5)
    X[:, 1] += np.where(y == 1, 1.0, -1.0)
    # spurious signal: dims [2:4] track the background (place)
    X[:, 2] += np.where(place == 1, 1.8, -1.8)
    X[:, 3] += np.where(place == 1, 1.2, -1.2)
    return _l2(X), y, place


def make_synthetic_phase0(*, n_train: int = 1200, n_test: int = 1200, d: int = 32,
                          rho_train: float = 0.95, rho_test: float = 0.95,
                          seed: int = 0) -> Phase0Data:
    """In-domain synthetic Waterbirds-like data (train and test at the SAME rho by default).

    Phase-0 is in-domain: train and test share the composited distribution (rho_train ==
    rho_test). The rho_test override exists only so the validator can also exercise the wiring
    under a shift; the H3 shift sweep itself is NOT part of Phase-0.
    """
    rng = np.random.default_rng(seed)
    Xtr, ytr, ptr = _make_split(n_train, rho_train, d, rng)
    Xte, yte, pte = _make_split(n_test, rho_test, d, rng)
    return Phase0Data(
        feats_train=Xtr, y_train=ytr, place_train=ptr,
        feats_test=Xte, y_test=yte, place_test=pte,
        n_classes=2, synthetic=True,
    )
