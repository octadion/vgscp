"""Dataset interface shared by all loaders.

Every dataset must expose group labels (class x spurious-attribute) so worst-group / minority
metrics and signal-contamination can be computed. The minority/conflict group is defined as
samples where the spurious attribute disagrees with the label.

Splits: train (fit f, NCV, ensembles) / d_learn (fit eta,zeta,beta,tau) / d_cal (conformal
quantile ONLY) / d_test (final eval). Disjoint, fixed by seed, logged.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

SPLITS = ("train", "d_learn", "d_cal", "d_test")


@dataclass
class SplitSpec:
    """Fractions of the non-train pool assigned to d_learn / d_cal / d_test."""

    d_learn: float = 0.25
    d_cal: float = 0.375
    d_test: float = 0.375


def make_group_id(y: np.ndarray, spurious_attr: np.ndarray, n_spurious_vals: int) -> np.ndarray:
    """group_id = y * n_spurious_vals + spurious_attr."""
    return y.astype(np.int64) * n_spurious_vals + spurious_attr.astype(np.int64)


def make_minority_mask(y: np.ndarray, spurious_attr: np.ndarray) -> np.ndarray:
    """Conflict group: spurious attribute disagrees with the label (binary-attr convention)."""
    return (y.astype(np.int64) != spurious_attr.astype(np.int64))


def split_indices(
    n: int,
    train_idx: np.ndarray,
    pool_idx: np.ndarray,
    spec: SplitSpec,
    seed: int,
) -> dict[str, np.ndarray]:
    """Deterministically partition a pool of indices into d_learn/d_cal/d_test by ``spec``.

    ``train_idx`` is provided by the dataset (its native train split). The remaining pool is
    shuffled with ``seed`` and partitioned. Returns disjoint index arrays keyed by split name.
    """
    rng = np.random.default_rng(seed)
    pool = pool_idx.copy()
    rng.shuffle(pool)
    n_pool = len(pool)
    n_learn = int(round(spec.d_learn * n_pool))
    n_cal = int(round(spec.d_cal * n_pool))
    d_learn = pool[:n_learn]
    d_cal = pool[n_learn : n_learn + n_cal]
    d_test = pool[n_learn + n_cal :]
    return {
        "train": np.asarray(train_idx),
        "d_learn": d_learn,
        "d_cal": d_cal,
        "d_test": d_test,
    }


class ImageDatasetBundle:
    """Container the precompute stage consumes.

    Holds torch Datasets / arrays per split plus group metadata. Concrete loaders populate it.
    """

    def __init__(self, name: str, n_classes: int):
        self.name = name
        self.n_classes = n_classes
        self.datasets: dict = {}        # split -> torch Dataset (image, label)
        self.y: dict = {}               # split -> np.ndarray labels
        self.spurious_attr: dict = {}   # split -> np.ndarray spurious attribute
        self.group_id: dict = {}        # split -> np.ndarray group id
        self.is_minority: dict = {}     # split -> np.ndarray bool
        self.concepts: dict = {}        # split -> optional np.ndarray ground-truth concepts
        self.meta: dict = {}
