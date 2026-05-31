"""Deep ensemble (M=5 ERM models, different seeds) — Section 6.

Training is just M independent ERM runs with different seeds (and shuffles). At inference the M
models are run over all splits ONCE and their probabilities cached (M, N, C); the ensemble
disagreement signal is then a vectorized op over the cache (signals/ensemble.py).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class EnsembleSpec:
    n_members: int = 5


def train_ensemble(
    build_and_train_fn: Callable[[int], "object"],
    spec: EnsembleSpec,
):
    """Train M members. ``build_and_train_fn(seed)`` returns a trained FeatureClassifier.

    Kept as a thin orchestrator so the heavy training reuses models/base_model.train_erm with a
    per-member seed (seed_everything is called inside per member for reproducibility).
    """
    return [build_and_train_fn(seed=m) for m in range(spec.n_members)]


def stack_member_probs(member_probs_list) -> np.ndarray:
    """Stack a list of (N, C) member-probability arrays into (M, N, C)."""
    return np.stack(member_probs_list, axis=0)
