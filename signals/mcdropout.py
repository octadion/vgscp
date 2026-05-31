"""MC-dropout reliability signal.

Given K stochastic forward passes' cached probabilities (K, N, C) — produced by enabling
dropout at eval time and batching the K passes (Section 13, never a python loop) — the signal
is the negative predictive variance, or the negative entropy of the mean prediction. Higher =
more reliable.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-12


def predictive_variance(pass_probs: np.ndarray) -> np.ndarray:
    """Mean-over-classes variance of p(y|x) across the K passes. Per sample."""
    return pass_probs.var(axis=0).mean(axis=1)


def mean_entropy(pass_probs: np.ndarray) -> np.ndarray:
    """Entropy of the mean prediction across passes. Per sample."""
    mean_p = np.clip(pass_probs.mean(axis=0), EPS, 1.0)
    return -(mean_p * np.log(mean_p)).sum(axis=1)


def mcdropout_signal(pass_probs: np.ndarray, mode: str = "variance") -> np.ndarray:
    """Reliability signal (higher = more reliable).

    mode='variance' -> -predictive_variance
    mode='entropy'  -> -mean_entropy
    """
    if mode == "variance":
        return -predictive_variance(pass_probs)
    if mode == "entropy":
        return -mean_entropy(pass_probs)
    raise ValueError(f"unknown mcdropout mode: {mode}")
