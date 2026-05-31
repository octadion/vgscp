"""Softmax-based reliability signals: MSP confidence, entropy, margin.

All operate on cached probability arrays ``probs`` of shape (N, C). Conventions:
"higher = more reliable". Entropy is returned NEGATED so higher = more reliable.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-12


def msp(probs: np.ndarray) -> np.ndarray:
    """Maximum softmax probability C(x) = max_y p(y|x). Higher = more reliable."""
    return probs.max(axis=1)


def entropy(probs: np.ndarray) -> np.ndarray:
    """Shannon entropy of p(y|x). Returned as-is (NOT a reliability signal by itself)."""
    p = np.clip(probs, EPS, 1.0)
    return -(p * np.log(p)).sum(axis=1)


def neg_entropy(probs: np.ndarray) -> np.ndarray:
    """Negative entropy. Higher = more reliable."""
    return -entropy(probs)


def margin(probs: np.ndarray) -> np.ndarray:
    """Top1 - top2 probability gap. Higher = more reliable."""
    part = np.partition(probs, -2, axis=1)
    return part[:, -1] - part[:, -2]


def p_true(probs: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """p(y_true | x), for logging."""
    return probs[np.arange(probs.shape[0]), y_true]
