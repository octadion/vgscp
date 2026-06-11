"""Per-group and worst-group accuracy on the Waterbirds 4-group structure.

Re-implemented here (small, group-count-agnostic) rather than imported: the v4 worst-group
helpers are private to legacy verdicts or hard-keyed to 2 groups (AUDIT_study.md §4).

Waterbirds group convention (matches experiments/shift_resampler.py):
    group_id = 2*y + place   with place land=0 / water=1
    g0=(y0,land) g1=(y0,water) g2=(y1,land) g3=(y1,water); concordant = {0,3}.
"""
from __future__ import annotations

import numpy as np

__all__ = ["group_ids", "per_group_accuracy", "worst_group_accuracy"]


def group_ids(y: np.ndarray, place: np.ndarray) -> np.ndarray:
    """group_id = 2*y + place."""
    return (2 * np.asarray(y).astype(int) + np.asarray(place).astype(int)).astype(int)


def per_group_accuracy(y_pred: np.ndarray, y_true: np.ndarray,
                       group_id: np.ndarray) -> dict:
    """{group_id: top-1 accuracy on that group}."""
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    g = np.asarray(group_id)
    correct = (y_pred == y_true)
    return {int(grp): float(correct[g == grp].mean()) for grp in np.unique(g)}


def worst_group_accuracy(y_pred: np.ndarray, y_true: np.ndarray,
                         group_id: np.ndarray) -> tuple[int, float]:
    """(worst_group_id, its accuracy). Ties broken by smallest group id."""
    acc = per_group_accuracy(y_pred, y_true, group_id)
    worst_g = min(acc, key=lambda g: (acc[g], g))
    return int(worst_g), float(acc[worst_g])
