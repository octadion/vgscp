"""§4 headline metric — cross-group conformity-score divergence (Wasserstein-1 + KS).

THIS DOES NOT EXIST ANYWHERE IN THE v1-v4 CODEBASE (AUDIT_study.md §3); it is written fresh.
The only prior divergence primitive was a TV distance between CAL vs TEST (a temporal-shift
diagnostic), not between PER-GROUP distributions.

Definition (spec §4): for a given (model x score function), on a held-out test split, compute
the per-group distribution of the TRUE-LABEL conformity score s(x, y_true), then the cross-group
divergence between the WORST group and the REST (pooled). We report Wasserstein-1 and the KS
statistic. Per the spec, results are reported PER SCORE FUNCTION separately and NEVER averaged
across score functions (APS/RAPS/THR live on different scales). Phase-0 reports APS only.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance

from conformal.scores import draw_randomization, scores_all, true_label_scores

__all__ = ["true_label_conformity_scores", "cross_group_divergence", "GroupDivergence"]


def true_label_conformity_scores(probs: np.ndarray, y_true: np.ndarray, *,
                                 score: str = "APS", seed: int = 0,
                                 randomize: bool = True) -> np.ndarray:
    """Per-sample true-label conformity score s(x, y_true) for a named score family.

    Reuses ``conformal.scores`` as-is. ``randomize`` draws the APS/RAPS uniform u once with a
    fixed seed (no effect on THR). Lower score = more conforming.
    """
    probs = np.asarray(probs, dtype=np.float64)
    y_true = np.asarray(y_true)
    u = draw_randomization(probs.shape[0], seed) if randomize else None
    s_all = scores_all(score, probs, u=u)
    return true_label_scores(s_all, y_true)


@dataclass
class GroupDivergence:
    score: str
    worst_group: int
    n_worst: int
    n_rest: int
    wasserstein1: float
    ks_stat: float
    ks_pvalue: float
    worst_mean: float
    rest_mean: float


def cross_group_divergence(scores: np.ndarray, group_id: np.ndarray, worst_group: int, *,
                           score_name: str = "APS") -> GroupDivergence:
    """Wasserstein-1 + KS between the worst group's score distribution and the pooled rest.

    ``worst_group`` is supplied by the caller (Phase-0 uses the lowest-accuracy group, so the
    burden metric and the worst-group-accuracy headline reference the same group). Raises if a
    side is empty.
    """
    scores = np.asarray(scores, dtype=np.float64)
    group_id = np.asarray(group_id)
    worst_mask = group_id == worst_group
    rest_mask = ~worst_mask
    s_worst = scores[worst_mask]
    s_rest = scores[rest_mask]
    if s_worst.size == 0 or s_rest.size == 0:
        raise ValueError(
            f"[divergence] empty side: |worst|={s_worst.size}, |rest|={s_rest.size} "
            f"for worst_group={worst_group}."
        )
    w1 = float(wasserstein_distance(s_worst, s_rest))
    ks = ks_2samp(s_worst, s_rest)
    return GroupDivergence(
        score=score_name, worst_group=int(worst_group),
        n_worst=int(s_worst.size), n_rest=int(s_rest.size),
        wasserstein1=w1, ks_stat=float(ks.statistic), ks_pvalue=float(ks.pvalue),
        worst_mean=float(s_worst.mean()), rest_mean=float(s_rest.mean()),
    )
