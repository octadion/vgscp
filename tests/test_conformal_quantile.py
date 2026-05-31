"""Unit tests for the conformal quantile and split-conformal coverage guarantee."""
import math

import numpy as np
import pytest

from conformal import scores as cscores
from conformal.split_conformal import (
    build_sets,
    conformal_quantile,
    covered,
    marginal_coverage,
)


def test_quantile_matches_kth_smallest():
    # qhat must equal the ceil((n+1)(1-alpha))-th smallest calibration score.
    rng = np.random.default_rng(0)
    s = rng.random(99)
    alpha = 0.1
    q = conformal_quantile(s, alpha)
    k = math.ceil((len(s) + 1) * (1 - alpha))  # = 90
    expected = np.sort(s)[k - 1]
    assert q == pytest.approx(expected)


def test_quantile_infinite_when_n_too_small():
    # With n < 1/alpha - 1 the level exceeds 1 -> full sets (qhat = +inf), still valid.
    assert conformal_quantile(np.array([0.2, 0.5]), alpha=0.1) == float("inf")


def test_split_conformal_marginal_coverage():
    # On exchangeable data, THR split-conformal must achieve >= 1-alpha marginal coverage.
    rng = np.random.default_rng(1)
    n, C, alpha = 4000, 4, 0.1

    def make_probs(m):
        logits = rng.normal(size=(m, C))
        e = np.exp(logits - logits.max(1, keepdims=True))
        p = e / e.sum(1, keepdims=True)
        y = np.array([rng.choice(C, p=p[i]) for i in range(m)])  # labels ~ true p
        return p, y

    cal_p, cal_y = make_probs(n)
    test_p, test_y = make_probs(n)
    cal_scores = cscores.thr_scores_all(cal_p)[np.arange(n), cal_y]
    qhat = conformal_quantile(cal_scores, alpha)
    membership = build_sets(cscores.thr_scores_all(test_p), qhat)
    cov = marginal_coverage(membership, test_y)
    # finite-sample: coverage >= 1-alpha within a few SE
    se = math.sqrt((1 - alpha) * alpha / n)
    assert cov >= (1 - alpha) - 3 * se


@pytest.mark.parametrize("score_name", ["THR", "APS", "RAPS"])
def test_scores_shape_and_coverage(score_name):
    rng = np.random.default_rng(2)
    n, C, alpha = 3000, 5, 0.1

    def make(m):
        logits = rng.normal(size=(m, C)) * 1.5
        e = np.exp(logits - logits.max(1, keepdims=True))
        p = e / e.sum(1, keepdims=True)
        y = np.array([rng.choice(C, p=p[i]) for i in range(m)])
        return p, y

    cal_p, cal_y = make(n)
    test_p, test_y = make(n)
    u_cal = cscores.draw_randomization(n, seed=7)
    u_test = cscores.draw_randomization(n, seed=8)
    cal_all = cscores.scores_all(score_name, cal_p, u=u_cal)
    test_all = cscores.scores_all(score_name, test_p, u=u_test)
    assert cal_all.shape == (n, C)
    qhat = conformal_quantile(cal_all[np.arange(n), cal_y], alpha)
    cov = marginal_coverage(build_sets(test_all, qhat), test_y)
    se = math.sqrt((1 - alpha) * alpha / n)
    assert cov >= (1 - alpha) - 4 * se
