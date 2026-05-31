"""Unit tests for the selective gate and the validity-protocol guards."""
import math

import numpy as np
import pytest

from conformal import scores as cscores
from conformal.selective import gate_threshold, retained_mask, selective_conformal
from conformal.validity import (
    ValidityError,
    assert_disjoint_splits,
    assert_gate_is_x_measurable,
    assert_hyperparams_from_learn,
    check_marginal_coverage,
)


def _exchangeable(n, C, rng):
    logits = rng.normal(size=(n, C))
    e = np.exp(logits - logits.max(1, keepdims=True))
    p = e / e.sum(1, keepdims=True)
    y = np.array([rng.choice(C, p=p[i]) for i in range(n)])
    return p, y


def test_gate_threshold_retains_top_fraction():
    g = np.arange(100.0)
    tau = gate_threshold(g, budget=0.2)
    assert retained_mask(g, tau).mean() == pytest.approx(0.8, abs=0.02)
    # budget 0 retains everything
    assert retained_mask(g, gate_threshold(g, 0.0)).all()


@pytest.mark.parametrize("budget", [0.0, 0.2, 0.4])
def test_selective_conformal_retained_coverage_valid(budget):
    # Marginal coverage is an EXPECTATION guarantee, so we average realized coverage over many
    # independent draws (a single draw shares one qhat and has inflated variance). With an
    # x-measurable gate, the mean retained coverage must be >= 1-alpha (up to small slack).
    alpha = 0.1
    covs = []
    for seed in range(25):
        rng = np.random.default_rng(seed)
        learn_p, learn_y = _exchangeable(2000, 4, rng)
        cal_p, cal_y = _exchangeable(2000, 4, rng)
        test_p, test_y = _exchangeable(2000, 4, rng)
        g_learn, g_cal, g_test = learn_p.max(1), cal_p.max(1), test_p.max(1)
        thr_cal = cscores.thr_scores_all(cal_p)
        thr_test = cscores.thr_scores_all(test_p)
        res = selective_conformal(g_learn, g_cal, g_test, thr_cal, cal_y,
                                  thr_test, test_y, alpha, budget)
        if np.isfinite(res.coverage):
            covs.append(res.coverage)
    mean_cov = float(np.mean(covs))
    assert mean_cov >= (1 - alpha) - 0.01, f"mean retained coverage {mean_cov} < target at b={budget}"


def test_disjoint_splits_guard():
    ok = {"train": np.array([0, 1, 2]), "cal": np.array([3, 4]), "test": np.array([5, 6])}
    assert_disjoint_splits(ok)  # no raise
    bad = {"train": np.array([0, 1, 2]), "cal": np.array([2, 3])}
    with pytest.raises(ValidityError):
        assert_disjoint_splits(bad)


def test_gate_x_measurable_guard_flags_label_leak():
    # A gate that is a deterministic function of the label must be rejected.
    y = np.array([0, 0, 1, 1, 2, 2])
    leaking_gate = y.astype(float) * 10.0  # one gate value per label
    with pytest.raises(ValidityError):
        assert_gate_is_x_measurable(leaking_gate, y)
    # a continuous gate not tied to the label passes
    rng = np.random.default_rng(0)
    assert_gate_is_x_measurable(rng.random(len(y)), y)


def test_hyperparams_source_guard():
    assert_hyperparams_from_learn("d_learn")  # no raise
    with pytest.raises(ValidityError):
        assert_hyperparams_from_learn("d_cal")
    with pytest.raises(ValidityError):
        assert_hyperparams_from_learn("d_test")


def test_coverage_check_passes_within_noise_fails_on_gross_undercoverage():
    # within finite-sample noise of target -> pass
    ok, _ = check_marginal_coverage(0.887, alpha=0.1, n=2700)
    assert ok
    # grossly under target with large n -> fail
    bad, _ = check_marginal_coverage(0.70, alpha=0.1, n=5000)
    assert not bad
