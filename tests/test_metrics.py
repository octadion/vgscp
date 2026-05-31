"""Unit tests for AUROC / AURC / contamination metrics."""
import numpy as np
import pytest

from eval import metrics


def test_auroc_perfect_and_random():
    score = np.array([0.1, 0.2, 0.3, 0.9, 0.95])
    label = np.array([0, 0, 0, 1, 1])
    assert metrics.auroc(score, label) == pytest.approx(1.0)
    # reversed score -> 0.0
    assert metrics.auroc(-score, label) == pytest.approx(0.0)


def test_auroc_matches_sklearn():
    sklearn = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(0)
    score = rng.random(500)
    label = (rng.random(500) < 0.4).astype(int)
    assert metrics.auroc(score, label) == pytest.approx(
        sklearn.roc_auc_score(label, score), abs=1e-9
    )


def test_aurc_lower_for_better_signal():
    rng = np.random.default_rng(1)
    n = 2000
    correct = (rng.random(n) < 0.8).astype(int)
    good = correct + rng.normal(0, 0.3, n)   # informative
    bad = rng.normal(0, 1, n)                # uninformative
    assert metrics.aurc(good, correct) < metrics.aurc(bad, correct)


def test_capture_rate_confidence_is_zero_by_construction():
    # confident-but-wrong are, by definition, high confidence -> confidence abstains ~none.
    rng = np.random.default_rng(2)
    n = 2000
    conf = rng.random(n)
    correct = (rng.random(n) < 0.7).astype(int)
    # gating by confidence itself abstains the LOW-confidence samples, so confident errors
    # (high conf) are essentially never captured.
    rate = metrics.confident_but_wrong_capture_rate(conf, correct, conf, budget=0.2)
    assert rate < 0.05


def test_contamination_conditional_detects_dependence():
    # signal = spurious attr (within each class) -> high conditional contamination.
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 2000)
    a = rng.integers(0, 2, 2000)
    sig = a + rng.normal(0, 0.1, 2000)
    assert metrics.contamination_auroc(sig, a, y) > 0.9
    # independent signal -> ~0.5
    indep = rng.normal(0, 1, 2000)
    assert abs(metrics.contamination_auroc(indep, a, y) - 0.5) < 0.06
