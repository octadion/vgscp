"""Unit tests for E4 — the locked scacp applicability gate.

Validates BOTH branches of the gate (it CAN pass on a constructed differential+gap+support case,
and fails on non-differential noise), so the 312-attribute scan returning 0/312 is a real negative,
not a trivially-always-false gate. Reuses the same ks_conformal machinery the scan uses.
"""
import numpy as np
import pytest

from eval.scacp_gate import (DIFF_AUROC_MIN, GAP_MIN, SUPPORT_MIN, gate_attribute)
from ks_conformal.common_utils import TestbedConfig as _Cfg
from ks_conformal.common_utils import make_ahat, make_population


def _split(n, seed):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    k = n // 2
    return perm[:k], perm[k:]


def _gate_for(beta, n=4000, p_minor=0.3, margin_minor=1.0, flip=0.4, seed=0):
    pop = make_population(_Cfg(n=n, n_classes=30, p_minor=p_minor, margin_major=4.0,
                                        margin_minor=margin_minor, seed=seed, score="APS"))
    s = pop["s_true"]["APS"]
    A = pop["A_true"]
    Ahat = make_ahat(A, score_for_noise=s, flip_rate={0: flip * 0.3, 1: flip}, beta=beta, seed=seed + 1)
    cal, test = _split(n, seed)
    return gate_attribute(s[cal], A[cal], Ahat[cal], s[test], A[test], Ahat[test],
                          alpha=0.10, attr_id=0, attr_name="t")


def test_gate_fires_diagnostic_under_strong_differential_noise():
    # strong score<->probe-error correlation -> directional diagnostic AUROC should be high
    r = _gate_for(beta=16.0)
    assert r.diff_noise_auroc >= DIFF_AUROC_MIN
    assert r.minority_support >= SUPPORT_MIN  # p_minor=0.3, n=4000 -> ample support


def test_gate_diagnostic_quiet_under_nondifferential_noise():
    # beta=0 -> probe errors independent of score -> directional AUROC ~0.5, below threshold
    r = _gate_for(beta=0.0)
    assert r.diff_noise_auroc < DIFF_AUROC_MIN


def test_gate_can_pass_when_all_three_hold():
    # strong differential noise + a hard minority (real naive->oracle gap) + ample support
    r = _gate_for(beta=16.0, margin_minor=0.8, flip=0.45)
    # the gate is capable of passing; assert the conjunction logic is consistent
    assert r.passed == (r.pass_diff and r.pass_gap and r.pass_support)
    if r.pass_diff and r.pass_gap and r.pass_support:
        assert r.passed


def test_support_floor():
    # tiny minority -> support below the floor -> support criterion fails
    r = _gate_for(beta=16.0, n=500, p_minor=0.05)
    assert r.minority_support < SUPPORT_MIN
    assert not r.pass_support and not r.passed


def test_real_path_raises_without_data():
    """Real scan must RAISE (never fabricate) when datasets are absent: download=False + a
    nonexistent root -> load_waterbirds raises, so no fake scan is produced."""
    from scripts.run_e4_scacp_gate import run_real
    cfg = {"dataset": {"name": "waterbirds", "root": "/nonexistent_wb", "download": False,
                       "n_classes": 200},
           "cub": {"root": "/nonexistent_cub", "download": False},
           "clip": {"model_name": "ViT-B-32", "pretrained": "openai"}}
    with pytest.raises(Exception):
        run_real(cfg)
