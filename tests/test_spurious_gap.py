"""Unit tests for the counterfactual support-gap kill-switch machinery.

Covers (1) the mandatory byte-identical guarantee of the masked Merlin selector when
``allowed_mask=None``; (2) that a mask actually restricts Merlin to the allowed dims; (3) the
spuriousness scorer rho separates a planted spurious dim from clean dims; (4) clean_mask_from_rho;
(5) gap_signals shape/convention; and (6) the pre-committed GREEN/RED verdict branches.
"""
import numpy as np
import pandas as pd
import pytest

from eval.killswitch_verdict import killswitch_verdict
from signals.spurious_gap import (clean_mask_from_rho, concept_spuriousness, gap_signals,
                                  rho_recovery_summary)

torch = pytest.importorskip("torch")

from models.verifier_adapter import ReimplNCV


def _planted(n, seed):
    """4 clean core concepts (track y) + 1 spurious (tracks place, flips on a 25% minority) + noise."""
    rng = np.random.default_rng(seed)
    d = 10
    y = rng.integers(0, 2, n)
    minority = rng.random(n) < 0.25
    place = np.where(minority, 1 - y, y)
    X = rng.normal(0, 1, (n, d)).astype(np.float32)
    sy, sp = (2 * y - 1).astype(np.float32), (2 * place - 1).astype(np.float32)
    for k in range(4):
        X[:, k] = 1.8 * sy + rng.normal(0, 0.6, n)
    X[:, 4] = 2.4 * sp + rng.normal(0, 0.3, n)  # spurious dim (index 4)
    return X, y, place


def _trained(X, y, seed=0):
    v = ReimplNCV(concept_dim=X.shape[1], n_classes=2, merlin_sparsity=3, morgana_sparsity=3,
                  hidden=32, device="cpu", morgana_enabled=True, epochs=20, lr=2e-3,
                  batch_size=128, n_train_max=None, standardize=False)
    v.train(X, y, seed=seed)
    return v


def test_allowed_mask_none_is_byte_identical():
    """allowed_mask=None MUST reproduce the original greedy selection exactly (mask + chosen),
    and an all-True mask must give the identical selection too."""
    X, y, _ = _planted(200, 0)
    v = _trained(X, y)
    xt = torch.as_tensor(X, dtype=torch.float32)
    yp = torch.zeros(len(y), dtype=torch.long)
    with torch.inference_mode():
        m_none, sel_none = v._greedy_select(xt, 3, yp, allowed_mask=None)
        m_all, sel_all = v._greedy_select(xt, 3, yp, allowed_mask=np.ones(X.shape[1], dtype=bool))
    assert torch.equal(m_none, m_all)
    assert sel_none == sel_all


def test_allowed_mask_restricts_selection():
    """A restrictive mask must keep every chosen concept inside the allowed set."""
    X, y, _ = _planted(200, 1)
    v = _trained(X, y, seed=1)
    xt = torch.as_tensor(X, dtype=torch.float32)
    yp = torch.zeros(len(y), dtype=torch.long)
    allowed = np.zeros(X.shape[1], dtype=bool)
    allowed[[0, 1, 2]] = True  # only the first three (clean) dims
    with torch.inference_mode():
        mask, sel = v._greedy_select(xt, 3, yp, allowed_mask=allowed)
    chosen = np.array(sel)
    assert set(np.unique(chosen)).issubset({0, 1, 2})
    # disallowed columns are never switched on
    assert mask[:, 3:].sum().item() == 0.0


def test_merlin_completeness_matches_predict_full():
    """merlin_completeness(allowed_mask=None) == V_comp from predict() (full-bank Merlin)."""
    X, y, _ = _planted(200, 2)
    v = _trained(X, y, seed=2)
    yp = (y ^ (np.arange(len(y)) % 3 == 0)).astype(np.int64)  # some flipped preds
    out = v.predict(X, yp)
    v_comp = out.pA_given_SM[np.arange(len(yp)), yp]
    full = v.merlin_completeness(X, yp, allowed_mask=None)
    assert np.allclose(full, v_comp, atol=1e-5)


def test_concept_spuriousness_flags_planted_dim():
    X, y, place = _planted(600, 3)
    rho = concept_spuriousness(X, place)
    assert rho[4] > 0.6                       # planted spurious dim is high
    assert rho[5:].max() < rho[4]             # noise dims lower
    assert rho[:4].max() < rho[4] + 1e-9      # clean dims not above the spurious one


def test_clean_mask_from_rho_and_recovery():
    rho = np.array([0.05, 0.08, 0.1, 0.9, 0.95])  # 3 clean, 2 spurious
    mask, tau = clean_mask_from_rho(rho)
    assert mask.tolist() == [True, True, True, False, False]
    assert 0.1 < tau < 0.9
    rec = rho_recovery_summary(rho, mask, n_clean_true=3)
    assert rec["clean_recall"] == 1.0 and rec["spurious_recall"] == 1.0
    # explicit tau overrides the data-driven choice
    mask2, tau2 = clean_mask_from_rho(rho, tau=0.5)
    assert tau2 == 0.5 and mask2.tolist() == [True, True, True, False, False]


def test_gap_signals_shapes_and_convention():
    X, y, place = _planted(300, 4)
    v = _trained(X, y, seed=4)
    rho = concept_spuriousness(X, place)
    clean_mask, _ = clean_mask_from_rho(rho)
    yp = y.astype(np.int64)
    gs = gap_signals(v, X, yp, clean_mask, lam=1.0)
    for k in ("support_clean", "V_gap", "V_gap_pure", "_support_full", "_gap"):
        assert gs[k].shape == (len(y),)
    assert np.allclose(gs["_gap"], gs["_support_full"] - gs["support_clean"])
    assert np.allclose(gs["V_gap"], gs["support_clean"] - 1.0 * gs["_gap"])
    assert np.allclose(gs["V_gap_pure"], -gs["_gap"])


# ---- verdict branches (no torch needed beyond the import above) ----
def _dose_df(correct, sig, n_majority=40):
    n = len(correct)
    rows = {"correct": list(correct) + [1] * n_majority,
            "is_minority": [1] * n + [0] * n_majority}
    rng = np.random.default_rng(0)
    for name, vals in sig.items():
        rows[name] = list(vals) + list(rng.random(n_majority))
    return pd.DataFrame(rows)


def _sep(correct, rng, strength):
    return np.asarray(correct) * strength + rng.normal(0, 1.0, len(correct))


def test_verdict_green():
    rng = np.random.default_rng(1)
    correct = rng.integers(0, 2, 300)
    # V_gap separates strongly; both trust_concept and support_clean separate weakly -> GREEN.
    sig = {"V_gap": _sep(correct, rng, 6.0), "support_clean": _sep(correct, rng, 0.5),
           "trust_concept": _sep(correct, rng, 0.5)}
    v = killswitch_verdict([{"dose": 16, "lam": 1.0, "df": _dose_df(correct, sig)}], n_resamples=400)
    assert v.label == "GREEN", v.rationale
    assert 16 in v.green_doses


def test_verdict_red_ties_trust_concept():
    rng = np.random.default_rng(2)
    correct = rng.integers(0, 2, 300)
    strong = _sep(correct, rng, 6.0)
    # trust_concept ties V_gap exactly -> never beats THE BAR -> RED.
    sig = {"V_gap": strong, "support_clean": _sep(correct, rng, 0.5), "trust_concept": strong.copy()}
    v = killswitch_verdict([{"dose": 16, "lam": 1.0, "df": _dose_df(correct, sig)}], n_resamples=400)
    assert v.label == "RED", v.rationale


def test_verdict_red_never_beats_support_clean():
    rng = np.random.default_rng(3)
    correct = rng.integers(0, 2, 300)
    strong = _sep(correct, rng, 6.0)
    # support_clean ties V_gap (gap adds nothing) though V_gap beats trust_concept -> RED.
    sig = {"V_gap": strong, "support_clean": strong.copy(), "trust_concept": _sep(correct, rng, 0.5)}
    v = killswitch_verdict([{"dose": 16, "lam": 1.0, "df": _dose_df(correct, sig)}], n_resamples=400)
    assert v.label == "RED", v.rationale
