"""Unit tests for the faithful Prover-Verifier-Game verifier (models/verifier_adapter.ReimplNCV).

The decisive sanity check (task Section 3): on a tiny synthetic concept space with a planted
spurious concept, after the alternating PVG training Arthur's accuracy UNDER MERLIN (cooperative,
helpful evidence for the true label) must exceed its accuracy UNDER MORGANA (adversarial,
misleading evidence). If the game is wired correctly, Merlin helps and Morgana hurts.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from models.verifier_adapter import ReimplNCV, build_verifier


def _planted_concept_space(n, seed):
    """Binary task. Concepts: 4 reliable core concepts aligned with y; 1 SPURIOUS concept that
    agrees with y on the majority but FLIPS on a 25% minority (so a sparse subset containing it
    can mislead Arthur); the rest are noise. Returns (concepts (n,12), y, is_minority)."""
    rng = np.random.default_rng(seed)
    d = 12
    y = rng.integers(0, 2, n)
    minority = rng.random(n) < 0.25
    place = np.where(minority, 1 - y, y)  # spurious attribute flips on the minority
    X = rng.normal(0.0, 1.0, size=(n, d)).astype(np.float32)
    sy = (2 * y - 1).astype(np.float32)
    sp = (2 * place - 1).astype(np.float32)
    for k in range(4):                      # core concepts: strong, honest y-signal
        X[:, k] = 1.8 * sy + rng.normal(0, 0.6, n)
    X[:, 4] = 2.2 * sp + rng.normal(0, 0.4, n)   # spurious concept (tracks place, not y)
    # cols 5..11 stay noise
    return X, y, minority


def test_pvg_merlin_beats_morgana():
    X, y, _ = _planted_concept_space(500, seed=0)
    v = ReimplNCV(concept_dim=X.shape[1], n_classes=2, merlin_sparsity=3, morgana_sparsity=3,
                  hidden=32, device="cpu", morgana_enabled=True, epochs=30, lr=2e-3,
                  batch_size=128, n_train_max=None)
    v.train(X, y, seed=0)
    m = v.intrinsic_metrics(X, y)
    # Merlin can support the true label from the core concepts -> high completeness.
    assert m["merlin_acc"] > 0.75, m
    # The game bites: Morgana's misleading set drives Arthur off the true label.
    assert m["merlin_acc"] > m["morgana_acc"] + 0.10, m


def test_pvg_predict_shapes_and_reject():
    X, y, _ = _planted_concept_space(200, seed=1)
    v = ReimplNCV(concept_dim=X.shape[1], n_classes=2, merlin_sparsity=3, morgana_sparsity=3,
                  hidden=32, device="cpu", morgana_enabled=True, epochs=10, batch_size=128,
                  n_train_max=None)
    v.train(X, y, seed=1)
    y_pred = np.zeros(len(y), dtype=np.int64)  # arbitrary downstream predictions
    out = v.predict(X, y_pred)
    assert out.pA_given_SM.shape == (len(y), 2)
    assert out.pA_given_SA.shape == (len(y), 2)
    assert out.reject_prob is not None and out.reject_prob.shape == (len(y),)
    # probabilities over classes are valid (<=1 row-sum since a reject mass may be siphoned off)
    assert np.all(out.pA_given_SM >= -1e-5) and np.all(out.pA_given_SM <= 1 + 1e-5)
    assert len(out.merlin_concepts) == len(y) and len(out.merlin_concepts[0]) == 3


def test_morgana_off_ablation_has_no_adversary():
    """morgana=off => no reject head, and S_A == S_M (V_sound carries no adversarial info)."""
    X, y, _ = _planted_concept_space(200, seed=2)
    v = build_verifier({"source": "reimpl", "morgana": "off", "epochs": 10, "hidden": 32,
                        "merlin_sparsity": 3, "morgana_sparsity": 3, "n_train_max": None},
                       concept_dim=X.shape[1], n_classes=2, device="cpu")
    assert v.morgana_enabled is False
    assert v.has_reject is False
    v.train(X, y, seed=2)
    out = v.predict(X, np.zeros(len(y), dtype=np.int64))
    # adversarial set collapses onto the cooperative set; reject signal is identically zero
    assert np.allclose(out.pA_given_SM, out.pA_given_SA)
    assert np.allclose(out.reject_prob, 0.0)
