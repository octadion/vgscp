"""Numerical-equivalence test for the optimized greedy prover selection.

The E2 perf fix rewrote ``ReimplNCV._greedy_select`` to (a) save/restore a single column instead of
cloning the whole (n,d) mask per candidate and (b) build ``chosen`` via one host transfer instead of
n*sparsity ``.item()`` calls. This test pins that the optimized version is BYTE-IDENTICAL to the
original clone-based algorithm (mask + chosen) on random inputs, incl. the allowed_mask branch — so
the speedup did not change the method.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from models.verifier_adapter import ReimplNCV


def _reference_greedy(v, concepts_t, sparsity, maximize_label, allowed_mask=None):
    """Literal reimplementation of the ORIGINAL clone-based greedy (the pre-optimization code)."""
    n, d = concepts_t.shape
    allowed = None
    if allowed_mask is not None:
        allowed = torch.as_tensor(np.asarray(allowed_mask, dtype=bool), device=concepts_t.device)
        sparsity = min(int(sparsity), int(allowed.sum().item()))
    mask = torch.zeros(n, d, device=concepts_t.device)
    chosen = [[] for _ in range(n)]
    rows = torch.arange(n, device=concepts_t.device)
    for _ in range(int(sparsity)):
        best_gain = torch.full((n,), -1e9, device=concepts_t.device)
        best_j = torch.zeros(n, dtype=torch.long, device=concepts_t.device)
        for j in range(d):
            if allowed is not None and not bool(allowed[j]):
                continue
            trial = mask.clone()
            trial[:, j] = 1.0
            p = v._arthur_probs(concepts_t, trial)[rows, maximize_label]
            already = mask[:, j] > 0
            gain = torch.where(already, torch.full_like(p, -1e9), p)
            upd = gain > best_gain
            best_gain = torch.where(upd, gain, best_gain)
            best_j = torch.where(upd, torch.full_like(best_j, j), best_j)
        mask[rows, best_j] = 1.0
        for i in range(n):
            chosen[i].append(int(best_j[i].item()))
    return mask, chosen


def _make_trained(n=60, d=14, n_classes=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, d)).astype(np.float32)
    y = rng.integers(0, n_classes, n)
    v = ReimplNCV(concept_dim=d, n_classes=n_classes, morgana_enabled=True, hidden=16, epochs=3,
                  merlin_sparsity=4, morgana_sparsity=4, batch_size=32, n_train_max=None,
                  standardize=False, device="cpu")
    v.train(X, y, seed=seed)
    xt = torch.as_tensor(X, dtype=torch.float32)
    return v, xt, y


@pytest.mark.parametrize("sparsity", [1, 4])
def test_greedy_equivalent_no_allowed(sparsity):
    v, xt, y = _make_trained()
    label = torch.as_tensor(y, dtype=torch.long)
    with torch.inference_mode():
        m_new, c_new = v._greedy_select(xt, sparsity, label)
        m_ref, c_ref = _reference_greedy(v, xt, sparsity, label)
    assert torch.equal(m_new, m_ref)
    assert c_new == c_ref


def test_greedy_equivalent_with_allowed_mask():
    v, xt, y = _make_trained(seed=1)
    label = torch.as_tensor(y, dtype=torch.long)
    allowed = np.zeros(xt.shape[1], dtype=bool)
    allowed[[1, 3, 5, 7, 9]] = True  # only a clean subset selectable (support-gap restriction)
    with torch.inference_mode():
        m_new, c_new = v._greedy_select(xt, 4, label, allowed_mask=allowed)
        m_ref, c_ref = _reference_greedy(v, xt, 4, label, allowed_mask=allowed)
    assert torch.equal(m_new, m_ref)
    assert c_new == c_ref
    # only allowed dims were ever chosen
    assert set(j for row in c_new for j in row) <= set(np.where(allowed)[0].tolist())


def test_return_chosen_false_matches_mask():
    v, xt, y = _make_trained(seed=2)
    label = torch.as_tensor(y, dtype=torch.long)
    with torch.inference_mode():
        m_full, c_full = v._greedy_select(xt, 4, label, return_chosen=True)
        m_skip, c_skip = v._greedy_select(xt, 4, label, return_chosen=False)
    assert torch.equal(m_full, m_skip)   # the mask is identical whether or not chosen is built
    assert c_skip is None
