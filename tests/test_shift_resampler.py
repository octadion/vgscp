"""Unit tests for the correlation-strength shift resampler.

Covers: target group fractions, realized rho hitting the target within tolerance across the sweep,
fixed-N counts, and the disjoint cal/test pool split.
"""
import numpy as np
import pytest

from experiments import shift_resampler as rs


def _pool(seed=0, n=6000):
    """Synthetic pool with all 4 Waterbirds groups well populated (rho ~ 0.7)."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    place = np.where(rng.random(n) < 0.7, y, 1 - y)  # ~70% concordant
    return (2 * y + place).astype(int)


def test_target_fractions_sum_to_one_and_symmetry():
    for rho in (0.95, 0.75, 0.5, 0.25):
        frac = rs.target_group_fractions(rho)
        assert frac.sum() == pytest.approx(1.0)
        assert frac[0] == pytest.approx(frac[3])      # concordant groups symmetric
        assert frac[1] == pytest.approx(frac[2])      # minority groups symmetric
        assert frac[0] == pytest.approx(rho / 2)


@pytest.mark.parametrize("rho", [0.95, 0.75, 0.50, 0.25])
def test_resample_hits_target_rho_within_tolerance(rho):
    group_pool = _pool()
    n = 2000
    res = rs.resample_to_rho(group_pool, rho, n, seed=1)
    assert res.idx.shape[0] == n                      # fixed N respected
    assert res.counts.sum() == n
    # realized rho = concordant fraction must match target within rounding tolerance
    assert abs(res.rho_realized - rho) <= 0.02
    # realized counts match the target counts exactly (deterministic largest-remainder rounding)
    np.testing.assert_array_equal(res.counts, res.target_counts)


def test_flipped_minority_regime_inverts_majority():
    group_pool = _pool()
    res = rs.resample_to_rho(group_pool, 0.25, 2000, seed=2)
    # at rho=0.25 the minority groups (1,2) dominate over the concordant groups (0,3)
    minority = res.counts[1] + res.counts[2]
    concordant = res.counts[0] + res.counts[3]
    assert minority > concordant
    assert res.rho_realized < 0.5


def test_fixed_n_across_rho():
    group_pool = _pool()
    sizes = {rho: rs.resample_to_rho(group_pool, rho, 1500, seed=3).idx.shape[0]
             for rho in (0.95, 0.75, 0.5, 0.25)}
    assert set(sizes.values()) == {1500}              # total test N fixed across the sweep


def test_split_pool_disjoint_and_complete():
    cal, test = rs.split_pool(1000, frac_cal=0.5, seed=7)
    assert set(cal.tolist()).isdisjoint(test.tolist())
    assert sorted(cal.tolist() + test.tolist()) == list(range(1000))
    assert len(cal) == 500


def test_missing_group_raises_without_replacement_headroom():
    # a pool missing group 1 cannot reach a rho that needs minority mass
    g = np.array([0, 0, 3, 3, 2, 2])   # no group 1
    with pytest.raises(ValueError):
        rs.resample_to_rho(g, rho=0.5, n=100, seed=0)
