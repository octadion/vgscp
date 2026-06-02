"""Controlled spurious-correlation-STRENGTH shift by group-mix resampling (the shift axis).

Waterbirds groups: ``group_id = 2*y + place`` with place land=0 / water=1, so
  g0 = (y=0, place=0)  concordant   (majority)
  g1 = (y=0, place=1)  discordant   (minority)
  g2 = (y=1, place=0)  discordant   (minority)
  g3 = (y=1, place=1)  concordant   (majority)
A sample is "concordant" when place == y (groups 0, 3). We define the correlation STRENGTH

    rho = P(place == y)

at FIXED, balanced class marginals P(y=0)=P(y=1)=1/2. The target group fractions are then

    g0 = g3 = rho/2 ,   g1 = g2 = (1-rho)/2 ,

so rho=0.95 is the train-like regime (5% minority), rho=0.5 is independence (place ⟂ y), and
rho=0.25 is the FLIPPED-minority regime (the former minority groups become the majority). We
resample a pool to hit a target rho at a fixed total N (with replacement within each group, so
extreme rho stay feasible from a finite pool), and verify the realized rho is within tolerance.

Calibration is built ONCE at ``rho_cal`` (train-like); test sets are built across a sweep of
``rho_test`` from a DISJOINT pool, so calibration and test never share a sample.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# group_id convention (kept explicit so callers/readers don't have to rederive it)
GROUPS = (0, 1, 2, 3)
CONCORDANT_GROUPS = (0, 3)   # place == y
MINORITY_GROUPS = (1, 2)     # place != y


def target_group_fractions(rho: float) -> np.ndarray:
    """Target fraction per group (g0..g3) for correlation strength ``rho`` at balanced classes."""
    if not (0.0 <= rho <= 1.0):
        raise ValueError(f"rho must be in [0,1], got {rho}")
    return np.array([rho / 2.0, (1.0 - rho) / 2.0, (1.0 - rho) / 2.0, rho / 2.0], dtype=np.float64)


def realized_rho(group_id: np.ndarray) -> float:
    """Empirical P(place == y) = fraction of samples in the concordant groups {0, 3}."""
    g = np.asarray(group_id)
    if g.size == 0:
        return float("nan")
    return float(np.isin(g, CONCORDANT_GROUPS).mean())


def _target_counts(rho: float, n: int) -> np.ndarray:
    """Per-group integer counts summing to exactly ``n`` (largest-remainder rounding)."""
    frac = target_group_fractions(rho)
    raw = frac * n
    counts = np.floor(raw).astype(int)
    remainder = n - counts.sum()
    if remainder > 0:
        # hand the leftover slots to the groups with the largest fractional parts
        order = np.argsort(-(raw - counts))
        for k in range(remainder):
            counts[order[k % len(counts)]] += 1
    return counts


def split_pool(n_total: int, frac_cal: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Seeded disjoint split of pooled indices into (cal_pool, test_pool)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_total)
    n_cal = int(round(frac_cal * n_total))
    return np.sort(perm[:n_cal]), np.sort(perm[n_cal:])


@dataclass
class ResampleResult:
    idx: np.ndarray            # indices INTO the pool that was passed in
    counts: np.ndarray         # realized per-group counts (g0..g3)
    target_counts: np.ndarray
    rho_target: float
    rho_realized: float


def resample_to_rho(
    group_id_pool: np.ndarray,
    rho: float,
    n: int,
    seed: int,
    replace: bool = True,
) -> ResampleResult:
    """Resample pool indices to a target correlation strength ``rho`` at fixed total ``n``.

    Draws ``target_counts[g]`` samples from each group's pool members (with replacement by
    default, so an extreme rho is reachable even when a group is scarce). Returns indices INTO
    ``group_id_pool``. Raises if a needed group is entirely absent from the pool.
    """
    rng = np.random.default_rng(seed)
    g = np.asarray(group_id_pool)
    counts = _target_counts(rho, n)
    chosen = []
    for grp in GROUPS:
        want = int(counts[grp])
        if want == 0:
            continue
        members = np.where(g == grp)[0]
        if members.size == 0:
            raise ValueError(f"pool has no samples in group {grp}; cannot hit rho={rho}")
        if (not replace) and want > members.size:
            raise ValueError(f"group {grp}: need {want} but only {members.size} available "
                             f"(set replace=True for extreme rho)")
        chosen.append(rng.choice(members, size=want, replace=replace))
    idx = np.concatenate(chosen)
    rng.shuffle(idx)
    realized_counts = np.array([(g[idx] == grp).sum() for grp in GROUPS], dtype=int)
    return ResampleResult(idx=idx, counts=realized_counts, target_counts=counts,
                          rho_target=float(rho), rho_realized=realized_rho(g[idx]))
