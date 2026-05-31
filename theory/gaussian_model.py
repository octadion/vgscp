"""Synthetic Gaussian shortcut model (Section 3, 5).

A controllable testbed where a *spurious* feature correlates with the label on the majority
group and FLIPS on a minority (conflict) group, while a *core* feature is genuinely predictive.
This lets us verify P1 (selective-conformal validity) and P2 (separation: confidence/trust are
contaminated by the spurious attribute and fail on the minority, while concept-space
verifiability V is approximately independent of the spurious attribute yet predicts correctness).

Feature layout per sample:  x = [ core (d_core) | spurious (d_spurious) | noise (d_noise) ].
  - core dims  ~ N( (2y-1) * core_sep/2 , 1 )                  # genuinely class-relevant
  - spurious   ~ N( (2a-1) * mu_spur ,    1 ),  a = spurious attribute
  - noise dims ~ N( 0, 1 )

Spurious attribute a equals the label on the majority and is flipped on the minority/conflict
group (fraction = minority_frac). ``shortcut_strength`` s in [0.5, 1] scales the spurious
feature's separation: mu_spur = spurious_sep/2 * (2s - 1). At s=0.5 the spurious feature carries
no class signal (no shortcut); at s=1 it is maximally separated and a model that latches onto it
becomes confident-but-wrong on the minority.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GaussianConfig:
    n: int = 8000
    d_core: int = 2
    d_spurious: int = 2
    d_noise: int = 4
    core_sep: float = 1.2
    spurious_sep: float = 2.5
    minority_frac: float = 0.2
    shortcut_strength: float = 0.9   # s in [0.5, 1.0]

    @property
    def d_total(self) -> int:
        return self.d_core + self.d_spurious + self.d_noise

    def core_slice(self):
        return slice(0, self.d_core)

    def spurious_slice(self):
        return slice(self.d_core, self.d_core + self.d_spurious)

    def noise_slice(self):
        return slice(self.d_core + self.d_spurious, self.d_total)


@dataclass
class GaussianSample:
    x: np.ndarray            # (n, d_total) full feature vector phi(x)
    y: np.ndarray            # (n,) label in {0,1}
    a: np.ndarray            # (n,) spurious attribute in {0,1}
    is_minority: np.ndarray  # (n,) bool: a != y (conflict group)
    group_id: np.ndarray     # (n,) class x attribute group id = 2*y + a

    @property
    def core(self):
        return self._cfg_slice("core")

    def _cfg_slice(self, which):  # set lazily by generator
        raise RuntimeError("use the slices returned by generate()")


def generate(cfg: GaussianConfig, seed: int) -> dict:
    """Generate one split. Returns a dict with features, labels, attributes, masks, slices."""
    rng = np.random.default_rng(seed)
    n = cfg.n

    y = rng.integers(0, 2, size=n)
    # assign minority (conflict) membership
    is_minority = rng.random(n) < cfg.minority_frac
    a = np.where(is_minority, 1 - y, y)

    mu_core = cfg.core_sep / 2.0
    mu_spur = cfg.spurious_sep / 2.0 * (2.0 * cfg.shortcut_strength - 1.0)

    x = np.empty((n, cfg.d_total), dtype=np.float64)
    core_sign = (2 * y - 1)[:, None]
    spur_sign = (2 * a - 1)[:, None]
    x[:, cfg.core_slice()] = rng.normal(0, 1, (n, cfg.d_core)) + mu_core * core_sign
    x[:, cfg.spurious_slice()] = rng.normal(0, 1, (n, cfg.d_spurious)) + mu_spur * spur_sign
    x[:, cfg.noise_slice()] = rng.normal(0, 1, (n, cfg.d_noise))

    group_id = 2 * y + a

    return {
        "x": x,
        "y": y,
        "a": a,
        "is_minority": is_minority,
        "group_id": group_id,
        "core_slice": cfg.core_slice(),
        "spurious_slice": cfg.spurious_slice(),
        "noise_slice": cfg.noise_slice(),
        "cfg": cfg,
    }


def generate_splits(cfg: GaussianConfig, seed: int, names=("train", "d_learn", "d_cal", "d_test")):
    """Generate disjoint i.i.d. splits (fresh draws => exchangeable, non-overlapping)."""
    return {name: generate(cfg, seed=seed * 100 + i) for i, name in enumerate(names)}
