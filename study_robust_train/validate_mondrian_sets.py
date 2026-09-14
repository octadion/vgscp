"""Check the group-conditional set construction on synthetic scores.

Three things must hold after the label-conditional fix in ``conformal.group_robust``:

  1. the set is computable at test time --- membership depends on the scores and on the spurious
     attribute, never on the test label;
  2. per-group coverage is exactly what the previous (own-group) construction gave, since the true
     label is compared against its own group's quantile either way;
  3. set sizes are NOT the same, which is why every set-size number has to be regenerated.

Run: python -m study_robust_train.validate_mondrian_sets
"""
from __future__ import annotations

import numpy as np

from conformal.group_robust import mondrian_build_sets, mondrian_quantiles


def _own_group_sets(test_scores_all: np.ndarray, test_group: np.ndarray, group_q: dict):
    """The superseded construction: one quantile per point, used for every candidate label."""
    qvec = np.array([group_q.get(int(g), float("inf")) for g in test_group])
    return test_scores_all <= qvec[:, None]


def main(n: int = 20000, alpha: float = 0.1, seed: int = 0) -> bool:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=2 * n)
    a = np.where(rng.random(2 * n) < 0.95, y, 1 - y)      # rho = 0.95, as in the study
    group = 2 * y + a
    # scores are easier on the aligned groups, so the per-group quantiles genuinely differ
    s_true = rng.beta(2, 8, size=2 * n) + 0.35 * (y != a)
    scores = np.stack([s_true, rng.beta(2, 8, size=2 * n) + 0.2], axis=1)
    scores[np.arange(2 * n), 1 - y] = scores[np.arange(2 * n), 1]
    scores[np.arange(2 * n), y] = s_true

    cal, test = slice(0, n), slice(n, 2 * n)
    gq = mondrian_quantiles(scores[cal][np.arange(n), y[cal]], group[cal], alpha)
    new = mondrian_build_sets(scores[test], group[test], gq)
    old = _own_group_sets(scores[test], group[test], gq)
    yt, gt = y[test], group[test]

    cov_new = {g: new[gt == g][np.arange((gt == g).sum()), yt[gt == g]].mean() for g in np.unique(gt)}
    cov_old = {g: old[gt == g][np.arange((gt == g).sum()), yt[gt == g]].mean() for g in np.unique(gt)}
    same_cov = all(abs(cov_new[g] - cov_old[g]) < 1e-12 for g in cov_new)

    # label independence: relabel every point, keep the attribute, rebuild
    flipped = 2 * (1 - yt) + (gt % 2)
    label_free = np.array_equal(new, mondrian_build_sets(scores[test], flipped, gq))
    sizes_move = float(np.abs(new.sum(1) - old.sum(1)).mean())

    print(f"per-group coverage unchanged : {same_cov}  {[round(float(v), 4) for v in cov_new.values()]}")
    print(f"membership free of the label : {label_free}")
    print(f"mean |size_new - size_old|   : {sizes_move:.4f}  "
          f"(old {old.sum(1).mean():.3f} -> new {new.sum(1).mean():.3f})")
    ok = same_cov and label_free and sizes_move > 0
    print("OK" if ok else "FAILED")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
