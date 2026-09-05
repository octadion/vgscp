"""Reviewer-requested statistics on the grid records (R2.3, R3.1, R3.2).

``stats.py`` holds the estimators; nothing connected them to the recorded results, so the paper
could not report what the reviewers asked for. This module is that wiring and nothing more -- it
recomputes no experiment and changes no number, it only attaches uncertainty to numbers already
measured.

  R2.3  cluster_bootstrap_ci on every cell of the main tables. Splits are nested within seeds, so
        the bootstrap resamples seeds first and splits within them; treating the 50 (seed, split)
        rows as independent would understate the interval.
  R3.1  spread_equivalence across methods. The paper says the efficiency range is "narrow"; TOST
        turns that adjective into a test with a stated margin.
  R3.2  correlation_ci on the (worst-group accuracy, worst-group set size) correlation that
        Table 2 reports per cell. Those correlations are computed over 4-5 methods, so the
        ``small_n`` flag matters more than the point estimate.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from .stats import cluster_bootstrap_ci, correlation_ci, spread_equivalence

__all__ = ["arm_means", "cell_table_with_ci", "correlation_with_ci", "efficiency_equivalence",
           "seed_variance_audit"]


def _filter(records, **eq):
    return [r for r in records if all(r.get(k) == v for k, v in eq.items())]


def arm_means(records, field, **eq):
    """{(backbone, dataset, method): (values, seeds, splits)} for one slice of the grid."""
    out = defaultdict(lambda: ([], [], []))
    for r in _filter(records, **eq):
        v, s, sp = out[(r["backbone"], r["dataset"], r["method"])]
        v.append(r[field]); s.append(r["train_seed"]); sp.append(r["split_seed"])
    return out


def cell_table_with_ci(records, field, *, B=2000, alpha=0.05, seed=0, **eq):
    """R2.3: mean of ``field`` per arm with a two-stage cluster bootstrap CI."""
    rows = {}
    for key, (vals, seeds, splits) in sorted(arm_means(records, field, **eq).items()):
        if not vals:
            continue
        ci = cluster_bootstrap_ci(np.asarray(vals, float), np.asarray(seeds),
                                  np.asarray(splits), B=B, alpha=alpha, seed=seed)
        rows[key] = {"mean": float(np.mean(vals)), "lo": ci["lo"], "hi": ci["hi"],
                     "n_seeds": len(set(seeds)), "n_rows": len(vals)}
    return rows


def correlation_with_ci(records, *, x_field="worst_group_acc", y_field="worst_group_set_size",
                        B=2000, alpha=0.05, seed=0, **eq):
    """R3.2: Table 2's per-cell correlation, with a CI and an explicit small-n flag.

    One point per METHOD, exactly as the table computes it -- so n is 4 or 5. The CI is the honest
    reading of that: with n=4 a Pearson r of -1.00 is not evidence of a deterministic relationship.
    """
    # Paired BY METHOD NAME, not by iteration order. Building the x and y lists from two separate
    # passes and zipping them by position would silently mispair the axes if the two passes ever
    # enumerated methods differently, and the resulting correlation would look perfectly ordinary.
    xm = {k: float(np.mean(v[0])) for k, v in arm_means(records, x_field, **eq).items()}
    ym = {k: float(np.mean(v[0])) for k, v in arm_means(records, y_field, **eq).items()}
    per_cell = defaultdict(lambda: ([], []))
    for key in sorted(set(xm) & set(ym)):
        per_cell[(key[0], key[1])][0].append(xm[key])
        per_cell[(key[0], key[1])][1].append(ym[key])
    out = {}
    for cell, (xs, ys) in sorted(per_cell.items()):
        if len(xs) != len(ys) or len(xs) < 3:
            out[cell] = {"n_methods": len(xs), "note": "too few methods for a correlation"}
            continue
        res = correlation_ci(np.asarray(xs), np.asarray(ys), B=B, alpha=alpha, seed=seed)
        res["n_methods"] = len(xs)
        out[cell] = res
    return out


def efficiency_equivalence(records, *, field="worst_group_set_size", margin=0.10,
                           B=2000, alpha=0.05, seed=0, **eq):
    """R3.1: is the across-method spread in ``field`` inside an equivalence margin?"""
    out = {}
    per_cell = defaultdict(dict)
    for (bb, ds, m), (vals, seeds, _sp) in arm_means(records, field, **eq).items():
        per_cell[(bb, ds)][m] = (np.asarray(vals, float), np.asarray(seeds))
    for cell, by_method in sorted(per_cell.items()):
        if len(by_method) < 2:
            continue
        out[cell] = spread_equivalence(by_method, margin=margin, B=B, alpha=alpha, seed=seed)
    return out


def seed_variance_audit(records, field="worst_group_acc"):
    """Which methods actually vary with the seed.

    ``erm`` and ``afr`` fit with lbfgs, whose ``random_state`` sklearn documents as used only by
    sag/saga/liblinear -- so their seed is inert and their across-seed SD is exactly zero. That is
    correct behaviour, not a defect, but reporting it as stability would be wrong, and a cluster
    bootstrap over seeds cannot widen an interval for an arm that has no seed variance.
    """
    per_arm = defaultdict(list)
    for r in records:
        per_arm[(r["backbone"], r["dataset"], r["method"], r["train_seed"])].append(r[field])
    per_method = defaultdict(list)
    for (bb, ds, m, s), vals in per_arm.items():
        per_method[(ds, m)].append((bb, s, float(np.mean(vals))))
    out = {}
    for (ds, m), triples in sorted(per_method.items()):
        by_bb = defaultdict(list)
        for bb, s, v in triples:
            by_bb[bb].append(v)
        sds = [float(np.std(v, ddof=1)) for v in by_bb.values() if len(v) > 1]
        # Tolerance, not `== 0.0`: a deterministic solver still leaves last-ulp noise in the
        # accumulated mean, so an exact comparison silently failed to flag arms that are in fact
        # seed-inert. 1e-12 is far below anything reportable and far above that noise.
        mx = max(sds) if sds else float("nan")
        out[(ds, m)] = {"max_sd_across_backbones": mx,
                        "seed_is_inert": bool(sds and mx < 1e-12)}
    return out
