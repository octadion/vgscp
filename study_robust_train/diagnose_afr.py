"""Why does AFR collapse on CelebA but not on Waterbirds?

Two hypotheses have already been tested and REFUTED, so this reports measurements rather than
arguing for a third:

  1. "The stage-1 ERM is fit in-sample on the reweighting split, inflating p_true."
     Refuted: fitting stage 1 on a separate split made worst-group accuracy WORSE (0.463 vs 0.732)
     and mean p_true was unchanged (0.9946 vs 0.9949).
  2. "CelebA's class imbalance plus an untuned gamma=2.0 degenerates the weights."
     Refuted on synthetic: AFR still improved over ERM (0.700 vs 0.600) and gamma made no
     difference at all across 0.5/1.0/2.0/4.0.

Neither synthetic reproduces the real collapse (wg 0.013-0.029 on CelebA, 0.688-0.890 on
Waterbirds), so the cause has to be measured on the real features. What is printed here is the
weight distribution AFR actually induces, the effective sample size it leaves, and what the fitted
head then predicts -- enough to say whether the head is degenerate and, if so, on what.
"""
from __future__ import annotations

import numpy as np

from .heads import fit_species_head
from .methods import fit_afr
from . import metrics

__all__ = ["diagnose"]


def _ess(w):
    """Kish effective sample size: how many equally-weighted points the weighting is worth."""
    w = np.asarray(w, float)
    return float(w.sum() ** 2 / np.sum(w ** 2)) if np.any(w) else 0.0


def diagnose(gd, *, gammas=(0.0, 0.5, 1.0, 2.0, 4.0), seed=0, verbose=True):
    """Measure AFR's stage-1 confidence, its induced weights, and the head those weights produce."""
    Xrw, yrw, grw = gd.reweight
    Xev, yev, gev = gd.eval_domain
    Xtr, ytr, gtr = gd.train
    out = {"backbone": gd.backbone, "dataset": gd.dataset, "n_reweight": len(yrw)}

    stage1 = fit_species_head(Xrw, yrw, seed=seed)
    p = stage1.predict_proba(Xrw)
    cls = list(stage1.classes_)
    p_true = np.array([p[i, cls.index(yrw[i])] for i in range(len(yrw))])
    out["p_true_quantiles"] = {q: float(np.quantile(p_true, q))
                               for q in (0.01, 0.10, 0.50, 0.90, 0.99)}

    groups, counts = np.unique(grw, return_counts=True)
    minority = groups[int(np.argmin(counts))]
    out["reweight_group_counts"] = {int(g): int(c) for g, c in zip(groups, counts)}
    out["minority_group"] = int(minority)

    def wg_of(head):
        return float(metrics.worst_group_accuracy(
            np.argmax(head.predict_proba(Xev), axis=1), yev, gev)[1])

    out["erm_reference_wg"] = wg_of(fit_species_head(Xrw, yrw, seed=seed))
    out["by_gamma"] = {}
    for gam in gammas:
        w = np.power(np.clip(1.0 - p_true, 1e-6, 1.0), gam)
        w = w / w.mean()
        head = fit_afr(Xrw, yrw, gamma=gam, seed=seed)
        pred = np.argmax(head.predict_proba(Xev), axis=1)
        vals, cts = np.unique(pred, return_counts=True)
        out["by_gamma"][gam] = {
            "wg_acc": wg_of(head),
            "ess": _ess(w),
            "ess_frac": _ess(w) / len(w),
            "weight_mass_on_minority": float(w[grw == minority].sum() / w.sum()),
            "minority_share_unweighted": float((grw == minority).mean()),
            "pred_class_share": {int(v): float(c / len(pred)) for v, c in zip(vals, cts)},
        }

    if verbose:
        print(f"\n=== AFR diagnostic: {gd.backbone} / {gd.dataset} ===")
        print(f"  reweight split n={len(yrw):,}  groups={out['reweight_group_counts']}  "
              f"minority=g{out['minority_group']} ({out['by_gamma'][gammas[0]]['minority_share_unweighted']:.2%})")
        q = out["p_true_quantiles"]
        print(f"  stage-1 p_true  p01={q[0.01]:.4f}  p10={q[0.10]:.4f}  med={q[0.50]:.4f}  "
              f"p90={q[0.90]:.4f}  p99={q[0.99]:.4f}")
        print(f"  plain ERM worst-group acc on eval = {out['erm_reference_wg']:.3f}")
        print(f"  {'gamma':>6s} {'wg acc':>8s} {'ESS':>10s} {'ESS/n':>8s} "
              f"{'w on minority':>14s}  predicted-class share")
        for gam, d in out["by_gamma"].items():
            share = ", ".join(f"{k}:{v:.3f}" for k, v in sorted(d["pred_class_share"].items()))
            print(f"  {gam:6.1f} {d['wg_acc']:8.3f} {d['ess']:10.1f} {d['ess_frac']:8.3f} "
                  f"{d['weight_mass_on_minority']:14.4f}  {share}")
        print("  gamma=0 is unweighted AFR == plain ERM on the reweight split; if wg_acc there is")
        print("  already near zero the weighting is not what breaks it.")
    return out
