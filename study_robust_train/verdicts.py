"""Pre-registered verdicts H1 / H2 / H3 (spec §6). Report ALL, even if negative.

H1 (Transfer): relative to ERM, robust training reduces the ACCURACY-MATCHED cross-group
  divergence with multi-seed CI excluding 0. GO = CI excludes 0 (reduction) at matched accuracy
  on >=2 of 3 score functions. Evaluated at the no-shift condition rho_test == rho_cal = 0.95
  (H3 handles the shift). The bar is the CI, not a magic threshold.

H2 (Ranking inversion — headline): rank methods by worst-group ACCURACY and by worst-group
  conformal BURDEN (coverage gap / set-size disparity). Report whether the TOP method differs.
  An inversion is the headline payload; "no inversion" is reported honestly.

H3 (Shift survival): calibrate at rho_cal=0.95, sweep rho_test; does any H1 reduction SURVIVE the
  sweep or collapse when test correlation != calibration? Report the curve per method x score.
"""
from __future__ import annotations

import numpy as np

from .accuracy_matching import matched_divergence, raw_divergence

__all__ = ["h1_verdict", "h2_verdict", "h3_verdict"]

SCORES = ("APS", "RAPS", "THR")


def h1_verdict(records, robust_methods, *, scores=SCORES, rho=0.95, reference="erm",
               metric="div_wasserstein1", n_boot=2000, seed=0) -> dict:
    """Per robust method: matched-divergence result per score + GO (reduction CI excludes 0 on
    >=2/3 scores). Also carries the uncontrolled raw divergence for side-by-side reporting."""
    out = {"rho": rho, "reference": reference, "metric": metric, "bar": ">=2/3 scores: CI excludes 0 (reduction)", "methods": {}}
    for m in robust_methods:
        per_score = {}
        n_reduce = 0
        for sc in scores:
            res = matched_divergence(records, m, sc, rho, reference=reference, metric=metric,
                                     n_boot=n_boot, seed=seed)
            res["raw_ref"] = raw_divergence(records, reference, sc, rho, metric=metric)
            res["raw_robust"] = raw_divergence(records, m, sc, rho, metric=metric)
            per_score[sc] = res
            if res.get("reduces"):
                n_reduce += 1
        out["methods"][m] = {"per_score": per_score, "n_scores_reduce": n_reduce,
                             "n_scores": len(scores), "GO": bool(n_reduce >= 2)}
    return out


def _mean_by_method(records, key, methods, rho, scores=SCORES):
    vals = {}
    for m in methods:
        xs = [r[key] for r in records
              if r["method"] == m and r["rho_test"] == rho and r["score"] in scores]
        vals[m] = float(np.mean(xs)) if xs else float("nan")
    return vals


def h2_verdict(records, methods, *, rho=0.95, burden_key="cov_gap") -> dict:
    """Two rankings: by worst-group ACCURACY (higher=better) and by conformal BURDEN
    (``burden_key`` lower=better, default coverage gap). Reports whether the top method differs."""
    wg_acc = _mean_by_method(records, "worst_group_acc", methods, rho)
    burden = _mean_by_method(records, burden_key, methods, rho)
    size_disp = _mean_by_method(records, "set_size_disparity", methods, rho)

    rank_acc = sorted(methods, key=lambda m: (-wg_acc[m], m))          # best worst-group acc first
    rank_burden = sorted(methods, key=lambda m: (burden[m], m))         # lowest burden first
    rank_size = sorted(methods, key=lambda m: (size_disp[m], m))

    return {
        "rho": rho, "burden_key": burden_key,
        "worst_group_acc": wg_acc, "burden": burden, "set_size_disparity": size_disp,
        "ranking_by_accuracy": rank_acc, "ranking_by_burden": rank_burden,
        "ranking_by_set_size_disparity": rank_size,
        "top_by_accuracy": rank_acc[0], "top_by_burden": rank_burden[0],
        "inversion": bool(rank_acc[0] != rank_burden[0]),
    }


def h3_verdict(records, robust_methods, *, scores=SCORES, rho_sweep=(0.95, 0.90, 0.80, 0.70, 0.60, 0.50),
               reference="erm", metric="div_wasserstein1", n_boot=2000, seed=0) -> dict:
    """Per method x score: the matched-divergence reduction across the rho sweep, and whether it
    SURVIVES (reduces stays True across the whole sweep) or collapses (and at which rho)."""
    out = {"rho_cal": 0.95, "rho_sweep": list(rho_sweep), "reference": reference,
           "metric": metric, "methods": {}}
    for m in robust_methods:
        per_score = {}
        for sc in scores:
            curve = []
            for rho in rho_sweep:
                res = matched_divergence(records, m, sc, rho, reference=reference, metric=metric,
                                         n_boot=n_boot, seed=seed)
                curve.append({"rho_test": rho, "matched": res.get("matched"),
                              "delta_matched": res.get("delta_matched"),
                              "ci": res.get("ci"), "reduces": bool(res.get("reduces"))})
            survives = all(c["reduces"] for c in curve)
            collapses_at = next((c["rho_test"] for c in curve if not c["reduces"]), None)
            per_score[sc] = {"curve": curve, "survives": survives, "collapses_at": collapses_at}
        out["methods"][m] = per_score
    return out
