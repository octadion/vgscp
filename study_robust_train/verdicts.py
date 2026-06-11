"""Pre-registered verdicts H1 / H2 / H3 (spec §6). Report ALL, even if negative.

H1 (Transfer): relative to ERM, robust training reduces the ACCURACY-MATCHED cross-group
  divergence with multi-seed CI excluding 0. GO = CI excludes 0 (reduction) at matched accuracy
  on >=2 of 3 score functions. Evaluated at the no-shift condition rho_test == rho_cal = 0.95.

H2 (Ranking inversion — headline): rank methods by worst-group ACCURACY and by worst-group
  conformal BURDEN (coverage gap). Report whether the TOP method differs AND whether that
  difference is real — the burden-top method's cov_gap must be SEPARATED (bootstrap CI of the
  cov_gap difference excludes 0) from the accuracy-top method's. A point-estimate inversion with
  overlapping CIs is noise, not signal.

H3 (Shift survival): calibrate at rho_cal=0.95, sweep rho_test. H3 is measured on the BURDEN
  quantities, NOT on coverage. Two channels, reported separately from coverage stability:
    (1) divergence survival : does the H1 accuracy-matched reduction hold across the sweep?
    (2) set-size disparity   : does the burden RELOCATE to set-size inflation as rho falls?
  Coverage stability (worst-group coverage vs rho) is reported alongside but is NOT the H3
  criterion — flat coverage with growing sets is "relocate, not remove" (burden2026), the
  expected story. Each method/score is labeled: survived / never_held / held_then_broke@rho /
  undefined@rho.
"""
from __future__ import annotations

import numpy as np

from .accuracy_matching import matched_divergence, raw_divergence

__all__ = ["h1_verdict", "h2_verdict", "h3_verdict"]

SCORES = ("APS", "RAPS", "THR")


# ---------------------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------------------
def _vals(records, method, key, rho, scores):
    return np.array([r[key] for r in records
                     if r["method"] == method and r["rho_test"] == rho and r["score"] in scores],
                    dtype=np.float64)


def _mean(x: np.ndarray) -> float:
    """Empty-safe mean (no RuntimeWarning on empty slices)."""
    return float(x.mean()) if x.size else float("nan")


def _bootstrap_mean_ci(x: np.ndarray, *, n_boot=2000, ci=0.95, seed=0):
    if x.size == 0:
        return float("nan"), (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(x, size=x.size, replace=True).mean() for _ in range(n_boot)])
    tail = (1.0 - ci) / 2.0
    return float(x.mean()), (float(np.quantile(means, tail)), float(np.quantile(means, 1 - tail)))


def _bootstrap_diff_ci(a: np.ndarray, b: np.ndarray, *, n_boot=2000, ci=0.95, seed=0):
    """CI of mean(a) - mean(b) by independent resampling (unpaired)."""
    if a.size == 0 or b.size == 0:
        return float("nan"), (float("nan"), float("nan")), False
    rng = np.random.default_rng(seed)
    diffs = np.array([rng.choice(a, size=a.size, replace=True).mean()
                      - rng.choice(b, size=b.size, replace=True).mean() for _ in range(n_boot)])
    tail = (1.0 - ci) / 2.0
    lo, hi = float(np.quantile(diffs, tail)), float(np.quantile(diffs, 1 - tail))
    return float(a.mean() - b.mean()), (lo, hi), bool(lo > 0 or hi < 0)


# ---------------------------------------------------------------------------------------
# H1
# ---------------------------------------------------------------------------------------
def h1_verdict(records, robust_methods, *, scores=SCORES, rho=0.95, reference="erm",
               metric="div_wasserstein1", n_boot=2000, seed=0) -> dict:
    out = {"rho": rho, "reference": reference, "metric": metric,
           "bar": ">=2/3 scores: CI excludes 0 (reduction)", "methods": {}}
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


# ---------------------------------------------------------------------------------------
# H2 — ranking inversion WITH CIs (Task B)
# ---------------------------------------------------------------------------------------
def h2_verdict(records, methods, *, rho=0.95, burden_key="cov_gap", scores=SCORES,
               n_boot=2000, ci=0.95, seed=0) -> dict:
    """Two rankings + an inversion that is only called REAL when the burden-top method's
    ``burden_key`` is bootstrap-separated from the accuracy-top method's (CI of the difference
    excludes 0). cov_gap points = per (train_seed x split x score) records at ``rho``."""
    wg_acc, burden, burden_ci, size_disp = {}, {}, {}, {}
    for m in methods:
        acc_pts = _vals(records, m, "worst_group_acc", rho, scores)
        wg_acc[m] = float(acc_pts.mean()) if acc_pts.size else float("nan")
        bpts = _vals(records, m, burden_key, rho, scores)
        mean_b, ci_b = _bootstrap_mean_ci(bpts, n_boot=n_boot, ci=ci, seed=seed)
        burden[m] = mean_b
        burden_ci[m] = list(ci_b)
        sd = _vals(records, m, "set_size_disparity", rho, scores)
        size_disp[m] = float(sd.mean()) if sd.size else float("nan")

    rank_acc = sorted(methods, key=lambda m: (-wg_acc[m], m))
    rank_burden = sorted(methods, key=lambda m: (burden[m], m))
    acc_top, burden_top = rank_acc[0], rank_burden[0]

    inversion_point = bool(acc_top != burden_top)
    # is the inversion real? compare cov_gap of acc_top vs burden_top (acc_top should be HIGHER burden)
    diff, diff_ci, separated = _bootstrap_diff_ci(
        _vals(records, acc_top, burden_key, rho, scores),
        _vals(records, burden_top, burden_key, rho, scores),
        n_boot=n_boot, ci=ci, seed=seed)
    return {
        "rho": rho, "burden_key": burden_key,
        "worst_group_acc": wg_acc, "burden": burden, "burden_ci": burden_ci,
        "set_size_disparity": size_disp,
        "ranking_by_accuracy": rank_acc, "ranking_by_burden": rank_burden,
        "top_by_accuracy": acc_top, "top_by_burden": burden_top,
        "inversion_point": inversion_point,
        "inversion_diff": diff, "inversion_diff_ci": list(diff_ci),
        "inversion_real": bool(inversion_point and separated),
    }


# ---------------------------------------------------------------------------------------
# H3 — burden survival (Task A): divergence channel + set-size-disparity channel
# ---------------------------------------------------------------------------------------
def _failure_type(curve, rho_cal=0.95):
    """Label the divergence-survival curve. ``curve`` entries have matched/reduces/rho_test."""
    held_at_cal = next((c for c in curve if c["rho_test"] == rho_cal), None)
    if held_at_cal is not None and not held_at_cal.get("matched"):
        # if undefined even at calibration rho
        return f"undefined@{rho_cal}"
    if held_at_cal is None or not held_at_cal.get("reduces"):
        return "never_held"
    # held at cal; find first rho (in sweep order) where it stops reducing or is undefined
    for c in curve:
        if c["rho_test"] == rho_cal:
            continue
        if not c.get("matched"):
            return f"undefined@{c['rho_test']}"
        if not c.get("reduces"):
            return f"held_then_broke@{c['rho_test']}"
    return "survived"


def h3_verdict(records, robust_methods, *, scores=SCORES,
               rho_sweep=(0.95, 0.90, 0.80, 0.70, 0.60, 0.50), reference="erm",
               metric="div_wasserstein1", n_boot=2000, seed=0) -> dict:
    out = {"rho_cal": 0.95, "rho_sweep": list(rho_sweep), "reference": reference, "metric": metric,
           "criterion": ("burden survival, NOT coverage. (1) divergence: H1 matched reduction holds "
                         "across sweep; (2) set-size disparity vs rho (relocation channel). Coverage "
                         "stability reported separately."),
           "methods": {}, "coverage_stability": {}, "setsize_disparity_by_method": {}}

    # coverage stability (reported, NOT the criterion) + raw set-size disparity per method (all methods incl ERM)
    all_methods = sorted(set(robust_methods) | {reference})
    for m in all_methods:
        cov_curve = [{"rho_test": rho, "worst_group_cov": _mean(_vals(records, m, "worst_group_cov", rho, scores))}
                     for rho in rho_sweep]
        covs = np.array([c["worst_group_cov"] for c in cov_curve])
        out["coverage_stability"][m] = {"curve": cov_curve,
                                        "range": float(np.nanmax(covs) - np.nanmin(covs)) if covs.size else float("nan"),
                                        "flat": bool(covs.size and (np.nanmax(covs) - np.nanmin(covs)) < 0.05)}
        out["setsize_disparity_by_method"][m] = [
            {"rho_test": rho, "set_size_disparity": _mean(_vals(records, m, "set_size_disparity", rho, scores)),
             "worst_group_set_size": _mean(_vals(records, m, "worst_group_set_size", rho, scores))}
            for rho in rho_sweep]

    erm_sd = {d["rho_test"]: d["set_size_disparity"] for d in out["setsize_disparity_by_method"][reference]}

    for m in robust_methods:
        per_score = {}
        for sc in scores:
            div_curve = []
            for rho in rho_sweep:
                res = matched_divergence(records, m, sc, rho, reference=reference, metric=metric,
                                         n_boot=n_boot, seed=seed)
                div_curve.append({"rho_test": rho, "matched": res.get("matched"),
                                  "delta_matched": res.get("delta_matched"), "ci": res.get("ci"),
                                  "reduces": bool(res.get("reduces"))})
            ftype = _failure_type(div_curve)
            per_score[sc] = {"divergence_curve": div_curve, "failure_type": ftype,
                             "survived": ftype == "survived"}
        # set-size relocation channel (per method, score-agnostic raw disparity vs ERM)
        sd_curve = []
        for d in out["setsize_disparity_by_method"][m]:
            rho = d["rho_test"]
            sd_curve.append({"rho_test": rho, "robust": d["set_size_disparity"],
                             "erm": erm_sd.get(rho, float("nan")),
                             "robust_minus_erm": d["set_size_disparity"] - erm_sd.get(rho, float("nan")),
                             "worst_group_set_size": d["worst_group_set_size"]})
        sizes = np.array([d["robust"] for d in sd_curve])
        # relocation flag: disparity grows as rho falls (correlation of disparity with (1-rho))
        inflates = bool(sizes.size >= 2 and sizes[-1] > sizes[0])
        out["methods"][m] = {"per_score": per_score, "setsize_disparity_curve": sd_curve,
                             "setsize_inflates_under_shift": inflates}
    return out
