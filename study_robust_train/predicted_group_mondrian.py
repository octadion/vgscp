"""Mondrian with PREDICTED groups — the deployment hit when the protected attribute is estimated,
not given. Reuses cached head posteriors + the recoverability probe (no retraining).

Group g = 2*y + a (y = task label, a = spurious attribute; y = g//2, a = g%2). The recoverability
probe (standardized, trained in-domain on TRAIN features -> a) predicts â for every eval point; the
PREDICTED stratum is 2*y + â (a swapped for â; y kept, consistent with the existing 4-group harness).

Three Mondrian conditions, coverage ALWAYS tallied on the TRUE groups:
  (a) TRUE        : strata = true g at cal AND test (the current result).
  (b) PRED-test   : thresholds from TRUE-group calibration; test points assigned to strata by â.
  (c) PRED-both   : â used at cal AND test (a never observed) — the fully deployable setting.
Per method: worst-group coverage + worst-group set size under (a)/(b)/(c) + the gap (a)-(c).

Pre-registered expectation: because â is highly accurate, (c) nearly matches (a) on worst-group
coverage (small gap) at a modest efficiency cost -> group-conditional calibration is deployable
without ground-truth group labels. Reported honestly if the gap is large.

Discipline: >=3 seeds × >=10 splits, 95% CIs, in-domain probe (TRAIN->eval), no leakage (â never
uses a test point's true a; coverage scored on true groups).
"""
from __future__ import annotations

import os

import numpy as np
from sklearn.metrics import roc_auc_score

from conformal.group_robust import mondrian_build_sets, mondrian_quantiles
from conformal.scores import draw_randomization, scores_all, true_label_scores
from experiments.shift_resampler import resample_to_rho, split_pool

from .calibration_ablation import _agg
from .conformal_eval import RHO_CAL
from .grid import worst_group_floor
from .heads import fit_species_head, head_probs
from .methods import METHODS, fit_method
from .recoverability import spurious_from_group
from .verdicts import SCORES

DEPLOYABLE_GAP = 0.02   # |cov(a) - cov(c)| <= this -> deployable without ground-truth groups
CONDITIONS = ("a_true", "b_pred_test", "c_pred_both")


def _per_group(values, group):
    return {int(g): float(values[group == g].mean()) for g in np.unique(group)}


def _eval_abc(probs, y, group, ahat, *, score, alpha, rho_test, rho_cal=RHO_CAL,
              split_seed=0, frac_cal=0.5):
    """One split: build Mondrian sets under (a)/(b)/(c); metrics on TRUE groups."""
    N = len(y)
    cal_pool, test_pool = split_pool(N, frac_cal=frac_cal, seed=split_seed)
    m = min(cal_pool.size, test_pool.size)
    cal_idx = cal_pool[resample_to_rho(group[cal_pool], rho_cal, m, seed=split_seed * 2 + 1).idx]
    test_idx = test_pool[resample_to_rho(group[test_pool], rho_test, m, seed=split_seed * 2 + 2).idx]

    u_cal = draw_randomization(cal_idx.size, split_seed * 7 + 1)
    cal_true = true_label_scores(scores_all(score, probs[cal_idx], u=u_cal), y[cal_idx])
    u_test = draw_randomization(test_idx.size, split_seed * 7 + 2)
    test_all = scores_all(score, probs[test_idx], u=u_test)
    y_test = y[test_idx]
    g_true_cal = group[cal_idx]
    g_true_test = group[test_idx]                                   # TRUE groups (coverage scored here)
    s_pred_cal = 2 * (g_true_cal // 2) + ahat[cal_idx]              # predicted stratum = 2*y + â
    s_pred_test = 2 * (g_true_test // 2) + ahat[test_idx]

    def metrics(membership):
        covered = membership[np.arange(test_idx.size), y_test].astype(float)
        size = membership.sum(axis=1).astype(float)
        cov_by = _per_group(covered, g_true_test)                  # TRUE-group coverage
        size_by = _per_group(size, g_true_test)
        wg = min(cov_by, key=lambda g: (cov_by[g], g))
        return {"worst_group_cov": cov_by[wg], "worst_group_set_size": size_by[wg],
                "marg_cov": float(covered.mean()), "mean_set_size": float(size.mean())}

    gq_true = mondrian_quantiles(cal_true, g_true_cal, alpha)
    gq_pred = mondrian_quantiles(cal_true, s_pred_cal, alpha)
    return {
        "a_true": metrics(mondrian_build_sets(test_all, g_true_test, gq_true)),
        "b_pred_test": metrics(mondrian_build_sets(test_all, s_pred_test, gq_true)),
        "c_pred_both": metrics(mondrian_build_sets(test_all, s_pred_test, gq_pred)),
    }


def run_predicted_group(data_by_key: dict, *, methods=METHODS, scores=SCORES, seeds=(0, 1, 2),
                        n_splits=10, alpha=0.1, rho_test=0.95, rho_cal=RHO_CAL,
                        method_hp=None) -> dict:
    method_hp = method_hp or {}
    records, excluded = [], []
    for key, gd in data_by_key.items():
        Xtr, ytr, gtr = gd.train
        Xev, yev, gev = gd.eval_domain
        a_tr, a_ev = spurious_from_group(gtr), spurious_from_group(gev)
        # recoverability probe (in-domain TRAIN->eval), one per seed, reused across methods
        probes = {}
        for seed in seeds:
            probe = fit_species_head(Xtr, a_tr, seed=seed)
            pcol = list(probe.classes_).index(1) if 1 in probe.classes_ else 0
            ap = probe.predict_proba(Xev)[:, pcol]
            auroc = float(roc_auc_score(a_ev, ap)) if len(np.unique(a_ev)) > 1 else float("nan")
            probes[seed] = ((ap >= 0.5).astype(int), auroc)
        for method in methods:
            for seed in seeds:
                head = fit_method(method, gd.train, gd.reweight, seed=seed, **method_hp.get(method, {}))
                probs = head_probs(head, Xev, gd.n_classes)
                from . import metrics as _m
                _, wg_acc = _m.worst_group_accuracy(np.argmax(probs, axis=1), yev, gev)
                if wg_acc < worst_group_floor(gd.dataset, method):
                    excluded.append({"backbone": gd.backbone, "dataset": gd.dataset, "method": method,
                                     "seed": seed, "worst_group_acc": wg_acc,
                                     "floor": worst_group_floor(gd.dataset, method)})
                    continue
                ahat, auroc = probes[seed]
                for score in scores:
                    for sp in range(n_splits):
                        res = _eval_abc(probs, yev, gev, ahat, score=score, alpha=alpha,
                                        rho_test=rho_test, rho_cal=rho_cal, split_seed=sp)
                        for cond, key_short in (("a_true", "a"), ("b_pred_test", "b"), ("c_pred_both", "c")):
                            r = res[cond]
                            records.append({"backbone": gd.backbone, "dataset": gd.dataset,
                                            "method": method, "train_seed": seed, "condition": cond,
                                            "score": score, "alpha": alpha, "rho_cal": rho_cal,
                                            "rho_test": rho_test, "split_seed": sp, "probe_auroc": auroc,
                                            "worst_group_acc": wg_acc, **r})
    verdicts = {}
    for key in sorted({(r["backbone"], r["dataset"]) for r in records}):
        recs = [r for r in records if (r["backbone"], r["dataset"]) == key]
        verdicts[key] = _verdict(recs, sorted({r["method"] for r in recs}), alpha=alpha,
                                 scores=scores, rho_test=rho_test)
    return {"records": records, "excluded": excluded, "verdicts": verdicts, "alpha": alpha,
            "scores": list(scores), "rho_test": rho_test}


def _verdict(recs, methods, *, alpha, scores=SCORES, score="APS", rho_test=0.95) -> dict:
    target = 1.0 - alpha
    out = {"target": target, "score": score, "rho_test": rho_test, "methods": {}}
    for m in methods:
        base = [r for r in recs if r["method"] == m and r["score"] == score and r["rho_test"] == rho_test]
        row = {}
        for cond in CONDITIONS:
            f = [r for r in base if r["condition"] == cond]
            row[cond] = {"cov": _agg(f, "worst_group_cov"), "size": _agg(f, "worst_group_set_size")}
        aurocs = [r["probe_auroc"] for r in base if r["probe_auroc"] == r["probe_auroc"]]
        row["probe_auroc"] = float(np.mean(aurocs)) if aurocs else float("nan")
        row["gap_cov_a_minus_c"] = row["a_true"]["cov"]["mean"] - row["c_pred_both"]["cov"]["mean"]
        row["gap_size_c_minus_a"] = row["c_pred_both"]["size"]["mean"] - row["a_true"]["size"]["mean"]
        row["deployable"] = bool(abs(row["gap_cov_a_minus_c"]) <= DEPLOYABLE_GAP)
        out["methods"][m] = row
    out["all_deployable"] = all(r["deployable"] for r in out["methods"].values())
    return out


# ---------------------------------------------------------------------------------------
# CSV + report
# ---------------------------------------------------------------------------------------
_CSV_COLS = ["backbone", "dataset", "method", "train_seed", "condition", "score", "alpha",
             "rho_cal", "rho_test", "split_seed", "probe_auroc", "worst_group_acc",
             "worst_group_cov", "worst_group_set_size", "marg_cov", "mean_set_size"]


def write_csv(records, path):
    import csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_COLS, extrasaction="ignore")
        w.writeheader()
        for r in records:
            w.writerow(r)


def write_md(out: dict, path: str, *, synthetic=False):
    target = 1.0 - out["alpha"]
    L = []
    if synthetic:
        L += ["> **SYNTHETIC LOGIC-VALIDATION ONLY — NOT a scientific result.** Toy numbers; they",
              "> validate the machinery, not the phenomenon.\n"]
    L.append("# PREDICTED_GROUP_MONDRIAN.md — Mondrian with predicted groups (deployment hit)\n")
    L.append(f"Group g=2·y+a; predicted stratum 2·y+â from the in-domain recoverability probe. "
             f"Coverage scored on TRUE groups in all conditions. Target 1-α={target:.2f}, "
             f"ρ_cal={RHO_CAL}, ρ_test={out['rho_test']}. Means ± 95% CI over ≥3 seeds × ≥10 splits. "
             f"Deployable bar: |cov(a)−cov(c)| ≤ {DEPLOYABLE_GAP}.\n")
    L.append("Conditions: **(a)** true groups · **(b)** predicted groups at TEST only "
             "(thresholds from true-group cal) · **(c)** predicted groups at cal AND test (a never observed).\n")
    if out["excluded"]:
        L.append("## Excluded arms (§2 worst-group acc floor; e.g. AFR on CelebA)\n")
        for e in out["excluded"]:
            L.append(f"- {e['backbone']}/{e['dataset']} {e['method']} seed{e['seed']}: "
                     f"worst-group acc {e['worst_group_acc']:.3f} < {e['floor']}")
        L.append("")

    def _tbl(vobj):
        rows = ["| method | probe AUROC | cov(a) [CI] | cov(b) | cov(c) [CI] | **gap(a−c)** | "
                "size(a) | size(c) | size cost(c−a) | deployable |",
                "|---|---|---|---|---|---|---|---|---|---|"]
        for mth, row in vobj["methods"].items():
            ca, cb, cc = row["a_true"]["cov"], row["b_pred_test"]["cov"], row["c_pred_both"]["cov"]
            sa, sc = row["a_true"]["size"], row["c_pred_both"]["size"]
            rows.append(
                f"| {mth} | {row['probe_auroc']:.3f} | {ca['mean']:.3f} [{ca['ci'][0]:.3f},{ca['ci'][1]:.3f}] | "
                f"{cb['mean']:.3f} | {cc['mean']:.3f} [{cc['ci'][0]:.3f},{cc['ci'][1]:.3f}] | "
                f"{row['gap_cov_a_minus_c']:+.3f} | {sa['mean']:.2f} | {sc['mean']:.2f} | "
                f"{row['gap_size_c_minus_a']:+.2f} | {row['deployable']} |")
        return rows

    for key, v in out["verdicts"].items():
        bb, ds = key
        L.append(f"## {bb} / {ds}\n")
        L.append(f"### APS (ρ_test={v['rho_test']}) — **all methods deployable: {v['all_deployable']}**\n")
        L += _tbl(v)
        L.append(f"\n_gap(a−c) = worst-group coverage lost when â replaces a everywhere; "
                 f"size cost = extra worst-group set size. all deployable = {v['all_deployable']}._\n")
        # RAPS/THR appendix
        recs = [r for r in out["records"] if (r["backbone"], r["dataset"]) == key]
        for sc in ("RAPS", "THR"):
            vsc = _verdict(recs, list(v["methods"]), alpha=out["alpha"], score=sc, rho_test=v["rho_test"])
            L.append(f"<details><summary>appendix — {sc} (all deployable: {vsc['all_deployable']})</summary>\n")
            L += _tbl(vsc)
            L.append("\n</details>\n")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
