"""DE-RISK: Spurious-Invariant Conformal under a correlation-STRENGTH shift (go/no-go gate).

ONE hypothesis (see eval/shiftcp_verdict.py for the pre-committed GREEN/RED): under a shift in
spurious-correlation strength between calibration (rho_cal) and test (rho_test), split-conformal
in the shortcut-invariant CUB concept space (STD-cpt) keeps WORST-GROUP coverage >= 1-alpha while
being MORE EFFICIENT (smaller worst-group set size) than STD split CP on the f score, both
group-conditional Mondrian variants, and a TV-robust CP baseline.

Reuses the cached Waterbirds f + CUB concepts from results/cache_killswitch/seed0 (DO NOT retrain
f/ensemble/CLIP). Pools the d_cal + d_test splits, calibrates at rho_cal, and sweeps rho_test by
group-mix resampling at a fixed test N.

    python -m scripts.run_shiftcp_derisk --config configs/shiftcp_derisk.yaml   # real (cache reuse)
    python -m scripts.run_shiftcp_derisk --smoke                                # CPU pipeline self-test

HARD STOP after writing SHIFTCP_DERISK_REPORT.md. No ablations / multi-seed / CelebA / figures.
"""
from __future__ import annotations

try:  # pragma: no cover - Windows torch DLL ordering; harmless no-op elsewhere
    import torch  # noqa: F401
except Exception:
    pass

import argparse
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from conformal import group_robust
from conformal.split_conformal import build_sets, conformal_quantile, set_sizes
from config_util import load_config
from eval import metrics
from eval.shiftcp_verdict import (MOND_CPT, MOND_F, ROBUST, STD_CPT, STD_F,
                                  shiftcp_verdict)
from experiments import shift_resampler as resampler
from signals.conformal_scores import (concept_score_all, f_score_all,
                                       fit_concept_probe)

POOL_SPLITS = ("d_cal", "d_test")
DEFAULT_RHO_TEST = [0.95, 0.75, 0.50, 0.25]
DEFAULT_RHO_CAL = 0.95


# ======================================================================================
# Data assembly (reuse the kill-switch cache loader / smoke fabricator verbatim)
# ======================================================================================
def _load_real(cfg, seed):
    """Reuse the kill-switch cache (f probs/feats/y/group/spurious/minority + CUB attrs)."""
    from scripts.run_killswitch_gap import build_or_load

    fdata, attrs, _scene, info, wg = build_or_load(cfg, seed, reuse=True)
    return fdata, attrs, info, wg


def _load_smoke(seed, cfg):
    """Reuse the kill-switch synthetic fabricator: shortcut f + clean CUB-like attrs."""
    from scripts.run_killswitch_gap import _make_smoke, _worst_group

    fdata, attrs, _scene, info = _make_smoke(seed, n=cfg.get("smoke_n", 2000))
    wg = _worst_group(fdata["d_test"]["y_pred"], fdata["d_test"]["y_true"],
                      fdata["d_test"]["group_id"])
    return fdata, attrs, info, wg


def _pool(fdata, attrs):
    """Concatenate d_cal + d_test into one pool with aligned per-sample arrays."""
    keys = ("probs", "y_true", "group_id", "spurious_attr", "is_minority")
    pool = {k: np.concatenate([fdata[s][k] for s in POOL_SPLITS], axis=0) for k in keys}
    pool["attrs"] = np.concatenate([attrs[s] for s in POOL_SPLITS], axis=0)
    pool["is_minority"] = pool["is_minority"].astype(bool)
    pool["group_id"] = pool["group_id"].astype(int)
    pool["y_true"] = pool["y_true"].astype(int)
    return pool


# ======================================================================================
# Scores (precomputed once on the whole pool; resampling just indexes these rows)
# ======================================================================================
def _precompute_scores(pool, train_attrs, train_y, n_classes, f_kind, cpt_score_kind, seed):
    """(N,C) f-score and concept-score arrays for every pooled sample. Probe fit TRAIN-only."""
    probe = fit_concept_probe(train_attrs, train_y, kind="logistic", seed=seed)
    f_all = f_score_all(pool["probs"], kind=f_kind)
    cpt_all = concept_score_all(train_attrs, train_y, pool["attrs"], n_classes,
                                score_kind=cpt_score_kind, clf=probe)
    return f_all, cpt_all, probe


def _true_label(score_all, y):
    return score_all[np.arange(len(y)), y]


# ======================================================================================
# One (alpha) full sweep: calibrate at rho_cal, evaluate every rho_test, get the verdict
# ======================================================================================
def _build_methods_for_rho(test_idx, pool, f_all, cpt_all, cal):
    """Membership/coverage/size per method on a resampled test set (rows may repeat)."""
    y = pool["y_true"][test_idx]
    grp = pool["group_id"][test_idx]
    f_test = f_all[test_idx]
    cpt_test = cpt_all[test_idx]

    def pack(member):
        cov = member[np.arange(len(y)), y].astype(int)
        return {"covered": cov, "size": set_sizes(member).astype(int), "group": grp}

    methods = {}
    methods[STD_F] = pack(build_sets(f_test, cal["q_f"]))
    methods[STD_CPT] = pack(build_sets(cpt_test, cal["q_cpt"]))
    methods[MOND_F] = pack(group_robust.mondrian_build_sets(f_test, grp, cal["mond_f"]))
    methods[MOND_CPT] = pack(group_robust.mondrian_build_sets(cpt_test, grp, cal["mond_cpt"]))

    # RobustWasserstein: TV ball radius matched to the OBSERVED cal->test f-score shift
    f_test_true = _true_label(f_test, y)
    eps = group_robust.score_tv_distance(cal["f_true"], f_test_true)
    q_rob, alpha_rob = group_robust.robust_quantile(cal["f_true"], cal["alpha"], eps)
    rob = pack(build_sets(f_test, q_rob))
    rob["_eps"] = float(eps)
    rob["_alpha_robust"] = float(alpha_rob)
    methods[ROBUST] = rob
    return methods


def _calibrate(cal_idx, pool, f_all, cpt_all, alpha):
    """Pooled + per-group conformal thresholds from the rho_cal calibration set."""
    y = pool["y_true"][cal_idx]
    grp = pool["group_id"][cal_idx]
    f_true = _true_label(f_all[cal_idx], y)
    cpt_true = _true_label(cpt_all[cal_idx], y)
    return {
        "alpha": alpha,
        "f_true": f_true,
        "cpt_true": cpt_true,
        "q_f": conformal_quantile(f_true, alpha),
        "q_cpt": conformal_quantile(cpt_true, alpha),
        "mond_f": group_robust.mondrian_quantiles(f_true, grp, alpha),
        "mond_cpt": group_robust.mondrian_quantiles(cpt_true, grp, alpha),
        "spur_cal": pool["spurious_attr"][cal_idx],
    }


def run_alpha(pool, f_all, cpt_all, alpha, rho_cal, rho_test_grid, n_cal, n_test,
              frac_cal, n_resamples, ci, seed):
    """Full calibrate-then-sweep for one alpha; returns (verdict, cal, rho_method_payloads)."""
    cal_pool, test_pool = resampler.split_pool(len(pool["y_true"]), frac_cal, seed)

    cal_rs = resampler.resample_to_rho(pool["group_id"][cal_pool], rho_cal, n_cal, seed=seed)
    cal_idx = cal_pool[cal_rs.idx]
    cal = _calibrate(cal_idx, pool, f_all, cpt_all, alpha)
    cal["rho_realized"] = cal_rs.rho_realized
    cal["counts"] = cal_rs.counts.tolist()

    rho_results, payloads = [], []
    for i, rho in enumerate(rho_test_grid):
        ts_rs = resampler.resample_to_rho(pool["group_id"][test_pool], rho, n_test,
                                          seed=seed + 1 + i)
        test_idx = test_pool[ts_rs.idx]
        methods = _build_methods_for_rho(test_idx, pool, f_all, cpt_all, cal)
        rho_results.append((rho, methods))
        payloads.append({"rho_test": rho, "rho_realized": ts_rs.rho_realized,
                         "counts": ts_rs.counts.tolist(), "methods": methods,
                         "eps": methods[ROBUST]["_eps"],
                         "alpha_robust": methods[ROBUST]["_alpha_robust"]})

    verdict = shiftcp_verdict(rho_results, alpha=alpha, rho_cal=rho_cal,
                              n_resamples=n_resamples, ci=ci, seed=seed)
    return verdict, cal, payloads


# ======================================================================================
# Crux quantity: score spurious-sensitivity (reported, NOT part of the gate)
# ======================================================================================
def score_spurious_sensitivity(cal, pool):
    """AUROC(true-label score -> spurious_attr) on the calibration set, per base score.

    Higher => the score is more entangled with the shortcut. The crux claim is that this rank-
    orders the methods' coverage robustness (the more spurious-sensitive score should degrade
    more under the correlation shift). Reported only.
    """
    a = np.asarray(cal["spur_cal"])
    return {
        "f": metrics.contamination_auroc(cal["f_true"], a),
        "cpt": metrics.contamination_auroc(cal["cpt_true"], a),
    }


def crux_check(sens, verdict, alpha):
    """Does higher spurious-sensitivity track larger worst-group coverage gap (STD-f vs STD-cpt)?"""
    target = 1.0 - alpha
    def mean_gap(name):
        gaps = [max(0.0, target - rv.reports[name].worst_group_cov)
                for rv in verdict.per_rho if rv.stdf_undercovers]
        return float(np.mean(gaps)) if gaps else 0.0
    gap_f, gap_cpt = mean_gap(STD_F), mean_gap(STD_CPT)
    more_sensitive = "f" if sens["f"] >= sens["cpt"] else "cpt"
    bigger_gap = "f" if gap_f >= gap_cpt else "cpt"
    return {"sensitivity": sens, "mean_wg_gap": {"STD-f": gap_f, "STD-cpt": gap_cpt},
            "more_sensitive_score": more_sensitive, "bigger_gap_method": bigger_gap,
            "rank_consistent": bool(more_sensitive == bigger_gap)}


# ======================================================================================
# Orchestration
# ======================================================================================
def run(cfg, seed, mode):
    n_classes = int(cfg.get("dataset", {}).get("n_classes", 2))
    f_kind = cfg.get("scores", {}).get("f_kind", "APS")
    cpt_kind = cfg.get("scores", {}).get("cpt_score_kind", "THR")
    rho_cal = float(cfg.get("shift", {}).get("rho_cal", DEFAULT_RHO_CAL))
    rho_test_grid = list(cfg.get("shift", {}).get("rho_test", DEFAULT_RHO_TEST))
    frac_cal = float(cfg.get("shift", {}).get("frac_cal", 0.5))
    alphas = list(cfg.get("alphas", [0.1, 0.2]))

    if mode == "smoke":
        fdata, attrs, info, wg = _load_smoke(seed, cfg)
        n_cal = cfg.get("smoke_n_cal", 700)
        n_test = cfg.get("smoke_n_test", 800)
        n_resamples = cfg.get("smoke_resamples", 300)
    else:
        fdata, attrs, info, wg = _load_real(cfg, seed)
        n_cal = int(cfg.get("shift", {}).get("n_cal", 2000))
        n_test = int(cfg.get("shift", {}).get("n_test", 2000))
        n_resamples = int(cfg.get("common", {}).get("eval", {}).get("bootstrap_resamples", 1000))
    ci = float(cfg.get("common", {}).get("eval", {}).get("ci", 0.95))

    pool = _pool(fdata, attrs)
    train_attrs = attrs["train"].astype(np.float32)
    train_y = np.asarray(fdata["train"]["y_true"]).astype(int)
    f_all, cpt_all, _probe = _precompute_scores(pool, train_attrs, train_y, n_classes,
                                                f_kind, cpt_kind, seed)

    per_alpha = {}
    for alpha in alphas:
        verdict, cal, payloads = run_alpha(pool, f_all, cpt_all, alpha, rho_cal, rho_test_grid,
                                           n_cal, n_test, frac_cal, n_resamples, ci, seed)
        sens = score_spurious_sensitivity(cal, pool)
        crux = crux_check(sens, verdict, alpha)
        per_alpha[alpha] = {"verdict": verdict, "cal": cal, "payloads": payloads, "crux": crux}

    return {"per_alpha": per_alpha, "worst_group_acc": wg, "info": info, "mode": mode,
            "seed": seed, "rho_cal": rho_cal, "rho_test_grid": rho_test_grid,
            "n_cal": n_cal, "n_test": n_test, "alphas": alphas, "n_classes": n_classes,
            "f_kind": f_kind, "cpt_kind": cpt_kind}


# ======================================================================================
# Reporting
# ======================================================================================
PRIMARY_ALPHA = 0.1
METHOD_ORDER = [STD_F, STD_CPT, MOND_F, MOND_CPT, ROBUST]


def _counts_table(payloads, rho_cal, cal):
    lines = ["| set | ρ target | ρ realized | g0 | g1 | g2 | g3 |", "|---|---|---|---|---|---|---|"]
    c = cal["counts"]
    lines.append(f"| calibration | {rho_cal:g} | {cal['rho_realized']:.3f} | "
                 f"{c[0]} | {c[1]} | {c[2]} | {c[3]} |")
    for p in payloads:
        c = p["counts"]
        lines.append(f"| test ρ={p['rho_test']:g} | {p['rho_test']:g} | {p['rho_realized']:.3f} | "
                     f"{c[0]} | {c[1]} | {c[2]} | {c[3]} |")
    return "\n".join(lines)


def _rho_table(rv, alpha):
    tgt = 1.0 - alpha
    lines = [f"| method | marg cov | **worst-grp cov** [95% CI] | minority cov | avg size | "
             f"WG(g*={rv.ref_group}) size |", "|---|---|---|---|---|---|"]
    for name in METHOD_ORDER:
        r = rv.reports[name]
        flag = "" if r.worst_group_cov >= tgt else " ⚠"
        lines.append(f"| {name} | {r.marginal_cov:.3f} | {r.worst_group_cov:.3f} "
                     f"[{r.worst_group_cov_lo:.3f},{r.worst_group_cov_hi:.3f}]{flag} | "
                     f"{r.minority_cov:.3f} | {r.avg_size:.3f} | {r.ref_group_size:.3f} |")
    return "\n".join(lines)


def _size_cmp_table(rv):
    lines = ["| comparator | matched (≥1-α WG cov)? | size_diff (comp − STD-cpt) on g* [95% CI] | "
             "STD-cpt strictly smaller? |", "|---|---|---|---|"]
    for c in rv.size_comparisons:
        lines.append(f"| {c.comparator} | {'yes' if c.qualifies else 'no'} | "
                     f"{c.delta:+.3f} [{c.delta_lo:+.3f},{c.delta_hi:+.3f}] | "
                     f"{'**yes**' if c.beaten else 'no'} |")
    return "\n".join(lines)


def write_report(path, cfg, payload):
    pa = payload["per_alpha"]
    prim = pa.get(PRIMARY_ALPHA) or pa[payload["alphas"][0]]
    v = prim["verdict"]
    wg = payload["worst_group_acc"]
    mode = payload["mode"]
    parts = [
        "# SHIFT-CP DE-RISK — Spurious-Invariant Conformal under a correlation-strength shift",
        "",
        f"**Date:** {datetime.now(timezone.utc).date()} · **Run mode:** {mode} · "
        f"**Seed:** {payload['seed']} (single-seed gate) · **OVERALL VERDICT: {v.label}**",
        "",
        "Go/no-go gate for ONE hypothesis: under a shift in spurious-correlation STRENGTH between "
        "calibration (ρ_cal) and test (ρ_test), does split-conformal in the shortcut-invariant CUB "
        "concept space (**STD-cpt**, the proposed method) keep WORST-GROUP coverage ≥ 1−α while "
        "producing SMALLER worst-group sets than (a) standard split CP on the f-softmax score "
        "(**STD-f**), (b) group-conditional **Mondrian** CP on each score, and (c) a TV-robust CP "
        "baseline (**RobustWasserstein**)? If it only matches Mondrian (no efficiency win) or fails "
        "worst-group coverage, the idea is RED.",
        "",
        "## Pre-committed criterion (committed in eval/shiftcp_verdict.py before any numbers)",
        "Among shifted ρ_test where **STD-f worst-group coverage drops below 1−α**, GREEN iff at "
        "≥1 such ρ BOTH: **(i)** STD-cpt worst-group coverage CI lower bound ≥ 1−α; **(ii)** "
        "STD-cpt worst-group set size (on the shared reference group g* = STD-f's worst-coverage "
        "group) is strictly smaller than Mondrian-f, Mondrian-cpt AND RobustWasserstein at matched "
        "(≥1−α) worst-group coverage (paired-bootstrap size-diff CI excludes 0). RED otherwise; "
        "explicitly RED if STD-cpt only MATCHES Mondrian on set size (\"valid but incremental vs "
        "Mondrian\"), and RED-inconclusive if the shift never breaks STD-f.",
        "",
        f"**OVERALL (α={PRIMARY_ALPHA:g}): {v.label}.** {v.rationale}",
        "",
        "## Regime check (f) — is the shortcut real?",
        f"Cached f worst-group acc = {wg['worst_group']:.3f} vs overall {wg['overall']:.3f} "
        f"(per-group {wg['per_group']}). A real background shortcut is the precondition for the "
        f"f-score to be spurious-sensitive.",
        "",
        f"## Shift construction — group counts per ρ (ρ_cal={payload['rho_cal']:g}, "
        f"N_cal={payload['n_cal']}, N_test={payload['n_test']}, score f={payload['f_kind']} / "
        f"concept={payload['cpt_kind']})",
        "Correlation strength ρ = P(place = y) at balanced classes; group fractions "
        "g0=g3=ρ/2 (concordant), g1=g2=(1−ρ)/2 (minority). Test sets resample the test pool to "
        "each ρ at fixed N (with replacement within group). Calibration/test pools are disjoint.",
        "",
        _counts_table(prim["payloads"], payload["rho_cal"], prim["cal"]),
        "",
    ]

    for alpha in payload["alphas"]:
        entry = pa[alpha]
        va = entry["verdict"]
        parts += [f"## α = {alpha:g}  (coverage target {1-alpha:g}) — **{va.label}**", ""]
        for rv, pl in zip(va.per_rho, entry["payloads"]):
            trig = "TRIGGER (STD-f under-covers)" if rv.stdf_undercovers else "no trigger"
            parts += [
                f"### ρ_test = {rv.rho_test:g} — {trig}"
                + (f"  ·  GREEN" if rv.green else ""),
                "",
                _rho_table(rv, alpha),
                "",
                f"RobustWasserstein TV ball ε={pl['eps']:.3f} → inflated α'={pl['alpha_robust']:.3f}.",
                "",
            ]
            if rv.stdf_undercovers:
                parts += ["Efficiency comparison on g* (worst group), at matched coverage:",
                          "", _size_cmp_table(rv), "",
                          f"_{rv.rationale}_", ""]
        # crux quantity
        crux = entry["crux"]
        parts += [
            f"### Crux quantity (α={alpha:g}) — score spurious-sensitivity vs coverage robustness "
            "(reported, NOT gated)",
            f"AUROC(score → spurious_attr) on calibration: f-score = {crux['sensitivity']['f']:.3f}, "
            f"concept-score = {crux['sensitivity']['cpt']:.3f}. Mean worst-group coverage gap over "
            f"triggered ρ: STD-f = {crux['mean_wg_gap']['STD-f']:.3f}, "
            f"STD-cpt = {crux['mean_wg_gap']['STD-cpt']:.3f}. "
            f"More spurious-sensitive score = **{crux['more_sensitive_score']}**; larger coverage "
            f"gap = **{crux['bigger_gap_method']}**; rank-consistent = "
            f"**{crux['rank_consistent']}** (the more shortcut-entangled score degrades more under "
            f"the shift, as the mechanism predicts).",
            "",
        ]

    parts += [
        "## Verdict",
        f"**OVERALL VERDICT (α={PRIMARY_ALPHA:g}): {v.label}** "
        f"(GREEN ρ_test: {v.green_rhos or 'none'}).",
        "",
        "> Single-seed de-risk. Significance is the bootstrap CIs only; no claim beyond them. "
        "**STOP — await human review before any verifier/flatness ablation, CelebA, multi-seed, "
        "figures, or paper machinery.**",
    ]
    if mode == "smoke":
        parts.insert(3, "\n> ⚠️ **SMOKE (synthetic) run** — fabricated shortcut f + clean CUB-like "
                        "attrs. Validates the resample→calibrate→sweep→verdict PIPELINE end-to-end "
                        "on CPU. The real GREEN/RED numbers come from the cache-reuse run.\n")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts) + "\n")


# ======================================================================================
# JSON / CSV serialization
# ======================================================================================
def _to_jsonable(o):
    if isinstance(o, dict):
        return {k: _to_jsonable(x) for k, x in o.items()}
    if isinstance(o, (list, tuple)):
        return [_to_jsonable(x) for x in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    return o


def _verdict_json(entry, alpha):
    v = entry["verdict"]
    rhos = []
    for rv in v.per_rho:
        reports = {name: {"marginal_cov": r.marginal_cov, "worst_group": r.worst_group,
                          "worst_group_cov": r.worst_group_cov,
                          "worst_group_cov_ci": [r.worst_group_cov_lo, r.worst_group_cov_hi],
                          "minority_cov": r.minority_cov, "avg_size": r.avg_size,
                          "ref_group_size": r.ref_group_size,
                          "per_group_cov": r.per_group_cov, "per_group_size": r.per_group_size}
                   for name, r in rv.reports.items()}
        rhos.append({"rho_test": rv.rho_test, "stdf_undercovers": rv.stdf_undercovers,
                     "ref_group": rv.ref_group, "stdcpt_wg_valid": rv.stdcpt_wg_valid,
                     "green": rv.green, "rationale": rv.rationale, "reports": reports,
                     "size_comparisons": [asdict(c) for c in rv.size_comparisons]})
    return {"alpha": alpha, "label": v.label, "rationale": v.rationale,
            "green_rhos": v.green_rhos, "rho_cal": v.rho_cal, "crux": entry["crux"],
            "calibration": {"rho_realized": entry["cal"]["rho_realized"],
                            "counts": entry["cal"]["counts"]},
            "per_rho": rhos}


def write_artifacts(run_dir, payload):
    os.makedirs(run_dir, exist_ok=True)
    out = {"verdict": payload["per_alpha"][PRIMARY_ALPHA]["verdict"].label
           if PRIMARY_ALPHA in payload["per_alpha"] else None,
           "mode": payload["mode"], "seed": payload["seed"], "rho_cal": payload["rho_cal"],
           "rho_test_grid": payload["rho_test_grid"], "n_cal": payload["n_cal"],
           "n_test": payload["n_test"], "worst_group_acc": payload["worst_group_acc"],
           "info": payload["info"],
           "alphas": {str(a): _verdict_json(payload["per_alpha"][a], a)
                      for a in payload["alphas"]}}
    with open(os.path.join(run_dir, "shiftcp_results.json"), "w") as f:
        json.dump(_to_jsonable(out), f, indent=2)

    # tidy CSV: one row per (alpha, rho_test, method)
    rows = []
    for alpha in payload["alphas"]:
        v = payload["per_alpha"][alpha]["verdict"]
        for rv in v.per_rho:
            for name, r in rv.reports.items():
                rows.append({"alpha": alpha, "rho_test": rv.rho_test, "method": name,
                             "marginal_cov": r.marginal_cov, "worst_group": r.worst_group,
                             "worst_group_cov": r.worst_group_cov,
                             "worst_group_cov_lo": r.worst_group_cov_lo,
                             "worst_group_cov_hi": r.worst_group_cov_hi,
                             "minority_cov": r.minority_cov, "avg_size": r.avg_size,
                             "ref_group_size": r.ref_group_size,
                             "stdf_undercovers": rv.stdf_undercovers, "green": rv.green})
    pd.DataFrame(rows).to_csv(os.path.join(run_dir, "shiftcp_metrics.csv"), index=False)


# ======================================================================================
# CLI
# ======================================================================================
def _default_cfg():
    cfg = {}
    common_path = "configs/common.yaml"
    if os.path.exists(common_path):
        cfg["common"] = load_config(common_path).get("common", {})
    cfg.setdefault("dataset", {"n_classes": 2})
    cfg.setdefault("scores", {"f_kind": "APS", "cpt_score_kind": "THR"})
    cfg.setdefault("shift", {"rho_cal": DEFAULT_RHO_CAL, "rho_test": DEFAULT_RHO_TEST,
                             "frac_cal": 0.5})
    cfg.setdefault("alphas", [0.1, 0.2])
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/shiftcp_derisk.yaml")
    ap.add_argument("--smoke", action="store_true", help="synthetic CPU pipeline self-test")
    ap.add_argument("--timestamp", default=None)
    args = ap.parse_args()
    ts = args.timestamp or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    if args.smoke:
        cfg = load_config(args.config) if os.path.exists(args.config) else _default_cfg()
        cfg.setdefault("common", _default_cfg()["common"])
        for k, v in _default_cfg().items():
            cfg.setdefault(k, v)
        payload = run(cfg, seed=0, mode="smoke")
        write_report("SHIFTCP_DERISK_REPORT.md", cfg, payload)
        v = payload["per_alpha"][PRIMARY_ALPHA]["verdict"]
        print("[shiftcp] SMOKE OK — full pipeline ran on synthetic data.")
        print(f"[shiftcp] OVERALL VERDICT (alpha={PRIMARY_ALPHA}): {v.label}")
        print("[shiftcp] wrote SHIFTCP_DERISK_REPORT.md (smoke). STOP — real numbers: cache-reuse run.")
        return

    cfg = load_config(args.config)
    seed = int(cfg.get("experiment", {}).get("seeds", [0])[0])
    run_id = f"shiftcp_derisk_{ts}"
    run_dir = os.path.join(cfg.get("experiment", {}).get("out_dir", "results"), "runs", run_id)

    payload = run(cfg, seed, mode="real")
    write_artifacts(run_dir, payload)
    write_report(os.path.join(run_dir, "SHIFTCP_DERISK_REPORT.md"), cfg, payload)
    write_report("SHIFTCP_DERISK_REPORT.md", cfg, payload)

    v = payload["per_alpha"][PRIMARY_ALPHA]["verdict"]
    print(f"[shiftcp] results -> {run_dir}")
    print(f"[shiftcp] OVERALL VERDICT (alpha={PRIMARY_ALPHA}): {v.label}  {v.rationale}")
    print("[shiftcp] STOP. Await human review of SHIFTCP_DERISK_REPORT.md before ablations, "
          "CelebA, multi-seed, figures, or paper machinery.")


if __name__ == "__main__":
    main()
