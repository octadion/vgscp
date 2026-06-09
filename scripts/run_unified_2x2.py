"""CORRECTED unified 2x2 run (study paper v2) -- REPLACES the separate E1 + E3.

Fixes the three confounds from the prior run (run spec v2 §1):
  #1  feature head: real path now fits/​reports a CLEAN-CUB feature head (see experiments.real_data;
      the head fix + both top-1s are surfaced here).
  #2  score function: the SAME conformal score is compared ACROSS representations (never feat-APS vs
      cpt-THR). APS is primary; RAPS/THR are appendix. (§2c)
  #3  the 2x2 is COMPLETE and in ONE run: representation {feature, concept} x scheme {split,
      Mondrian}, all four cells, one finer rho sweep, >=10 seeds. (§2d)

§2b: the concept score is IMAGE-DERIVED (predicted), never the ground-truth MTurk attributes. The
concept source is selectable (cbm primary / zeroshot appendix / gt_attrs_leaky = the prior invalid
behaviour, kept only for the leakage demonstration).

Two views are produced from this ONE run (§2d): (i) coverage-gap-vs-rho for all four cells (former
E3), (ii) worst-group-coverage-vs-mean-set-size frontier scatter per rho (former E1 figure).

The pre-committed group-free-substitution verdict + kill-switch live in eval/unified_verdict.py
(committed before any numbers). The efficiency (secondary) claim carries the §2e accuracy control.

    python -m scripts.run_unified_2x2 --smoke --seeds 10                       # CPU self-test
    python -m scripts.run_unified_2x2 --smoke --seeds 10 --concept-source zeroshot
    python -m scripts.run_unified_2x2 --config configs/cub200_frontier.yaml --seeds 10   # Colab/GPU
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
from datetime import datetime, timezone

import numpy as np
import pandas as pd

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from conformal import group_robust
from conformal import scores as cscores
from conformal.split_conformal import build_sets, conformal_quantile, set_sizes
from eval.unified_verdict import PRIMARY_SCORE, unified_verdict
from experiments import cub200_frontier as cf

SCORE_FNS = ("APS", "RAPS", "THR")
DEFAULT_RHO_TEST = [0.95, 0.90, 0.80, 0.70, 0.60, 0.50]
DEFAULT_RHO_CAL = 0.95
# the COMPLETE 2x2 (the missing feat+split is now present)
CELLS = (("feature", "split"), ("feature", "Mondrian"),
         ("concept", "split"), ("concept", "Mondrian"))


# ======================================================================================
# Score precompute (once per (seed, score) on the whole pool; resampling indexes rows)
# ======================================================================================
def _score_arrays(pop, score, seed):
    n = pop["feat_probs"].shape[0]
    u = cscores.draw_randomization(n, seed=seed + 7) if score in ("APS", "RAPS") else None
    feat = cscores.scores_all(score, pop["feat_probs"], u=u)
    cpt = cscores.scores_all(score, pop["cpt_probs"], u=u)
    return feat, cpt


def _true(score_all, y):
    return score_all[np.arange(len(y)), y]


# ======================================================================================
# §2e accuracy control: classes where the two heads' clean top-1 are matched
# ======================================================================================
def matched_class_subset(pop, tol=0.10, min_classes=5):
    """Classes where per-class top-1 of the feature & concept heads are within ``tol``.

    Isolates the set-size comparison from the head-accuracy gap: efficiency is re-evaluated on test
    points whose species fall in this subset. Returns (set_of_classes, info). If too few classes
    match, the caller labels the efficiency result "confounded by accuracy" (per §2e)."""
    y = pop["species"]
    fp, cp = pop["feat_probs"].argmax(1), pop["cpt_probs"].argmax(1)
    matched = []
    per = {}
    for c in np.unique(y):
        m = y == c
        if m.sum() < 10:
            continue
        fa, ca = float((fp[m] == c).mean()), float((cp[m] == c).mean())
        per[int(c)] = (fa, ca)
        if abs(fa - ca) <= tol:
            matched.append(int(c))
    feasible = len(matched) >= min_classes
    info = {"n_matched_classes": len(matched), "tol": tol, "feasible": bool(feasible),
            "matched_feat_top1": float(np.mean([per[c][0] for c in matched])) if matched else float("nan"),
            "matched_cpt_top1": float(np.mean([per[c][1] for c in matched])) if matched else float("nan")}
    return set(matched), info


# ======================================================================================
# Metrics on a (resampled) test set -- worst-group coverage = coverage on the ATYPICAL group
# ======================================================================================
def _method_metrics(member, species, typ, matched_mask=None):
    cov = member[np.arange(len(species)), species].astype(int)
    sizes = set_sizes(member)
    per_cov = {int(g): float(cov[typ == g].mean()) for g in np.unique(typ)}
    atyp = per_cov.get(cf.ATYPICAL, float("nan"))
    cov_gap = (max(per_cov.values()) - min(per_cov.values())) if len(per_cov) > 1 else 0.0
    # §2e: mean set size restricted to the accuracy-matched class subset (NaN if no matched points)
    sz_matched = float(sizes[matched_mask].mean()) if (matched_mask is not None
                                                       and matched_mask.any()) else float("nan")
    return {"worst_cov": atyp, "mean_set_size": float(sizes.mean()),
            "marg_cov": float(cov.mean()), "cov_gap": float(cov_gap),
            "mean_set_size_accmatched": sz_matched}


# ======================================================================================
# One seed: calibrate at rho_cal, sweep test rho, ALL FOUR cells x ALL scores
# ======================================================================================
def run_seed(pop, seed, rho_cal, rho_test_grid, n_cal, n_test, frac_cal, alpha, matched_classes):
    species, typ = pop["species"], pop["typicality"]
    cal_pool, test_pool = cf.split_pool(len(species), frac_cal, seed)
    cal_counts = cf.reference_species_counts(species[cal_pool], n_cal)
    test_counts = cf.reference_species_counts(species[test_pool], n_test)
    cal_rs = cf.resample_to_rho_multiclass(species[cal_pool], typ[cal_pool], rho_cal, n_cal,
                                           seed=seed, species_counts=cal_counts)
    cal_idx = cal_pool[cal_rs.idx]

    records = []
    diag = {"cal_rho_realized": cal_rs.rho_realized, "cal_n_empty_atypical": cal_rs.n_empty_atypical,
            "test_rho_realized": {}}
    for score in SCORE_FNS:
        feat_all, cpt_all = _score_arrays(pop, score, seed)
        y_cal, t_cal = species[cal_idx], typ[cal_idx]
        # SAME score function for both representations; split + Mondrian thresholds for each (§2c)
        q_feat = conformal_quantile(_true(feat_all[cal_idx], y_cal), alpha)
        q_cpt = conformal_quantile(_true(cpt_all[cal_idx], y_cal), alpha)
        mq_feat = group_robust.mondrian_quantiles(_true(feat_all[cal_idx], y_cal), t_cal, alpha)
        mq_cpt = group_robust.mondrian_quantiles(_true(cpt_all[cal_idx], y_cal), t_cal, alpha)

        for i, rho in enumerate(rho_test_grid):
            ts = cf.resample_to_rho_multiclass(species[test_pool], typ[test_pool], rho, n_test,
                                               seed=seed + 1000 + i, species_counts=test_counts)
            test_idx = test_pool[ts.idx]
            y_t, t_t = species[test_idx], typ[test_idx]
            feat_t, cpt_t = feat_all[test_idx], cpt_all[test_idx]
            mmask = np.isin(y_t, list(matched_classes)) if matched_classes else None
            diag["test_rho_realized"][round(rho, 4)] = ts.rho_realized
            membership = {
                ("feature", "split"): build_sets(feat_t, q_feat),
                ("feature", "Mondrian"): group_robust.mondrian_build_sets(feat_t, t_t, mq_feat),
                ("concept", "split"): build_sets(cpt_t, q_cpt),
                ("concept", "Mondrian"): group_robust.mondrian_build_sets(cpt_t, t_t, mq_cpt),
            }
            for (rep, scheme) in CELLS:
                m = _method_metrics(membership[(rep, scheme)], y_t, t_t, mmask)
                records.append({"test_corr": float(rho), "rho_realized": ts.rho_realized,
                                "score": score, "representation": rep, "scheme": scheme,
                                "seed": int(seed), "worst_cov": m["worst_cov"],
                                "cov_gap": m["cov_gap"], "marg_cov": m["marg_cov"],
                                "mean_set_size": m["mean_set_size"],
                                "mean_set_size_accmatched": m["mean_set_size_accmatched"]})
    return records, diag


# ======================================================================================
# Orchestration (population built ONCE; seeds = random cal/test splits)
# ======================================================================================
def run(cfg, mode, n_seeds, concept_source, seeds=None):
    rho_cal = float(cfg.get("shift", {}).get("rho_cal", DEFAULT_RHO_CAL))
    rho_test_grid = list(cfg.get("shift", {}).get("rho_test", DEFAULT_RHO_TEST))
    frac_cal = float(cfg.get("shift", {}).get("frac_cal", 0.5))
    alpha = float(cfg.get("alpha", 0.1))
    seeds = seeds if seeds is not None else list(range(n_seeds))

    if mode == "smoke":
        sc = cf.SmokeConfig(
            n=int(cfg.get("smoke", {}).get("n", 9000)),
            n_classes=int(cfg.get("smoke", {}).get("n_classes", 40)),
            concept_source=concept_source,
            seed=int(cfg.get("smoke", {}).get("pop_seed", 12345)))
        pop = cf.make_smoke_population(sc)
        n_cal = int(cfg.get("smoke", {}).get("n_cal", 2000))
        n_test = int(cfg.get("smoke", {}).get("n_test", 2000))
    else:
        cfg = dict(cfg)
        cfg["concept_source"] = concept_source        # plumb the predicted-concept choice to real
        pop = cf.load_real_population(cfg, seed=int(cfg.get("pop_seed", 0)))
        n_cal = int(cfg.get("shift", {}).get("n_cal", 3000))
        n_test = int(cfg.get("shift", {}).get("n_test", 3000))

    matched_classes, acc_info = matched_class_subset(pop)

    all_records, diags = [], {}
    for s in seeds:
        recs, diag = run_seed(pop, s, rho_cal, rho_test_grid, n_cal, n_test, frac_cal, alpha,
                              matched_classes)
        all_records.extend(recs)
        diags[s] = diag

    verdicts = {sc_: unified_verdict(all_records, rho_cal=rho_cal, alpha=alpha, score=sc_)
                for sc_ in SCORE_FNS}
    return {"records": all_records, "verdicts": verdicts, "diags": diags, "mode": mode,
            "seeds": seeds, "rho_cal": rho_cal, "rho_test_grid": rho_test_grid, "alpha": alpha,
            "n_cal": n_cal, "n_test": n_test, "n_classes": pop["n_classes"],
            "concept_source": pop.get("concept_source", concept_source),
            "feat_top1": pop.get("feat_top1"), "cpt_top1": pop.get("cpt_top1"),
            "acc_control": acc_info, "pop_info": pop.get("info")}


# ======================================================================================
# §2e efficiency at matched worst-group coverage (concept vs feature, SAME score)
# ======================================================================================
def efficiency_summary(df, payload, score=PRIMARY_SCORE):
    """Concept-vs-feature mean set size under Mondrian (both hold worst-group cov ~1-a), reported
    RAW (honest accuracies) and on the accuracy-MATCHED class subset (§2e)."""
    sub = df[df["score"] == score]
    out = {"score": score, "feat_top1": payload["feat_top1"], "cpt_top1": payload["cpt_top1"],
           "acc_control": payload["acc_control"], "per_rho": []}
    for rho in sorted(sub["test_corr"].unique()):
        r = sub[abs(sub["test_corr"] - rho) < 1e-9]
        def cell(rep, scheme, col):
            c = r[(r.representation == rep) & (r.scheme == scheme)]
            return float(c[col].mean()) if len(c) else float("nan")
        out["per_rho"].append({
            "rho_test": float(rho),
            "feat_size": cell("feature", "Mondrian", "mean_set_size"),
            "cpt_size": cell("concept", "Mondrian", "mean_set_size"),
            "feat_size_accmatched": cell("feature", "Mondrian", "mean_set_size_accmatched"),
            "cpt_size_accmatched": cell("concept", "Mondrian", "mean_set_size_accmatched"),
            "feat_worst_cov": cell("feature", "Mondrian", "worst_cov"),
            "cpt_worst_cov": cell("concept", "Mondrian", "worst_cov")})
    return out


# ======================================================================================
# Aggregation / serialization
# ======================================================================================
def _agg_table(df, score):
    sub = df[df["score"] == score]
    return sub.groupby(["representation", "scheme", "test_corr"]).agg(
        worst_cov_m=("worst_cov", "mean"), worst_cov_s=("worst_cov", "std"),
        size_m=("mean_set_size", "mean"), size_s=("mean_set_size", "std"),
        marg_m=("marg_cov", "mean"), gap_m=("cov_gap", "mean"), gap_s=("cov_gap", "std"),
        n=("seed", "count")).reset_index()


def write_csv(path, records):
    pd.DataFrame(records).sort_values(
        ["score", "representation", "scheme", "test_corr", "seed"]).to_csv(path, index=False)


def make_figures(df, fig_dir, payload, score=PRIMARY_SCORE):
    """View (i) gap-vs-rho (4 cells) and view (ii) worst-cov-vs-set-size frontier scatter."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"[u2x2] matplotlib unavailable ({e}); skipping figures")
        return []
    os.makedirs(fig_dir, exist_ok=True)
    g = _agg_table(df, score)
    target = 1.0 - payload["alpha"]
    style = {("feature", "split"): ("#d62728", "s", "feat+split"),
             ("feature", "Mondrian"): ("#ff9896", "s", "feat+Mondrian"),
             ("concept", "split"): ("#1f77b4", "^", "cpt+split"),
             ("concept", "Mondrian"): ("#2ca02c", "o", "cpt+Mondrian")}
    paths = []
    src = payload["concept_source"]

    # view (i): coverage gap vs rho
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for cell, (col, mk, lab) in style.items():
        s = g[(g.representation == cell[0]) & (g.scheme == cell[1])].sort_values("test_corr")
        if s.empty:
            continue
        ax.plot(s["test_corr"], s["gap_m"], marker=mk, color=col, label=lab, lw=1.6, ms=6)
    ax.set_xlabel("test correlation ρ  (← stronger shift)")
    ax.set_ylabel("coverage gap (max−min group coverage)")
    ax.invert_xaxis()
    ax.set_title(f"Coverage-gap vs ρ — {score} (concept={src}, α={payload['alpha']:g})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        p = os.path.join(fig_dir, f"u2x2_gap_vs_rho_{score}.{ext}")
        fig.savefig(p, dpi=150)
        paths.append(p)
    plt.close(fig)

    # view (ii): worst-group coverage vs mean set size frontier
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for cell, (col, mk, lab) in style.items():
        s = g[(g.representation == cell[0]) & (g.scheme == cell[1])].sort_values("test_corr")
        if s.empty:
            continue
        ax.plot(s["size_m"], s["worst_cov_m"], marker=mk, color=col, label=lab, lw=1.4, ms=6)
        for _, row in s.iterrows():
            ax.annotate(f"{row['test_corr']:.2f}", (row["size_m"], row["worst_cov_m"]),
                        fontsize=6, alpha=0.6, xytext=(3, 3), textcoords="offset points")
    ax.axhline(target, ls="--", color="gray", lw=1, label=f"target 1−α={target:g}")
    ax.set_xlabel("mean set size  (efficiency →)")
    ax.set_ylabel("worst-group (atypical) coverage")
    ax.set_title(f"Frontier — {score} (concept={src}, points labeled by ρ_test)")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        p = os.path.join(fig_dir, f"u2x2_frontier_{score}.{ext}")
        fig.savefig(p, dpi=150)
        paths.append(p)
    plt.close(fig)
    return paths


def _verdict_json(v):
    return {"label": v.label, "green": v.green, "score": v.score, "alpha": v.alpha,
            "rho_cal": v.rho_cal, "n_shifted": v.n_shifted, "n_recovered": v.n_recovered,
            "sweep_mean_R": v.sweep_mean_R, "sweep_mean_mech_feat": v.sweep_mean_mech_feat,
            "sweep_mean_mech_cpt": v.sweep_mean_mech_cpt, "rationale": v.rationale,
            "per_rho": [{"rho_test": r.rho_test, "shifted": r.shifted, "R": r.R,
                         "repr_significant": r.repr_significant, "recovers": r.recovers,
                         "gap_feat_split": r.gap_feat_split["mean"],
                         "gap_cpt_split": r.gap_cpt_split["mean"],
                         "gap_feat_mond": r.gap_feat_mond["mean"],
                         "gap_cpt_mond": r.gap_cpt_mond["mean"],
                         "d_repr_lo": r.d_repr["lo"], "d_repr_hi": r.d_repr["hi"]}
                        for r in v.per_rho]}


def write_json(path, payload, eff):
    out = {k: payload[k] for k in ("mode", "seeds", "rho_cal", "rho_test_grid", "alpha", "n_cal",
                                   "n_test", "n_classes", "concept_source", "feat_top1", "cpt_top1",
                                   "acc_control")}
    out["verdicts"] = {s: _verdict_json(v) for s, v in payload["verdicts"].items()}
    out["efficiency"] = eff
    out["diags"] = {str(k): v for k, v in payload["diags"].items()}
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=float)


# ======================================================================================
# Report
# ======================================================================================
def _pm(m, s):
    return f"{m:.3f}±{s:.3f}" if np.isfinite(m) else "--"


def write_report(path, df, payload, eff, fig_paths):
    v = payload["verdicts"][PRIMARY_SCORE]
    mode, src = payload["mode"], payload["concept_source"]
    f1, c1 = payload["feat_top1"], payload["cpt_top1"]
    lines = [
        "# Unified 2×2 — CORRECTED coverage / gap run (study paper v2)",
        "",
        f"**Date:** {datetime.now(timezone.utc).date()} · **Run mode:** {mode} · "
        f"**Seeds:** {len(payload['seeds'])} · **α={payload['alpha']:g}** · "
        f"**ρ_cal={payload['rho_cal']:g}** · **ρ_test:** {payload['rho_test_grid']} · "
        f"**classes:** {payload['n_classes']} · **concept source:** `{src}` (image-derived)",
        "",
    ]
    if mode == "smoke":
        lines += ["> ⚠️ **SMOKE (synthetic) run** — fabricated CUB-200-like population validating the "
                  "corrected construct→calibrate→4-cell→verdict pipeline on CPU. Real numbers come "
                  "from the Colab/GPU cached-feature run. **Not a scientific result.**", ""]
    lines += [
        "## Pre-committed claim (eval/unified_verdict.py — fixed before any numbers)",
        "Group-free substitution: an invariant (predicted) concept score under **pooled split** "
        "recovers fraction `R = (gap[feat,split]−gap[cpt,split]) / (gap[feat,split]−gap[feat,Mondrian])` "
        "of the worst-group gap reduction that the **Mondrian mechanism** gives. GREEN iff R≥0.5 at a "
        "majority of shifted ρ with the paired gap[feat,split]−gap[cpt,split] CI excluding 0. "
        "Kill-switch fallback: *\"group-conditional calibration is the binding lever; the "
        "representation contributes little beyond efficiency.\"*",
        "",
        "## Classifier heads (top-1 accuracy) — §2a",
        f"- feature-space head: **{f1:.3f}**" if f1 is not None else "- feature-space head: (real run)",
        f"- concept-space head (`{src}`): **{c1:.3f}**" if c1 is not None else
        "- concept-space head: (real run)",
        "",
        f"## Verdict (primary score {PRIMARY_SCORE}): **{v.label}**",
        f"{v.rationale}",
        "",
        "### Recovered fraction R and mechanism main effect, by ρ_test (APS)",
        "| ρ_test | gap feat+split | gap cpt+split | gap feat+Mond | R | repr.CI≠0 | recovers |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in v.per_rho:
        if not r.shifted:
            continue
        lines.append(f"| {r.rho_test:.2f} | {r.gap_feat_split['mean']:.3f} | "
                     f"{r.gap_cpt_split['mean']:.3f} | {r.gap_feat_mond['mean']:.3f} | "
                     f"{r.R:.2f} | {'yes' if r.repr_significant else 'no'} | "
                     f"{'✓' if r.recovers else '·'} |")
    lines += ["", f"Mechanism main effect (Mondrian gap reduction, sweep mean): "
              f"feature **{v.sweep_mean_mech_feat:+.3f}**, concept **{v.sweep_mean_mech_cpt:+.3f}** "
              f"— reported as part of the honest 2×2, not suppressed.", ""]

    # full 2x2 tables per score
    for score in SCORE_FNS:
        g = _agg_table(df, score)
        tag = " (primary)" if score == PRIMARY_SCORE else " (appendix)"
        lines += [f"### {score}{tag} — worst-group cov / mean set size / coverage gap by ρ_test", "",
                  "| representation | scheme | ρ | worst-grp cov | mean set size | marg cov | cov gap |",
                  "|---|---|---|---|---|---|---|"]
        for (rep, scheme) in CELLS:
            s = g[(g.representation == rep) & (g.scheme == scheme)].sort_values(
                "test_corr", ascending=False)
            for _, r in s.iterrows():
                lines.append(f"| {rep} | {scheme} | {r['test_corr']:.2f} | "
                             f"{_pm(r['worst_cov_m'], r['worst_cov_s'])} | "
                             f"{_pm(r['size_m'], r['size_s'])} | {r['marg_m']:.3f} | "
                             f"{_pm(r['gap_m'], r['gap_s'])} |")
        lines.append("")

    # §2e efficiency control
    lines += ["## Efficiency (secondary) with the §2e accuracy control",
              f"Concept-vs-feature mean set size under Mondrian (both hold worst-group cov ≈ 1−α). "
              f"Head top-1: feature **{f1:.3f}**, concept **{c1:.3f}**.",
              ""]
    ac = payload["acc_control"]
    if ac["feasible"]:
        lines.append(f"Accuracy-matched control: **{ac['n_matched_classes']} classes** with |Δtop-1|≤"
                     f"{ac['tol']:g} (matched top-1 feature {ac['matched_feat_top1']:.3f} / concept "
                     f"{ac['matched_cpt_top1']:.3f}); efficiency re-evaluated on that subset below.")
    else:
        lines.append(f"⚠️ Accuracy matching **infeasible** (only {ac['n_matched_classes']} classes "
                     f"within |Δtop-1|≤{ac['tol']:g}); the raw efficiency result below is "
                     f"**confounded by accuracy** (§2e) and labelled as such.")
    lines += ["", "| ρ_test | feat size | cpt size | feat size (acc-matched) | cpt size (acc-matched) |",
              "|---|---|---|---|---|"]
    for r in eff["per_rho"]:
        lines.append(f"| {r['rho_test']:.2f} | {r['feat_size']:.2f} | {r['cpt_size']:.2f} | "
                     f"{r['feat_size_accmatched']:.2f} | {r['cpt_size_accmatched']:.2f} |")
    lines.append("")

    if fig_paths:
        lines += ["## Figures", ""]
        for p in fig_paths:
            if p.endswith(".pdf"):
                lines.append(f"- {os.path.relpath(p)}")
        lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ======================================================================================
# CLI
# ======================================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/cub200_frontier.yaml")
    ap.add_argument("--smoke", action="store_true", help="synthetic CPU pipeline self-test")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--concept-source", default="cbm",
                    choices=list(cf.CONCEPT_SOURCES), help="image-derived concept source (§2b)")
    ap.add_argument("--out", default="results/unified")
    args = ap.parse_args()

    cfg = {}
    if os.path.exists(args.config):
        from config_util import load_config
        cfg = load_config(args.config)

    mode = "smoke" if args.smoke else "real"
    payload = run(cfg, mode=mode, n_seeds=args.seeds, concept_source=args.concept_source)
    df = pd.DataFrame(payload["records"])
    eff = efficiency_summary(df, payload)

    os.makedirs(args.out, exist_ok=True)
    write_csv(os.path.join(args.out, "unified_2x2.csv"), payload["records"])
    fig_paths = make_figures(df, "results/figures", payload)
    write_json(os.path.join(args.out, "unified_results.json"), payload, eff)
    write_report(os.path.join(args.out, "UNIFIED_REPORT.md"), df, payload, eff, fig_paths)

    v = payload["verdicts"][PRIMARY_SCORE]
    print(f"[u2x2] mode={mode} concept={args.concept_source} seeds={len(payload['seeds'])} -> {args.out}")
    print(f"[u2x2] heads top-1: feature={payload['feat_top1']}, concept={payload['cpt_top1']}")
    print(f"[u2x2] VERDICT ({PRIMARY_SCORE}): {v.label}")
    print(f"[u2x2]   {v.rationale}")
    if mode == "smoke":
        print("[u2x2] SMOKE OK — corrected pipeline validated on synthetic. Real numbers: Colab/GPU.")


if __name__ == "__main__":
    main()
