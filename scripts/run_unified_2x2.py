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
from eval.unified_verdict import PRIMARY_SCORE, combined_decision, unified_verdict
from experiments import cub200_frontier as cf
from experiments.real_data import GATE_MIN_TOP1, FeatureHeadGateError

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
def run(cfg, mode, n_seeds, concept_source, seeds=None, diagnostic=False):
    rho_cal = float(cfg.get("shift", {}).get("rho_cal", DEFAULT_RHO_CAL))
    rho_test_grid = list(cfg.get("shift", {}).get("rho_test", DEFAULT_RHO_TEST))
    frac_cal = float(cfg.get("shift", {}).get("frac_cal", 0.5))
    alpha = float(cfg.get("alpha", 0.1))
    seeds = seeds if seeds is not None else list(range(n_seeds))

    if mode == "smoke":
        smk = cfg.get("smoke", {})
        sc_kw = {}
        # optional feature-margin overrides so a DEGRADED-head smoke can exercise the gate/diagnostic
        for k in ("feat_margin_typical", "feat_margin_atypical", "feat_spurious_kappa"):
            if k in smk:
                sc_kw[k] = float(smk[k])
        sc = cf.SmokeConfig(
            n=int(smk.get("n", 9000)), n_classes=int(smk.get("n_classes", 40)),
            concept_source=concept_source, seed=int(smk.get("pop_seed", 12345)), **sc_kw)
        pop = cf.make_smoke_population(sc)
        n_cal = int(cfg.get("smoke", {}).get("n_cal", 2000))
        n_test = int(cfg.get("smoke", {}).get("n_test", 2000))
    else:
        cfg = dict(cfg)
        cfg["concept_source"] = concept_source        # plumb the predicted-concept choice to real
        if diagnostic:
            cfg["bypass_gate"] = True                 # v4b: one-off no-verdict measurement at the head
        pop = cf.load_real_population(cfg, seed=int(cfg.get("pop_seed", 0)))
        n_cal = int(cfg.get("shift", {}).get("n_cal", 3000))
        n_test = int(cfg.get("shift", {}).get("n_test", 3000))

    # §4 HARD GATE: the head must clear the floor BEFORE any verdict run. v4b --diagnostic-no-verdict
    # BYPASSES the gate for a one-off measurement (no verdict emitted; the gate stays in force for
    # verdict runs). Otherwise enforce it (real mode also raised inside assemble; this re-checks smoke).
    gate_top1 = pop.get("feat_top1_indomain_typical") or pop.get("feat_top1")
    if diagnostic:
        print(f"[u2x2 DIAGNOSTIC] §4 gate BYPASSED for this ONE-OFF no-verdict run; in-domain/proxy "
              f"typical top-1={gate_top1:.3f} (gate ≥ {GATE_MIN_TOP1} stays in force for verdict runs).")
    else:
        enforce_feature_gate(pop, mode)

    matched_classes, acc_info = matched_class_subset(pop)

    all_records, diags = [], {}
    for s in seeds:
        recs, diag = run_seed(pop, s, rho_cal, rho_test_grid, n_cal, n_test, frac_cal, alpha,
                              matched_classes)
        all_records.extend(recs)
        diags[s] = diag

    if diagnostic:
        verdicts = combined = None                    # v4b: NO verdict, measurements only
        diag_summary = diagnostic_summary(all_records, rho_cal, pop["n_classes"])
    else:
        verdicts = {sc_: unified_verdict(all_records, rho_cal=rho_cal, alpha=alpha, score=sc_)
                    for sc_ in SCORE_FNS}
        combined = combined_decision(verdicts)        # headline GREEN only if >=2/3 score fns GREEN
        diag_summary = None
    return {"records": all_records, "verdicts": verdicts, "combined": combined, "diags": diags,
            "diagnostic": diagnostic, "diagnostic_summary": diag_summary,
            "mode": mode, "seeds": seeds, "rho_cal": rho_cal, "rho_test_grid": rho_test_grid,
            "alpha": alpha, "n_cal": n_cal, "n_test": n_test, "n_classes": pop["n_classes"],
            "concept_source": pop.get("concept_source", concept_source),
            "feat_top1": pop.get("feat_top1"), "cpt_top1": pop.get("cpt_top1"),
            "feat_top1_indomain_typical": pop.get("feat_top1_indomain_typical"),
            "feat_top1_cleancub": pop.get("feat_top1_cleancub"),
            "baseline_top1": pop.get("baseline_top1"), "diag": pop.get("diag"),
            "acc_control": acc_info, "pop_info": pop.get("info")}


def enforce_feature_gate(pop, mode):
    """§4 (v4)/§0b HARD HALT. Gate on the IN-DOMAIN species top-1 on TYPICAL composited d_test (real)
    or the synthetic head top-1 (smoke) -- the IN-EXPERIMENT distribution, not the clean-CUB anchor
    (the v3 mistake). Raise FeatureHeadGateError if below floor -> orchestrator writes BLOCKERS + no
    verdict. (Real mode already raised inside assemble_e1_population; this is the belt-and-suspenders.)"""
    gate_top1 = pop.get("feat_top1_indomain_typical")
    if gate_top1 is None:
        gate_top1 = pop.get("feat_top1")          # smoke proxy (in-domain by construction)
    if gate_top1 is not None and gate_top1 < GATE_MIN_TOP1:
        raise FeatureHeadGateError(
            gate_top1, where=f"{mode} orchestrator in-domain gate",
            diagnosis=("In-domain typical species top-1 below the 0.55 floor; a verdict here would be "
                       "computed on a weak head (the v3 error ran at 0.246). Halting per §0b: NO 2x2 "
                       "and NO verdict."))


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
# ======================================================================================
# v4b DIAGNOSTIC summary (NO verdict) — is the worst-group gap resolvable at this accuracy?
# ======================================================================================
GAP_RESOLVED = 0.15           # feat+split worst-group gap that counts as "clearly non-trivial"
GAP_WASHED = 0.05
SETFRAC_RESOLVED = 25.0 / 200  # mean set size < ~25/200 of the label space => non-degenerate
SETFRAC_WASHED = 40.0 / 200    # mean set size >= ~40/200 => sets too large, gap washed out


def diagnostic_summary(records, rho_cal, n_classes, score="APS"):
    """At the LARGEST shift (min shifted rho, e.g. 0.5): feature+split worst-group coverage, coverage
    gap, and mean set size (mean ± std over seeds), and which branch holds. NO verdict, NO R."""
    shifted = sorted({r["test_corr"] for r in records
                      if r["score"] == score and r["test_corr"] < rho_cal - 1e-9})
    rho = min(shifted) if shifted else float("nan")
    rows = [r for r in records if r["score"] == score and r["representation"] == "feature"
            and r["scheme"] == "split" and abs(r["test_corr"] - rho) < 1e-9]
    def ms(key):
        v = np.array([r[key] for r in rows], float)
        return (float(v.mean()), float(v.std(ddof=1)) if v.size > 1 else 0.0) if v.size else (float("nan"), 0.0)
    wc_m, wc_s = ms("worst_cov")
    gap_m, gap_s = ms("cov_gap")
    sz_m, sz_s = ms("mean_set_size")
    set_frac = sz_m / n_classes if n_classes else float("nan")
    if gap_m >= GAP_RESOLVED and set_frac < SETFRAC_RESOLVED:
        branch = "RESOLVED"
        rationale = (f"feat+split gap {gap_m:.3f} ≥ {GAP_RESOLVED} with non-degenerate sets "
                     f"({sz_m:.1f}/{n_classes} = {set_frac:.3f} < {SETFRAC_RESOLVED:.3f}): the construction "
                     f"works at this accuracy; the ≥0.55 gate was conservative. Recommend a proper "
                     f"hardened-verdict run with a 'moderate-accuracy' caveat AND re-examining the gate "
                     f"threshold WITH THE ADVISOR (transparently). [recommendation only — no action taken]")
    elif set_frac >= SETFRAC_WASHED and gap_m < GAP_WASHED:
        branch = "STILL WASHED OUT"
        rationale = (f"sets ≳ {SETFRAC_WASHED:.3f}·{n_classes} ({sz_m:.1f}) with feat+split gap "
                     f"{gap_m:.3f} < {GAP_WASHED}: this accuracy is insufficient, the 200-way ceiling is "
                     f"binding. Recommend raising accuracy via coarser labels (~20–50 classes / CUB "
                     f"families) [primary] or a stronger CLIP backbone (ViT-L/14) [keeps 200-way]; the "
                     f"researcher chooses. [recommendation only — no action taken]")
    else:
        branch = "IN BETWEEN"
        rationale = (f"feat+split gap {gap_m:.3f}, sets {sz_m:.1f}/{n_classes} ({set_frac:.3f}) fall "
                     f"between the RESOLVED and WASHED-OUT thresholds. Reporting both options (coarser "
                     f"labels OR stronger backbone); no call forced. [no action taken]")
    return {"rho_test": rho, "score": score, "n_classes": n_classes,
            "feat_split_worst_cov": (wc_m, wc_s), "feat_split_gap": (gap_m, gap_s),
            "feat_split_set_size": (sz_m, sz_s), "set_frac": set_frac,
            "branch": branch, "rationale": rationale}


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


def make_figures(df, fig_dir, payload, score=PRIMARY_SCORE, prefix="u2x2"):
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
        p = os.path.join(fig_dir, f"{prefix}_gap_vs_rho_{score}.{ext}")
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
        p = os.path.join(fig_dir, f"{prefix}_frontier_{score}.{ext}")
        fig.savefig(p, dpi=150)
        paths.append(p)
    plt.close(fig)
    return paths


def _verdict_json(v):
    return {"label": v.label, "green": v.green, "score": v.score, "alpha": v.alpha,
            "rho_cal": v.rho_cal, "n_shifted": v.n_shifted, "n_recovered": v.n_recovered,
            "majority": v.majority, "hardest_recovers": v.hardest_recovers,
            "sweep_mean_R": v.sweep_mean_R, "sweep_mean_mech_feat": v.sweep_mean_mech_feat,
            "sweep_mean_mech_cpt": v.sweep_mean_mech_cpt, "rationale": v.rationale,
            "per_rho": [{"rho_test": r.rho_test, "shifted": r.shifted, "is_hardest": r.is_hardest,
                         "R": r.R, "repr_significant": r.repr_significant, "recovers": r.recovers,
                         "gap_feat_split": r.gap_feat_split["mean"],
                         "gap_cpt_split": r.gap_cpt_split["mean"],
                         "gap_feat_mond": r.gap_feat_mond["mean"],
                         "gap_cpt_mond": r.gap_cpt_mond["mean"],
                         "d_repr_lo": r.d_repr["lo"], "d_repr_hi": r.d_repr["hi"]}
                        for r in v.per_rho]}


def write_json(path, payload, eff):
    out = {k: payload[k] for k in ("mode", "seeds", "rho_cal", "rho_test_grid", "alpha", "n_cal",
                                   "n_test", "n_classes", "concept_source", "feat_top1", "cpt_top1",
                                   "feat_top1_indomain_typical", "feat_top1_cleancub", "baseline_top1",
                                   "diag", "acc_control")}
    out["combined"] = payload["combined"]
    out["verdicts"] = {s: _verdict_json(v) for s, v in payload["verdicts"].items()}
    out["efficiency"] = eff
    out["diags"] = {str(k): v for k, v in payload["diags"].items()}
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=float)


def write_blockers_v4(err: FeatureHeadGateError, args):
    """§0b/§4 mandatory: on a gate failure (or branch B/C), write BLOCKERS_v4.md and produce NO 2x2 /
    NO verdict / skip E2."""
    txt = f"""# BLOCKERS v4 — §4 IN-DOMAIN GATE FAILED → run HALTED (no 2×2, no verdict, E2 skipped)

**Date:** {datetime.now(timezone.utc).date()} · **Run:** `run_unified_2x2` mode=\
{'smoke' if args.smoke else 'real'} concept_source={args.concept_source}

## Why this run produced no result
The IN-DOMAIN feature head did not clear the pre-committed **§4 gate** (in-domain species top-1 on the
TYPICAL composited test ≥ {GATE_MIN_TOP1}). Per spec v4 §4/§0b the run **halts here and emits NO 2×2,
NO verdict, and SKIPS E2** — a verdict on a weak head is a reporting error, not a result (v3 wrongly
ran at an in-domain 0.246 head measured by the wrong, clean-CUB, gate).

- **Gate location:** {err.where}
- **Observed in-domain typical top-1:** **{err.top1:.3f}**  (required ≥ {GATE_MIN_TOP1})

## Diagnosis (incl. §2 decision-table branch)
{err.diagnosis}

## What to do (per the §2 decision table)
- **Branch B (H2, encoding bug):** fix the composited CLIP feature extraction to use the canonical
  open_clip transform; re-verify §1; re-run. Do NOT lower the gate.
- **Branch C (H1, construction destroys species signal):** STOP; recommend dropping to coarser label
  granularity (~20–50 classes / CUB families) and return for human review.
Return for human review either way.
"""
    with open("BLOCKERS_v4.md", "w", encoding="utf-8") as f:
        f.write(txt)


def write_diagnostic_report(path, df, payload, fig_paths):
    """v4b DIAGNOSTIC report — clearly labelled, NO verdict. The 2x2 tables, the feat+split largest-
    shift numbers, and which branch (RESOLVED / STILL WASHED OUT / IN BETWEEN) holds."""
    s = payload["diagnostic_summary"]
    fit = payload.get("feat_top1_indomain_typical")
    f1, c1 = payload["feat_top1"], payload["cpt_top1"]
    wc, gap, sz = s["feat_split_worst_cov"], s["feat_split_gap"], s["feat_split_set_size"]
    lines = [
        "# Unified 2×2 — DIAGNOSTIC ONLY (study paper v4b) — **NO VERDICT**",
        "",
        f"**Date:** {datetime.now(timezone.utc).date()} · **Run mode:** {payload['mode']} · "
        f"**Seeds:** {len(payload['seeds'])} · **α={payload['alpha']:g}** · "
        f"**ρ_cal={payload['rho_cal']:g}** · **classes:** {payload['n_classes']} · "
        f"**concept source:** `{payload['concept_source']}`",
        "",
        "> ⚠️ **DIAGNOSTIC, NOT A VERDICT.** The §4 accuracy gate was BYPASSED for this **one-off** "
        "measurement (in-domain typical top-1 was below the 0.55 floor). **No GREEN/FALLBACK label, no "
        "R metric, no pre-registered outcome is emitted.** The ≥0.55 gate stays in force for any "
        "verdict run. This run only measures whether the worst-group gap is resolvable at this accuracy.",
        "",
    ] + (["> ⚠️ **SMOKE (synthetic) run** — fabricated population validating the diagnostic machinery "
          "(gate-bypass → 2×2 → branch, no verdict) on CPU. **The branch below is a MACHINERY check, "
          "NOT the scientific call** — the real numbers (the 0.426 in-domain head) come from the "
          "Colab/GPU run.", ""] if payload["mode"] == "smoke" else []) + [
        f"- in-domain typical top-1 (the head used): **{fit:.3f}**" if fit is not None else
        f"- head proxy top-1: **{f1:.3f}**",
        f"- pool feature top-1 **{f1:.3f}** · concept (`{payload['concept_source']}`) top-1 **{c1:.3f}**",
        "",
        "## Diagnostic question (largest shift, ρ_test = "
        f"{s['rho_test']:.2f}): is the worst-group gap resolvable here?",
        f"- **feature+split worst-group coverage** = {wc[0]:.3f} ± {wc[1]:.3f}",
        f"- **feature+split coverage gap** = {gap[0]:.3f} ± {gap[1]:.3f}  "
        f"(non-trivial if ≥ ~{GAP_RESOLVED})",
        f"- **feature+split mean set size** = {sz[0]:.2f} ± {sz[1]:.2f}  "
        f"(= {s['set_frac']:.3f}·{payload['n_classes']}; non-degenerate if < ~{SETFRAC_RESOLVED:.3f}·n)",
        "",
        f"## Decision branch: **{s['branch']}**  *(report only — no action taken; return for human review)*",
        s["rationale"],
        "",
    ]
    for score in SCORE_FNS:
        g = _agg_table(df, score)
        tag = " (primary)" if score == PRIMARY_SCORE else " (appendix)"
        lines += [f"### {score}{tag} — worst-group cov / mean set size / coverage gap by ρ_test", "",
                  "| representation | scheme | ρ | worst-grp cov | mean set size | marg cov | cov gap |",
                  "|---|---|---|---|---|---|---|"]
        for (rep, scheme) in CELLS:
            sub = g[(g.representation == rep) & (g.scheme == scheme)].sort_values(
                "test_corr", ascending=False)
            for _, r in sub.iterrows():
                lines.append(f"| {rep} | {scheme} | {r['test_corr']:.2f} | "
                             f"{_pm(r['worst_cov_m'], r['worst_cov_s'])} | "
                             f"{_pm(r['size_m'], r['size_s'])} | {r['marg_m']:.3f} | "
                             f"{_pm(r['gap_m'], r['gap_s'])} |")
        lines.append("")
    if fig_paths:
        lines += ["## Figures", ""] + [f"- {os.path.relpath(p)}" for p in fig_paths if p.endswith(".pdf")]
        lines.append("")
    lines += [
        "## Reproduce (real numbers: Colab/GPU; the §4 gate stays in force for verdict runs)",
        "```",
        "python -m scripts.run_unified_2x2 --config configs/cub200_frontier.yaml --seeds 10 \\",
        "       --concept-source cbm --diagnostic-no-verdict --out results/unified_diagnostic",
        "```",
        "This BYPASSES the gate for THIS one-off measurement only. Decision branches (report, do NOT "
        "act — return for human review): **RESOLVED** → propose a hardened-verdict run + re-examine the "
        "gate threshold with the advisor; **STILL WASHED OUT** → coarser labels (~20–50 families) or "
        "ViT-L/14; **IN BETWEEN** → report both options.",
        "",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ======================================================================================
# Report
# ======================================================================================
def _pm(m, s):
    return f"{m:.3f}±{s:.3f}" if np.isfinite(m) else "--"


def write_report(path, df, payload, eff, fig_paths):
    v = payload["verdicts"][PRIMARY_SCORE]
    comb = payload["combined"]
    mode, src = payload["mode"], payload["concept_source"]
    f1, c1 = payload["feat_top1"], payload["cpt_top1"]
    fit, fcc = payload.get("feat_top1_indomain_typical"), payload.get("feat_top1_cleancub")
    diag = payload.get("diag")
    lines = [
        "# Unified 2×2 — IN-DOMAIN head run (study paper v4)",
        "",
        f"**Date:** {datetime.now(timezone.utc).date()} · **Run mode:** {mode} · "
        f"**Seeds:** {len(payload['seeds'])} · **α={payload['alpha']:g}** · "
        f"**ρ_cal={payload['rho_cal']:g}** · **ρ_test:** {payload['rho_test_grid']} · "
        f"**classes:** {payload['n_classes']} · **concept source:** `{src}` (image-derived, in-domain)",
        "",
    ]
    if mode == "smoke":
        lines += ["> ⚠️ **SMOKE (synthetic) run** — fabricated CUB-200-like population validating the "
                  "in-domain-gate→4-cell→verdict pipeline on CPU. Real numbers come from the Colab/GPU "
                  "cached-feature run. **Not a scientific result.**", ""]
    if diag:
        lines += ["## §1 diagnostic (reported before any patch)",
                  f"- clean→clean (anchor): **{diag['clean_to_clean']:.3f}**",
                  f"- clean→composited (the v3 mismatch reproduced): **{diag['clean_to_composited']:.3f}**",
                  f"- **in-domain** composited→composited d_test — all **{diag['indomain_all']:.3f}** / "
                  f"typical **{diag['indomain_typical']:.3f}** / atypical **{diag['indomain_atypical']:.3f}**",
                  ""]
    lines += [
        "## §4 IN-DOMAIN accuracy gate (HARD HALT — checked before any 2×2/verdict)",
        f"- gate metric = **in-domain typical** species top-1 (composited d_test, is_minority==0): "
        f"**{fit:.3f}** (gate ≥ {GATE_MIN_TOP1}; **PASSED** — else HALTED with no verdict)"
        if fit is not None else
        f"- gate metric (real run); smoke proxy feat_top1={f1:.3f} ≥ {GATE_MIN_TOP1}",
        f"- secondary anchor: clean-CUB clean→clean top-1 = **{fcc:.3f}**" if fcc is not None else
        "- secondary clean-CUB anchor: (real run)",
        "",
        "## Pre-committed HARDENED claim (eval/unified_verdict.py — fixed before any numbers)",
        "Group-free substitution: an invariant (predicted) concept score under **pooled split** "
        "recovers fraction `R = (gap[feat,split]−gap[cpt,split]) / (gap[feat,split]−gap[feat,Mondrian])`. "
        "**v3 per-score GREEN** iff R≥0.5 (paired CI≠0) at a majority of shifted ρ **AND at the largest "
        "shift ρ=0.5**. **v3 headline GREEN** iff per-score GREEN for **≥2/3** score functions. "
        "Kill-switch fallback: *\"group-conditional calibration is the binding lever for worst-group "
        "coverage; an invariant representation does not robustly substitute for it.\"*",
        "",
        "## Classifier heads (top-1 accuracy)",
        f"- feature-space head (pool): **{f1:.3f}**" if f1 is not None else
        "- feature-space head: (real run)",
        f"- concept-space head (`{src}`): **{c1:.3f}**" if c1 is not None else
        "- concept-space head: (real run)",
        "",
        f"## HEADLINE verdict (v3 combined, ≥{comb['min_required']}/3 scores): **{comb['label']}**",
        f"{comb['rationale']}",
        "",
        "| score | per-score verdict | majority recovers | ρ=0.5 recovers | sweep R |",
        "|---|---|---|---|---|",
    ]
    for s in SCORE_FNS:
        vs = payload["verdicts"][s]
        lines.append(f"| {s} | {'GREEN' if vs.green else 'FALLBACK'} | "
                     f"{'yes' if vs.majority else 'no'} | "
                     f"{'yes' if vs.hardest_recovers else 'NO'} | {vs.sweep_mean_R:.2f} |")
    lines += [
        "",
        f"### Recovered fraction R and mechanism main effect, by ρ_test ({PRIMARY_SCORE})",
        "| ρ_test | gap feat+split | gap cpt+split | gap feat+Mond | R | repr.CI≠0 | recovers |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in v.per_rho:
        if not r.shifted:
            continue
        rho_lbl = f"**{r.rho_test:.2f}** (hardest)" if r.is_hardest else f"{r.rho_test:.2f}"
        lines.append(f"| {rho_lbl} | {r.gap_feat_split['mean']:.3f} | "
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
    ap.add_argument("--diagnostic-no-verdict", action="store_true",
                    help="v4b: one-off measurement at the in-domain head — BYPASS the §4 gate and "
                         "emit NO verdict (the gate stays in force for verdict runs)")
    ap.add_argument("--out", default="results/unified")
    args = ap.parse_args()

    cfg = {}
    if os.path.exists(args.config):
        from config_util import load_config
        cfg = load_config(args.config)

    mode = "smoke" if args.smoke else "real"
    diagnostic = args.diagnostic_no_verdict

    # §4/§0b HARD HALT (verdict runs only): if the head fails the gate, write BLOCKERS_v4, no verdict.
    try:
        payload = run(cfg, mode=mode, n_seeds=args.seeds, concept_source=args.concept_source,
                      diagnostic=diagnostic)
    except FeatureHeadGateError as e:
        write_blockers_v4(e, args)
        print(f"[u2x2] §4 IN-DOMAIN GATE FAILED: {e}")
        print("[u2x2] HALTED per §0b — wrote BLOCKERS_v4.md; NO 2×2, NO verdict, E2 skipped.")
        sys.exit(2)

    df = pd.DataFrame(payload["records"])
    os.makedirs(args.out, exist_ok=True)
    write_csv(os.path.join(args.out, "unified_2x2.csv"), payload["records"])

    if diagnostic:
        # v4b: measurements only — NO verdict, NO combined decision, NO efficiency-claim verdict.
        fig_paths = make_figures(df, "results/figures", payload, prefix="u2x2_diag")
        write_diagnostic_report("RESULTS_v4b_diagnostic.md", df, payload, fig_paths)
        write_diagnostic_report(os.path.join(args.out, "DIAGNOSTIC_REPORT.md"), df, payload, fig_paths)
        s = payload["diagnostic_summary"]
        gt = payload.get("feat_top1_indomain_typical") or payload["feat_top1"]
        print(f"[u2x2 DIAGNOSTIC] mode={mode} concept={args.concept_source} "
              f"seeds={len(payload['seeds'])} -> {args.out} (NO VERDICT)")
        print(f"[u2x2 DIAGNOSTIC] in-domain typical top-1={gt:.3f} (gate BYPASSED, one-off) | "
              f"feat+split @ρ={s['rho_test']:.2f}: worst_cov={s['feat_split_worst_cov'][0]:.3f} "
              f"gap={s['feat_split_gap'][0]:.3f} set_size={s['feat_split_set_size'][0]:.2f}"
              f"/{payload['n_classes']}")
        print(f"[u2x2 DIAGNOSTIC] BRANCH: {s['branch']} — {s['rationale']}")
        return

    eff = efficiency_summary(df, payload)
    fig_paths = make_figures(df, "results/figures", payload)
    write_json(os.path.join(args.out, "unified_results.json"), payload, eff)
    write_report(os.path.join(args.out, "UNIFIED_REPORT.md"), df, payload, eff, fig_paths)

    comb = payload["combined"]
    print(f"[u2x2] mode={mode} concept={args.concept_source} seeds={len(payload['seeds'])} -> {args.out}")
    print(f"[u2x2] §4 gate: in-domain typical top-1="
          f"{payload.get('feat_top1_indomain_typical') or payload['feat_top1']:.3f} >= {GATE_MIN_TOP1} "
          f"(PASSED) | heads: feature(pool)={payload['feat_top1']:.3f} concept={payload['cpt_top1']:.3f}")
    for s in SCORE_FNS:
        vs = payload["verdicts"][s]
        print(f"[u2x2]   {s}: {'GREEN' if vs.green else 'FALLBACK'} "
              f"(majority={vs.majority}, ρ=0.5 recovers={vs.hardest_recovers}, R={vs.sweep_mean_R:.2f})")
    print(f"[u2x2] HEADLINE (≥{comb['min_required']}/3 scores): {comb['label']} — {comb['rationale']}")
    if mode == "smoke":
        print("[u2x2] SMOKE OK — hardened pipeline validated on synthetic. Real numbers: Colab/GPU.")


if __name__ == "__main__":
    main()
