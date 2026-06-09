"""E1 — CUB-200 multiclass coverage--efficiency frontier (PRIORITY positive anchor).

Builds the multiclass pipeline (the new infrastructure) and, at EACH test correlation strength,
measures worst-group coverage vs mean set size for the three methods the spec names:
  (a) feature-space score + Mondrian      [feat+Mondrian]
  (b) concept-space score + pooled split  [cpt+split]
  (c) concept-space score + Mondrian      [cpt+Mondrian]
across the APS / RAPS / THR score functions, calibrating at rho_cal and sweeping test rho, with
>=10 random calibration/test splits (seeds) reported mean +/- std. The construction, group
definition, and rho sweep follow experiments/cub200_frontier.py (spec §2b) EXACTLY.

Pre-committed claim + kill-switch live in eval/e1_verdict.py (committed before any numbers).

    python -m scripts.run_cub200_frontier --smoke              # CPU pipeline self-test (synthetic)
    python -m scripts.run_cub200_frontier --config configs/cub200_frontier.yaml   # real (Colab/GPU)

Outputs (under results/e1/ + results/figures/): tidy CSV (test_corr, score, scheme, seed ->
worst_cov, mean_set_size, marg_cov, cov_gap), per-score-function frontier figures, a results JSON
(incl. both heads' top-1 accuracy), and E1_REPORT.md.
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

# Windows consoles default to cp1252; our verdict rationale uses Δ/α/ρ. Make stdout/stderr UTF-8 so
# console prints never crash (report files are already written with encoding="utf-8").
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from conformal import group_robust
from conformal import scores as cscores
from conformal.split_conformal import build_sets, conformal_quantile, set_sizes
from eval.e1_verdict import (CPT_MOND, CPT_SPLIT, FEAT_MOND, PRIMARY_SCORE,
                             e1_verdict)
from experiments import cub200_frontier as cf

SCORE_FNS = ("APS", "RAPS", "THR")
DEFAULT_RHO_TEST = [0.95, 0.90, 0.80, 0.70, 0.60, 0.50]
DEFAULT_RHO_CAL = 0.95
SCHEMES = (FEAT_MOND, CPT_SPLIT, CPT_MOND)


# ======================================================================================
# Score precompute (once per (seed, score) on the whole pool; resampling indexes rows)
# ======================================================================================
def _score_arrays(pop, score, seed):
    """(N,C) feature & concept nonconformity arrays for one score fn. APS/RAPS share a fixed u."""
    n = pop["feat_probs"].shape[0]
    u = cscores.draw_randomization(n, seed=seed + 7) if score in ("APS", "RAPS") else None
    feat = cscores.scores_all(score, pop["feat_probs"], u=u)
    cpt = cscores.scores_all(score, pop["cpt_probs"], u=u)
    return feat, cpt


def _true(score_all, y):
    return score_all[np.arange(len(y)), y]


# ======================================================================================
# Metrics on a (resampled) test set -- worst-group coverage = coverage on the ATYPICAL group
# ======================================================================================
def _method_metrics(member, species, typ):
    cov = member[np.arange(len(species)), species].astype(int)
    sizes = set_sizes(member)
    per_cov = {int(g): float(cov[typ == g].mean()) for g in np.unique(typ)}
    atyp = per_cov.get(cf.ATYPICAL, float("nan"))
    cov_gap = (max(per_cov.values()) - min(per_cov.values())) if len(per_cov) > 1 else 0.0
    return {
        "worst_cov": atyp,                       # spec: worst-group coverage = atypical coverage
        "mean_set_size": float(sizes.mean()),
        "marg_cov": float(cov.mean()),
        "cov_gap": float(cov_gap),
        "per_group_cov": per_cov,
    }


# ======================================================================================
# One seed: calibrate at rho_cal, sweep test rho, all schemes x scores
# ======================================================================================
def run_seed(pop, seed, rho_cal, rho_test_grid, n_cal, n_test, frac_cal, alpha):
    species = pop["species"]
    typ = pop["typicality"]
    n_classes = pop["n_classes"]
    cal_pool, test_pool = cf.split_pool(len(species), frac_cal, seed)

    # fixed per-species draw counts (held across rho so the 200-way class balance is constant)
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
        # calibration thresholds
        y_cal, t_cal = species[cal_idx], typ[cal_idx]
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
            diag["test_rho_realized"][round(rho, 4)] = ts.rho_realized

            membership = {
                FEAT_MOND: group_robust.mondrian_build_sets(feat_t, t_t, mq_feat),
                CPT_SPLIT: build_sets(cpt_t, q_cpt),
                CPT_MOND: group_robust.mondrian_build_sets(cpt_t, t_t, mq_cpt),
            }
            for scheme in SCHEMES:
                m = _method_metrics(membership[scheme], y_t, t_t)
                records.append({"test_corr": float(rho), "score": score, "scheme": scheme,
                                "seed": int(seed), "worst_cov": m["worst_cov"],
                                "mean_set_size": m["mean_set_size"], "marg_cov": m["marg_cov"],
                                "cov_gap": m["cov_gap"]})
    return records, diag


# ======================================================================================
# Orchestration (population built ONCE; seeds = random cal/test splits, per spec §2)
# ======================================================================================
def run(cfg, mode, n_seeds, seeds=None):
    rho_cal = float(cfg.get("shift", {}).get("rho_cal", DEFAULT_RHO_CAL))
    rho_test_grid = list(cfg.get("shift", {}).get("rho_test", DEFAULT_RHO_TEST))
    frac_cal = float(cfg.get("shift", {}).get("frac_cal", 0.5))
    alpha = float(cfg.get("alpha", 0.1))
    seeds = seeds if seeds is not None else list(range(n_seeds))

    if mode == "smoke":
        sc = cf.SmokeConfig(
            n=int(cfg.get("smoke", {}).get("n", 9000)),
            n_classes=int(cfg.get("smoke", {}).get("n_classes", 40)),
            seed=int(cfg.get("smoke", {}).get("pop_seed", 12345)))
        pop = cf.make_smoke_population(sc)
        n_cal = int(cfg.get("smoke", {}).get("n_cal", 2000))
        n_test = int(cfg.get("smoke", {}).get("n_test", 2000))
    else:
        pop = cf.load_real_population(cfg, seed=int(cfg.get("pop_seed", 0)))
        n_cal = int(cfg.get("shift", {}).get("n_cal", 3000))
        n_test = int(cfg.get("shift", {}).get("n_test", 3000))

    all_records, diags = [], {}
    for s in seeds:
        recs, diag = run_seed(pop, s, rho_cal, rho_test_grid, n_cal, n_test, frac_cal, alpha)
        all_records.extend(recs)
        diags[s] = diag

    verdicts = {sc_: e1_verdict(all_records, rho_cal=rho_cal, alpha=alpha, score=sc_)
                for sc_ in SCORE_FNS}
    return {"records": all_records, "verdicts": verdicts, "diags": diags, "mode": mode,
            "seeds": seeds, "rho_cal": rho_cal, "rho_test_grid": rho_test_grid, "alpha": alpha,
            "n_cal": n_cal, "n_test": n_test, "n_classes": pop["n_classes"],
            "feat_top1": pop.get("feat_top1"), "cpt_top1": pop.get("cpt_top1")}


# ======================================================================================
# Aggregation / serialization
# ======================================================================================
def _agg_table(df, score):
    """Mean +/- std over seeds for each (scheme, rho) at one score; tidy DataFrame for the report."""
    sub = df[df["score"] == score]
    g = sub.groupby(["scheme", "test_corr"]).agg(
        worst_cov_m=("worst_cov", "mean"), worst_cov_s=("worst_cov", "std"),
        size_m=("mean_set_size", "mean"), size_s=("mean_set_size", "std"),
        marg_m=("marg_cov", "mean"), gap_m=("cov_gap", "mean"), gap_s=("cov_gap", "std"),
        n=("seed", "count")).reset_index()
    return g


def write_csv(path, records):
    pd.DataFrame(records).sort_values(["score", "scheme", "test_corr", "seed"]).to_csv(
        path, index=False)


def make_figures(df, fig_dir, payload):
    """Per-score-function frontier: worst-group coverage (y) vs mean set size (x), one line/scheme."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"[e1] matplotlib unavailable ({e}); skipping figures")
        return []
    os.makedirs(fig_dir, exist_ok=True)
    target = 1.0 - payload["alpha"]
    colors = {FEAT_MOND: "#d62728", CPT_SPLIT: "#1f77b4", CPT_MOND: "#2ca02c"}
    markers = {FEAT_MOND: "s", CPT_SPLIT: "^", CPT_MOND: "o"}
    paths = []
    for score in SCORE_FNS:
        g = _agg_table(df, score)
        fig, ax = plt.subplots(figsize=(6.2, 4.6))
        for scheme in SCHEMES:
            s = g[g["scheme"] == scheme].sort_values("test_corr")
            if s.empty:
                continue
            ax.plot(s["size_m"], s["worst_cov_m"], marker=markers[scheme], color=colors[scheme],
                    label=scheme, lw=1.6, ms=6)
            for _, row in s.iterrows():
                ax.annotate(f"{row['test_corr']:.2f}", (row["size_m"], row["worst_cov_m"]),
                            fontsize=6, alpha=0.6, xytext=(3, 3), textcoords="offset points")
        ax.axhline(target, ls="--", color="gray", lw=1, label=f"target 1-α={target:g}")
        ax.set_xlabel("mean set size  (efficiency →)")
        ax.set_ylabel("worst-group (atypical) coverage")
        ax.set_title(f"E1 frontier — {score} (α={payload['alpha']:g}, "
                     f"ρ_cal={payload['rho_cal']:g}, points labeled by ρ_test)")
        ax.legend(fontsize=8, loc="best")
        fig.tight_layout()
        for ext in ("pdf", "png"):
            p = os.path.join(fig_dir, f"e1_frontier_{score}.{ext}")
            fig.savefig(p, dpi=150)
            paths.append(p)
        plt.close(fig)
    return paths


def _verdict_json(v):
    return {"label": v.label, "green": v.green, "score": v.score, "alpha": v.alpha,
            "rho_cal": v.rho_cal, "n_shifted": v.n_shifted, "n_relocated": v.n_relocated,
            "sweep_mean_d_cov": v.sweep_mean_d_cov, "sweep_mean_d_size": v.sweep_mean_d_size,
            "rationale": v.rationale,
            "per_rho": [{"rho_test": r.rho_test, "shifted": r.shifted, "relocates": r.relocates,
                         "size_matched": r.size_matched, "cov_not_worse": r.cov_not_worse,
                         "better_cov": r.better_cov, "better_size": r.better_size,
                         "d_cov": r.d_cov, "d_size": r.d_size,
                         "feat_worst_cov": r.feat_worst_cov, "cpt_worst_cov": r.cpt_worst_cov,
                         "feat_gap": r.feat_gap, "cpt_gap": r.cpt_gap} for r in v.per_rho]}


def write_json(path, payload):
    out = {k: payload[k] for k in ("mode", "seeds", "rho_cal", "rho_test_grid", "alpha",
                                   "n_cal", "n_test", "n_classes", "feat_top1", "cpt_top1")}
    out["verdicts"] = {s: _verdict_json(v) for s, v in payload["verdicts"].items()}
    out["diags"] = {str(k): v for k, v in payload["diags"].items()}
    with open(path, "w") as f:
        json.dump(out, f, indent=2)


# ======================================================================================
# Report
# ======================================================================================
def _pm(m, s):
    return f"{m:.3f}±{s:.3f}" if np.isfinite(m) else "--"


def write_report(path, df, payload, fig_paths):
    v_prim = payload["verdicts"][PRIMARY_SCORE]
    mode = payload["mode"]
    lines = [
        "# E1 — CUB-200 multiclass coverage–efficiency frontier",
        "",
        f"**Date:** {datetime.now(timezone.utc).date()} · **Run mode:** {mode} · "
        f"**Seeds:** {len(payload['seeds'])} (random cal/test splits) · "
        f"**α={payload['alpha']:g}** · **ρ_cal={payload['rho_cal']:g}** · "
        f"**ρ_test sweep:** {payload['rho_test_grid']} · **classes:** {payload['n_classes']}",
        "",
    ]
    if mode == "smoke":
        lines += [
            "> ⚠️ **SMOKE (synthetic) run** — a fabricated CUB-200-like multiclass population "
            "(spurious-sensitive feature head + shortcut-invariant concept head with a small "
            "residual). Validates the full construct→resample→calibrate→score→frontier→verdict "
            "pipeline end-to-end on CPU. The real frontier numbers come from the Colab/GPU "
            "cached-feature run (`experiments/cub200_frontier.load_real_population`).",
            "",
        ]
    lines += [
        "## Pre-committed claim (eval/e1_verdict.py, fixed before any numbers)",
        "The concept-score curve **relocates** the frontier toward better worst-group (atypical) "
        "coverage at matched mean set size. **No claim of strict dominance to the ideal corner** "
        "(residual non-invariance forbids reaching absolute worst-group coverage 1−α). Kill-switch: "
        "if no consistent relocation, fall back to *\"mechanism doesn't help; representation gives "
        "only relative-gap robustness, not absolute coverage.\"*",
        "",
        "## Classifier heads (top-1 accuracy)",
        f"- feature-space head (CLIP features → {payload['n_classes']} species): "
        f"**{payload['feat_top1']:.3f}**" if payload.get("feat_top1") is not None else
        "- feature-space head: (real run)",
        f"- concept-space head (CUB attributes → {payload['n_classes']} species): "
        f"**{payload['cpt_top1']:.3f}**" if payload.get("cpt_top1") is not None else
        "- concept-space head: (real run)",
        "",
        f"## Verdict (primary score {PRIMARY_SCORE}): **{v_prim.label}**",
        f"{v_prim.rationale}",
        "",
    ]
    # per-score verdict summary (Mondrian equalizes worst-group COVERAGE; the win is in SET SIZE)
    lines += ["| score | verdict | relocated/shifted ρ | sweep Δ worst-cov | sweep Δ set-size "
              "(cpt−feat, Mondrian) |", "|---|---|---|---|---|"]
    for s in SCORE_FNS:
        v = payload["verdicts"][s]
        lines.append(f"| {s} | {v.label} | {v.n_relocated}/{v.n_shifted} | "
                     f"{v.sweep_mean_d_cov:+.3f} | {v.sweep_mean_d_size:+.2f} |")
    lines.append("")

    # per-score frontier tables
    for score in SCORE_FNS:
        g = _agg_table(df, score)
        lines += [f"### {score} — worst-group coverage / mean set size / coverage gap by ρ_test",
                  "", "| scheme | ρ_test | worst-grp cov | mean set size | marg cov | cov gap |",
                  "|---|---|---|---|---|---|"]
        for scheme in SCHEMES:
            sub = g[g["scheme"] == scheme].sort_values("test_corr", ascending=False)
            for _, r in sub.iterrows():
                lines.append(f"| {scheme} | {r['test_corr']:.2f} | "
                             f"{_pm(r['worst_cov_m'], r['worst_cov_s'])} | "
                             f"{_pm(r['size_m'], r['size_s'])} | {r['marg_m']:.3f} | "
                             f"{_pm(r['gap_m'], r['gap_s'])} |")
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
    ap.add_argument("--seeds", type=int, default=10, help="number of random cal/test splits")
    ap.add_argument("--out", default="results/e1")
    args = ap.parse_args()

    cfg = {}
    if os.path.exists(args.config):
        from config_util import load_config
        cfg = load_config(args.config)

    mode = "smoke" if args.smoke else "real"
    payload = run(cfg, mode=mode, n_seeds=args.seeds)

    os.makedirs(args.out, exist_ok=True)
    df = pd.DataFrame(payload["records"])
    csv_path = os.path.join(args.out, "e1_frontier.csv")
    write_csv(csv_path, payload["records"])
    write_json(os.path.join(args.out, "e1_results.json"), payload)
    fig_paths = make_figures(df, "results/figures", payload)
    write_report(os.path.join(args.out, "E1_REPORT.md"), df, payload, fig_paths)
    write_report("E1_REPORT.md", df, payload, fig_paths)

    v = payload["verdicts"][PRIMARY_SCORE]
    print(f"[e1] mode={mode} seeds={len(payload['seeds'])} -> {args.out}")
    print(f"[e1] heads top-1: feature={payload['feat_top1']}, concept={payload['cpt_top1']}")
    print(f"[e1] VERDICT ({PRIMARY_SCORE}): {v.label}  {v.rationale}")
    print(f"[e1] CSV={csv_path}  figures={'yes' if fig_paths else 'no'}")
    if mode == "smoke":
        print("[e1] SMOKE OK — pipeline validated on synthetic data. Real numbers: Colab/GPU run.")


if __name__ == "__main__":
    main()
