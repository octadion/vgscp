"""E3 — Correlation-strength shift, CLEAN MULTI-SEED regeneration.

Calibrate at high correlation (rho_cal), evaluate across the correlation sweep, for concept-space
vs feature-space scores, under BOTH split and Mondrian, reporting:
  * absolute WORST-GROUP coverage  (min over the 4 Waterbirds groups), and
  * the coverage GAP  (max - min group coverage; the shift-robustness axis),
as mean ± std over >=10 random seeds. This is NOT the single-seed go/no-go gate
(scripts/run_shiftcp_derisk.py) -- it REUSES that script's pool / score / calibration / per-rho
method machinery VERBATIM and only adds the multi-seed loop + aggregation the spec asks for.

Pre-committed expectation (report whatever the clean multi-seed run produces, regardless of whether
it matches the prior single-seed kill-check):
  * concept score markedly MORE shift-robust in the GAP than the f-score
    (prior single-seed: max-min gap 0.598 -> 0.109), BUT
  * does NOT hold absolute worst-group coverage (prior single-seed ~0.68-0.71 vs 0.90 target), and
  * Mondrian does not repair the absolute coverage shortfall.

    python -m scripts.run_e3_shift_multiseed --smoke --seeds 10        # CPU synthetic self-test
    python -m scripts.run_e3_shift_multiseed --config configs/shiftcp_derisk.yaml --seeds 10  # real
"""
from __future__ import annotations

try:  # pragma: no cover
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

from config_util import load_config
from experiments import shift_resampler as resampler
# Reuse the single-seed gate's machinery verbatim (extend, don't rebuild).
from scripts.run_shiftcp_derisk import (_build_methods_for_rho, _calibrate,
                                        _load_smoke, _pool, _precompute_scores)
from eval.shiftcp_verdict import MOND_CPT, MOND_F, STD_CPT, STD_F

# (score, scheme) labelling of the four methods E3 reports.
METHOD_AXES = {STD_F: ("feature", "split"), STD_CPT: ("concept", "split"),
               MOND_F: ("feature", "Mondrian"), MOND_CPT: ("concept", "Mondrian")}
METHODS = (STD_F, STD_CPT, MOND_F, MOND_CPT)
DEFAULT_RHO_TEST = [0.95, 0.90, 0.80, 0.70, 0.60, 0.50, 0.25]
DEFAULT_RHO_CAL = 0.95


def _group_metrics(covered, group):
    """Per-group coverage + worst-group coverage + max-min coverage gap."""
    per = {int(g): float(covered[group == g].mean()) for g in np.unique(group)}
    vals = list(per.values())
    return {"worst_group_cov": float(min(vals)), "cov_gap": float(max(vals) - min(vals)),
            "marg_cov": float(covered.mean()), "per_group_cov": per}


def run_seed(fdata, attrs, n_classes, f_kind, cpt_kind, rho_cal, rho_grid,
             n_cal, n_test, frac_cal, alpha, seed):
    pool = _pool(fdata, attrs)
    train_attrs = attrs["train"].astype(np.float32)
    train_y = np.asarray(fdata["train"]["y_true"]).astype(int)
    f_all, cpt_all, _ = _precompute_scores(pool, train_attrs, train_y, n_classes,
                                           f_kind, cpt_kind, seed)
    cal_pool, test_pool = resampler.split_pool(len(pool["y_true"]), frac_cal, seed)
    cal_rs = resampler.resample_to_rho(pool["group_id"][cal_pool], rho_cal, n_cal, seed=seed)
    cal = _calibrate(cal_pool[cal_rs.idx], pool, f_all, cpt_all, alpha)

    rows = []
    for i, rho in enumerate(rho_grid):
        ts = resampler.resample_to_rho(pool["group_id"][test_pool], rho, n_test, seed=seed + 1 + i)
        test_idx = test_pool[ts.idx]
        methods = _build_methods_for_rho(test_idx, pool, f_all, cpt_all, cal)
        for name in METHODS:
            m = methods[name]
            gm = _group_metrics(m["covered"], m["group"])
            score, scheme = METHOD_AXES[name]
            rows.append({"rho_test": float(rho), "rho_realized": ts.rho_realized,
                         "score": score, "scheme": scheme, "method": name, "seed": int(seed),
                         "worst_group_cov": gm["worst_group_cov"], "cov_gap": gm["cov_gap"],
                         "marg_cov": gm["marg_cov"]})
    return rows


def run(cfg, mode, n_seeds):
    n_classes = int(cfg.get("dataset", {}).get("n_classes", 2))
    f_kind = cfg.get("scores", {}).get("f_kind", "APS")
    cpt_kind = cfg.get("scores", {}).get("cpt_score_kind", "THR")
    rho_cal = float(cfg.get("shift", {}).get("rho_cal", DEFAULT_RHO_CAL))
    rho_grid = list(cfg.get("shift", {}).get("rho_test", DEFAULT_RHO_TEST))
    frac_cal = float(cfg.get("shift", {}).get("frac_cal", 0.5))
    alpha = float(cfg.get("alphas", [0.1])[0])

    # REAL: build the binary f + CUB attributes ONCE on cached CLIP features (the §2b runtime
    # trick); seeds then vary ONLY the cal/test resampling (per spec §2 "≥10 random splits"). SMOKE:
    # regenerate the synthetic shortcut f per seed.
    real_fdata = real_attrs = None
    if mode == "real":
        from experiments.real_data import build_binary_fdata, load_real_bundle
        bundle = load_real_bundle(cfg, seed=int(cfg.get("pop_seed", 0)))
        real_fdata = build_binary_fdata(bundle, cfg, seed=int(cfg.get("pop_seed", 0)),
                                        with_ensemble_mc=False)
        real_attrs = bundle.attrs
        n_cal = int(cfg.get("shift", {}).get("n_cal", 2000))
        n_test = int(cfg.get("shift", {}).get("n_test", 2000))

    rows = []
    for s in range(n_seeds):
        if mode == "smoke":
            fdata, attrs, info, _wg = _load_smoke(s, cfg)
            n_cal = cfg.get("smoke_n_cal", 700)
            n_test = cfg.get("smoke_n_test", 800)
        else:
            fdata, attrs = real_fdata, real_attrs
        rows.extend(run_seed(fdata, attrs, n_classes, f_kind, cpt_kind, rho_cal, rho_grid,
                             n_cal, n_test, frac_cal, alpha, s))
    return {"rows": rows, "alpha": alpha, "rho_cal": rho_cal, "rho_grid": rho_grid,
            "n_seeds": n_seeds, "mode": mode, "f_kind": f_kind, "cpt_kind": cpt_kind}


# -------------------------------------------------------------------- aggregation / report
def _agg(df):
    return df.groupby(["score", "scheme", "rho_test"]).agg(
        wg_m=("worst_group_cov", "mean"), wg_s=("worst_group_cov", "std"),
        gap_m=("cov_gap", "mean"), gap_s=("cov_gap", "std"),
        marg_m=("marg_cov", "mean"), n=("seed", "count")).reset_index()


def write_report(path, df, payload):
    g = _agg(df)
    tgt = 1.0 - payload["alpha"]
    lines = [
        "# E3 — Correlation-strength shift (clean multi-seed regeneration)",
        "",
        f"**Date:** {datetime.now(timezone.utc).date()} · **Run mode:** {payload['mode']} · "
        f"**Seeds:** {payload['n_seeds']} · **α={payload['alpha']:g}** (target {tgt:g}) · "
        f"**ρ_cal={payload['rho_cal']:g}** · **ρ_test:** {payload['rho_grid']} · "
        f"**f-score={payload['f_kind']} / concept-score={payload['cpt_kind']}**",
        "",
    ]
    if payload["mode"] == "smoke":
        lines += ["> ⚠️ **SMOKE (synthetic) run** — fabricated shortcut f + clean CUB-like attrs. "
                  "Validates the multi-seed pipeline on CPU. Real numbers come from the cache-reuse "
                  "run on Colab/GPU.", ""]
    lines += [
        "## Pre-committed expectation (report regardless)",
        "Concept score markedly more shift-robust in the **coverage GAP** (prior single-seed "
        "0.598→0.109) but does **not** hold absolute worst-group coverage (~0.68–0.71 vs target); "
        "Mondrian does not repair the absolute shortfall.",
        "",
        "## Absolute worst-group coverage (mean ± std over seeds)",
        "| score | scheme | " + " | ".join(f"ρ={r:g}" for r in payload["rho_grid"]) + " |",
        "|---|---|" + "---|" * len(payload["rho_grid"]),
    ]
    for score in ("feature", "concept"):
        for scheme in ("split", "Mondrian"):
            cells = []
            for r in payload["rho_grid"]:
                row = g[(g.score == score) & (g.scheme == scheme) & (abs(g.rho_test - r) < 1e-9)]
                cells.append(f"{row.wg_m.iloc[0]:.3f}±{row.wg_s.iloc[0]:.3f}" if len(row) else "--")
            lines.append(f"| {score} | {scheme} | " + " | ".join(cells) + " |")
    lines += ["", "## Coverage gap = max−min group coverage (mean ± std; lower = more shift-robust)",
              "| score | scheme | " + " | ".join(f"ρ={r:g}" for r in payload["rho_grid"]) + " |",
              "|---|---|" + "---|" * len(payload["rho_grid"])]
    for score in ("feature", "concept"):
        for scheme in ("split", "Mondrian"):
            cells = []
            for r in payload["rho_grid"]:
                row = g[(g.score == score) & (g.scheme == scheme) & (abs(g.rho_test - r) < 1e-9)]
                cells.append(f"{row.gap_m.iloc[0]:.3f}±{row.gap_s.iloc[0]:.3f}" if len(row) else "--")
            lines.append(f"| {score} | {scheme} | " + " | ".join(cells) + " |")
    lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/shiftcp_derisk.yaml")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--out", default="results/e3")
    args = ap.parse_args()

    cfg = load_config(args.config) if os.path.exists(args.config) else {}
    mode = "smoke" if args.smoke else "real"
    payload = run(cfg, mode=mode, n_seeds=args.seeds)

    os.makedirs(args.out, exist_ok=True)
    df = pd.DataFrame(payload["rows"])
    df.sort_values(["score", "scheme", "rho_test", "seed"]).to_csv(
        os.path.join(args.out, "e3_shift_metrics.csv"), index=False)
    with open(os.path.join(args.out, "e3_results.json"), "w") as f:
        json.dump({k: payload[k] for k in payload if k != "rows"}, f, indent=2)
    write_report(os.path.join(args.out, "E3_REPORT.md"), df, payload)
    write_report("E3_REPORT.md", df, payload)

    g = _agg(df)
    print(f"[e3] mode={mode} seeds={payload['n_seeds']} -> {args.out}")
    for score in ("feature", "concept"):
        row = g[(g.score == score) & (g.scheme == "split")].sort_values("rho_test")
        if len(row):
            lo, hi = row.iloc[0], row.iloc[-1]
            print(f"[e3] {score:7s} split: gap ρ={hi.rho_test:g}->{lo.rho_test:g}: "
                  f"{hi.gap_m:.3f} -> {lo.gap_m:.3f} | worst-cov {lo.wg_m:.3f} at ρ={lo.rho_test:g}")
    if mode == "smoke":
        print("[e3] SMOKE OK — multi-seed pipeline validated. Real numbers: cache-reuse run on Colab.")


if __name__ == "__main__":
    main()
