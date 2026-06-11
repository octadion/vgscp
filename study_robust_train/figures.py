"""Figures for RESULTS_study.md (spec §8 deliverable 3). matplotlib Agg, no display needed.

  - burden vs training method (accuracy-matched Delta, per score)         -> burden_<key>.png
  - worst-group coverage & mean set size vs rho (the shift curves)        -> shift_<key>.png
  - accuracy-vs-conformal-burden ranking scatter (the H2 plot)            -> h2_ranking_<key>.png
"""
from __future__ import annotations

import os

import numpy as np

from .accuracy_matching import matched_divergence
from .verdicts import SCORES


def _agg(records, method, key, field, rho, score):
    xs = [r[field] for r in records if (r["backbone"], r["dataset"]) == key
          and r["method"] == method and r["rho_test"] == rho and r["score"] == score]
    return float(np.mean(xs)) if xs else float("nan")


def make_figures(out: dict, outdir: str) -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(outdir, exist_ok=True)
    records = out["records"]
    written = []

    for key, v in out["verdicts"].items():
        if "note" in v:
            continue
        bb, ds = key
        methods = v["h2"]["ranking_by_accuracy"]
        robust = [m for m in methods if m != "erm"]

        # 1) accuracy-matched burden Delta per method (APS), with CI
        fig, ax = plt.subplots(figsize=(7, 4))
        names, deltas, errs = [], [], []
        for m in robust:
            res = matched_divergence(records, m, "APS", 0.95)
            if res.get("matched"):
                names.append(m); deltas.append(res["delta_matched"])
                errs.append([res["delta_matched"] - res["ci"][0], res["ci"][1] - res["delta_matched"]])
        if names:
            err = np.array(errs).T
            ax.bar(names, deltas, yerr=err, capsize=4, color="#4C72B0")
            ax.axhline(0, color="k", lw=0.8)
            ax.set_ylabel("accuracy-matched Δ divergence (APS)\n(>0 = robust lowers burden)")
            ax.set_title(f"H1 accuracy-matched burden — {bb}/{ds} @ ρ=0.95")
            plt.xticks(rotation=20); plt.tight_layout()
            p = os.path.join(outdir, f"burden_{bb}_{ds}.png"); fig.savefig(p, dpi=120); written.append(p)
        plt.close(fig)

        # 2) coverage stability (NOT the H3 criterion) + the two burden channels vs rho
        rhos = sorted({r["rho_test"] for r in records}, reverse=True)
        fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(16, 4.2))
        for m in methods:
            cov = [_agg(records, m, key, "worst_group_cov", rho, "APS") for rho in rhos]
            sz = [_agg(records, m, key, "mean_set_size", rho, "APS") for rho in rhos]
            disp = [_agg(records, m, key, "set_size_disparity", rho, "APS") for rho in rhos]
            lw = 2.4 if m == "erm" else 1.4
            a1.plot(rhos, cov, marker="o", label=m, lw=lw)
            a2.plot(rhos, sz, marker="o", label=m, lw=lw)
            a3.plot(rhos, disp, marker="o", label=m, lw=lw)
        a1.axhline(0.9, color="k", ls="--", lw=0.8, label="target 1-α")
        for a in (a1, a2, a3):
            a.set_xlabel("ρ_test"); a.invert_xaxis()
        a1.set_ylabel("worst-group coverage"); a2.set_ylabel("mean set size")
        a3.set_ylabel("set-size disparity (max−min group)")
        a1.set_title("coverage stability (reported, not H3 criterion)")
        a2.set_title("set size vs shift")
        a3.set_title("BURDEN RELOCATION: set-size disparity vs shift")
        a1.legend(fontsize=7); plt.tight_layout()
        p = os.path.join(outdir, f"shift_{bb}_{ds}.png"); fig.savefig(p, dpi=120); written.append(p)
        plt.close(fig)

        # 3) accuracy vs burden ranking scatter (H2) with burden CIs (Task B)
        h2 = v["h2"]
        fig, ax = plt.subplots(figsize=(6.2, 5))
        for m in methods:
            x = h2["worst_group_acc"][m]; y = h2["burden"][m]; ci = h2["burden_ci"][m]
            ax.errorbar(x, y, yerr=[[y - ci[0]], [ci[1] - y]], fmt="o", ms=7, capsize=4)
            ax.annotate(m, (x, y), fontsize=8, xytext=(5, 4), textcoords="offset points")
        ax.set_xlabel("worst-group accuracy (higher better)")
        ax.set_ylabel(f"conformal burden: {h2['burden_key']} (lower better) ±95% CI")
        tag = "[INVERSION-REAL]" if h2["inversion_real"] else (
              "[inversion (CIs overlap → noise)]" if h2["inversion_point"] else "[no inversion]")
        ax.set_title(f"H2 accuracy vs burden — {bb}/{ds}  {tag}")
        plt.tight_layout()
        p = os.path.join(outdir, f"h2_ranking_{bb}_{ds}.png"); fig.savefig(p, dpi=120); written.append(p)
        plt.close(fig)

    return written
