"""KS-0 -- shared prerequisite: does the predicted-vs-true attribute gap even exist?

Both candidate CP directions are motivated by a gap between conditioning on the PREDICTED attribute
A_hat and the TRUE attribute A. If Mondrian-on-A_hat already covers the true groups, both
directions are dead. We calibrate Mondrian thresholds two ways -- on A_true (oracle) and on A_hat
(naive) -- and measure TRUE-attribute-conditional coverage + mean set size for both, over >=10
random calibration/test splits, at alpha in {0.10, 0.05}.

PRE-COMMITTED GO/NO-GO: proceed to KS-1/KS-2 only if Mondrian-on-A_hat under-covers the worst true
group by >= 2-3 coverage points vs nominal (and vs oracle), robustly across splits at both alpha.
If naive-on-A_hat already covers the true groups -> NO-GO, both directions dead.

    python -m ks_conformal.ks0_problem_exists                # synthetic testbed (runs here)
    python -m ks_conformal.ks0_problem_exists --real         # CUB-200/CLIP (BLOCKED: see RESULTS.md)
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from conformal import group_robust as gr
from conformal.split_conformal import conformal_quantile, build_sets
from ks_conformal import common_utils as cu
from ks_conformal import plotting

OUT = os.path.join("results", "ks_conformal")
FLIP_RATE = {0: 0.10, 1: 0.35}     # P(A_hat != a | A=a): probe errs more on the scarce minority
NOISE_BETA = 6.0                   # differential noise (realistic); the gap exists for beta=0 too


def run(real: bool = False, n_splits: int = cu.DEFAULT_N_SPLITS, score: str = "APS"):
    if real:
        cu.load_real_population({}, seed=0)        # raises BLOCKED with the explanation
    cfg = cu.TestbedConfig(seed=0, score=score)
    pop = cu.make_population(cfg)
    A = pop["A_true"]
    s_true = pop["s_true"][score]
    sa = pop["scores_all"][score]
    Ahat = cu.make_ahat(A, s_true, FLIP_RATE, beta=NOISE_BETA, seed=101)
    print(f"[ks0] realized confusion (counts a_hat x a):")
    _, _, counts = cu.confusion_matrix_MW(A, Ahat)
    print(counts.astype(int))

    out = {}
    for alpha in cu.ALPHAS:
        rows = {m: {"worst_cov": [], "worst_gap": [], "marg_cov": [], "overall_size": [],
                    "per_cov": {0: [], 1: []}, "per_size": {0: [], 1: []}}
                for m in ("oracle", "naive_Ahat", "marginal")}
        for sp in cu.iter_splits(len(A), n_splits, (0.0, 0.5, 0.5), seed0=0):
            cal, test = sp["cal"], sp["test"]
            # oracle: Mondrian on A_true
            qO = gr.mondrian_quantiles(s_true[cal], A[cal], alpha)
            memO = cu.sets_from_per_true_group_q(sa[test], A[test], qO)
            # naive: Mondrian on A_hat (calibrate per A_hat group, apply by test A_hat)
            qN = gr.mondrian_quantiles(s_true[cal], Ahat[cal], alpha)
            memN = gr.mondrian_build_sets(sa[test], Ahat[test], qN)
            # marginal: a single global quantile (reference)
            qM = conformal_quantile(s_true[cal], alpha)
            memM = build_sets(sa[test], qM)
            for name, mem in (("oracle", memO), ("naive_Ahat", memN), ("marginal", memM)):
                rep = cu.coverage_report(mem, pop["y_true"][test], A[test], alpha)
                rows[name]["worst_cov"].append(rep.worst_cov)
                rows[name]["worst_gap"].append(rep.worst_gap)
                rows[name]["marg_cov"].append(rep.marginal_cov)
                rows[name]["overall_size"].append(rep.overall_size)
                for g in (0, 1):
                    rows[name]["per_cov"][g].append(rep.per_group_cov[g])
                    rows[name]["per_size"][g].append(rep.per_group_size[g])
        out[alpha] = {m: {
            "worst_cov": cu.agg(rows[m]["worst_cov"]),
            "worst_gap": cu.agg(rows[m]["worst_gap"]),
            "marg_cov": cu.agg(rows[m]["marg_cov"]),
            "overall_size": cu.agg(rows[m]["overall_size"]),
            "per_cov": {g: cu.agg(rows[m]["per_cov"][g]) for g in (0, 1)},
            "per_size": {g: cu.agg(rows[m]["per_size"][g]) for g in (0, 1)},
        } for m in rows}
    verdict = _verdict(out)
    return {"out": out, "verdict": verdict, "counts": counts.tolist(),
            "flip_rate": FLIP_RATE, "beta": NOISE_BETA, "score": score, "n_splits": n_splits,
            "n_classes": cfg.n_classes, "synthetic": True}


def _verdict(out: dict) -> dict:
    """GO iff naive-on-A_hat under-covers the worst TRUE group by >= 2 pts vs nominal AND vs oracle,
    robustly (CI upper bound of the under-coverage stays negative) at BOTH alpha."""
    ok = True
    detail = []
    for alpha in cu.ALPHAS:
        naive = out[alpha]["naive_Ahat"]["worst_gap"]       # worst_cov - (1-alpha)
        oracle = out[alpha]["oracle"]["worst_cov"]["mean"]
        naive_wc = out[alpha]["naive_Ahat"]["worst_cov"]["mean"]
        gap_vs_nominal = naive["mean"]                       # negative = under-coverage
        ci_excludes0 = naive["hi"] < 0
        below_oracle = (oracle - naive_wc) >= 0.02
        cond = (gap_vs_nominal <= -0.02) and ci_excludes0 and below_oracle
        ok = ok and cond
        detail.append(f"alpha={alpha}: naive worst-true-grp cov={naive_wc:.3f} "
                      f"(gap {gap_vs_nominal:+.3f}, 95%CI hi {naive['hi']:+.3f}), "
                      f"oracle={oracle:.3f}, below-oracle-by={oracle-naive_wc:+.3f} -> "
                      f"{'meets' if cond else 'FAILS'} GO bar")
    label = "GO (problem exists; proceed to KS-1/KS-2)" if ok else \
            "NO-GO (naive-on-A_hat covers true groups; both directions dead)"
    return {"go": ok, "label": label, "detail": detail}


def write_report(payload: dict, path: str):
    v = payload["verdict"]
    L = ["# KS-0 -- Does the predicted-vs-true attribute gap exist?", "",
         f"**Verdict: {v['label']}**", "",
         "Synthetic multiclass testbed (C={} classes, minority A=1 is harder). Â is a noisy probe "
         "with per-group flip rates {} and differential noise β={} (the gap is present for β=0 too "
         "-- it is a *labelling* gap, not a noise-correlation artefact). {} random cal/test splits; "
         "evaluation target is TRUE-attribute-conditional coverage."
         .format(payload["n_classes"], payload["flip_rate"], payload["beta"], payload["n_splits"]),
         "", f"Realized confusion counts (rows Â, cols A_true): {payload['counts']}", ""]
    for d in v["detail"]:
        L.append(f"- {d}")
    for alpha in cu.ALPHAS:
        L += ["", f"## alpha = {alpha} (nominal coverage {1-alpha:.2f})", "",
              "| scheme | cov A=0 | cov A=1 (minority) | worst-grp cov | worst gap | size A=0 | size A=1 | overall size |",
              "|---|---|---|---|---|---|---|---|"]
        for m in ("oracle", "naive_Ahat", "marginal"):
            r = payload["out"][alpha][m]
            L.append("| {} | {} | {} | {} | {} | {} | {} | {} |".format(
                m, cu.fmt(r["per_cov"][0]), cu.fmt(r["per_cov"][1]), cu.fmt(r["worst_cov"]),
                cu.fmt(r["worst_gap"]), cu.fmt(r["per_size"][0]), cu.fmt(r["per_size"][1]),
                cu.fmt(r["overall_size"])))
    L += ["", "> Numbers are mean±std over splits. Synthetic testbed; the real CUB-200/CLIP number "
          "is BLOCKED in this environment (no torch/open_clip/datasets). See RESULTS.md."]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


def make_plots(payload: dict):
    for alpha in cu.ALPHAS:
        pg = {}
        for m in ("oracle", "naive_Ahat", "marginal"):
            r = payload["out"][alpha][m]
            pg[m] = {g: (r["per_cov"][g]["mean"], r["per_cov"][g]["std"]) for g in (0, 1)}
        plotting.plot_group_coverage(
            pg, alpha, os.path.join(OUT, f"ks0_coverage_a{int(alpha*100)}.pdf"),
            f"KS-0 true-conditional coverage (α={alpha})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--splits", type=int, default=cu.DEFAULT_N_SPLITS)
    ap.add_argument("--score", default="APS")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    payload = run(real=args.real, n_splits=args.splits, score=args.score)
    write_report(payload, os.path.join(OUT, "KS0_REPORT.md"))
    make_plots(payload)
    with open(os.path.join(OUT, "ks0_results.json"), "w") as f:
        json.dump(payload, f, indent=2, default=float)
    print("\n[ks0] VERDICT:", payload["verdict"]["label"])
    for d in payload["verdict"]["detail"]:
        print("   ", d)
    print("[ks0] wrote", os.path.join(OUT, "KS0_REPORT.md"))


if __name__ == "__main__":
    main()
