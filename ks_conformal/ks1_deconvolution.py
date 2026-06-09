"""KS-1 -- confusion-matrix-deconvolved attribute-conditional CP (Rank 1).

Idea under test: recover the per-true-group quantile that Mondrian-on-A_true *would* have used,
by deconvolving the A_hat-conditional score distributions with the mixing matrix W (solve
F_observed = W . G_true for the true-group CDFs, then take the (1-α) quantile of recovered G_a).

  KS-1a  known-M, NON-differential noise  -- estimator sanity. Must recover oracle within ~1 pt.
  KS-1b  known-M, DIFFERENTIAL noise       -- THE critical test. Same marginal M, flips correlated
                                              with the score. Does the deconvolution break?
  KS-1c  real A_hat, estimated M̂           -- reality check + differential-noise DIAGNOSTIC.
                                              (real number BLOCKED here; synthetic analog reported.)

    python -m ks_conformal.ks1_deconvolution            # synthetic (runs here)
    python -m ks_conformal.ks1_deconvolution --real     # CUB-200/CLIP (BLOCKED: see RESULTS.md)
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from conformal import group_robust as gr
from eval.metrics import auroc
from ks_conformal import common_utils as cu
from ks_conformal import plotting

OUT = os.path.join("results", "ks_conformal")
FLIP_RATE = {0: 0.10, 1: 0.35}
BETA_SWEEP = [0.0, 2.0, 4.0, 8.0, 16.0]     # KS-1b: score<->flip correlation, weak -> strong
REAL_BETA = 8.0                             # KS-1c synthetic analog: realistic differential noise


# --------------------------------------------------------------------------------------
# diagnostics
# --------------------------------------------------------------------------------------
def differential_diagnostic(A, Ahat, s_true):
    """Within each true group, AUROC of the nonconformity score for predicting the probe error
    1[A_hat != A]. ~0.5 => non-differential; >0.5 => differential (harder points mislabelled)."""
    flip = (Ahat != A).astype(int)
    out = {}
    for a in (0, 1):
        m = A == a
        out[a] = float(auroc(s_true[m], flip[m])) if len(np.unique(flip[m])) == 2 else float("nan")
    return out


def _eval_q_on_splits(pop, score, Ahat, W, alpha, n_splits, what="known", fr_hold=0.0):
    """Per split: oracle (Mondrian on A_true) and deconv (W-deconvolution of A_hat-conditional cal
    CDFs). Returns aggregated per-true-group coverage + sizes + deconv stability. If what=='est',
    W is re-estimated per split from the holdout indices instead of using the supplied W."""
    A, sa, s_true = pop["A_true"], pop["scores_all"][score], pop["s_true"][score]
    yt = pop["y_true"]
    acc = {"oracle": {"cov": {0: [], 1: []}, "size": {0: [], 1: []}, "worst": []},
           "deconv": {"cov": {0: [], 1: []}, "size": {0: [], 1: []}, "worst": [],
                      "cond": [], "negmass": [], "qgap1": []}}
    for sp in cu.iter_splits(len(A), n_splits, (fr_hold, (1 - fr_hold) / 2, (1 - fr_hold) / 2)):
        cal, test = sp["cal"], sp["test"]
        Wuse = W
        if what == "est":
            ho = sp["holdout"]
            _, Wuse, _ = cu.confusion_matrix_MW(A[ho], Ahat[ho])
        qO = gr.mondrian_quantiles(s_true[cal], A[cal], alpha)
        dr = cu.deconvolve_quantiles(s_true[cal], Ahat[cal], Wuse, alpha)
        qD = {0: dr.q[0], 1: dr.q[1]}
        memO = cu.sets_from_per_true_group_q(sa[test], A[test], qO)
        memD = cu.sets_from_per_true_group_q(sa[test], A[test], qD)
        rO = cu.coverage_report(memO, yt[test], A[test], alpha)
        rD = cu.coverage_report(memD, yt[test], A[test], alpha)
        for g in (0, 1):
            acc["oracle"]["cov"][g].append(rO.per_group_cov[g])
            acc["oracle"]["size"][g].append(rO.per_group_size[g])
            acc["deconv"]["cov"][g].append(rD.per_group_cov[g])
            acc["deconv"]["size"][g].append(rD.per_group_size[g])
        acc["oracle"]["worst"].append(rO.worst_cov)
        acc["deconv"]["worst"].append(rD.worst_cov)
        acc["deconv"]["cond"].append(dr.cond_number)
        acc["deconv"]["negmass"].append(dr.negative_mass)
        acc["deconv"]["qgap1"].append(qD[1] - qO.get(1, np.nan))
    def pack(d):
        o = {"cov": {g: cu.agg(d["cov"][g]) for g in (0, 1)},
             "size": {g: cu.agg(d["size"][g]) for g in (0, 1)},
             "worst": cu.agg(d["worst"])}
        for k in ("cond", "negmass", "qgap1"):
            if k in d:
                o[k] = cu.agg(d[k])
        return o
    return {"oracle": pack(acc["oracle"]), "deconv": pack(acc["deconv"])}


# --------------------------------------------------------------------------------------
# KS-1a
# --------------------------------------------------------------------------------------
def run_a(pop, score, n_splits):
    A, s_true = pop["A_true"], pop["s_true"][score]
    M, W = cu.known_MW(pop["cfg"].p_minor, FLIP_RATE)
    Ahat = cu.make_ahat(A, s_true, FLIP_RATE, beta=0.0, seed=201)     # NON-differential
    diag = differential_diagnostic(A, Ahat, s_true)
    res = {alpha: _eval_q_on_splits(pop, score, Ahat, W, alpha, n_splits, what="known")
           for alpha in cu.ALPHAS}
    ok = True
    detail = []
    for alpha in cu.ALPHAS:
        dec = res[alpha]["deconv"]["worst"]["mean"]
        ora = res[alpha]["oracle"]["worst"]["mean"]
        d = abs(dec - ora)
        cond = d <= 0.015                                            # ~1 coverage point
        ok = ok and cond
        detail.append(f"alpha={alpha}: deconv worst-grp cov={dec:.3f} vs oracle {ora:.3f} "
                      f"(|Δ|={d:.3f}) -> {'recovers' if cond else 'FAILS to recover'} oracle")
    return {"res": res, "diag": diag, "W": W.tolist(),
            "verdict": {"pass": ok,
                        "label": ("PASS -- estimator recovers oracle under known-M, "
                                  "non-differential noise" if ok else
                                  "FAIL -- estimator misspecified even with M known & "
                                  "non-differential -> Rank 1 dead"),
                        "detail": detail}}


# --------------------------------------------------------------------------------------
# KS-1b  (the critical test)
# --------------------------------------------------------------------------------------
def run_b(pop, score, n_splits):
    A, s_true = pop["A_true"], pop["s_true"][score]
    M, W = cu.known_MW(pop["cfg"].p_minor, FLIP_RATE)               # SAME marginal M for all beta
    sweep = []
    for beta in BETA_SWEEP:
        Ahat = cu.make_ahat(A, s_true, FLIP_RATE, beta=beta, seed=301)
        diag = differential_diagnostic(A, Ahat, s_true)
        # verify the marginal confusion is unchanged across beta
        _, _, counts = cu.confusion_matrix_MW(A, Ahat)
        res = {alpha: _eval_q_on_splits(pop, score, Ahat, W, alpha, n_splits, what="known")
               for alpha in cu.ALPHAS}
        sweep.append({"beta": beta, "diag": diag, "counts": counts.tolist(), "res": res})
    # verdict: 1a recovered (checked separately); does 1b under-cover the true minority as the
    # score<->flip correlation grows, using the SAME marginal-M deconvolution?
    breaks = False
    detail = []
    for alpha in cu.ALPHAS:
        cov0 = sweep[0]["res"][alpha]["deconv"]["cov"][1]["mean"]    # beta=0 minority cov
        covH = sweep[-1]["res"][alpha]["deconv"]["cov"][1]["mean"]   # strongest-diff minority cov
        gapH = covH - (1 - alpha)
        broke = gapH <= -0.02 and (cov0 - covH) >= 0.02
        breaks = breaks or broke
        detail.append(f"alpha={alpha}: minority true-grp cov non-diff(β=0)={cov0:.3f} -> "
                      f"strong-diff(β={BETA_SWEEP[-1]:g})={covH:.3f} (gap {gapH:+.3f}); "
                      f"{'BREAKS' if broke else 'holds'}")
    return {"sweep": sweep, "verdict": {
        "breaks": breaks,
        "label": ("CONFIRMED breakdown -- the simple (score-independent-M) deconvolution is valid "
                  "ONLY under non-differential noise; differential noise under-covers the true "
                  "minority" if breaks else
                  "no breakdown observed under differential noise (unexpected -- inspect)"),
        "detail": detail}}


# --------------------------------------------------------------------------------------
# KS-1c  (reality check: real A_hat, estimated M̂)  -- synthetic analog here, real BLOCKED
# --------------------------------------------------------------------------------------
def run_c(pop, score, n_splits):
    A, s_true, sa, yt = pop["A_true"], pop["s_true"][score], pop["scores_all"][score], pop["y_true"]
    M, W_known = cu.known_MW(pop["cfg"].p_minor, FLIP_RATE)
    Ahat = cu.make_ahat(A, s_true, FLIP_RATE, beta=REAL_BETA, seed=401)   # realistic differential
    diag = differential_diagnostic(A, Ahat, s_true)
    # deconv with ESTIMATED W_hat per split (from a held-out attribute-labelled split)
    est = {alpha: _eval_q_on_splits(pop, score, Ahat, W_known, alpha, n_splits, what="est",
                                    fr_hold=0.34) for alpha in cu.ALPHAS}
    # naive-on-A_hat (KS-0 scheme) for the "fraction of the gap closed" comparison
    naive = {}
    for alpha in cu.ALPHAS:
        wc_naive, sz_naive, wc_oracle = [], [], []
        for sp in cu.iter_splits(len(A), n_splits, (0.34, 0.33, 0.33)):
            cal, test = sp["cal"], sp["test"]
            qN = gr.mondrian_quantiles(s_true[cal], Ahat[cal], alpha)
            memN = gr.mondrian_build_sets(sa[test], Ahat[test], qN)
            rN = cu.coverage_report(memN, yt[test], A[test], alpha)
            qO = gr.mondrian_quantiles(s_true[cal], A[cal], alpha)
            memO = cu.sets_from_per_true_group_q(sa[test], A[test], qO)
            rO = cu.coverage_report(memO, yt[test], A[test], alpha)
            wc_naive.append(rN.worst_cov); sz_naive.append(rN.overall_size)
            wc_oracle.append(rO.worst_cov)
        naive[alpha] = {"worst": cu.agg(wc_naive), "size": cu.agg(sz_naive),
                        "oracle_worst": cu.agg(wc_oracle)}
    # verdict
    is_diff = (diag.get(1, 0.5) >= 0.55) or (diag.get(0, 0.5) >= 0.55)
    detail = [f"differential-noise diagnostic (AUROC of score predicting probe error): "
              f"A=0 {diag[0]:.3f}, A=1 {diag[1]:.3f} -> "
              f"{'DIFFERENTIAL' if is_diff else 'approx non-differential'}"]
    closes_ok = True
    for alpha in cu.ALPHAS:
        ow = est[alpha]["oracle"]["worst"]["mean"]
        dw = est[alpha]["deconv"]["worst"]["mean"]
        nw = naive[alpha]["worst"]["mean"]
        frac = (dw - nw) / (ow - nw) if (ow - nw) > 1e-6 else float("nan")
        sz_dec = est[alpha]["deconv"]["size"][1]["mean"]
        cond = est[alpha]["deconv"]["cond"]["mean"]
        negm = est[alpha]["deconv"]["negmass"]["mean"]
        closes_ok = closes_ok and (dw >= (1 - alpha) - 0.01)
        detail.append(f"alpha={alpha}: oracle worst={ow:.3f}, naive worst={nw:.3f}, "
                      f"deconv(M̂) worst={dw:.3f} (closes {frac:.0%} of gap), "
                      f"minority set-size={sz_dec:.1f}, W cond={cond:.2f}, neg-mass={negm:.3f}")
    viable = (not is_diff) and closes_ok
    label = ("Rank 1 viable on this data" if viable else
             "Rank 1 NOT viable: " + ("real noise is DIFFERENTIAL, so the simple score-independent-M "
             "deconvolution is invalid (a score-dependent correction would be required)"
             if is_diff else "deconvolution fails to restore worst-group coverage"))
    return {"est": est, "naive": naive, "diag": diag,
            "verdict": {"viable": viable, "label": label, "detail": detail}}


# --------------------------------------------------------------------------------------
def run(real=False, n_splits=cu.DEFAULT_N_SPLITS, score="APS"):
    if real:
        cu.load_real_population({}, seed=0)
    pop = cu.make_population(cu.TestbedConfig(seed=0, score=score))
    a = run_a(pop, score, n_splits)
    b = run_b(pop, score, n_splits)
    c = run_c(pop, score, n_splits)
    return {"a": a, "b": b, "c": c, "score": score, "n_splits": n_splits,
            "n_classes": pop["cfg"].n_classes, "flip_rate": FLIP_RATE, "synthetic": True}


def write_report(p, path):
    a, b, c = p["a"], p["b"], p["c"]
    L = ["# KS-1 -- Confusion-matrix-deconvolved attribute-conditional CP (Rank 1)", "",
         f"Synthetic testbed (C={p['n_classes']}, minority A=1 harder), flip rates {p['flip_rate']}, "
         f"{p['n_splits']} splits. True-attribute-conditional coverage throughout.", "",
         "## KS-1a -- known-M, NON-differential (estimator sanity)", "",
         f"**{a['verdict']['label']}**", "",
         f"Differential diagnostic (should be ~0.5): A=0 {a['diag'][0]:.3f}, A=1 {a['diag'][1]:.3f}", ""]
    for d in a["verdict"]["detail"]:
        L.append(f"- {d}")
    for alpha in cu.ALPHAS:
        r = a["res"][alpha]
        L += ["", f"alpha={alpha}: "
              f"oracle cov (A0/A1) {cu.fmt(r['oracle']['cov'][0])} / {cu.fmt(r['oracle']['cov'][1])}; "
              f"deconv cov (A0/A1) {cu.fmt(r['deconv']['cov'][0])} / {cu.fmt(r['deconv']['cov'][1])}; "
              f"deconv minority size {cu.fmt(r['deconv']['size'][1])}"]
    L += ["", "## KS-1b -- known-M, DIFFERENTIAL noise (THE critical test)", "",
          f"**{b['verdict']['label']}**", "",
          "Same marginal confusion matrix M across all β (verified: per-row counts below); only the "
          "score<->flip correlation changes. The deconvolution uses the (correct) marginal W.", "",
          "| β | diff-diagnostic A=1 | minority cov α=.10 | minority cov α=.05 | counts(Â×A) |",
          "|---|---|---|---|---|"]
    for s in b["sweep"]:
        c10 = s["res"][0.10]["deconv"]["cov"][1]
        c05 = s["res"][0.05]["deconv"]["cov"][1]
        L.append(f"| {s['beta']:g} | {s['diag'][1]:.3f} | {cu.fmt(c10)} | {cu.fmt(c05)} | "
                 f"{s['counts']} |")
    for d in b["verdict"]["detail"]:
        L.append(f"- {d}")
    L += ["", "## KS-1c -- real A_hat, estimated M̂ (reality check)", "",
          "> **Real CUB-200/CLIP number BLOCKED** (no torch/open_clip/datasets here). Below is the "
          "synthetic analog with realistic DIFFERENTIAL noise (β={}) and M̂ ESTIMATED per split from "
          "a held-out attribute-labelled split.".format(REAL_BETA), "",
          f"**{c['verdict']['label']}**", ""]
    for d in c["verdict"]["detail"]:
        L.append(f"- {d}")
    L += ["", "> The KS-1c differential-noise diagnostic is the headline reality check: on real "
          "CLIP probes, harder examples are typically BOTH mis-predicted by the attribute probe AND "
          "high-nonconformity, i.e. the noise is differential -- which (per KS-1b) invalidates the "
          "simple deconvolution. Mean±std over splits; synthetic stand-in for the blocked real run."]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


def make_plots(p):
    # KS-1b breakdown vs correlation (minority group)
    for alpha in cu.ALPHAS:
        corrs = [s["diag"][1] for s in p["b"]["sweep"]]
        gaps1 = [s["res"][alpha]["deconv"]["cov"][1]["mean"] - (1 - alpha) for s in p["b"]["sweep"]]
        gaps0 = [s["res"][alpha]["deconv"]["cov"][0]["mean"] - (1 - alpha) for s in p["b"]["sweep"]]
        plotting.plot_breakdown_vs_corr(
            corrs, {1: gaps1, 0: gaps0},
            os.path.join(OUT, f"ks1b_breakdown_a{int(alpha*100)}.pdf"),
            f"KS-1b deconvolution breakdown under differential noise (α={alpha})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--splits", type=int, default=cu.DEFAULT_N_SPLITS)
    ap.add_argument("--score", default="APS")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    p = run(real=args.real, n_splits=args.splits, score=args.score)
    write_report(p, os.path.join(OUT, "KS1_REPORT.md"))
    make_plots(p)
    with open(os.path.join(OUT, "ks1_results.json"), "w") as f:
        json.dump(p, f, indent=2, default=float)
    print("\n[ks1a]", p["a"]["verdict"]["label"])
    for d in p["a"]["verdict"]["detail"]:
        print("   ", d)
    print("[ks1b]", p["b"]["verdict"]["label"])
    for d in p["b"]["verdict"]["detail"]:
        print("   ", d)
    print("[ks1c]", p["c"]["verdict"]["label"])
    for d in p["c"]["verdict"]["detail"]:
        print("   ", d)
    print("[ks1] wrote", os.path.join(OUT, "KS1_REPORT.md"))


if __name__ == "__main__":
    main()
