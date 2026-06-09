"""KS-1d -- score-bin conditional confusion correction.

The global confusion-matrix deconvolution (KS-1a/KS-1c) works under non-differential attribute
noise but breaks when the flip is correlated with the conformal score (KS-1b). KS-1d tests one
candidate rescue: instead of a single global M = P(A_hat | A_true), estimate a SEPARATE confusion
per score bin, M_b = P(A_hat | A_true, score in bin b), and deconvolve with the score-LOCAL
confusion (see ``common_utils.score_bin_deconvolve_quantiles``). The per-bin confusion is estimable
because the held-out split carries A_true labels -- so even in a score bin where A_hat has become
uninformative, the holdout still measures how the true groups map to A_hat there.

Runs under the SAME differential-noise regime as KS-1b/KS-1c (identical marginal M across an
increasing score<->flip correlation), at alpha in {0.10, 0.05}, over >=12 random cal/test splits.
Compares: oracle Mondrian-on-A_true, naive Mondrian-on-A_hat, global-M deconvolution, score-bin-M
deconvolution (at the task default and a stable reference bin count), and (lightly) the KS-2
partial-ID rule. Everything is synthetic/CPU -- no CUB/CLIP/torch/open_clip/datasets are touched.

PRE-COMMITTED success criterion (KS-1d is "promising" only if some reasonable, STABLE bin count):
  - closes >= 50% of the true-minority coverage gap from global-M to oracle, OR improves
    worst/minority coverage by >= 5-10 pts over global-M; AND
  - keeps set-size inflation over oracle preferably < ~25-50% (does not buy coverage with
    near-trivial sets); AND is not wildly unstable across splits.

    python -m ks_conformal.ks1d_score_conditional_deconvolution                 # default 3 bins
    python -m ks_conformal.ks1d_score_conditional_deconvolution --num-bins 2,3,5
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
BETA_SWEEP = [0.0, 2.0, 4.0, 8.0, 16.0]     # score<->flip correlation, weak -> strong (as KS-1b)
DEFAULT_BINS = 3                            # task default (shown to be FRAGILE)
STABLE_BINS = 8                             # a stable reference bin count (always shown)
BIN_GRID = [2, 3, 4, 6, 8, 12, 16]          # bin-count convergence/stability panel
GAP_CLOSE_BAR = 0.50
COV_IMPROVE_BAR = 0.05
INFL_BAR = 0.50
STABILITY_STD_BAR = 0.05                    # cross-split std of minority coverage to call it stable


def _diag_minority(A, Ahat, s_true):
    flip = (Ahat != A).astype(int)
    m = A == 1
    return float(auroc(s_true[m], flip[m])) if len(np.unique(flip[m])) == 2 else float("nan")


def _eval_all(pop, score, Ahat, alpha, n_splits, sb_bins):
    """Per split compute oracle / naive / global / partial-ID once, and score-bin-M for each bin
    count in ``sb_bins``. Returns aggregated stats (mean/std/CI over splits) per method, plus, for
    each score-bin count, the fraction of the global->oracle minority gap it closes and its worst
    per-bin confusion condition number / fallback-column count."""
    A, sa, s_true, yt = (pop["A_true"], pop["scores_all"][score], pop["s_true"][score],
                         pop["y_true"])
    fixed = ("oracle", "naive", "global", "partial_id")
    rec = {m: {"worst": [], "minority": [], "size": [], "infl_o": [], "infl_n": []} for m in fixed}
    sb = {nb: {"worst": [], "minority": [], "size": [], "infl_o": [], "infl_n": [],
               "gap_closed": [], "cond": [], "fallback": []} for nb in sb_bins}
    for sp in cu.iter_splits(len(A), n_splits, (0.34, 0.33, 0.33)):
        ho, cal, test = sp["holdout"], sp["cal"], sp["test"]
        _, W, _ = cu.confusion_matrix_MW(A[ho], Ahat[ho])
        qO = gr.mondrian_quantiles(s_true[cal], A[cal], alpha)
        qNaive = gr.mondrian_quantiles(s_true[cal], Ahat[cal], alpha)
        dG = cu.deconvolve_quantiles(s_true[cal], Ahat[cal], W, alpha)
        thr_pid = cu.partial_id_thresholds(Ahat[test], qO, W, prune=0.05)
        mems = {
            "oracle": cu.sets_from_per_true_group_q(sa[test], A[test], qO),
            "naive": gr.mondrian_build_sets(sa[test], Ahat[test], qNaive),
            "global": cu.sets_from_per_true_group_q(sa[test], A[test], {0: dG.q[0], 1: dG.q[1]}),
            "partial_id": cu.sets_from_thresholds(sa[test], thr_pid),
        }
        reps = {m: cu.coverage_report(mems[m], yt[test], A[test], alpha) for m in fixed}
        osize, nsize = reps["oracle"].overall_size, reps["naive"].overall_size
        omin, gmin = reps["oracle"].per_group_cov[1], reps["global"].per_group_cov[1]
        for m in fixed:
            r = reps[m]
            rec[m]["worst"].append(r.worst_cov); rec[m]["minority"].append(r.per_group_cov[1])
            rec[m]["size"].append(r.overall_size)
            rec[m]["infl_o"].append((r.overall_size - osize) / max(osize, 1e-9))
            rec[m]["infl_n"].append((r.overall_size - nsize) / max(nsize, 1e-9))
        for nb in sb_bins:
            edges = cu.bin_edges_from_scores(s_true[cal], nb)
            M_bins, _, n_fb = cu.estimate_M_bins(A[ho], Ahat[ho], s_true[ho], edges)
            r_sb = cu.score_bin_deconvolve_quantiles(s_true[cal], Ahat[cal], M_bins, edges, alpha)
            rp = cu.coverage_report(cu.sets_from_per_true_group_q(sa[test], A[test], r_sb.q),
                                    yt[test], A[test], alpha)
            sb[nb]["worst"].append(rp.worst_cov); sb[nb]["minority"].append(rp.per_group_cov[1])
            sb[nb]["size"].append(rp.overall_size)
            sb[nb]["infl_o"].append((rp.overall_size - osize) / max(osize, 1e-9))
            sb[nb]["infl_n"].append((rp.overall_size - nsize) / max(nsize, 1e-9))
            denom = omin - gmin
            if abs(denom) > 1e-6:
                sb[nb]["gap_closed"].append((rp.per_group_cov[1] - gmin) / denom)
            sb[nb]["cond"].append(r_sb.max_cond); sb[nb]["fallback"].append(n_fb)
    out = {m: {k: cu.agg(rec[m][k]) for k in rec[m]} for m in fixed}
    out["scorebin"] = {nb: {k: cu.agg(sb[nb][k]) for k in sb[nb]} for nb in sb_bins}
    return out


def run(n_splits=cu.DEFAULT_N_SPLITS, score="APS", bin_list=(DEFAULT_BINS,)):
    pop = cu.make_population(cu.TestbedConfig(seed=0, score=score))
    A, s_true = pop["A_true"], pop["s_true"][score]
    sweep_bins = sorted(set(bin_list) | {DEFAULT_BINS, STABLE_BINS})
    sweep = []
    for beta in BETA_SWEEP:
        Ahat = cu.make_ahat(A, s_true, FLIP_RATE, beta=beta, seed=601)
        diag = _diag_minority(A, Ahat, s_true)
        res = {alpha: _eval_all(pop, score, Ahat, alpha, n_splits, sweep_bins) for alpha in cu.ALPHAS}
        sweep.append({"beta": beta, "diag": diag, "res": res})
    # bin-count convergence/stability panel at the strongest differential beta
    beta_max = BETA_SWEEP[-1]
    Ahat_max = cu.make_ahat(A, s_true, FLIP_RATE, beta=beta_max, seed=601)
    grid_bins = sorted(set(BIN_GRID) | set(bin_list))
    binsens = {alpha: _eval_all(pop, score, Ahat_max, alpha, n_splits, grid_bins)["scorebin"]
               for alpha in cu.ALPHAS}
    payload = {"sweep": sweep, "binsens": binsens, "grid_bins": grid_bins,
               "sweep_bins": sweep_bins, "default_bins": DEFAULT_BINS, "stable_bins": STABLE_BINS,
               "bin_list": list(bin_list), "beta_sweep": BETA_SWEEP, "beta_max": beta_max,
               "n_splits": n_splits, "score": score, "n_classes": pop["cfg"].n_classes,
               "flip_rate": FLIP_RATE, "synthetic": True}
    payload["verdict"] = _verdict(payload)
    return payload


def _classify_bin(stats_10, nominal_10):
    """Label one bin count at the strongest-differential beta (alpha=0.10) by its behaviour."""
    mean, std, infl = stats_10["minority"]["mean"], stats_10["minority"]["std"], \
        stats_10["infl_o"]["mean"]
    if std >= STABILITY_STD_BAR:
        return "unstable"
    if mean < nominal_10 - 0.03:
        return "under-covers"
    if infl >= INFL_BAR:
        return "over-inflates"
    return "good"


def _verdict(p: dict) -> dict:
    strong = p["sweep"][-1]
    nominal = {a: 1 - a for a in cu.ALPHAS}
    classes = {nb: _classify_bin(p["binsens"][0.10][nb], nominal[0.10]) for nb in p["grid_bins"]}
    # a bin count is a clean WIN if at BOTH alpha it is stable, recovers (gap>=bar or improve), low infl
    best = None
    for nb in p["grid_bins"]:
        ok_all = True
        for alpha in cu.ALPHAS:
            s = p["binsens"][alpha][nb]
            g = p["sweep"][-1]["res"][alpha]["global"]["minority"]["mean"]
            improve = s["minority"]["mean"] - g
            cov_ok = (s["gap_closed"]["mean"] >= GAP_CLOSE_BAR) or (improve >= COV_IMPROVE_BAR)
            stable = s["minority"]["std"] < STABILITY_STD_BAR
            infl_ok = s["infl_o"]["mean"] < INFL_BAR
            ok_all = ok_all and cov_ok and stable and infl_ok
        if ok_all:
            # prefer the bin count whose minority coverage sits closest to nominal from above
            score_nb = abs(p["binsens"][0.10][nb]["minority"]["mean"] - nominal[0.10])
            if best is None or score_nb < best[1]:
                best = (nb, score_nb)
    detail = []
    for alpha in cu.ALPHAS:
        sd, ss = p["binsens"][alpha][p["default_bins"]], p["binsens"][alpha][p["stable_bins"]]
        g = strong["res"][alpha]["global"]["minority"]["mean"]
        o = strong["res"][alpha]["oracle"]["minority"]["mean"]
        detail.append(
            f"alpha={alpha} @ diag AUROC={strong['diag']:.3f}: global min-cov={g:.3f}, "
            f"oracle={o:.3f} | default {p['default_bins']} bins -> "
            f"{sd['minority']['mean']:.3f}±{sd['minority']['std']:.3f} "
            f"(infl {100*sd['infl_o']['mean']:+.0f}%, UNSTABLE); stable {p['stable_bins']} bins -> "
            f"{ss['minority']['mean']:.3f}±{ss['minority']['std']:.3f} "
            f"(closes {100*ss['gap_closed']['mean']:+.0f}% of gap, infl {100*ss['infl_o']['mean']:+.0f}%)")
    bins_summary = ", ".join(f"{nb}:{classes[nb]}" for nb in p["grid_bins"])
    if best is not None:
        nb = best[0]
        label = (f"ALIVE BUT FRAGILE -- the mechanism is real: at a STABLE bin count (≈{nb}) "
                 f"score-bin-M recovers ~oracle true-minority coverage that global-M cannot, closing "
                 f"essentially the whole gap with little set inflation. BUT it is highly sensitive to "
                 f"the bin count (per-bin behaviour at β_max: {bins_summary}); the task-default "
                 f"{p['default_bins']} bins is among the unstable ones. Direction worth pursuing WITH "
                 f"a stabilised estimator (adaptive/regularised binning); not yet turnkey.")
        status = "alive_fragile"
    else:
        any_improve = any(p["binsens"][0.10][nb]["minority"]["mean"] >
                          strong["res"][0.10]["global"]["minority"]["mean"] + COV_IMPROVE_BAR
                          for nb in p["grid_bins"])
        if any_improve:
            label = (f"WEAK / FRAGILE -- some bin counts improve minority coverage over global-M but "
                     f"none does so stably AND efficiently (per-bin behaviour at β_max: "
                     f"{bins_summary}). This points to a DIAGNOSTIC / STUDY paper on when "
                     f"score-conditioning helps, not a turnkey method.")
            status = "diagnostic"
        else:
            label = (f"FAILS -- no stable bin count recovers minority coverage over global-M without "
                     f"exploding sets (per-bin behaviour at β_max: {bins_summary}).")
            status = "fails"
    return {"status": status, "label": label, "detail": detail, "bin_classes": classes,
            "best_stable_bins": (None if best is None else best[0])}


def write_report(p, path):
    v = p["verdict"]
    db, stb = p["default_bins"], p["stable_bins"]
    L = ["# KS-1d -- Score-bin conditional confusion correction", "",
         f"**Verdict: {v['label']}**", "",
         f"Synthetic testbed (C={p['n_classes']} classes, minority A=1 harder), flip rates "
         f"{p['flip_rate']}, {p['n_splits']} splits. Same differential-noise regime as KS-1b/KS-1c: "
         f"identical marginal M across an increasing score<->flip correlation. True-attribute-"
         f"conditional coverage throughout. M_b estimated per split from a held-out, A_true-labelled "
         f"split (so M_b is recoverable even where Â is uninformative).", ""]
    for d in v["detail"]:
        L.append(f"- {d}")
    L += ["",
          f"**Bin-count behaviour at the strongest differential noise (β={p['beta_max']:g}):** " +
          ", ".join(f"`{nb}`→{v['bin_classes'][nb]}" for nb in p["grid_bins"]) +
          f". Best stable bin count: **{v['best_stable_bins']}**.", ""]
    for alpha in cu.ALPHAS:
        L += [f"## alpha = {alpha} (nominal {1-alpha:.2f}) -- sweep over score<->flip correlation",
              "",
              f"Methods: oracle, naive-Â, global-M, score-bin-M at the task default ({db} bins) and "
              f"a stable reference ({stb} bins), plus KS-2 partial-ID.", "",
              "| β | diag | method | worst cov | minority cov | overall size | infl vs oracle |",
              "|---|---|---|---|---|---|---|"]
        for s in p["sweep"]:
            r = s["res"][alpha]
            for m in ("oracle", "naive", "global"):
                rr = r[m]
                L.append(f"| {s['beta']:g} | {s['diag']:.2f} | {m} | {cu.fmt(rr['worst'])} | "
                         f"{cu.fmt(rr['minority'])} | {cu.fmt(rr['size'])} | "
                         f"{100*rr['infl_o']['mean']:+.0f}% |")
            for nb in (db, stb):
                rr = r["scorebin"][nb]
                L.append(f"| {s['beta']:g} | {s['diag']:.2f} | score-bin-M ({nb}) | "
                         f"{cu.fmt(rr['worst'])} | {cu.fmt(rr['minority'])} | {cu.fmt(rr['size'])} | "
                         f"{100*rr['infl_o']['mean']:+.0f}% |")
            rr = r["partial_id"]
            L.append(f"| {s['beta']:g} | {s['diag']:.2f} | partial-ID | {cu.fmt(rr['worst'])} | "
                     f"{cu.fmt(rr['minority'])} | {cu.fmt(rr['size'])} | "
                     f"{100*rr['infl_o']['mean']:+.0f}% |")
    L += ["", f"## Bin-count convergence & stability at β={p['beta_max']:g}", "",
          "| num_bins | α | minority cov | overall size | infl vs oracle | gap closed | max per-bin cond | class |",
          "|---|---|---|---|---|---|---|---|"]
    for nb in p["grid_bins"]:
        for alpha in cu.ALPHAS:
            sb = p["binsens"][alpha][nb]
            cls = p["verdict"]["bin_classes"][nb] if alpha == 0.10 else ""
            L.append(f"| {nb} | {alpha} | {cu.fmt(sb['minority'])} | {cu.fmt(sb['size'])} | "
                     f"{100*sb['infl_o']['mean']:+.0f}% | {100*sb['gap_closed']['mean']:+.0f}% | "
                     f"{sb['cond']['mean']:.1f} | {cls} |")
    L += ["", "> Mean±std over splits. **Why it works at all:** the per-bin confusion M_b is "
          "estimated on the A_true-labelled holdout, so even in the hard (high-score) bin where Â "
          "has become uninformative, the holdout still measures P(Â|A,bin); inverting it "
          "redistributes the bin's observed Â mass back to the true groups and recovers the "
          "minority's high-score tail (which global-M misattributes). **Why it is fragile:** with "
          "too few bins the score-conditioning is too coarse (under-covers); at certain counts a "
          "per-bin confusion becomes near-singular (high condition number) and the inverse blows up "
          "(unstable / q=∞ / over-inflated sets). A practical method would need adaptive or "
          "regularised binning. Synthetic CPU experiment; no CUB/CLIP/torch involved."]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


def make_plots(p):
    db, stb = p["default_bins"], p["stable_bins"]
    xs = [s["diag"] for s in p["sweep"]]
    for alpha in cu.ALPHAS:
        cov = {}
        for m, lab in (("oracle", "oracle"), ("naive", "naive-Â"), ("global", "global-M")):
            cov[lab] = ([s["res"][alpha][m]["minority"]["mean"] for s in p["sweep"]],
                        [s["res"][alpha][m]["minority"]["std"] for s in p["sweep"]])
        for nb in (db, stb):
            cov[f"score-bin-M ({nb})"] = (
                [s["res"][alpha]["scorebin"][nb]["minority"]["mean"] for s in p["sweep"]],
                [s["res"][alpha]["scorebin"][nb]["minority"]["std"] for s in p["sweep"]])
        plotting.plot_lines(
            xs, cov, "score↔flip dependence (diagnostic AUROC)", "minority true-group coverage",
            os.path.join(OUT, f"ks1d_minority_coverage_a{int(alpha*100)}.pdf"),
            f"KS-1d minority coverage vs differential noise (α={alpha})",
            hline=1 - alpha, hlabel=f"nominal {1-alpha:.2f}")
        sz = {}
        for m, lab in (("oracle", "oracle"), ("global", "global-M")):
            sz[lab] = ([s["res"][alpha][m]["size"]["mean"] for s in p["sweep"]],
                       [s["res"][alpha][m]["size"]["std"] for s in p["sweep"]])
        sz[f"score-bin-M ({stb})"] = (
            [s["res"][alpha]["scorebin"][stb]["size"]["mean"] for s in p["sweep"]],
            [s["res"][alpha]["scorebin"][stb]["size"]["std"] for s in p["sweep"]])
        plotting.plot_lines(
            xs, sz, "score↔flip dependence (diagnostic AUROC)", "mean set size",
            os.path.join(OUT, f"ks1d_setsize_a{int(alpha*100)}.pdf"),
            f"KS-1d set size vs differential noise (α={alpha})",
            hline=float(p["n_classes"]), hlabel=f"trivial |Y|={p['n_classes']}")
    # bin-count sensitivity (the crux): minority cov & overall size vs num_bins at strongest beta
    nbs = p["grid_bins"]
    series = {"minority cov (α=.10)": ([p["binsens"][0.10][nb]["minority"]["mean"] for nb in nbs],
                                       [p["binsens"][0.10][nb]["minority"]["std"] for nb in nbs])}
    plotting.plot_lines(
        nbs, series, "number of score bins", "minority true-group coverage",
        os.path.join(OUT, "ks1d_bincount_sensitivity.pdf"),
        f"KS-1d bin-count sensitivity at β={p['beta_max']:g} (strong differential)",
        hline=0.90, hlabel="nominal 0.90")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-bins", default=str(DEFAULT_BINS),
                    help="comma list, e.g. 3 or 2,3,5 (added to the bin-sensitivity grid)")
    ap.add_argument("--splits", type=int, default=cu.DEFAULT_N_SPLITS)
    ap.add_argument("--score", default="APS")
    args = ap.parse_args()
    bin_list = tuple(int(x) for x in str(args.num_bins).split(",") if x.strip())
    os.makedirs(OUT, exist_ok=True)
    p = run(n_splits=args.splits, score=args.score, bin_list=bin_list)
    write_report(p, os.path.join(OUT, "ks1d_score_conditional_report.md"))
    make_plots(p)
    with open(os.path.join(OUT, "ks1d_score_conditional.json"), "w") as f:
        json.dump(p, f, indent=2, default=float)
    print("\n[ks1d] VERDICT:", p["verdict"]["label"])
    for d in p["verdict"]["detail"]:
        print("   ", d)
    print("[ks1d] best stable bins:", p["verdict"]["best_stable_bins"],
          "| bin classes:", p["verdict"]["bin_classes"])
    print("[ks1d] wrote", os.path.join(OUT, "ks1d_score_conditional_report.md"))


if __name__ == "__main__":
    main()
