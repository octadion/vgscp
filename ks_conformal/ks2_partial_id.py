"""KS-2 -- test-time-only noisy attribute, partial-identification set sizes (Rank 2).

Idea under test: A_true is observed CLEANLY at calibration (so the per-true-group thresholds q_a
are correct), but only A_hat is available at TEST. Build sets that GUARANTEE the lower-coverage
envelope P(Y in C(X) | A_true=a) >= 1-α for every true group, using W and the clean q_a, and check
whether the guaranteed sets are USEFUL (not near-trivial).

Conservative rule we commit to (documented in common_utils.partial_id_thresholds): for a test point
with A_hat=a_hat, the plausible true groups are { a : W[a_hat,a] >= prune }; use the LARGEST clean
q_a among them (cover the worst plausible true group). prune=0 => global-max threshold (guaranteed
but loosest). A tighter LP partial-ID bound is noted but not implemented (kept simple/correct).

PRE-COMMITTED GO/NO-GO: Rank 2 is alive only if a rule (a) delivers >= 1-α true-conditional
coverage for EVERY true group AND (b) mean set-size inflation over oracle < ~30-50% and far from
|Y|. If guaranteeing coverage forces near-trivial / explosive sets -> Rank 2 useless -> dead.

    python -m ks_conformal.ks2_partial_id            # synthetic (runs here)
    python -m ks_conformal.ks2_partial_id --real     # CUB-200/CLIP (BLOCKED: see RESULTS.md)
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from conformal import group_robust as gr
from ks_conformal import common_utils as cu
from ks_conformal import plotting

OUT = os.path.join("results", "ks_conformal")
FLIP_RATE = {0: 0.10, 1: 0.35}
TEST_BETA = 8.0                         # realistic differential test-time noise
PRUNE_SWEEP = [0.0, 0.05, 0.10, 0.20]   # partial-ID pruning thresholds
INFLATION_BAR = 0.50                    # pre-committed: alive needs < ~30-50% inflation over oracle


def run(real=False, n_splits=cu.DEFAULT_N_SPLITS, score="APS"):
    if real:
        cu.load_real_population({}, seed=0)
    pop = cu.make_population(cu.TestbedConfig(seed=0, score=score))
    A, sa, s_true, yt = (pop["A_true"], pop["scores_all"][score], pop["s_true"][score],
                         pop["y_true"])
    C = pop["cfg"].n_classes
    Ahat = cu.make_ahat(A, s_true, FLIP_RATE, beta=TEST_BETA, seed=501)

    res = {}
    for alpha in cu.ALPHAS:
        ref = {"oracle": {"worst": [], "size": []}, "naive": {"worst": [], "size": []}}
        prunes = {pr: {"worst": [], "cov0": [], "cov1": [], "size": [], "size0": [], "size1": [],
                       "infl": []} for pr in PRUNE_SWEEP}
        for sp in cu.iter_splits(len(A), n_splits, (0.34, 0.33, 0.33)):
            ho, cal, test = sp["holdout"], sp["cal"], sp["test"]
            _, What, _ = cu.confusion_matrix_MW(A[ho], Ahat[ho])     # estimate W from holdout
            # clean per-true-group quantiles from calibration (A_true observed at cal)
            qclean = gr.mondrian_quantiles(s_true[cal], A[cal], alpha)
            # oracle reference (uses A_true at test too -- not available to Rank 2; reference only)
            memO = cu.sets_from_per_true_group_q(sa[test], A[test], qclean)
            rO = cu.coverage_report(memO, yt[test], A[test], alpha)
            ref["oracle"]["worst"].append(rO.worst_cov)
            ref["oracle"]["size"].append(rO.overall_size)
            # naive: just trust A_hat (apply clean q by the test A_hat) -- the efficient-but-invalid bound
            memN = cu.sets_from_per_true_group_q(sa[test], Ahat[test], qclean)
            rN = cu.coverage_report(memN, yt[test], A[test], alpha)
            ref["naive"]["worst"].append(rN.worst_cov)
            ref["naive"]["size"].append(rN.overall_size)
            for pr in PRUNE_SWEEP:
                thr = cu.partial_id_thresholds(Ahat[test], qclean, What, prune=pr)
                memP = cu.sets_from_thresholds(sa[test], thr)
                rP = cu.coverage_report(memP, yt[test], A[test], alpha)
                prunes[pr]["worst"].append(rP.worst_cov)
                prunes[pr]["cov0"].append(rP.per_group_cov[0])
                prunes[pr]["cov1"].append(rP.per_group_cov[1])
                prunes[pr]["size"].append(rP.overall_size)
                prunes[pr]["size0"].append(rP.per_group_size[0])
                prunes[pr]["size1"].append(rP.per_group_size[1])
                prunes[pr]["infl"].append((rP.overall_size - rO.overall_size) /
                                          max(rO.overall_size, 1e-9))
        res[alpha] = {
            "oracle": {"worst": cu.agg(ref["oracle"]["worst"]), "size": cu.agg(ref["oracle"]["size"])},
            "naive": {"worst": cu.agg(ref["naive"]["worst"]), "size": cu.agg(ref["naive"]["size"])},
            "prunes": {pr: {k: cu.agg(prunes[pr][k]) for k in prunes[pr]} for pr in PRUNE_SWEEP},
            "trivial": float(C),
        }
    verdict = _verdict(res)
    return {"res": res, "verdict": verdict, "trivial": float(C), "n_classes": C,
            "flip_rate": FLIP_RATE, "test_beta": TEST_BETA, "prune_sweep": PRUNE_SWEEP,
            "n_splits": n_splits, "score": score, "synthetic": True}


def _verdict(res: dict) -> dict:
    """Alive iff SOME prune rule is valid (worst true-group cov >= 1-α-0.01 at BOTH α) AND its mean
    set-size inflation over oracle < INFLATION_BAR and far from trivial. We report the BEST valid
    rule (smallest inflation among the valid ones)."""
    detail = []
    best = None
    for pr in PRUNE_SWEEP:
        valid = all(res[a]["prunes"][pr]["worst"]["mean"] >= (1 - a) - 0.01 for a in cu.ALPHAS)
        infl = max(res[a]["prunes"][pr]["infl"]["mean"] for a in cu.ALPHAS)
        sz = max(res[a]["prunes"][pr]["size"]["mean"] for a in cu.ALPHAS)
        triv = res[cu.ALPHAS[0]]["trivial"]
        far_from_trivial = sz < 0.5 * triv
        usable = valid and (infl < INFLATION_BAR) and far_from_trivial
        detail.append(f"prune={pr:g}: valid(all groups, both α)={valid}, max set-size inflation "
                      f"over oracle={infl:+.0%}, max overall size={sz:.1f} (|Y|={triv:g}) -> "
                      f"{'USABLE' if usable else 'not usable'}")
        if valid:
            if best is None or infl < best[1]:
                best = (pr, infl, usable, sz)
    if best is None:
        return {"alive": False, "label": "NO-GO (Rank 2 dead): no pruning rule achieves valid "
                "true-conditional coverage for every group -- partial identification cannot be "
                "guaranteed under this differential test noise.", "detail": detail, "best": None}
    pr, infl, usable, sz = best
    if usable:
        label = (f"GO (Rank 2 alive): conservative partial-ID at prune={pr:g} guarantees coverage "
                 f"for every true group with {infl:+.0%} set-size inflation over oracle (size {sz:.1f} "
                 f"<< |Y|={res[cu.ALPHAS[0]]['trivial']:g}).")
        alive = True
    else:
        label = (f"NO-GO (Rank 2 useless): the cheapest VALID rule (prune={pr:g}) costs {infl:+.0%} "
                 f"set-size inflation over oracle (size {sz:.1f}), exceeding the pre-committed "
                 f"~{int(INFLATION_BAR*100)}% bar -- guaranteeing coverage forces near-trivial sets.")
        alive = False
    return {"alive": alive, "label": label, "detail": detail, "best_prune": pr}


def write_report(p, path):
    v = p["verdict"]
    L = ["# KS-2 -- Test-time-only noisy attribute, partial-identification set sizes (Rank 2)", "",
         f"**Verdict: {v['label']}**", "",
         f"Synthetic testbed (C={p['n_classes']} classes => |Y|={p['trivial']:g}); A_true clean at "
         f"calibration (clean per-group q_a), only A_hat at test (differential noise β={p['test_beta']}, "
         f"flip rates {p['flip_rate']}); W estimated per split from a held-out attribute split. "
         f"{p['n_splits']} splits. Conservative rule: use the largest clean q_a over plausible groups "
         f"{{a : W[â,a] >= prune}}.", ""]
    for d in v["detail"]:
        L.append(f"- {d}")
    for alpha in cu.ALPHAS:
        r = p["res"][alpha]
        L += ["", f"## alpha = {alpha} (nominal {1-alpha:.2f}, |Y|={p['trivial']:g})", "",
              "| scheme | cov A=0 | cov A=1 | worst-grp cov | overall size | infl. vs oracle |",
              "|---|---|---|---|---|---|",
              f"| oracle (Mondrian-on-A_true, ref) | -- | -- | {cu.fmt(r['oracle']['worst'])} | "
              f"{cu.fmt(r['oracle']['size'])} | 0% |",
              f"| naive (trust A_hat, ref) | -- | -- | {cu.fmt(r['naive']['worst'])} | "
              f"{cu.fmt(r['naive']['size'])} | -- |"]
        for pr in p["prune_sweep"]:
            pp = r["prunes"][pr]
            L.append(f"| partial-ID prune={pr:g} | {cu.fmt(pp['cov0'])} | {cu.fmt(pp['cov1'])} | "
                     f"{cu.fmt(pp['worst'])} | {cu.fmt(pp['size'])} | "
                     f"{100*pp['infl']['mean']:+.0f}% |")
    L += ["", "> Mean±std over splits. The tension is structural: the 35% of the true minority that "
          "the probe mislabels as majority (A_hat=0) is only covered if group-1 is NOT pruned for "
          "A_hat=0 (prune <= P(A=1|A_hat=0)); but not pruning it forces the easy majority points to "
          "the large minority threshold, inflating their sets. Differential test noise worsens this "
          "(high-score minority points are exactly the ones hidden in A_hat=0).",
          "", "> Real CUB-200/CLIP number BLOCKED here (no torch/open_clip/datasets); synthetic "
          "mechanism demonstration. See RESULTS.md."]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


def make_plots(p):
    for alpha in cu.ALPHAS:
        r = p["res"][alpha]
        pg = {"oracle": {0: (np.nan, 0), 1: (r["oracle"]["worst"]["mean"], r["oracle"]["worst"]["std"])}}
        cov = {}
        for pr in p["prune_sweep"]:
            pp = r["prunes"][pr]
            cov[f"prune={pr:g}"] = {0: (pp["cov0"]["mean"], pp["cov0"]["std"]),
                                    1: (pp["cov1"]["mean"], pp["cov1"]["std"])}
        plotting.plot_group_coverage(
            cov, alpha, os.path.join(OUT, f"ks2_coverage_a{int(alpha*100)}.pdf"),
            f"KS-2 partial-ID true-conditional coverage (α={alpha})")
        sizes = {"oracle": (r["oracle"]["size"]["mean"], r["oracle"]["size"]["std"]),
                 "naive": (r["naive"]["size"]["mean"], r["naive"]["size"]["std"])}
        for pr in p["prune_sweep"]:
            pp = r["prunes"][pr]
            sizes[f"prune={pr:g}"] = (pp["size"]["mean"], pp["size"]["std"])
        plotting.plot_setsize(sizes, os.path.join(OUT, f"ks2_setsize_a{int(alpha*100)}.pdf"),
                              f"KS-2 mean set size (α={alpha})", trivial=p["trivial"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--splits", type=int, default=cu.DEFAULT_N_SPLITS)
    ap.add_argument("--score", default="APS")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    p = run(real=args.real, n_splits=args.splits, score=args.score)
    write_report(p, os.path.join(OUT, "KS2_REPORT.md"))
    make_plots(p)
    with open(os.path.join(OUT, "ks2_results.json"), "w") as f:
        json.dump(p, f, indent=2, default=float)
    print("\n[ks2] VERDICT:", p["verdict"]["label"])
    for d in p["verdict"]["detail"]:
        print("   ", d)
    print("[ks2] wrote", os.path.join(OUT, "KS2_REPORT.md"))


if __name__ == "__main__":
    main()
