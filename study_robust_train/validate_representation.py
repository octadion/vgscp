"""Local logic-validation for the representation-level experiment and the hierarchical statistics.

Runs the entire numpy side — ``stats`` (cluster bootstrap, TOST, correlation CIs, the min-over-k
simulation) and ``representation`` (experiment loop, verdict, report) — on synthetic GridData, with
no torch, no GPU and no dataset. Exits 0 only if every check passes.

This is a CORRECTNESS check on the machinery, not a result: the synthetic numbers mean nothing.
Its job is to make sure a Colab session never fails after the expensive fine-tune has already run.

    python -m study_robust_train.validate_representation
"""
from __future__ import annotations

import sys

import numpy as np

from .representation import (REPR_CALIBRATIONS, representation_verdict,
                             run_representation_experiment, write_representation_md)
from .stats import (cluster_bootstrap_ci, correlation_ci, min_of_k_shortfall,
                    paired_cluster_diff_ci, simulate_min_coverage, spread_equivalence,
                    tost_equivalence)
from .synthetic import make_synthetic_griddata

FAILS = []


def check(name: str, cond: bool, detail: str = ""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  — {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


def validate_stats():
    print("\n[1] stats: cluster bootstrap")
    rng = np.random.default_rng(0)
    # 4 seeds x 25 splits, with real between-seed offsets: the flat bootstrap should be too narrow
    seeds, vals = [], []
    for s in range(4):
        offset = rng.normal(0, 0.05)
        for _ in range(25):
            seeds.append(s); vals.append(0.87 + offset + rng.normal(0, 0.005))
    vals, seeds = np.array(vals), np.array(seeds)

    cl = cluster_bootstrap_ci(vals, seeds, B=800, seed=1)
    flat = cluster_bootstrap_ci(vals, np.zeros_like(seeds), B=800, seed=1)
    check("cluster CI is wider than flat CI (between-seed variance propagates)",
          (cl["hi"] - cl["lo"]) > (flat["hi"] - flat["lo"]),
          f"cluster {cl['hi']-cl['lo']:.4f} vs flat {flat['hi']-flat['lo']:.4f}")
    check("cluster method reported", cl["method"] == "cluster" and cl["n_seeds"] == 4)
    check("single-seed input degrades to flat and says so", flat["method"] == "flat (1 seed)")
    check("point estimate unaffected by clustering", abs(cl["point"] - vals.mean()) < 1e-12)

    print("\n[2] stats: paired difference")
    a = vals + 0.02
    d = paired_cluster_diff_ci(a, vals, seeds, B=800, seed=2)
    check("paired diff recovers a known +0.02 offset", abs(d["point"] - 0.02) < 1e-9)
    check("paired diff CI excludes zero for a real offset", d["excludes_zero"])
    d0 = paired_cluster_diff_ci(vals, vals, seeds, B=800, seed=2)
    check("identical inputs give a zero-width, non-significant diff", not d0["excludes_zero"])

    print("\n[3] stats: TOST equivalence")
    tight = 0.90 + rng.normal(0, 0.002, 200)
    eq = tost_equivalence(tight, np.repeat(np.arange(4), 50), margin=0.03, center=0.90, B=800)
    check("tight data around center is EQUIVALENT within margin", eq["equivalent"], eq["verdict"])
    wide = 0.90 + rng.normal(0.05, 0.002, 200)
    neq = tost_equivalence(wide, np.repeat(np.arange(4), 50), margin=0.03, center=0.90, B=800)
    check("data offset beyond margin is NOT equivalent", not neq["equivalent"], neq["verdict"])

    print("\n[4] stats: spread equivalence across methods")
    flat_methods = {m: (0.87 + rng.normal(0, 0.003, 60), np.repeat(np.arange(3), 20))
                    for m in ("erm", "dfr", "gdro")}
    sp = spread_equivalence(flat_methods, margin=0.03, B=600)
    check("flat methods -> spread EQUIVALENT", sp["equivalent"],
          f"spread {sp['spread']:.4f}, hi {sp['hi']:.4f}")
    spread_methods = dict(flat_methods)
    spread_methods["erm"] = (0.60 + rng.normal(0, 0.003, 60), np.repeat(np.arange(3), 20))
    sp2 = spread_equivalence(spread_methods, margin=0.03, B=600)
    check("one collapsed method -> spread NOT equivalent", not sp2["equivalent"],
          f"spread {sp2['spread']:.4f}, hi {sp2['hi']:.4f}")
    check("spread needs >=2 methods", not spread_equivalence({"a": (np.ones(3), np.zeros(3))},
                                                             margin=0.03)["equivalent"])

    print("\n[5] stats: correlation CI")
    x = np.array([0.51, 0.81, 0.69, 0.64, 0.66])
    c = correlation_ci(x, -2.0 * x + 1.5, B=600)
    check("perfect negative relation recovers r ~ -1", abs(c["r"] + 1.0) < 1e-6, f"r={c['r']:.3f}")
    check("n=5 is flagged small_n", c["small_n"], "guards against reading r=-1.00 as precise")
    check("correlation rejects misaligned inputs",
          _raises(lambda: correlation_ci(np.zeros(4), np.zeros(5))))

    print("\n[6] stats: min-over-k sub-target law (the R1.1 / R3 answer)")
    sim = simulate_min_coverage(n_cal=300, n_test=300, k_groups=4, alpha=0.1, n_draws=400, seed=0)
    check("mean-over-groups coverage sits at the 0.90 target",
          abs(sim["mean_over_groups"] - 0.90) < 0.01, f"{sim['mean_over_groups']:.4f}")
    check("min-over-4 is materially below target with no model involved",
          sim["expected_min"] < 0.89, f"E[min]={sim['expected_min']:.4f}")
    check("shortfall/SD lands on the expected-minimum constant (~1.03)",
          0.85 < sim["shortfall_over_sd"] < 1.25, f"{sim['shortfall_over_sd']:.2f}")
    big = simulate_min_coverage(n_cal=3000, n_test=3000, k_groups=4, alpha=0.1, n_draws=300, seed=0)
    check("shortfall shrinks as calibration data grows",
          big["shortfall"] < sim["shortfall"],
          f"{sim['shortfall']:.4f} (n=300) -> {big['shortfall']:.4f} (n=3000)")
    check("closed form agrees with the simulation",
          abs(min_of_k_shortfall(4, sim["per_group_sd"]) - sim["shortfall"]) < 0.01)


def validate_representation():
    print("\n[7] representation: experiment loop")
    data = {}
    for rp, spur in (("erm", 1.9), ("groupdro", 0.7), ("reweight", 1.0)):
        for ft_seed in (0, 1, 2):
            data[("synthetic", rp, ft_seed)] = make_synthetic_griddata(
                backbone=f"ft_{rp}", dataset="synthetic", n_pool=2400, d=16,
                core_scale=1.2, spur_scale=spur, seed=100 * ft_seed + int(spur * 10))
    out = run_representation_experiment(data, heads=("erm", "dfr"), scores=("APS",),
                                        n_splits=4, alpha=0.1)
    recs = out["records"]
    expected = len(data) * 2 * 1 * len(REPR_CALIBRATIONS) * 4
    check("record count matches the grid", len(recs) == expected, f"{len(recs)} == {expected}")
    check("both calibration policies present",
          {r["calibration"] for r in recs} == set(REPR_CALIBRATIONS))
    check("ft_seed carried on every record (the clustering unit)",
          all(isinstance(r["ft_seed"], int) for r in recs))
    check("all three representations present",
          {r["representation"] for r in recs} == {"erm", "groupdro", "reweight"})

    print("\n[8] representation: new conformal_eval fields")
    check("mean_group_cov emitted", all("mean_group_cov" in r for r in recs))
    check("cov_range emitted (paper Eq. 1: max-min)", all("cov_range" in r for r in recs))
    check("n_cal_worst_group emitted", all(r["n_cal_worst_group"] >= 0 for r in recs))
    check("mean_group_cov >= worst_group_cov always (it is a mean over groups)",
          all(r["mean_group_cov"] >= r["worst_group_cov"] - 1e-9 for r in recs))
    mond = [r for r in recs if r["calibration"] == "mondrian"]
    check("Mondrian mean-over-groups exceeds its worst-group minimum",
          np.mean([r["mean_group_cov"] for r in mond])
          > np.mean([r["worst_group_cov"] for r in mond]))

    print("\n[9] representation: verdict structure")
    V = out["verdicts"]
    check("one verdict per (dataset, score)", set(V) == {"synthetic/APS"}, str(set(V)))
    v = V["synthetic/APS"]
    check("per-representation flatness computed for all three",
          set(v["per_representation"]) == {"erm", "groupdro", "reweight"})
    check("two-lever comparison produced a decision",
          v["levers"]["verdict"] != "undetermined", v["levers"]["verdict"])
    check("lever difference carries a CI", "lo" in v["levers"]["diff_worst_mondrian_minus_best_marginal"])
    check("primary lever holds the head fixed at the plain ERM head",
          v["levers"]["primary_head"] == "erm")
    check("a head-fixed lever is reported for every head",
          set(v["levers"]["by_head"]) == {"erm", "dfr"})
    check("head-fixed levers vary only the representation",
          all(len(hv["best_marginal_cell"]) == 1 for hv in v["levers"]["by_head"].values()),
          "cells are representations, not (repr, head) pairs")
    check("joint lever kept as a separate, stricter bound",
          v["levers"]["joint"]["scope"].startswith("joint")
          and len(v["levers"]["joint"]["best_marginal_cell"]) == 2)
    check("marginal representation effect computed against the ERM representation",
          any(k.startswith("groupdro_vs_erm") for k in v["marginal_repr_effect"]))
    # NB: no check that Mondrian is flatter here. On this toy pool the eval split leaves the
    # minority groups ~10 calibration points, so Mondrian's per-group thresholds are noisier than
    # the pooled one and Mondrian is legitimately *worse* than marginal. That is the finite-sample
    # regime of [12], not a defect -- and asserting the paper's substantive result on meaningless
    # synthetic numbers would be exactly the wrong thing for this validator to do.
    check("flatness spreads are finite and non-negative",
          all(0 <= d["mondrian_flatness"]["spread"] < 1 for d in v["per_representation"].values()))

    print("\n[9b] representation: manipulation check gates the interpretation")
    from .representation import manipulation_check
    man = out["manipulation"]["synthetic"]
    check("manipulation check produced a verdict", man["verdict"].startswith(("PASS", "FAIL")),
          man["verdict"][:60])
    check("per-head rows carry the reference wg acc",
          all("reference_wg_acc" in r for r in man["per_head"].values()))
    # wg-acc is one number per (repr, head, seed); the record list repeats it across splits x
    # scores x calibrations, so n must be the seed count, not the record count.
    n_seeds = len({r["ft_seed"] for r in recs})
    any_cmp = next(v for r in man["per_head"].values() for v in r.values() if isinstance(v, dict))
    check("wg-acc de-duplicated to one value per seed (not inflated by splits)",
          any_cmp["n_obs"] == n_seeds, f"n_obs={any_cmp['n_obs']} == {n_seeds} seeds")
    rigged_fail = [dict(r, worst_group_acc=(0.9 if r["representation"] == "erm" else 0.2))
                   for r in recs]
    check("a robust arm WORSE than the reference -> FAIL",
          manipulation_check(rigged_fail)["synthetic"]["verdict"].startswith("FAIL"))
    rigged_pass = [dict(r, worst_group_acc=(0.2 if r["representation"] == "erm" else 0.9))
                   for r in recs]
    check("a robust arm clearly better -> PASS",
          manipulation_check(rigged_pass)["synthetic"]["verdict"].startswith("PASS"))

    print("\n[10] representation: report emitter")
    import tempfile, os
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "REPRESENTATION.md")
        text = write_representation_md(out, p)
        check("report written", os.path.exists(p) and len(text) > 400)
        for token in ("Primary lever comparison", "Sub-target diagnostic", "equivalence @ margin",
                      "mean *over-groups* coverage", "Did the fine-tune change the representation?",
                      "Same comparison with each head held fixed",
                      "Manipulation check (READ FIRST)"):
            check(f"report contains {token!r}", token in text)

    print("\n[11] representation: the verdict discriminates in BOTH directions")
    # A one-sided rig proves nothing when the unrigged verdict already reads NARROW, so drive the
    # verdict to each outcome in turn and require it to follow.
    def _rig(marg, mond, jitter=0.002):
        rng2 = np.random.default_rng(7)
        out2 = []
        for r in recs:
            r2 = dict(r)
            base = marg if r2["calibration"] == "marginal_split" else mond
            r2["worst_group_cov"] = float(base + rng2.normal(0, jitter))
            out2.append(r2)
        return representation_verdict(out2, scores=("APS",))["synthetic/APS"]["levers"]

    narrow = _rig(marg=0.99, mond=0.60)
    check("marginal-wins data -> NARROW THE TITLE", "NARROW" in narrow["verdict"],
          narrow["verdict"])
    dominate = _rig(marg=0.60, mond=0.89)
    check("Mondrian-wins data -> CALIBRATION LEVER DOMINATES",
          "DOMINATES" in dominate["verdict"], dominate["verdict"])
    check("dominating verdict is backed by a CI that excludes zero",
          dominate["diff_worst_mondrian_minus_best_marginal"]["excludes_zero"])
    tie = _rig(marg=0.85, mond=0.85, jitter=0.05)
    check("indistinguishable levers do NOT claim dominance", "NARROW" in tie["verdict"],
          tie["verdict"])

    print("\n[12] the two components of the sub-target level (real per-group counts)")
    from .stats import group_counts_at_rho
    counts = group_counts_at_rho(2000, 0.95)
    check("rho=0.95 yields a ~19x majority/minority imbalance",
          counts.max() // max(counts.min(), 1) >= 15, f"counts={counts.tolist()}")
    mon = simulate_min_coverage(counts, counts, policy="mondrian", n_draws=400, seed=0)
    mar = simulate_min_coverage(counts, counts, policy="marginal", n_draws=400, seed=0)
    check("Mondrian under-covers more than marginal when minority cal counts are tiny",
          mon["expected_min"] < mar["expected_min"],
          f"mondrian {mon['expected_min']:.3f} vs marginal {mar['expected_min']:.3f}")
    check("that Mondrian shortfall shrinks as the pool grows",
          simulate_min_coverage(group_counts_at_rho(40000, 0.95),
                                group_counts_at_rho(40000, 0.95), policy="mondrian",
                                n_draws=300, seed=0)["shortfall"] < mon["shortfall"])
    check("per-group counts must match k_groups",
          _raises(lambda: simulate_min_coverage([10, 10], [10, 10], k_groups=4)))
    check("unknown policy rejected",
          _raises(lambda: simulate_min_coverage(100, 100, policy="bogus")))


def _raises(fn):
    try:
        fn()
        return False
    except Exception:
        return True


def main():
    print("=" * 78)
    print("validate_representation — logic only, synthetic inputs, no torch / GPU / datasets")
    print("=" * 78)
    validate_stats()
    validate_representation()
    print("\n" + "=" * 78)
    if FAILS:
        print(f"FAILED ({len(FAILS)}): " + ", ".join(FAILS))
        return 1
    print("ALL CHECKS PASSED — safe to run the Colab notebook")
    return 0


if __name__ == "__main__":
    sys.exit(main())
