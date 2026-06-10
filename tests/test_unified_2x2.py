"""Tests for the CORRECTED unified 2x2 run (study paper v2): complete 2x2, unified score, the
group-free-substitution verdict (both GREEN and FALLBACK branches), and the §2b predicted-concept
sources. Pure-numpy / synthetic -- no CLIP or datasets needed."""
import numpy as np
import pytest

from eval.unified_verdict import combined_decision, unified_verdict, R_MIN
from experiments import cub200_frontier as cf
from experiments.real_data import FeatureHeadGateError, GATE_MIN_TOP1
import scripts.run_unified_2x2 as u

RHO = [0.95, 0.90, 0.80, 0.70, 0.60, 0.50]


def _mk_records(score, gap_by_rho_and_cell, n_seeds=10):
    """Build tidy records from {rho: {(rep,scheme): gap}} (tiny per-seed jitter so CIs are finite)."""
    recs = []
    for rho, cells in gap_by_rho_and_cell.items():
        for (rep, scheme), gap in cells.items():
            for s in range(n_seeds):
                recs.append({"test_corr": rho, "score": score, "representation": rep,
                             "scheme": scheme, "seed": s, "worst_cov": 0.9 - gap,
                             "cov_gap": gap + 1e-4 * ((s % 3) - 1), "marg_cov": 0.9,
                             "mean_set_size": 5.0})
    return recs


def _run(concept_kwargs, seeds=8, n_classes=30):
    sc = cf.SmokeConfig(n=7000, n_classes=n_classes, seed=999, **concept_kwargs)
    pop = cf.make_smoke_population(sc)
    matched, _ = u.matched_class_subset(pop)
    recs = []
    for s in range(seeds):
        r, _ = u.run_seed(pop, s, 0.95, RHO, 1500, 1500, 0.5, 0.1, matched)
        recs.extend(r)
    return pop, recs


def test_all_four_cells_present():
    """The 2x2 is COMPLETE: feat+split must exist (the cell missing from the prior E1)."""
    _, recs = _run({"concept_source": "cbm"}, seeds=2)
    cells = {(r["representation"], r["scheme"]) for r in recs}
    assert cells == {("feature", "split"), ("feature", "Mondrian"),
                     ("concept", "split"), ("concept", "Mondrian")}


def test_same_score_compared_across_representations():
    """Every score appears for BOTH representations (no feat-APS vs cpt-THR confound)."""
    _, recs = _run({"concept_source": "cbm"}, seeds=2)
    for score in ("APS", "RAPS", "THR"):
        reps = {r["representation"] for r in recs if r["score"] == score}
        assert reps == {"feature", "concept"}


def test_green_when_concept_invariant():
    """Clean (invariant) predicted concept -> group-free substitution holds (GREEN)."""
    _, recs = _run({"concept_source": "cbm"})
    v = unified_verdict(recs, rho_cal=0.95, score="APS")
    assert v.green and v.sweep_mean_R >= R_MIN
    # the mechanism main effect is large and REPORTED (not suppressed)
    assert v.sweep_mean_mech_feat > 0.3


def test_fallback_when_concept_contaminated():
    """A concept score as contaminated as the feature score -> kill-switch FALLBACK fires."""
    _, recs = _run({"concept_source": "cbm", "cpt_margin": 1.2,
                    "cpt_residual_delta": 2.2, "cpt_spurious_kappa": 1.8})
    v = unified_verdict(recs, rho_cal=0.95, score="APS")
    assert not v.green


def test_feature_split_undercovers_atypical_under_shift():
    """Sanity: the contaminated feature score under pooled split badly under-covers the worst group
    (the phenomenon Mondrian / an invariant concept score are meant to address)."""
    _, recs = _run({"concept_source": "cbm"})
    fs = [r["worst_cov"] for r in recs
          if r["representation"] == "feature" and r["scheme"] == "split"
          and abs(r["test_corr"] - 0.5) < 1e-9]
    fm = [r["worst_cov"] for r in recs
          if r["representation"] == "feature" and r["scheme"] == "Mondrian"
          and abs(r["test_corr"] - 0.5) < 1e-9]
    assert np.mean(fs) < 0.6 < np.mean(fm)   # split under-covers; Mondrian restores it


@pytest.mark.parametrize("src", ["cbm", "zeroshot", "gt_attrs_leaky"])
def test_concept_sources_run(src):
    """All three §2b concept sources produce a valid population + verdict."""
    pop, recs = _run({"concept_source": src}, seeds=3)
    assert pop["concept_source"] == src
    v = unified_verdict(recs, rho_cal=0.95, score="APS")
    assert v.n_shifted == 5


# ---------------------------------------------------------------------------- v3 hardening
def test_gate_halts_below_floor():
    """§4: the orchestrator HALTS (FeatureHeadGateError) when the IN-DOMAIN head is below 0.55."""
    u.enforce_feature_gate({"feat_top1": 0.70}, "smoke")                       # OK, no raise
    with pytest.raises(FeatureHeadGateError):
        u.enforce_feature_gate({"feat_top1": 0.162}, "smoke")                  # smoke proxy below floor
    with pytest.raises(FeatureHeadGateError):
        u.enforce_feature_gate({"feat_top1_indomain_typical": 0.246}, "real")  # the v3 in-domain value
    # the clean-CUB anchor is NO LONGER the gate: a high anchor with a low in-domain head still halts
    with pytest.raises(FeatureHeadGateError):
        u.enforce_feature_gate({"feat_top1_indomain_typical": 0.40,
                                "feat_top1_cleancub": 0.70}, "real")


def test_degraded_smoke_head_halts_run():
    """End-to-end: a degraded synthetic head makes run() raise (no verdict computed)."""
    cfg = {"smoke": {"n": 5000, "n_classes": 30, "feat_margin_typical": 0.2,
                     "feat_margin_atypical": 0.1, "feat_spurious_kappa": 0.0}}
    with pytest.raises(FeatureHeadGateError):
        u.run(cfg, mode="smoke", n_seeds=3, concept_source="cbm")


def test_verdict_requires_hardest_shift():
    """v3 per-score GREEN needs the LARGEST shift (rho=0.5) to recover, not just a majority."""
    # recover everywhere EXCEPT rho=0.5 (cpt gap ~ feat gap there -> R~0)
    cells = {}
    for rho in RHO:
        recovers = rho > 0.5      # fails only at the hardest shift
        cells[rho] = {("feature", "split"): 0.70,
                      ("feature", "Mondrian"): 0.05,
                      ("concept", "split"): 0.15 if recovers else 0.69,
                      ("concept", "Mondrian"): 0.04}
    v = unified_verdict(_mk_records("APS", cells), rho_cal=0.95, score="APS")
    assert v.majority and not v.hardest_recovers and not v.green   # majority alone is NOT enough


def test_diagnostic_no_verdict_bypasses_gate():
    """v4b: --diagnostic-no-verdict on a sub-threshold head does NOT halt, emits NO verdict, and
    returns a diagnostic summary with a branch (RESOLVED/STILL WASHED OUT/IN BETWEEN)."""
    cfg = {"smoke": {"n": 6000, "n_classes": 30, "feat_margin_typical": 0.9,
                     "feat_margin_atypical": 0.4, "feat_spurious_kappa": 1.8}}
    payload = u.run(cfg, mode="smoke", n_seeds=6, concept_source="cbm", diagnostic=True)
    assert payload["diagnostic"] is True
    assert payload["verdicts"] is None and payload["combined"] is None      # NO verdict
    s = payload["diagnostic_summary"]
    assert s["branch"] in ("RESOLVED", "STILL WASHED OUT", "IN BETWEEN")
    assert abs(s["rho_test"] - 0.5) < 1e-9                                   # largest shift
    # feat+split numbers are present
    assert np.isfinite(s["feat_split_gap"][0]) and np.isfinite(s["feat_split_set_size"][0])


def test_diagnostic_branch_thresholds():
    """The branch classifier maps (gap, set_size/n_classes) to the spec thresholds."""
    from scripts.run_unified_2x2 import diagnostic_summary
    n = 200
    def mk(gap, setsz):
        return [{"test_corr": 0.5, "score": "APS", "representation": "feature", "scheme": "split",
                 "seed": i, "worst_cov": 0.9 - gap, "cov_gap": gap, "marg_cov": 0.9,
                 "mean_set_size": setsz} for i in range(5)]
    assert diagnostic_summary(mk(0.30, 15), 0.95, n)["branch"] == "RESOLVED"        # gap>=.15, <25/200
    assert diagnostic_summary(mk(0.02, 60), 0.95, n)["branch"] == "STILL WASHED OUT"  # gap<.05, >=40/200
    assert diagnostic_summary(mk(0.30, 60), 0.95, n)["branch"] == "IN BETWEEN"       # big gap AND big sets


def test_combined_requires_two_of_three_scores():
    """v3 headline GREEN requires per-score GREEN for >= 2 of 3 score functions."""
    good = {rho: {("feature", "split"): 0.70, ("feature", "Mondrian"): 0.05,
                  ("concept", "split"): 0.15, ("concept", "Mondrian"): 0.04} for rho in RHO}
    bad = {rho: {("feature", "split"): 0.70, ("feature", "Mondrian"): 0.05,
                 ("concept", "split"): 0.69, ("concept", "Mondrian"): 0.04} for rho in RHO}
    vg = lambda c, s: unified_verdict(_mk_records(s, c), rho_cal=0.95, score=s)
    # 1/3 green -> FALLBACK
    d1 = combined_decision({"APS": vg(good, "APS"), "RAPS": vg(bad, "RAPS"), "THR": vg(bad, "THR")})
    assert not d1["green"] and d1["n_green_scores"] == 1
    # 2/3 green -> GREEN
    d2 = combined_decision({"APS": vg(good, "APS"), "RAPS": vg(good, "RAPS"), "THR": vg(bad, "THR")})
    assert d2["green"] and d2["n_green_scores"] == 2
