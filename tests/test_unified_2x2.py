"""Tests for the CORRECTED unified 2x2 run (study paper v2): complete 2x2, unified score, the
group-free-substitution verdict (both GREEN and FALLBACK branches), and the §2b predicted-concept
sources. Pure-numpy / synthetic -- no CLIP or datasets needed."""
import numpy as np
import pytest

from eval.unified_verdict import unified_verdict, R_MIN
from experiments import cub200_frontier as cf
import scripts.run_unified_2x2 as u

RHO = [0.95, 0.90, 0.80, 0.70, 0.60, 0.50]


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
