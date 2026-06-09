"""Unit tests for E1 — CUB-200 multiclass frontier construction, resampler, and verdict.

Pure-numpy / CPU; no torch / open_clip / datasets needed. These pin the §2b invariants:
class balance is held FIXED across the rho sweep, rho is hit within tolerance, the typicality group
and species parsing are correct, and the pre-committed verdict returns GREEN/FALLBACK as specified.
"""
import numpy as np
import pytest

from experiments import cub200_frontier as cf
from eval import e1_verdict as ev


# ---------------------------------------------------------------------------- construction
def test_typicality_group_and_rho():
    place = np.array([0, 1, 0, 1])
    stype = np.array([0, 1, 1, 0])  # typical when place == stype -> [T, T, F, F]
    typ = cf.typicality_group(place, stype)
    assert typ.tolist() == [cf.TYPICAL, cf.TYPICAL, cf.ATYPICAL, cf.ATYPICAL]
    assert cf.realized_rho(typ) == pytest.approx(0.5)


def test_species_from_waterbirds_paths():
    paths = [
        "/x/waterbird_complete95/001.Black_footed_Albatross/Black_Footed_Albatross_0046_18.jpg",
        "data\\raw\\200.Common_Yellowthroat\\Common_Yellowthroat_0003_190521.jpg",
    ]
    sp = cf.species_from_waterbirds_paths(paths)
    assert sp.tolist() == [0, 199]  # 1-based CUB class -> 0-based species id


# ---------------------------------------------------------------------------- resampler
def _toy_pool(n=4000, C=20, seed=0, p_typ=0.8):
    rng = np.random.default_rng(seed)
    stype = (np.arange(C) % 2)
    species = rng.integers(0, C, n)
    is_typ = rng.random(n) < p_typ
    place = np.where(is_typ, stype[species], 1 - stype[species])
    typ = cf.typicality_group(place, stype[species])
    return species, typ


def test_resampler_holds_class_balance_fixed_across_rho():
    species, typ = _toy_pool()
    counts = cf.reference_species_counts(species, n=2000)
    per_species = {}
    for rho in (0.95, 0.7, 0.5):
        rs = cf.resample_to_rho_multiclass(species, typ, rho, 2000, seed=1, species_counts=counts)
        # the per-species count must be IDENTICAL across rho (only the typ/atyp mix changes)
        sp_counts = dict(zip(*np.unique(species[rs.idx], return_counts=True)))
        per_species[rho] = sp_counts
    base = per_species[0.95]
    for rho in (0.7, 0.5):
        assert per_species[rho] == base, "class balance changed across rho"


def test_resampler_hits_target_rho():
    species, typ = _toy_pool()
    for rho in (0.95, 0.8, 0.6, 0.5):
        rs = cf.resample_to_rho_multiclass(species, typ, rho, 3000, seed=2)
        assert abs(rs.rho_realized - rho) < 0.03, (rho, rs.rho_realized)


def test_reference_counts_sum_to_n():
    species, _ = _toy_pool()
    counts = cf.reference_species_counts(species, n=1234)
    assert sum(counts.values()) == 1234


# ---------------------------------------------------------------------------- smoke population
def test_smoke_population_shapes_and_contamination():
    pop = cf.make_smoke_population(cf.SmokeConfig(n=3000, n_classes=15, seed=7))
    C = pop["n_classes"]
    assert pop["feat_probs"].shape == (3000, C)
    assert pop["cpt_probs"].shape == (3000, C)
    # both heads classify better than chance
    assert pop["feat_top1"] > 2.0 / C
    assert pop["cpt_top1"] > 2.0 / C
    # feature head is background-contaminated: its accuracy on the atypical group should be LOWER
    # than on the typical group (the spurious boost hurts atypical), unlike the concept head.
    typ = pop["typicality"]
    feat_pred = pop["feat_probs"].argmax(1)
    cpt_pred = pop["cpt_probs"].argmax(1)
    sp = pop["species"]
    feat_atyp = (feat_pred[typ == cf.ATYPICAL] == sp[typ == cf.ATYPICAL]).mean()
    feat_typ = (feat_pred[typ == cf.TYPICAL] == sp[typ == cf.TYPICAL]).mean()
    cpt_atyp = (cpt_pred[typ == cf.ATYPICAL] == sp[typ == cf.ATYPICAL]).mean()
    cpt_typ = (cpt_pred[typ == cf.TYPICAL] == sp[typ == cf.TYPICAL]).mean()
    assert feat_typ - feat_atyp > 0.10, "feature head should be much worse on atypical"
    assert abs(cpt_typ - cpt_atyp) < feat_typ - feat_atyp, "concept head should be more invariant"


def test_load_real_population_raises_without_data():
    """Real loader must RAISE (never fabricate) when the datasets are absent. With download=False
    and a nonexistent root, load_waterbirds raises FileNotFoundError -- no fake population returned."""
    cfg = {"dataset": {"name": "waterbirds", "root": "/nonexistent_wb", "download": False,
                       "n_classes": 200},
           "cub": {"root": "/nonexistent_cub", "download": False},
           "clip": {"model_name": "ViT-B-32", "pretrained": "openai"}}
    with pytest.raises(Exception):
        cf.load_real_population(cfg, seed=0)


# ---------------------------------------------------------------------------- verdict
def _records(d_size, d_cov, rhos=(0.95, 0.9, 0.8, 0.7, 0.6, 0.5), seeds=10):
    """Synthesize tidy records where cpt+Mondrian beats feat+Mondrian by d_size (size) / d_cov (cov)
    at every shifted rho, with small seed noise so CIs are finite."""
    recs = []
    rng = np.random.default_rng(0)
    for rho in rhos:
        for s in range(seeds):
            feat_sz, feat_cov = 10.0, 0.90
            cpt_sz = feat_sz + (d_size if rho < 0.95 else 0.0) + rng.normal(0, 0.05)
            cpt_cov = feat_cov + (d_cov if rho < 0.95 else 0.0) + rng.normal(0, 0.002)
            for scheme, sz, cov in [(ev.FEAT_MOND, feat_sz, feat_cov),
                                    (ev.CPT_MOND, cpt_sz, cpt_cov),
                                    (ev.CPT_SPLIT, cpt_sz, cpt_cov - 0.03)]:
                recs.append({"test_corr": rho, "score": "APS", "scheme": scheme, "seed": s,
                             "worst_cov": cov, "mean_set_size": sz, "marg_cov": 0.9,
                             "cov_gap": 0.02})
    return recs


def test_verdict_green_on_size_win():
    # concept sets 3.0 smaller at matched coverage -> weak-Pareto relocation -> GREEN
    v = ev.e1_verdict(_records(d_size=-3.0, d_cov=0.0), rho_cal=0.95)
    assert v.green and v.n_relocated == v.n_shifted == 5
    assert v.sweep_mean_d_size < -2.0


def test_verdict_fallback_when_no_improvement():
    # concept identical on both axes -> no strict improvement -> FALLBACK (kill-switch)
    v = ev.e1_verdict(_records(d_size=0.0, d_cov=0.0), rho_cal=0.95)
    assert not v.green
    assert "relative-gap robustness" in v.fallback_claim


def test_verdict_fallback_when_size_much_worse():
    # concept covers a hair better but sets 5x bigger -> not weak-Pareto -> FALLBACK
    v = ev.e1_verdict(_records(d_size=+40.0, d_cov=0.01), rho_cal=0.95)
    assert not v.green
