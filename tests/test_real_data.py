"""Tests for the REAL-data downstream logic (head fitting, §2b construction, fdata assembly).

The CLIP-encoding + dataset-loading steps need the real environment, so here we INJECT a fabricated
``RealBundle`` (planted so the logistic heads can learn) and verify every pure-numpy/sklearn step:
column-aligned head posteriors, the binary fdata structure (incl. ensemble/MC shapes), the species-
type map, and E1's ``load_real_population`` assembling a valid population dict from a bundle. No
open_clip / datasets are touched.
"""
import numpy as np
import pytest

from experiments import cub200_frontier as cf
from experiments import real_data as rd


def _planted_bundle(n_classes=10, d_feat=16, d_attr=12, per_split=400, seed=0):
    """Fabricate a RealBundle with features/attrs that predict species, and a fixed species type."""
    rng = np.random.default_rng(seed)
    species_type = (np.arange(n_classes) % 2).astype(np.int64)  # t(species)
    splits = ("train", "d_learn", "d_cal", "d_test")
    features, attrs, species, y, place, group_id, is_minority, paths = ({} for _ in range(8))
    # per-class planted centroids so a logistic head separates them
    feat_centroids = rng.normal(0, 3, (n_classes, d_feat))
    attr_centroids = rng.normal(0, 3, (n_classes, d_attr))
    for sp in splits:
        s = rng.integers(0, n_classes, per_split)
        features[sp] = (feat_centroids[s] + rng.normal(0, 1.0, (per_split, d_feat))).astype(np.float32)
        attrs[sp] = (attr_centroids[s] + rng.normal(0, 1.0, (per_split, d_attr))).astype(np.float32)
        species[sp] = s.astype(np.int64)
        yy = species_type[s]
        y[sp] = yy.astype(np.int64)
        pl = rng.integers(0, 2, per_split)
        place[sp] = pl.astype(np.int64)
        group_id[sp] = (2 * yy + pl).astype(np.int64)
        is_minority[sp] = (yy != pl)
        paths[sp] = [f"{sp}/{i}.jpg" for i in range(per_split)]
    return rd.RealBundle(features=features, attrs=attrs, species=species, y=y, place=place,
                         group_id=group_id, is_minority=is_minority, paths=paths,
                         species_type=species_type, attr_names=[f"a{i}" for i in range(d_attr)],
                         n_classes=n_classes, info={})


def test_head_probs_column_alignment():
    rng = np.random.default_rng(1)
    X = rng.normal(0, 1, (200, 8))
    y = rng.integers(0, 5, 200)
    # drop class 4 from training -> head_probs must still return 6 columns (missing class -> 0)
    keep = y != 4
    clf = rd.fit_logistic_head(X[keep], y[keep], seed=0)
    p = rd.head_probs(clf, X, n_classes=6)
    assert p.shape == (200, 6)
    assert np.allclose(p[:, 4], 0.0) and np.allclose(p[:, 5], 0.0)  # classes 4,5 absent in train
    assert np.allclose(p.sum(1), 1.0, atol=1e-6)


def test_species_type_map():
    species = np.array([0, 0, 1, 1, 2])
    y = np.array([1, 1, 0, 0, 1])
    t = rd._species_type_map(species, y, n_classes=3)
    assert t.tolist() == [1, 0, 1]


def test_build_binary_fdata_structure():
    b = _planted_bundle()
    cfg = {"heads": {}, "common": {"ensemble": {"n_members": 3}, "mc_dropout": {"n_passes": 5}}}
    fd = rd.build_binary_fdata(b, cfg, seed=0, with_ensemble_mc=True)
    for sp in ("train", "d_learn", "d_cal", "d_test"):
        d = fd[sp]
        n = len(b.species[sp])
        assert d["probs"].shape == (n, 2)
        assert d["features"].shape[0] == n
        assert d["member_probs"].shape == (3, n, 2)
        assert d["mc_pass_probs"].shape == (5, n, 2)
        assert set(np.unique(d["y_true"])) <= {0, 1}
    # f is a real classifier: train accuracy above chance on the planted data
    acc = (fd["train"]["y_pred"] == fd["train"]["y_true"]).mean()
    assert acc > 0.6


def test_load_real_population_from_injected_bundle(monkeypatch):
    b = _planted_bundle(n_classes=10)
    monkeypatch.setattr(rd, "load_real_bundle", lambda cfg, seed, **kw: b)
    cfg = {"dataset": {"n_classes": 10}, "heads": {}}
    pop = cf.load_real_population(cfg, seed=0)
    pool_n = sum(len(b.species[s]) for s in ("d_learn", "d_cal", "d_test"))
    assert pop["species"].shape == (pool_n,)
    assert pop["feat_probs"].shape == (pool_n, 10)
    assert pop["cpt_probs"].shape == (pool_n, 10)
    assert not pop["synthetic"]
    # typicality matches (place == species_type) by construction
    expect = cf.typicality_group(pop["place"], pop["species_type"])
    assert np.array_equal(pop["typicality"], expect)
    # heads learned the planted structure -> top-1 well above chance (0.1)
    assert pop["feat_top1"] > 0.4 and pop["cpt_top1"] > 0.4
    assert pop["info"]["pool_n_atypical"] >= 0


def test_load_real_population_feeds_orchestrator(monkeypatch):
    """End-to-end: an injected real bundle flows through the E1 orchestrator's per-seed runner."""
    from scripts.run_cub200_frontier import run_seed
    b = _planted_bundle(n_classes=12, per_split=600)
    monkeypatch.setattr(rd, "load_real_bundle", lambda cfg, seed, **kw: b)
    pop = cf.load_real_population({"dataset": {"n_classes": 12}, "heads": {}}, seed=0)
    recs, diag = run_seed(pop, seed=0, rho_cal=0.95, rho_test_grid=[0.95, 0.7],
                          n_cal=400, n_test=400, frac_cal=0.5, alpha=0.1)
    assert len(recs) == 3 * 2 * 3  # schemes x rhos x score_fns
    assert all(0.0 <= r["worst_cov"] <= 1.0 for r in recs)
