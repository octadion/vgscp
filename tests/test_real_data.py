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


def _l2(x):
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


def _inject_corrected_seams(monkeypatch, b):
    """Monkeypatch the §2a/§2b CLIP+data seams so the CORRECTED load_real_population is exercised on
    injected arrays (no open_clip / CUB images). Features must be L2-normalized (the §2a guard)."""
    import data.cub_attributes as cub
    for sp in b.features:
        b.features[sp] = _l2(b.features[sp])
    monkeypatch.setattr(rd, "load_real_bundle", lambda cfg, seed, **kw: b)
    monkeypatch.setattr(cub, "prepare_cub", lambda *a, **kw: "DUMMY_CUB_ROOT")
    # clean-CUB encode -> reuse the (L2-normalized) injected features for the matching split
    def fake_clip(paths, *a, **kw):
        sp = (kw.get("tag", "") or "").replace("cleancub_", "")
        return b.features[sp]
    monkeypatch.setattr(rd, "clip_image_features", fake_clip)


def test_load_real_population_from_injected_bundle(monkeypatch):
    """CORRECTED §2a/§2b path: clean-CUB feature head + image-derived concept, from an injected
    bundle. Uses gt_attrs_leaky here so the concept head is the pure attrs->species probe (the CBM
    predicted path is unit-tested separately in test_cbm_attribute_probe)."""
    b = _planted_bundle(n_classes=10)
    _inject_corrected_seams(monkeypatch, b)
    cfg = {"dataset": {"n_classes": 10}, "heads": {}, "cub": {"root": "x"}, "clip": {},
           "concept_source": "gt_attrs_leaky"}
    pop = cf.load_real_population(cfg, seed=0)
    pool_n = sum(len(b.species[s]) for s in ("d_learn", "d_cal", "d_test"))
    assert pop["species"].shape == (pool_n,)
    assert pop["feat_probs"].shape == (pool_n, 10)
    assert pop["cpt_probs"].shape == (pool_n, 10)
    assert not pop["synthetic"]
    assert pop["concept_source"] == "gt_attrs_leaky"
    assert "feat_top1_cleancub" in pop          # §2a sanity number is reported
    expect = cf.typicality_group(pop["place"], pop["species_type"])
    assert np.array_equal(pop["typicality"], expect)
    assert pop["feat_top1"] > 0.4 and pop["cpt_top1"] > 0.4
    assert pop["info"]["pool_n_atypical"] >= 0


def test_cbm_attribute_probe(monkeypatch):
    """§2b CBM: the image-derived predicted-concept path assembles a valid population (predicted
    attributes are noisier than GT, so we assert validity + above-chance, not a high bar)."""
    b = _planted_bundle(n_classes=8, per_split=500)
    _inject_corrected_seams(monkeypatch, b)
    cfg = {"dataset": {"n_classes": 8}, "heads": {}, "cub": {"root": "x"}, "clip": {},
           "concept_source": "cbm"}
    pop = cf.load_real_population(cfg, seed=0)
    assert pop["concept_source"] == "cbm"
    assert pop["cpt_probs"].shape[1] == 8
    assert np.allclose(pop["cpt_probs"].sum(1), 1.0, atol=1e-5)
    assert pop["feat_top1"] > 0.4               # clean-CUB-trained feature head still competent


def test_label_alignment_guard():
    """§1.1: the label/path desync guard fires on a mismatch, passes on matched basenames."""
    rd.assert_label_alignment(["a/1.jpg", "b/2.jpg"], ["x/1.jpg", "y/2.jpg"], tag="ok")  # basenames match
    with pytest.raises(ValueError):
        rd.assert_label_alignment(["a/1.jpg", "b/2.jpg"], ["a/1.jpg", "b/9.jpg"], tag="bad")


def test_known_good_baseline_gate():
    """§1.2: the baseline passes on separable features, fails (passes=False) on unlearnable noise."""
    rng = np.random.default_rng(0)
    C, d = 10, 32
    cent = rng.normal(0, 3, (C, d))
    ytr, yte = rng.integers(0, C, 800), rng.integers(0, C, 400)
    Xtr = _l2(cent[ytr] + rng.normal(0, 1, (800, d)))
    Xte = _l2(cent[yte] + rng.normal(0, 1, (400, d)))
    assert rd.clip_linear_probe_baseline(Xtr, ytr, Xte, yte)["passes"]
    # pure noise -> below the 0.55 floor
    Nt, Ne = _l2(rng.normal(0, 1, (800, d))), _l2(rng.normal(0, 1, (400, d)))
    assert not rd.clip_linear_probe_baseline(Nt, ytr, Ne, yte)["passes"]


def test_gate_halts_when_features_uninformative(monkeypatch):
    """§1.4: assemble_e1_population HALTS (FeatureHeadGateError) when clean features can't ID species
    (the v2 failure mode), emitting no population for a verdict to be computed on."""
    b = _planted_bundle(n_classes=10)
    # overwrite features with pure noise -> the head cannot clear the gate
    rng = np.random.default_rng(1)
    for sp in b.features:
        b.features[sp] = _l2(rng.normal(0, 1, b.features[sp].shape).astype(np.float32))
    _inject_corrected_seams(monkeypatch, b)
    cfg = {"dataset": {"n_classes": 10}, "heads": {}, "cub": {"root": "x"}, "clip": {},
           "concept_source": "cbm"}
    with pytest.raises(rd.FeatureHeadGateError):
        cf.load_real_population(cfg, seed=0)


def test_load_real_population_feeds_orchestrator(monkeypatch):
    """End-to-end: an injected bundle flows through the CORRECTED unified-2x2 per-seed runner with
    the COMPLETE 2x2 (4 cells)."""
    from scripts.run_unified_2x2 import matched_class_subset, run_seed
    b = _planted_bundle(n_classes=12, per_split=600)
    _inject_corrected_seams(monkeypatch, b)
    pop = cf.load_real_population({"dataset": {"n_classes": 12}, "heads": {}, "cub": {"root": "x"},
                                  "clip": {}, "concept_source": "cbm"}, seed=0)
    matched, _ = matched_class_subset(pop)
    recs, diag = run_seed(pop, seed=0, rho_cal=0.95, rho_test_grid=[0.95, 0.7],
                          n_cal=400, n_test=400, frac_cal=0.5, alpha=0.1,
                          matched_classes=matched)
    assert len(recs) == 4 * 2 * 3  # CELLS x rhos x score_fns (the complete 2x2)
    assert all(0.0 <= r["worst_cov"] <= 1.0 for r in recs)
