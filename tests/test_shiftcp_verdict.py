"""Unit tests for the pre-committed shift-CP GREEN/RED verdict.

Constructs synthetic per-method per-sample arrays to exercise each branch:
  - GREEN: STD-f under-covers worst group; STD-cpt holds WG coverage AND beats all comparators
    on worst-group set size at matched coverage.
  - RED "valid but incremental": STD-cpt valid but only TIES a Mondrian variant on set size.
  - RED "regime not triggered": the shift never pushes STD-f below 1-alpha.
  - RED "invalid": STD-cpt fails worst-group coverage under the shift.
"""
import numpy as np

from eval.shiftcp_verdict import (MOND_CPT, MOND_F, ROBUST, STD_CPT, STD_F,
                                  shiftcp_verdict)

ALPHA = 0.1
TARGET = 1.0 - ALPHA


def _groups(per_group_n=600):
    """Balanced 4-group test layout; returns (group_array, {g: index_mask})."""
    group = np.repeat([0, 1, 2, 3], per_group_n).astype(int)
    masks = {g: group == g for g in (0, 1, 2, 3)}
    return group, masks


def _covered(group, masks, cov_by_group):
    """Deterministic covered array hitting the per-group coverage exactly."""
    cov = np.zeros(len(group), dtype=int)
    for g, c in cov_by_group.items():
        idx = np.where(masks[g])[0]
        k = int(round(c * len(idx)))
        cov[idx[:k]] = 1
    return cov


def _sized(group, masks, size_by_group):
    size = np.ones(len(group), dtype=int)
    for g, s in size_by_group.items():
        size[masks[g]] = s
    return size


def _method(group, masks, cov_by_group, size_by_group):
    return {"covered": _covered(group, masks, cov_by_group),
            "size": _sized(group, masks, size_by_group),
            "group": group}


def _all_groups(value):
    return {0: value, 1: value, 2: value, 3: value}


def test_green_branch():
    group, masks = _groups()
    # STD-f under-covers group 1 (the minority worst group); g* := 1
    stdf = _method(group, masks, {0: 0.97, 1: 0.70, 2: 0.96, 3: 0.97}, _all_groups(2))
    # STD-cpt holds coverage everywhere and has SMALL sets on g*
    stdcpt = _method(group, masks, _all_groups(0.96), {0: 1, 1: 1, 2: 1, 3: 1})
    # comparators: valid worst-group coverage but LARGER sets on g*=1
    mond_f = _method(group, masks, _all_groups(0.95), _all_groups(3))
    mond_cpt = _method(group, masks, _all_groups(0.95), _all_groups(3))
    robust = _method(group, masks, _all_groups(0.99), _all_groups(4))
    methods = {STD_F: stdf, STD_CPT: stdcpt, MOND_F: mond_f, MOND_CPT: mond_cpt, ROBUST: robust}

    v = shiftcp_verdict([(0.25, methods)], alpha=ALPHA, rho_cal=0.95, n_resamples=300, seed=0)
    assert v.label == "GREEN"
    assert 0.25 in v.green_rhos
    rv = v.per_rho[0]
    assert rv.stdf_undercovers and rv.ref_group == 1 and rv.stdcpt_wg_valid
    assert all(c.beaten for c in rv.size_comparisons)


def test_red_valid_but_incremental_vs_mondrian():
    group, masks = _groups()
    stdf = _method(group, masks, {0: 0.97, 1: 0.70, 2: 0.96, 3: 0.97}, _all_groups(2))
    stdcpt = _method(group, masks, _all_groups(0.96), {0: 1, 1: 2, 2: 1, 3: 1})  # size 2 on g*
    mond_f = _method(group, masks, _all_groups(0.95), _all_groups(2))            # TIES cpt on g*
    mond_cpt = _method(group, masks, _all_groups(0.95), _all_groups(3))
    robust = _method(group, masks, _all_groups(0.99), _all_groups(4))
    methods = {STD_F: stdf, STD_CPT: stdcpt, MOND_F: mond_f, MOND_CPT: mond_cpt, ROBUST: robust}

    v = shiftcp_verdict([(0.25, methods)], alpha=ALPHA, rho_cal=0.95, n_resamples=300, seed=0)
    assert v.label == "RED"
    assert "incremental" in v.rationale.lower()
    # STD-cpt is valid, but the Mondrian-f size tie means not all comparators are beaten
    rv = v.per_rho[0]
    assert rv.stdcpt_wg_valid
    assert not all(c.beaten for c in rv.size_comparisons)


def test_red_regime_not_triggered():
    group, masks = _groups()
    # STD-f holds worst-group coverage everywhere -> no trigger
    stdf = _method(group, masks, _all_groups(0.95), _all_groups(2))
    stdcpt = _method(group, masks, _all_groups(0.96), _all_groups(1))
    mond_f = _method(group, masks, _all_groups(0.95), _all_groups(2))
    mond_cpt = _method(group, masks, _all_groups(0.95), _all_groups(2))
    robust = _method(group, masks, _all_groups(0.99), _all_groups(3))
    methods = {STD_F: stdf, STD_CPT: stdcpt, MOND_F: mond_f, MOND_CPT: mond_cpt, ROBUST: robust}

    v = shiftcp_verdict([(0.95, methods)], alpha=ALPHA, rho_cal=0.95, n_resamples=300, seed=0)
    assert v.label == "RED"
    assert "regime not triggered" in v.rationale.lower()


def test_red_stdcpt_invalid_under_shift():
    group, masks = _groups()
    stdf = _method(group, masks, {0: 0.97, 1: 0.70, 2: 0.96, 3: 0.97}, _all_groups(2))
    # STD-cpt ALSO under-covers the worst group -> not valid
    stdcpt = _method(group, masks, {0: 0.96, 1: 0.70, 2: 0.96, 3: 0.96}, _all_groups(1))
    mond_f = _method(group, masks, _all_groups(0.95), _all_groups(3))
    mond_cpt = _method(group, masks, _all_groups(0.95), _all_groups(3))
    robust = _method(group, masks, _all_groups(0.99), _all_groups(4))
    methods = {STD_F: stdf, STD_CPT: stdcpt, MOND_F: mond_f, MOND_CPT: mond_cpt, ROBUST: robust}

    v = shiftcp_verdict([(0.25, methods)], alpha=ALPHA, rho_cal=0.95, n_resamples=300, seed=0)
    assert v.label == "RED"
    assert not v.per_rho[0].stdcpt_wg_valid
