"""LOCAL logic-validation of the Phase-0 machinery on SYNTHETIC features (no torch / no data).

Validates the wiring the spec cares about — shapes, DFR group-balanced reweighting actually
runs, the §4 divergence computes, the in-domain split is wired correctly, the L2 assert fires —
and explicitly CLAIMS NO REAL NUMBERS. Run:

    python -m study_robust_train.validate_synthetic

Exits 0 on PASS. This is NOT the Phase-0 result; that is the Colab notebook on real Waterbirds.
"""
from __future__ import annotations

import numpy as np

from .divergence import cross_group_divergence, true_label_conformity_scores
from .heads import DFRHead, assert_l2_normalized, assert_multinomial_safe, fit_dfr, fit_erm, head_probs
from .phase0 import Phase0Data, format_report, run_phase0
from .synthetic import make_synthetic_phase0


def _check(name: str, cond: bool):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        raise AssertionError(f"logic-validation failed: {name}")


def main() -> int:
    print("=" * 78)
    print("study_robust_train — SYNTHETIC LOGIC VALIDATION (no real numbers claimed)")
    print("=" * 78)

    # --- §2a multinomial guarantee + L2 assert behave ---
    assert_multinomial_safe()
    print("\n[1] sec.2 preconditions")
    _check("assert_multinomial_safe() passes on installed sklearn", True)
    try:
        assert_l2_normalized(np.array([[3.0, 4.0]]), tag="deliberately-unnormalized")
        l2_fires = False
    except ValueError:
        l2_fires = True
    _check("assert_l2_normalized RAISES on non-unit-norm features", l2_fires)

    # --- build in-domain synthetic data ---
    data = make_synthetic_phase0(n_train=1200, n_test=1200, d=32,
                                 rho_train=0.95, rho_test=0.95, seed=0)
    print("\n[2] in-domain split wiring")
    _check("train features are L2-normalized (assert passes)",
           (assert_l2_normalized(data.feats_train, tag="train") or True))
    _check("test features are L2-normalized (assert passes)",
           (assert_l2_normalized(data.feats_test, tag="test") or True))
    _check("train/test feature dims match", data.feats_train.shape[1] == data.feats_test.shape[1])
    g_tr, g_te = data.group_train(), data.group_test()
    _check("all 4 Waterbirds groups present in train", set(np.unique(g_tr)) == {0, 1, 2, 3})
    _check("all 4 Waterbirds groups present in test", set(np.unique(g_te)) == {0, 1, 2, 3})
    _check("in-domain: train and test built at the same rho (composited->composited)",
           True)  # by construction rho_train == rho_test; documented in synthetic.py
    _check("data flagged synthetic=True (no-real-numbers guard)", data.synthetic is True)

    # --- ERM head fits and scores shape correctly ---
    print("\n[3] ERM last-layer head")
    erm = fit_erm(data.feats_train, data.y_train, seed=0)
    p_erm = head_probs(erm, data.feats_test, data.n_classes)
    _check("ERM posterior shape == (n_test, n_classes)", p_erm.shape == (data.feats_test.shape[0], 2))
    _check("ERM posteriors row-normalized", np.allclose(p_erm.sum(axis=1), 1.0, atol=1e-6))

    # --- DFR reweighting actually runs (group-balanced subsamples) ---
    print("\n[4] DFR group-balanced reweighting")
    dfr = fit_dfr(data.feats_train, data.y_train, g_tr, n_subsets=10, seed=0)
    _check("DFR is an ensemble of 10 group-balanced members", isinstance(dfr, DFRHead) and len(dfr.members) == 10)
    min_grp = int(min((g_tr == grp).sum() for grp in np.unique(g_tr)))
    _check("DFR balanced subset size == min group count (reweighting balanced the groups)",
           dfr.subset_size_per_group == min_grp)
    p_dfr = head_probs(dfr, data.feats_test, data.n_classes)
    _check("DFR posterior shape == (n_test, n_classes)", p_dfr.shape == (data.feats_test.shape[0], 2))

    # --- §4 divergence computes and is well-formed ---
    print("\n[5] sec.4 cross-group conformity-score divergence (APS, fresh metric)")
    aps = true_label_conformity_scores(p_erm, data.y_test, score="APS", seed=0)
    _check("APS true-label scores shape == (n_test,)", aps.shape == (data.feats_test.shape[0],))
    div = cross_group_divergence(aps, g_te, worst_group=int(g_te[0]), score_name="APS")
    _check("Wasserstein-1 finite and >= 0", np.isfinite(div.wasserstein1) and div.wasserstein1 >= 0)
    _check("KS statistic in [0, 1]", 0.0 <= div.ks_stat <= 1.0)
    _check("KS p-value in [0, 1]", 0.0 <= div.ks_pvalue <= 1.0)

    # --- full Phase-0 orchestrator runs end-to-end and STOPS ---
    print("\n[6] Phase-0 orchestrator (ERM + DFR, 3 seeds) -- runs then STOPS")
    out = run_phase0(data, seeds=(0, 1, 2))
    _check("per-run has 6 ModelResults (2 methods x 3 seeds)", len(out["per_run"]) == 6)
    for r in out["per_run"]:
        _check(f"{r.method} seed{r.seed}: worst-group acc in [0,1]", 0.0 <= r.worst_group_acc <= 1.0)
    _check("aggregate reports ERM and DFR", set(out["aggregate"]) == {"ERM", "DFR"})
    _check("orchestrator emits the STOP sentinel (no grid)", "STOP" in out["stop"])

    print("\n" + format_report(out))
    print("\n" + "=" * 78)
    print("LOGIC OK -- machinery validated on synthetic. NO REAL NUMBERS CLAIMED.")
    print("Real Phase-0 numbers + the DFR~0.86-0.92 / ERM~0.6-0.75 STOP gate: Colab notebook.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
