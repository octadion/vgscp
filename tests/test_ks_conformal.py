"""Unit tests for the attribute-conditional-CP kill-switches (ks_conformal/).

These pin the two structural facts the whole study rests on, on a small fast testbed:
  - KS-1a: known-M, NON-differential deconvolution recovers the oracle per-true-group quantile.
  - KS-1b: with the SAME marginal M but score-correlated (differential) flips, deconvolution
    UNDER-covers the true minority -> the simple deconvolution is non-differential-only.
  - confusion matrices: M columns and W rows are valid conditional distributions.
"""
import numpy as np

from ks_conformal import common_utils as cu


def _small_pop():
    return cu.make_population(cu.TestbedConfig(n=4000, n_classes=20, seed=3))


def test_known_MW_is_valid_distribution():
    M, W = cu.known_MW(0.2, {0: 0.1, 1: 0.35})
    assert np.allclose(M.sum(axis=0), 1.0)      # P(A_hat|A) sums over a_hat
    assert np.allclose(W.sum(axis=1), 1.0)      # P(A|A_hat) sums over a


def test_ks1a_nondifferential_recovers_oracle():
    from conformal import group_robust as gr
    pop = _small_pop()
    A, s_true = pop["A_true"], pop["s_true"]["APS"]
    M, W = cu.known_MW(pop["cfg"].p_minor, {0: 0.1, 1: 0.35})
    Ahat = cu.make_ahat(A, s_true, {0: 0.1, 1: 0.35}, beta=0.0, seed=11)
    sp = next(cu.iter_splits(len(A), 1, (0.0, 0.5, 0.5)))
    cal = sp["cal"]
    qO = gr.mondrian_quantiles(s_true[cal], A[cal], 0.1)
    dr = cu.deconvolve_quantiles(s_true[cal], Ahat[cal], W, 0.1)
    # recovered minority quantile within a small tolerance of the oracle
    assert abs(dr.q[1] - qO[1]) <= 0.05
    assert np.isfinite(dr.cond_number)


def test_ks1b_differential_breaks_minority_coverage():
    from conformal import group_robust as gr
    pop = _small_pop()
    A, s_true, sa, yt = (pop["A_true"], pop["s_true"]["APS"], pop["scores_all"]["APS"],
                         pop["y_true"])
    M, W = cu.known_MW(pop["cfg"].p_minor, {0: 0.1, 1: 0.35})

    def minority_cov(beta):
        covs = []
        for sp in cu.iter_splits(len(A), 5, (0.0, 0.5, 0.5)):
            cal, test = sp["cal"], sp["test"]
            dr = cu.deconvolve_quantiles(s_true[cal], cu.make_ahat(A, s_true, {0: 0.1, 1: 0.35},
                                         beta=beta, seed=22)[cal], W, 0.1)
            mem = cu.sets_from_per_true_group_q(sa[test], A[test], {0: dr.q[0], 1: dr.q[1]})
            covs.append(cu.coverage_report(mem, yt[test], A[test], 0.1).per_group_cov[1])
        return float(np.mean(covs))

    cov_nondiff = minority_cov(0.0)
    cov_diff = minority_cov(16.0)
    assert cov_diff < cov_nondiff - 0.02       # differential noise breaks the recovery


# --------------------------------------------------------------------------------------
# KS-1d: score-bin conditional confusion correction
# --------------------------------------------------------------------------------------
def test_estimate_M_bins_columns_sum_to_one_and_track_bin():
    # construct data where the flip rate clearly rises across score bins; M_b columns must be
    # valid conditional distributions and the high-bin minority flip rate must exceed the low-bin's
    pop = cu.make_population(cu.TestbedConfig(n=6000, n_classes=20, seed=5))
    A, s_true = pop["A_true"], pop["s_true"]["APS"]
    Ahat = cu.make_ahat(A, s_true, {0: 0.1, 1: 0.35}, beta=16.0, seed=9)
    edges = cu.bin_edges_from_scores(s_true, 4)
    M_bins, gM, _ = cu.estimate_M_bins(A, Ahat, s_true, edges, min_count=1)
    for Mb in M_bins:
        assert np.allclose(Mb.sum(axis=0), 1.0)                 # P(A_hat|A,bin) columns sum to 1
    # minority (a=1) flip prob P(A_hat=0|A=1,bin) should grow from the low to the high score bin
    low = M_bins[0][0, 1]
    high = M_bins[-1][0, 1]
    assert high > low


def test_scorebin_matches_global_under_nondifferential():
    from conformal import group_robust as gr
    pop = _small_pop()
    A, s_true = pop["A_true"], pop["s_true"]["APS"]
    Ahat = cu.make_ahat(A, s_true, {0: 0.1, 1: 0.35}, beta=0.0, seed=13)  # non-differential
    sp = next(cu.iter_splits(len(A), 1, (0.34, 0.33, 0.33)))
    ho, cal = sp["holdout"], sp["cal"]
    qO = gr.mondrian_quantiles(s_true[cal], A[cal], 0.1)
    edges = cu.bin_edges_from_scores(s_true[cal], 4)
    M_bins, _, _ = cu.estimate_M_bins(A[ho], Ahat[ho], s_true[ho], edges)
    sb = cu.score_bin_deconvolve_quantiles(s_true[cal], Ahat[cal], M_bins, edges, 0.1)
    # under non-differential noise the score-bin quantile ~ the oracle/global quantile
    assert abs(sb.q[1] - qO[1]) <= 0.06


def test_scorebin_improves_over_global_under_differential_at_enough_bins():
    """At a stable bin count the score-bin correction recovers minority coverage that the global
    deconvolution loses under differential noise (the KS-1d positive finding)."""
    from conformal import group_robust as gr
    pop = cu.make_population(cu.TestbedConfig(seed=0))
    A, s_true, sa, yt = (pop["A_true"], pop["s_true"]["APS"], pop["scores_all"]["APS"],
                         pop["y_true"])
    Ahat = cu.make_ahat(A, s_true, {0: 0.1, 1: 0.35}, beta=16.0, seed=601)

    def mincov(builder):
        covs = []
        for sp in cu.iter_splits(len(A), 6, (0.34, 0.33, 0.33)):
            ho, cal, test = sp["holdout"], sp["cal"], sp["test"]
            q = builder(ho, cal)
            mem = cu.sets_from_per_true_group_q(sa[test], A[test], q)
            covs.append(cu.coverage_report(mem, yt[test], A[test], 0.1).per_group_cov[1])
        return float(np.mean(covs))

    def global_q(ho, cal):
        _, W, _ = cu.confusion_matrix_MW(A[ho], Ahat[ho])
        dr = cu.deconvolve_quantiles(s_true[cal], Ahat[cal], W, 0.1)
        return {0: dr.q[0], 1: dr.q[1]}

    def scorebin_q(ho, cal):
        edges = cu.bin_edges_from_scores(s_true[cal], 8)
        M_bins, _, _ = cu.estimate_M_bins(A[ho], Ahat[ho], s_true[ho], edges)
        return cu.score_bin_deconvolve_quantiles(s_true[cal], Ahat[cal], M_bins, edges, 0.1).q

    assert mincov(scorebin_q) > mincov(global_q) + 0.05


# --------------------------------------------------------------------------------------
# KS-1e: stabilized (adaptive + regularized) score-conditional correction
# --------------------------------------------------------------------------------------
def test_regularize_Mb_columns_and_shrinkage():
    counts = np.array([[40.0, 5.0], [10.0, 25.0]])          # counts[a_hat, a]
    gM = np.array([[0.8, 0.3], [0.2, 0.7]])
    M0 = cu.regularize_Mb_from_counts(counts, gM, lam=0.0, smooth=1.0)
    Mh = cu.regularize_Mb_from_counts(counts, gM, lam=0.5, smooth=1.0)
    assert np.allclose(M0.sum(axis=0), 1.0) and np.allclose(Mh.sum(axis=0), 1.0)
    # shrinkage pulls toward the global matrix
    assert np.all(np.abs(Mh - gM) <= np.abs(M0 - gM) + 1e-9)


def test_adaptive_bins_are_well_conditioned():
    pop = cu.make_population(cu.TestbedConfig(n=8000, n_classes=20, seed=7))
    A, s_true = pop["A_true"], pop["s_true"]["APS"]
    Ahat = cu.make_ahat(A, s_true, {0: 0.1, 1: 0.35}, beta=16.0, seed=3)
    gM, _, _ = cu.confusion_matrix_MW(A, Ahat)
    edges, M_bins, info = cu.adaptive_score_bins(A, Ahat, s_true, gM, lam=0.0,
                                                 start_bins=16, min_count_per_A=60, cond_max=20.0)
    # adaptive merging keeps every per-bin confusion well-conditioned (or collapses to 1 bin)
    assert info["n_bins"] >= 1
    assert info["max_cond"] <= 20.0 * 1.05 or info["n_bins"] == 1
    assert info["min_count_per_A"] >= 60 or info["n_bins"] == 1
    for Mb in M_bins:
        assert np.allclose(Mb.sum(axis=0), 1.0)


def test_ks1e_adaptive_beats_global_and_is_stable_under_differential():
    """KS-1e adaptive (λ=0) beats global-M under strong differential noise AND is far more stable
    across splits than the fragile fixed-3-bin KS-1d estimator (the KS-1e stabilization claim)."""
    pop = cu.make_population(cu.TestbedConfig(seed=0))
    A, s_true, sa, yt = (pop["A_true"], pop["s_true"]["APS"], pop["scores_all"]["APS"],
                         pop["y_true"])
    Ahat = cu.make_ahat(A, s_true, {0: 0.1, 1: 0.35}, beta=16.0, seed=601)

    def minority_covs(builder):
        out = []
        for sp in cu.iter_splits(len(A), 8, (0.34, 0.33, 0.33)):
            ho, cal, test = sp["holdout"], sp["cal"], sp["test"]
            q = builder(ho, cal)
            mem = cu.sets_from_per_true_group_q(sa[test], A[test], q)
            out.append(cu.coverage_report(mem, yt[test], A[test], 0.1).per_group_cov[1])
        return np.array(out)

    def global_q(ho, cal):
        _, W, _ = cu.confusion_matrix_MW(A[ho], Ahat[ho])
        d = cu.deconvolve_quantiles(s_true[cal], Ahat[cal], W, 0.1)
        return {0: d.q[0], 1: d.q[1]}

    def fixed3_q(ho, cal):
        edges = cu.bin_edges_from_scores(s_true[cal], 3)
        Mb, _, _ = cu.estimate_M_bins(A[ho], Ahat[ho], s_true[ho], edges)
        return cu.score_bin_deconvolve_quantiles(s_true[cal], Ahat[cal], Mb, edges, 0.1).q

    def adaptive_q(ho, cal):
        gM, _, _ = cu.confusion_matrix_MW(A[ho], Ahat[ho])
        edges, Mb, _ = cu.adaptive_score_bins(A[ho], Ahat[ho], s_true[ho], gM, lam=0.0,
                                              start_bins=16, min_count_per_A=60, cond_max=20.0)
        return cu.score_bin_deconvolve_quantiles(s_true[cal], Ahat[cal], Mb, edges, 0.1).q

    adaptive = minority_covs(adaptive_q)
    glob = minority_covs(global_q)
    fixed3 = minority_covs(fixed3_q)
    assert adaptive.mean() > glob.mean() + 0.05          # recovers coverage global-M loses
    assert adaptive.std() < fixed3.std()                 # and is more stable than fragile 3-bin


def test_tune_lambda_returns_valid_choice():
    pop = cu.make_population(cu.TestbedConfig(n=8000, n_classes=20, seed=1))
    A, s_true = pop["A_true"], pop["s_true"]["APS"]
    Ahat = cu.make_ahat(A, s_true, {0: 0.1, 1: 0.35}, beta=8.0, seed=2)
    gM, _, _ = cu.confusion_matrix_MW(A, Ahat)
    lam, table = cu.tune_lambda_holdout(A, Ahat, s_true, [0.0, 0.1, 0.5], 0.1, gM, seed=0,
                                        min_count_per_A=60)
    assert lam in (0.0, 0.1, 0.5)
    assert set(table) == {0.0, 0.1, 0.5}
