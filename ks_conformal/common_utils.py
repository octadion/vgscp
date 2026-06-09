"""Shared machinery for the attribute-conditional-CP kill-switch experiments (KS-0/1/2).

This module deliberately reuses the repository's existing conformal abstractions:
  - ``conformal.scores``          : THR / APS / RAPS nonconformity scores (identical across KS-*)
  - ``conformal.split_conformal`` : finite-sample conformal quantile, set building, coverage
  - ``conformal.group_robust``    : Mondrian (group-conditional) quantiles
  - ``eval.bootstrap.CI``         : the CI dataclass used elsewhere in the repo

It adds ONLY the pieces the kill-switches need on top of that:
  - a controlled multiclass *score generator* (the synthetic testbed) with a true binary
    attribute A whose minority group is genuinely harder, so the per-true-group conformal
    quantiles differ (Mondrian is non-trivial);
  - a noisy-attribute / confusion model with an exactly-controllable score<->flip correlation
    (so KS-1a non-differential vs KS-1b differential differ ONLY in that correlation, with an
    IDENTICAL marginal confusion matrix M by construction);
  - the confusion matrix M[a_hat,a] and the mixing/posterior matrix W[a|a_hat];
  - the M-deconvolution estimator (solve F_obs = W . G_true for the true-group CDFs);
  - the KS-2 conservative partial-identification assignment rule;
  - true-attribute-conditional coverage / set-size metrics and a >=10-split aggregation loop.

REAL-DATA PATH (CUB-200 multiclass / Waterbirds, frozen CLIP, logistic heads) is wired through
``load_real_population`` against the repo loaders, but it requires torch + open_clip + the
datasets, which are UNAVAILABLE in this environment (see RESULTS.md). It raises a clear blocked
error rather than fabricating anything. The synthetic testbed is what actually runs here and is
sufficient for the estimator-level verdicts (KS-1a/KS-1b); the data-dependent verdicts
(KS-0/KS-1c/KS-2) are reported with the synthetic mechanism demonstration AND flagged BLOCKED for
the real number.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# Windows consoles default to cp1252; our reports use ±, Δ, β, etc. Make stdout/stderr UTF-8 so
# console prints never crash (the report files are already written with encoding="utf-8").
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from conformal import scores as score_mod
from conformal import split_conformal as sc
from conformal import group_robust as gr
from eval.bootstrap import CI

ALPHAS = (0.10, 0.05)
DEFAULT_N_SPLITS = 12          # >= 10 random calibration/test splits (task requirement)


# ======================================================================================
# Synthetic multiclass testbed  --  the "score generator"
# ======================================================================================
@dataclass
class TestbedConfig:
    n: int = 14000             # population size (split into holdout / cal / test per split)
    n_classes: int = 50        # label space (set size is meaningful, unlike binary)
    p_minor: float = 0.20      # P(A_true = 1)  -- the scarce minority group
    margin_major: float = 4.0  # true-class logit margin for the easy (majority) group
    margin_minor: float = 1.1  # ...for the hard (minority) group -> larger scores, larger q
    margin_sd: float = 1.1
    seed: int = 0
    score: str = "APS"         # primary score; THR also computed for a sanity cross-check


def make_population(cfg: TestbedConfig) -> dict:
    """Generate a population of softmax predictions with a true binary attribute A.

    The minority group (A=1) is harder: its true-class logit margin is smaller, so its
    nonconformity scores are stochastically larger and its oracle (Mondrian-on-A_true) conformal
    quantile is larger. A single marginal quantile therefore *under-covers* the minority group --
    exactly the regime the two CP directions target. Returns a dict of per-sample arrays plus
    pre-computed (N,C) score matrices and true-label scores for THR and APS.
    """
    rng = np.random.default_rng(cfg.seed)
    n, C = cfg.n, cfg.n_classes
    A = (rng.random(n) < cfg.p_minor).astype(np.int64)
    y = rng.integers(0, C, n)
    margin_mu = np.where(A == 1, cfg.margin_minor, cfg.margin_major)
    margin = np.maximum(0.05, rng.normal(margin_mu, cfg.margin_sd))
    logits = rng.normal(0.0, 1.0, size=(n, C))
    logits[np.arange(n), y] += margin
    logits -= logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)
    probs = probs.astype(np.float32)
    y_pred = probs.argmax(axis=1)

    u = score_mod.draw_randomization(n, seed=cfg.seed + 7)  # APS randomization, fixed per sample
    sa_aps = score_mod.aps_scores_all(probs, u)
    sa_thr = score_mod.thr_scores_all(probs)
    pop = {
        "probs": probs, "y_true": y, "y_pred": y_pred,
        "correct": (y_pred == y).astype(np.int64), "A_true": A, "u": u,
        "scores_all": {"APS": sa_aps, "THR": sa_thr},
        "s_true": {"APS": score_mod.true_label_scores(sa_aps, y),
                   "THR": score_mod.true_label_scores(sa_thr, y)},
        "n_classes": C, "synthetic": True, "cfg": cfg,
    }
    return pop


def load_real_population(cfg: dict, seed: int) -> dict:
    """REAL CUB-200 multiclass population from frozen CLIP features + linear heads.

    Intended Colab/GPU path (NO large-model training -- a 200-way logistic head and a logistic
    attribute probe on FROZEN CLIP features only):
      1. Build the dataset bundle + per-image paths via the repo loaders.
      2. Encode every image once with the repo's ``models.concept_extractor_clip`` CLIP backbone
         to frozen features (cache to disk -- precompute-once, like the rest of the repo).
      3. Fit a multinomial logistic head feats->200 species on the train split; softmax probs are
         f's predictions used by the conformal scores.
      4. Pick a discrete attribute A (a CUB attribute group yielding a clear minority, or the
         Waterbirds land/water background) and fit a logistic probe feats->A; A_hat = hard pred.
      5. Hold out an attribute-labeled split to estimate M / W.

    This requires torch + open_clip + the CUB/Waterbirds image data, which are NOT available in
    this environment. We raise rather than fabricate. The code path above is the documented
    Colab recipe; see RESULTS.md.
    """
    raise RuntimeError(
        "load_real_population: BLOCKED in this environment. The real CUB-200/CLIP path needs "
        "torch (broken: c10.dll init failure in .venv), open_clip (not installed), and the "
        "CUB-200/Waterbirds image datasets (not present on disk). Run the synthetic testbed here; "
        "run real on Colab/GPU per the recipe in this function's docstring and RESULTS.md."
    )


# ======================================================================================
# Noisy predicted attribute A_hat + confusion / mixing matrices
# ======================================================================================
def _gumbel_topk_mask(scores: np.ndarray, k: int, beta: float, rng) -> np.ndarray:
    """Choose EXACTLY k indices, with selection probability increasing in ``scores`` as beta grows.

    beta = 0 -> uniform (selection independent of score = NON-differential).
    beta large -> deterministically the top-k highest-score points (maximally differential).
    Always selects exactly k (so the marginal flip rate -- hence the marginal confusion matrix --
    is identical across beta). Implemented via Gumbel-top-k sampling over beta*rank(score).
    """
    n = len(scores)
    if k <= 0:
        return np.zeros(n, dtype=bool)
    if k >= n:
        return np.ones(n, dtype=bool)
    order = np.argsort(scores, kind="mergesort")
    rank = np.empty(n)
    rank[order] = np.linspace(0.0, 1.0, n)            # within-group score percentile in [0,1]
    keys = beta * rank + rng.gumbel(size=n)
    idx = np.argpartition(-keys, k - 1)[:k]
    mask = np.zeros(n, dtype=bool)
    mask[idx] = True
    return mask


def make_ahat(A_true: np.ndarray, score_for_noise: np.ndarray, flip_rate: dict, beta: float,
              seed: int) -> np.ndarray:
    """Synthesise a noisy binary predicted attribute A_hat.

    ``flip_rate[a]`` = P(A_hat != a | A_true = a) -- the MARGINAL per-group flip rate, held FIXED
    while ``beta`` varies the score<->flip correlation. For each true group we flip exactly
    round(flip_rate[a] * n_a) members; WHICH ones is chosen by ``_gumbel_topk_mask`` (beta=0
    non-differential; large beta = flip the highest-score members = differential). Binary A, so a
    flip means A_hat = 1 - A_true.
    """
    rng = np.random.default_rng(seed)
    Ahat = A_true.copy()
    for a in (0, 1):
        idx = np.where(A_true == a)[0]
        if len(idx) == 0:
            continue
        k = int(round(flip_rate[a] * len(idx)))
        flip_local = _gumbel_topk_mask(score_for_noise[idx], k, beta, rng)
        Ahat[idx[flip_local]] = 1 - a
    return Ahat


def confusion_matrix_MW(A_true: np.ndarray, Ahat: np.ndarray, n_groups: int = 2,
                        smooth: float = 1e-9):
    """Estimate M[a_hat, a] = P(A_hat=a_hat | A=a) and W[a | a_hat] = P(A=a | A_hat=a_hat).

    M columns (over a_hat) sum to 1; W rows (over a, for fixed a_hat) sum to 1. ``smooth`` avoids
    division by zero for empty cells. Returns (M, W, counts).
    """
    counts = np.zeros((n_groups, n_groups), dtype=np.float64)  # counts[a_hat, a]
    for ah, a in zip(Ahat, A_true):
        counts[int(ah), int(a)] += 1
    col = counts.sum(axis=0, keepdims=True) + smooth          # over a_hat -> P(A=a) * N
    M = (counts + smooth) / col                                # P(A_hat | A)
    row_joint = counts + smooth
    W = row_joint / row_joint.sum(axis=1, keepdims=True)       # P(A | A_hat), rows index a_hat
    return M, W, counts


def known_MW(p_minor: float, flip_rate: dict):
    """Analytic (population) M and W from base rates + marginal flip rates -- the 'known M' used by
    KS-1a/KS-1b. M[a_hat,a]; W[a_hat, a] = P(A=a | A_hat=a_hat)."""
    pi = np.array([1.0 - p_minor, p_minor])                    # P(A=0), P(A=1)
    M = np.array([[1 - flip_rate[0], flip_rate[1]],            # rows a_hat, cols a
                  [flip_rate[0], 1 - flip_rate[1]]])
    joint = M * pi[None, :]                                     # joint[a_hat, a]
    W = joint / joint.sum(axis=1, keepdims=True)               # P(A | A_hat)
    return M, W


# ======================================================================================
# Confusion-matrix deconvolution  (KS-1)
# ======================================================================================
@dataclass
class DeconvResult:
    q: np.ndarray              # recovered per-true-group quantile, indexed by a
    cond_number: float         # condition number of W (inverse stability)
    g_recovered: np.ndarray    # (n_groups, n_grid) recovered CDFs (clipped + isotonic)
    grid: np.ndarray
    negative_mass: float       # how much probability mass the raw inverse pushed below 0 (a
                               # numerical-instability diagnostic; 0 = clean)


def deconvolve_quantiles(cal_scores: np.ndarray, cal_ahat: np.ndarray, W: np.ndarray,
                         alpha: float, n_groups: int = 2) -> DeconvResult:
    """Recover the per-true-group (1-alpha) quantile by deconvolving the A_hat-conditional score
    CDFs with the mixing matrix W.

    Model (exact iff noise is NON-differential): for each threshold s,
        F_obs[a_hat](s) = sum_a  W[a_hat, a] * G_true[a](s).
    Stacking over a_hat:  F = W . G  -> G = W^{-1} F. We solve on the grid of observed calibration
    scores, clip to [0,1] and enforce monotonicity (isotonic via cumulative max), then read off
    q_a = inf{ s : G_a(s) >= 1-alpha }.
    """
    grid = np.unique(cal_scores)
    if grid.size == 0:
        return DeconvResult(np.full(n_groups, np.inf), np.inf,
                            np.zeros((n_groups, 0)), grid, 0.0)
    F = np.zeros((n_groups, grid.size))                        # F[a_hat, grid]
    for ah in range(n_groups):
        s = cal_scores[cal_ahat == ah]
        if s.size == 0:
            F[ah] = 1.0                                         # no info -> degenerate, flagged via cond
            continue
        # empirical CDF of group-a_hat scores evaluated on the shared grid
        F[ah] = np.searchsorted(np.sort(s), grid, side="right") / s.size
    try:
        Winv = np.linalg.inv(W)
        cond = float(np.linalg.cond(W))
    except np.linalg.LinAlgError:
        Winv = np.linalg.pinv(W)
        cond = float("inf")
    G_raw = Winv @ F                                           # (n_groups, grid) recovered CDFs
    negative_mass = float(np.clip(-G_raw, 0, None).max(initial=0.0))
    G = np.clip(G_raw, 0.0, 1.0)
    G = np.maximum.accumulate(G, axis=1)                       # isotonic (nondecreasing) CDF
    q = np.full(n_groups, np.inf)
    target = 1.0 - alpha
    for a in range(n_groups):
        hit = np.where(G[a] >= target)[0]
        if hit.size:
            q[a] = float(grid[hit[0]])
    return DeconvResult(q, cond, G, grid, negative_mass)


# ======================================================================================
# Score-bin conditional confusion correction  (KS-1d)
# ======================================================================================
# The global deconvolution above assumes a single confusion matrix M = P(A_hat | A_true). Under
# DIFFERENTIAL noise the flip probability rises with the nonconformity score, so M is really
# score-dependent, M_s = P(A_hat | A_true, score=s), and the global solve is biased (KS-1b). KS-1d
# tests whether estimating a SEPARATE confusion per score bin, M_b = P(A_hat | A_true, score in b),
# repairs this. Within a narrow bin the mixing is approximately score-independent, so a per-bin
# count deconvolution recovers how much TRUE mass of each group lives in each bin; the per-true-group
# quantile is then read from the reconstructed (cross-bin) score distribution.
def bin_edges_from_scores(scores: np.ndarray, num_bins: int) -> np.ndarray:
    """Equal-count (quantile) bin edges over scores; outer edges are +-inf so every point lands."""
    qs = np.linspace(0.0, 1.0, num_bins + 1)
    edges = np.quantile(scores, qs)
    edges = np.unique(edges)
    if edges.size < 2:                                 # degenerate (all-equal scores)
        edges = np.array([scores.min(), scores.max() + 1e-9])
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def assign_bins(scores: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Bin index in [0, len(edges)-2] for each score, using the interior edges."""
    return np.clip(np.digitize(scores, edges[1:-1]), 0, len(edges) - 2)


def estimate_M_bins(A_true: np.ndarray, Ahat: np.ndarray, scores: np.ndarray, edges: np.ndarray,
                    n_groups: int = 2, smooth: float = 1e-6, min_count: int = 20,
                    global_M: Optional[np.ndarray] = None):
    """Per-bin confusion matrices M_b[a_hat, a] = P(A_hat=a_hat | A=a, score in bin b).

    Estimated on an attribute-labelled split. Columns (over a_hat) sum to 1. A bin/group cell with
    fewer than ``min_count`` true-group members falls back to the global column (scarce-minority
    bins are otherwise wildly noisy) -- this fallback is recorded so the report can flag how often
    the per-bin estimate was actually usable. Returns (M_bins, global_M, n_fallback_cols).
    """
    if global_M is None:
        global_M, _, _ = confusion_matrix_MW(A_true, Ahat, n_groups)
    bins = assign_bins(scores, edges)
    nb = len(edges) - 1
    M_bins, n_fallback = [], 0
    for b in range(nb):
        m = bins == b
        counts = np.zeros((n_groups, n_groups), dtype=np.float64)   # counts[a_hat, a]
        for ah, a in zip(Ahat[m], A_true[m]):
            counts[int(ah), int(a)] += 1
        col = counts.sum(axis=0)                                    # true-group count in this bin
        Mb = np.zeros((n_groups, n_groups))
        for a in range(n_groups):
            if col[a] >= min_count:
                Mb[:, a] = (counts[:, a] + smooth) / (col[a] + n_groups * smooth)
            else:
                Mb[:, a] = global_M[:, a]                            # fallback to global column
                n_fallback += 1
        M_bins.append(Mb)
    return M_bins, global_M, n_fallback


@dataclass
class ScoreBinDeconvResult:
    q: dict                    # recovered per-true-group quantile
    bin_true_counts: np.ndarray  # (n_bins, n_groups) recovered true-group mass per bin
    total_true: np.ndarray     # (n_groups,) recovered total mass per true group
    max_cond: float            # worst per-bin confusion condition number (stability diagnostic)


def score_bin_deconvolve_quantiles(cal_scores: np.ndarray, cal_ahat: np.ndarray, M_bins: list,
                                   edges: np.ndarray, alpha: float,
                                   n_groups: int = 2) -> ScoreBinDeconvResult:
    """Score-conditional deconvolution of the per-true-group (1-alpha) quantile.

    The local confusion M_b is treated as constant only WITHIN coarse bin b, but is applied at FINE
    (grid) resolution inside the bin: the reconstructed cumulative true-group count at score s is

        C_a(s) = sum_b  sum_{a_hat}  (M_b^{-1})[a, a_hat] * #{ bin=b, A_hat=a_hat, score <= s },

    which keeps the group-specific within-bin SHAPE (no pooled-shape approximation) while only the
    confusion is binned. G_a(s) = C_a(s)/C_a(inf); q_a = inf{ s : G_a(s) >= 1-alpha }. At one bin
    this reduces to the global count-deconvolution; with more bins the per-bin M_b tracks the
    score<->flip correlation, at the cost of noisier (and possibly near-singular) per-bin inverses.
    """
    bins = assign_bins(cal_scores, edges)
    nb = len(M_bins)
    grid = np.unique(cal_scores)
    Minv, max_cond = [], 1.0
    for Mb in M_bins:
        try:
            Minv.append(np.linalg.inv(Mb))
            max_cond = max(max_cond, float(np.linalg.cond(Mb)))
        except np.linalg.LinAlgError:
            Minv.append(np.linalg.pinv(Mb))
            max_cond = float("inf")
    bin_true_counts = np.zeros((nb, n_groups))
    if grid.size == 0:
        return ScoreBinDeconvResult({a: float("inf") for a in range(n_groups)},
                                    bin_true_counts, np.zeros(n_groups), max_cond)
    C = np.zeros((n_groups, grid.size))
    for b in range(nb):
        m = bins == b
        n_full = np.bincount(cal_ahat[m].astype(int), minlength=n_groups).astype(np.float64)
        bin_true_counts[b] = np.clip(Minv[b] @ n_full, 0.0, None)
        for ah in range(n_groups):
            s_ah = np.sort(cal_scores[m & (cal_ahat == ah)])
            cum = np.searchsorted(s_ah, grid, side="right").astype(np.float64)  # obs cum within bin
            for a in range(n_groups):
                C[a] += Minv[b][a, ah] * cum
    T = C[:, -1].copy()
    q = {a: float("inf") for a in range(n_groups)}
    for a in range(n_groups):
        if T[a] <= 0:
            continue
        G = np.clip(C[a] / T[a], 0.0, 1.0)
        G = np.maximum.accumulate(G)                       # isotonic CDF
        hit = np.where(G >= 1.0 - alpha)[0]
        if hit.size:
            q[a] = float(grid[hit[0]])
    return ScoreBinDeconvResult(q, bin_true_counts, T, max_cond)


# ======================================================================================
# Stabilized score-conditional confusion correction  (KS-1e)
# ======================================================================================
# KS-1d's fixed-bin score-conditioning recovers oracle coverage but is fragile: too-few bins
# under-resolve, and some bin counts yield near-singular per-bin confusions that blow up. KS-1e
# stabilizes it WITHOUT hand-picking the bin count, via (a) regularized/shrunk per-bin confusion
# matrices and (b) adaptive binning that merges neighbours until every bin has enough samples per
# true group AND a well-conditioned confusion.
def regularize_Mb_from_counts(counts: np.ndarray, global_M: np.ndarray, lam: float,
                              smooth: float, n_groups: int = 2) -> np.ndarray:
    """Build one regularized per-bin confusion M_b[a_hat, a] from a count matrix counts[a_hat, a].

    Laplace/Dirichlet smoothing (``smooth``) on each column, empty columns fall back to the global
    column, then shrink toward the global confusion: M_b = (1-lam) M_b + lam M_global. A convex
    combination of column-stochastic matrices is column-stochastic, so no renormalization needed.
    """
    col = counts.sum(axis=0)
    Mb = np.zeros((n_groups, n_groups))
    for a in range(n_groups):
        if col[a] > 0:
            Mb[:, a] = (counts[:, a] + smooth) / (col[a] + n_groups * smooth)
        else:
            Mb[:, a] = global_M[:, a]
    return (1.0 - lam) * Mb + lam * global_M


def _counts_for_mask(A_true, Ahat, mask, n_groups):
    counts = np.zeros((n_groups, n_groups), dtype=np.float64)
    for ah, a in zip(Ahat[mask].astype(int), A_true[mask].astype(int)):
        counts[ah, a] += 1
    return counts


def adaptive_score_bins(A_true: np.ndarray, Ahat: np.ndarray, scores: np.ndarray,
                        global_M: np.ndarray, lam: float = 0.1, start_bins: int = 16,
                        min_count_per_A: int = 50, cond_max: float = 20.0, smooth: float = 1.0,
                        n_groups: int = 2):
    """Greedily merge fine quantile bins until every bin is usable, then return (edges, M_bins,info).

    Start from ``start_bins`` equal-count bins. Repeatedly find a bin whose per-true-group count is
    below ``min_count_per_A`` OR whose regularized confusion M_b has condition number above
    ``cond_max``, and merge it with its (smaller) neighbour. Terminate when all bins pass or only one
    bin remains. M_bins are the regularized confusions (shrinkage ``lam`` toward ``global_M``). This
    auto-selects the bin count per split -- no manual choice. ``info`` records n_bins, max cond, and
    the smallest per-group bin count.
    """
    edges0 = bin_edges_from_scores(scores, start_bins)
    fine = assign_bins(scores, edges0)
    n_fine = len(edges0) - 1
    groups = [[i] for i in range(n_fine)]                 # contiguous fine-bin index groups

    def stats(g):
        counts = _counts_for_mask(A_true, Ahat, np.isin(fine, g), n_groups)
        Mb = regularize_Mb_from_counts(counts, global_M, lam, smooth, n_groups)
        try:
            cond = float(np.linalg.cond(Mb))
        except np.linalg.LinAlgError:
            cond = float("inf")
        return counts.sum(axis=0), Mb, cond

    safety = 0
    while len(groups) > 1 and safety < 4 * n_fine:
        safety += 1
        bad = None
        for j, g in enumerate(groups):
            col, _, cond = stats(g)
            if col.min() < min_count_per_A or not np.isfinite(cond) or cond > cond_max:
                bad = j
                break
        if bad is None:
            break
        # merge the bad bin with its smaller-count neighbour
        if bad == 0:
            nb = 1
        elif bad == len(groups) - 1:
            nb = bad - 1
        else:
            nb = bad - 1 if stats(groups[bad - 1])[0].min() <= stats(groups[bad + 1])[0].min() \
                else bad + 1
        lo, hi = min(bad, nb), max(bad, nb)
        groups[lo:hi + 1] = [sum(groups[lo:hi + 1], [])]
    # final edges from the surviving group boundaries (cumulative fine-bin counts)
    bounds, acc = [], 0
    for g in groups[:-1]:
        acc += len(g)
        bounds.append(edges0[acc])
    edges = np.array([-np.inf, *bounds, np.inf])
    M_bins, cols, max_cond = [], [], 1.0
    for g in groups:
        col, Mb, cond = stats(g)
        M_bins.append(Mb)
        cols.append(col.min())
        max_cond = max(max_cond, cond)
    info = {"n_bins": len(groups), "max_cond": float(max_cond), "min_count_per_A": float(min(cols))}
    return edges, M_bins, info


def tune_lambda_holdout(A_ho: np.ndarray, Ahat_ho: np.ndarray, score_ho: np.ndarray,
                        lambdas: list, alpha: float, global_M: np.ndarray, seed: int = 0,
                        n_groups: int = 2, n_folds: int = 3, **bin_kw):
    """Pick the shrinkage lambda using ONLY the holdout (A_true-labelled) -- never test coverage.

    K-fold within the holdout: for each fold, estimate the adaptive bins + regularized M_b on the
    other folds, deconvolve the held-out fold's A_hat scores, and compare the recovered per-true-
    group quantiles to that fold's ORACLE (A_true) quantiles. The criterion is the worst-group
    QUANTILE reconstruction error |q_deconv - q_oracle| (smoother / less noisy than a coverage
    step), averaged over folds. Ties are broken toward the SMALLEST lambda, because shrinkage is
    toward the (differential-noise-biased) global M, so less shrinkage is preferred when it is not
    clearly worse. Returns (lam_star, table) with table[lam] = mean CV error.
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(A_ho))
    folds = np.array_split(idx, n_folds)
    table = {}
    for lam in lambdas:
        errs = []
        for k in range(n_folds):
            val = folds[k]
            fit = np.concatenate([folds[j] for j in range(n_folds) if j != k])
            gM_fit, _, _ = confusion_matrix_MW(A_ho[fit], Ahat_ho[fit], n_groups)
            edges, M_bins, _ = adaptive_score_bins(A_ho[fit], Ahat_ho[fit], score_ho[fit], gM_fit,
                                                  lam=lam, n_groups=n_groups, **bin_kw)
            rec = score_bin_deconvolve_quantiles(score_ho[val], Ahat_ho[val], M_bins, edges, alpha,
                                                 n_groups)
            e = 0.0
            for a in range(n_groups):
                mv = A_ho[val] == a
                if mv.sum() == 0:
                    continue
                q_or = conformal_quantile_safe(score_ho[val][mv], alpha)
                if np.isfinite(q_or) and np.isfinite(rec.q[a]):
                    e = max(e, abs(rec.q[a] - q_or))
                elif rec.q[a] != q_or:
                    e = max(e, 1.0)               # one side hit +inf -> large penalty
            errs.append(e)
        table[lam] = float(np.mean(errs))
    best_err = min(table.values())
    tol = 0.01                                    # within 0.01 quantile units counts as a tie
    lam_star = min(l for l in lambdas if table[l] <= best_err + tol)   # smallest near-best lambda
    return lam_star, table


def conformal_quantile_safe(s: np.ndarray, alpha: float) -> float:
    """Thin wrapper around the repo conformal quantile that returns +inf for empty input."""
    return sc.conformal_quantile(np.asarray(s), alpha) if len(s) else float("inf")


# ======================================================================================
# Partial-identification assignment  (KS-2)
# ======================================================================================
def partial_id_thresholds(test_ahat: np.ndarray, clean_q: dict, W: np.ndarray,
                          prune: float = 0.05, n_groups: int = 2) -> np.ndarray:
    """Conservative partial-ID per-test-point threshold (the rule we commit to).

    Calibration gives CLEAN per-true-group quantiles ``clean_q[a]`` (computed with A_true). At test
    only A_hat is known, so the true group is uncertain. For a test point with A_hat = a_hat, the
    plausible true groups are { a : W[a_hat, a] >= prune }. We use the LARGEST clean_q among them
    (= cover the worst plausible true group). With prune=0 every group is plausible and the rule
    reduces to the global-max quantile (guaranteed but loose); a small prune drops negligible
    groups to recover efficiency. Returns a per-test-point threshold vector.

    NOTE on tightness: this is the conservative envelope rule the task names as the simple
    implementable baseline. A tighter LP-based discrete partial-identification bound would shrink
    sets where one group only *slightly* dominates; we implement the conservative rule and report
    the efficiency it achieves (the LP gap is noted, not implemented, to keep the kill-switch
    simple and correct).
    """
    q_by_group = np.array([clean_q.get(a, np.inf) for a in range(n_groups)])
    thr = np.empty(len(test_ahat))
    for i, ah in enumerate(test_ahat):
        plausible = np.where(W[int(ah)] >= prune)[0]
        if plausible.size == 0:
            plausible = np.arange(n_groups)                    # never prune everything
        thr[i] = q_by_group[plausible].max()
    return thr


# ======================================================================================
# Coverage / set-size metrics  --  ALWAYS true-attribute-conditional
# ======================================================================================
@dataclass
class CoverageReport:
    per_group_cov: dict        # a -> empirical P(Y in C(X) | A_true = a)
    worst_cov: float
    worst_gap: float           # worst_cov - (1-alpha); negative = under-coverage
    marginal_cov: float
    per_group_size: dict       # a -> mean set size
    overall_size: float


def coverage_report(membership: np.ndarray, y_true: np.ndarray, A_true: np.ndarray,
                    alpha: float) -> CoverageReport:
    cov = sc.covered(membership, y_true)
    sizes = sc.set_sizes(membership)
    groups = np.unique(A_true)
    per_cov = {int(a): float(cov[A_true == a].mean()) for a in groups}
    per_sz = {int(a): float(sizes[A_true == a].mean()) for a in groups}
    worst = float(min(per_cov.values()))
    return CoverageReport(per_cov, worst, worst - (1.0 - alpha), float(cov.mean()),
                          per_sz, float(sizes.mean()))


# ======================================================================================
# Build prediction sets for the various schemes (all use repo conformal primitives)
# ======================================================================================
def sets_from_per_true_group_q(test_scores_all: np.ndarray, test_A_true: np.ndarray,
                               q_by_group: dict) -> np.ndarray:
    """Membership using each test point's TRUE-group quantile (oracle/deconv evaluation)."""
    qvec = np.array([q_by_group.get(int(a), float("inf")) for a in test_A_true])
    return test_scores_all <= qvec[:, None]


def sets_from_thresholds(test_scores_all: np.ndarray, thr_vec: np.ndarray) -> np.ndarray:
    return test_scores_all <= thr_vec[:, None]


# ======================================================================================
# Split aggregation (>= 10 random calibration/test splits; mean +/- std and 95% CI)
# ======================================================================================
def iter_splits(n: int, n_splits: int, fracs: tuple, seed0: int = 0):
    """Yield dicts of disjoint index arrays {'holdout','cal','test'} for each split.

    ``fracs`` = (holdout, cal, test) fractions (need not sum to 1; remainder unused). A fresh
    permutation per split, seeded by seed0 + split index, so every split is reproducible.
    """
    fr_hold, fr_cal, fr_test = fracs
    for s in range(n_splits):
        rng = np.random.default_rng(seed0 + s)
        perm = rng.permutation(n)
        n_h = int(round(fr_hold * n))
        n_c = int(round(fr_cal * n))
        n_t = int(round(fr_test * n))
        yield {"holdout": perm[:n_h], "cal": perm[n_h:n_h + n_c],
               "test": perm[n_h + n_c:n_h + n_c + n_t], "split": s}


def agg(values: np.ndarray) -> dict:
    """mean / std / 95% CI (normal approx across splits) for a 1-D array of per-split numbers."""
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=np.float64)
    if v.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "lo": float("nan"),
                "hi": float("nan"), "n": 0}
    mean = float(v.mean())
    std = float(v.std(ddof=1)) if v.size > 1 else 0.0
    half = 1.96 * std / np.sqrt(v.size) if v.size > 1 else 0.0
    return {"mean": mean, "std": std, "lo": mean - half, "hi": mean + half, "n": int(v.size)}


def fmt(a: dict, pct: bool = False) -> str:
    if a["n"] == 0:
        return "--"
    m, s = (a["mean"], a["std"])
    if pct:
        return f"{100*m:.1f}±{100*s:.1f}"
    return f"{m:.3f}±{s:.3f}"
