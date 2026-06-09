"""Pre-committed verdict for E1 — the CUB-200 multiclass coverage--efficiency frontier.

THIS FILE ENCODES THE CLAIM AND ITS KILL-SWITCH BEFORE ANY NUMBERS ARE PRINTED.

Pre-committed claim (from the run spec, E1):
  The concept-score curve RELOCATES the frontier toward better WORST-GROUP coverage at matched mean
  set size. We DO NOT claim strict dominance to the ideal corner (residual non-invariance of the
  concept representation forbids reaching absolute worst-group coverage = 1-alpha).

The frontier is inherently 2-D (worst-group coverage vs mean set size), and Mondrian conditions on
the typicality group, so BOTH (a) feature+Mondrian and (c) concept+Mondrian hold worst-group
coverage ~1-alpha by construction; the representation difference then shows up as SET SIZE (the
contaminated feature score's atypical-group quantile inflates under shift -> large sets, while the
shortcut-invariant concept score stays tight). So "relocates the frontier toward the better corner"
is tested as a weak-PARETO improvement in the (worst-group coverage UP, mean set size DOWN) plane,
not on the coverage axis alone.

Operationalization (committed here, not tuned to the result):
  * Primary contrast is the MATCHED-scheme pair on the primary score (APS) at alpha=0.1:
        (c) concept-space score + Mondrian   vs   (a) feature-space score + Mondrian.
    (b) concept-space + pooled split is reported as support.
  * Seeds are the replication unit. Per shifted test rho (rho_test < rho_cal, where the spurious
    shift actually bites), and PAIRED across seeds:
        d_cov(rho)  = mean_seed[ worst_cov(c) - worst_cov(a) ]    (>0 => concept covers better)
        d_size(rho) = mean_seed[ mean_set_size(c) - mean_set_size(a) ]  (<0 => concept tighter)
  * A rho "RELOCATES" (toward the better corner) iff the concept method is a weak-Pareto improvement
    -- NOT materially worse on EITHER axis -- AND strictly better on at least one axis with that
    axis's across-seed 95% CI excluding 0:
        not-worse:   d_cov(rho) >= -COV_TOL  AND  size(c) <= (1+SIZE_TOL_REL)*size(a)
        strictly better on coverage:  d_cov mean > 0 AND CI-lo > 0
        OR strictly better on size:   d_size mean < 0 AND CI-hi < 0  (concept sets smaller)

  GREEN  (claim MET):  relocation holds at a MAJORITY of shifted test rho. Reported as a frontier
                       relocation toward the better (high-coverage / low-size) corner -- NOT strict
                       dominance to the ideal corner (no claim of absolute worst-group cov = 1-alpha).
  FALLBACK (kill-switch FIRED): otherwise. The paper then falls back to the narrower claim
                       "mechanism doesn't help; representation gives only relative-gap robustness,
                       not absolute coverage." Either way the numbers are reported.

The verdict consumes tidy per-(score, rho, scheme, seed) frontier metrics and is purely a function
of them -- it never re-tunes anything.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Scheme names (must match scripts/run_cub200_frontier.py)
FEAT_MOND = "feat+Mondrian"     # (a)
CPT_SPLIT = "cpt+split"         # (b)
CPT_MOND = "cpt+Mondrian"       # (c)

PRIMARY_SCORE = "APS"
SIZE_TOL_REL = 0.10             # concept may use up to +10% mean set size and still count "not-worse"
COV_TOL = 0.02                  # concept worst-group cov may dip up to 2 pts and still count "not-worse"


def _agg(values) -> dict:
    """mean / std / 95% CI (normal approx) over per-seed numbers; ignores NaN/None."""
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=np.float64)
    if v.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "lo": float("nan"),
                "hi": float("nan"), "n": 0}
    mean = float(v.mean())
    std = float(v.std(ddof=1)) if v.size > 1 else 0.0
    half = 1.96 * std / np.sqrt(v.size) if v.size > 1 else 0.0
    return {"mean": mean, "std": std, "lo": mean - half, "hi": mean + half, "n": int(v.size)}


@dataclass
class RhoRelocation:
    rho_test: float
    shifted: bool                     # rho_test < rho_cal
    d_cov: dict                       # paired across-seed worst-cov improvement (c - a)
    d_size: dict                      # paired across-seed mean-set-size diff (c - a)
    feat_worst_cov: dict              # (a) absolute worst-group coverage across seeds
    cpt_worst_cov: dict               # (c) absolute worst-group coverage across seeds
    feat_gap: dict                    # (a) coverage gap (max-min group cov)
    cpt_gap: dict                     # (c) coverage gap
    size_matched: bool                # concept mean size <= (1+tol) * feature mean size
    cov_not_worse: bool               # concept worst-cov not materially below feature
    better_cov: bool                  # strictly better on coverage (CI-backed)
    better_size: bool                 # strictly smaller sets (CI-backed)
    relocates: bool                   # the per-rho weak-Pareto relocation decision


@dataclass
class E1Verdict:
    label: str                        # "GREEN (frontier relocates)" or "FALLBACK (kill-switch)"
    green: bool
    score: str
    alpha: float
    rho_cal: float
    per_rho: list = field(default_factory=list)
    n_shifted: int = 0
    n_relocated: int = 0
    sweep_mean_d_cov: float = float("nan")
    sweep_mean_d_size: float = float("nan")
    rationale: str = ""
    fallback_claim: str = ("mechanism doesn't help; representation gives only relative-gap "
                           "robustness, not absolute coverage")


def _pivot(records, score, scheme, rho, key):
    """All per-seed values of ``key`` for one (score, scheme, rho) cell."""
    return [r[key] for r in records
            if r["score"] == score and r["scheme"] == scheme
            and abs(r["test_corr"] - rho) < 1e-9]


def e1_verdict(records: list, rho_cal: float, alpha: float = 0.1,
               score: str = PRIMARY_SCORE) -> E1Verdict:
    """Compute the pre-committed E1 relocation verdict from tidy frontier records.

    ``records``: list of dicts with keys test_corr, score, scheme, seed, worst_cov, mean_set_size,
    marg_cov, cov_gap (one per (score, scheme, rho, seed)). Returns an ``E1Verdict``.
    """
    rhos = sorted({r["test_corr"] for r in records if r["score"] == score})
    per_rho, d_cov_means, d_size_means = [], [], []
    n_shifted = n_reloc = 0
    for rho in rhos:
        feat_cov = _pivot(records, score, FEAT_MOND, rho, "worst_cov")
        cpt_cov = _pivot(records, score, CPT_MOND, rho, "worst_cov")
        feat_sz = _pivot(records, score, FEAT_MOND, rho, "mean_set_size")
        cpt_sz = _pivot(records, score, CPT_MOND, rho, "mean_set_size")
        feat_gap = _pivot(records, score, FEAT_MOND, rho, "cov_gap")
        cpt_gap = _pivot(records, score, CPT_MOND, rho, "cov_gap")
        # paired per-seed deltas (seeds aligned by sorted order; all schemes share seed set)
        m = min(len(feat_cov), len(cpt_cov))
        d_cov = _agg([cpt_cov[i] - feat_cov[i] for i in range(m)])
        ms = min(len(feat_sz), len(cpt_sz))
        d_size = _agg([cpt_sz[i] - feat_sz[i] for i in range(ms)])
        fa, ca = _agg(feat_sz), _agg(cpt_sz)
        size_matched = bool(np.isfinite(ca["mean"]) and np.isfinite(fa["mean"])
                            and ca["mean"] <= (1.0 + SIZE_TOL_REL) * fa["mean"])
        cov_not_worse = bool(d_cov["n"] > 0 and d_cov["mean"] >= -COV_TOL)
        better_cov = bool(d_cov["n"] > 0 and d_cov["mean"] > 0 and d_cov["lo"] > 0)
        better_size = bool(d_size["n"] > 0 and d_size["mean"] < 0 and d_size["hi"] < 0)
        shifted = rho < rho_cal - 1e-9
        # weak-Pareto relocation toward the better corner: not worse on either axis, strictly
        # better (CI-backed) on at least one.
        relocates = bool(shifted and cov_not_worse and size_matched
                         and (better_cov or better_size))
        if shifted:
            n_shifted += 1
            d_cov_means.append(d_cov["mean"])
            d_size_means.append(d_size["mean"])
            if relocates:
                n_reloc += 1
        per_rho.append(RhoRelocation(
            rho_test=float(rho), shifted=shifted, d_cov=d_cov, d_size=d_size,
            feat_worst_cov=_agg(feat_cov), cpt_worst_cov=_agg(cpt_cov),
            feat_gap=_agg(feat_gap), cpt_gap=_agg(cpt_gap),
            size_matched=size_matched, cov_not_worse=cov_not_worse,
            better_cov=better_cov, better_size=better_size, relocates=relocates))

    sweep_mean = float(np.mean(d_cov_means)) if d_cov_means else float("nan")
    sweep_mean_size = float(np.mean(d_size_means)) if d_size_means else float("nan")
    majority = n_shifted > 0 and n_reloc * 2 > n_shifted
    green = bool(majority)
    if green:
        label = "GREEN (frontier relocates)"
        rationale = (
            f"Concept+Mondrian relocates the worst-group/efficiency frontier vs feature+Mondrian at "
            f"{n_reloc}/{n_shifted} shifted rho (majority): at matched worst-group coverage "
            f"(Δcov {sweep_mean:+.3f}) the concept score's mean set size is {sweep_mean_size:+.2f} "
            f"smaller on average. Reported as a frontier relocation toward the better corner; NO "
            f"claim of reaching absolute worst-group coverage 1-alpha (residual non-invariance).")
    else:
        label = "FALLBACK (kill-switch)"
        rationale = (
            f"No consistent relocation: {n_reloc}/{n_shifted} shifted rho relocated (sweep mean "
            f"Δcov {sweep_mean:+.3f}, Δsize {sweep_mean_size:+.2f}). Falling back to the narrower "
            f"claim: \"{E1Verdict.fallback_claim}\".")
    return E1Verdict(label=label, green=green, score=score, alpha=alpha, rho_cal=rho_cal,
                     per_rho=per_rho, n_shifted=n_shifted, n_relocated=n_reloc,
                     sweep_mean_d_cov=sweep_mean, sweep_mean_d_size=sweep_mean_size,
                     rationale=rationale)
