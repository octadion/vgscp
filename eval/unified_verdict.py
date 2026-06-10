"""Pre-committed verdict for the CORRECTED unified 2x2 run (study paper, v2).

THIS FILE ENCODES THE RE-PRE-REGISTERED CLAIM AND ITS KILL-SWITCH BEFORE ANY NUMBERS ARE PRINTED.
It REPLACES the E1 relocation verdict (eval/e1_verdict.py), which tested the now-FALSIFIED
"representation, not mechanism" framing. The falsification stands and is reported; this file tests
the corrected, group-free-substitution claim (run spec v2 §4).

----------------------------------------------------------------------------------------------------
Primary claim (GROUP-FREE SUBSTITUTION):
  When the spurious group is unknown (so Mondrian cannot be used), an invariant CONCEPT score
  recovers MOST of the worst-group coverage-gap reduction that group-conditional calibration
  (Mondrian, a mechanism) would provide.

  Metric, per test correlation rho, SAME score function (APS primary):
        R(rho) = (gap[feat,split] - gap[cpt,split]) / (gap[feat,split] - gap[feat,Mondrian])
  where gap = max-min group coverage (the shift-robustness axis). Numerator = how much the
  concept REPRESENTATION closes the gap while still using a group-free (pooled split) scheme;
  denominator = how much the Mondrian MECHANISM closes it on the feature representation (the
  reference lever). R in [0,1] => the representation substitute recovers that fraction of the
  mechanism's benefit; R>=1 => it matches or beats the mechanism.

  v3 HARDENED criterion (tightened after a fragile, score-function-dependent v2 pass; can only make
  GREEN HARDER, never easier). PER-SCORE GREEN iff:
    * R(rho) >= R_MIN (0.5) with the paired across-seed 95% CI for [gap(feat,split)-gap(cpt,split)]
      excluding 0 at a MAJORITY of shifted rho (rho_test < rho_cal), AND
    * the SAME holds at the LARGEST shift (rho_test = 0.5) -- v2 passed only by failing at the two
      largest shifts, exactly where the spurious effect bites hardest; this closes that hole.
  COMBINED (headline) GREEN iff the per-score verdict is GREEN for >= 2 of 3 score functions
  {APS, RAPS, THR} (see ``combined_decision``) -- robustness across score functions, not APS alone.

  KILL-SWITCH (be willing to kill the NEW claim too): otherwise the paper falls back to
    "group-conditional calibration is the binding lever for worst-group coverage; an invariant
     representation does not robustly substitute for it under the strengthened criterion."
  Report it either way. The verdict is evaluated ONLY on a feature head that cleared the §1.4 gate.

Also reported plainly (NOT suppressed -- it is part of the honest 2x2):
  * the MECHANISM MAIN EFFECT = Mondrian's gap reduction on each representation
    (gap[feat,split]-gap[feat,Mondrian] and gap[cpt,split]-gap[cpt,Mondrian]). Expected large.

Secondary claim (EFFICIENCY) is evaluated separately in the orchestrator with the §2e accuracy
control; it is NOT part of this go/no-go (kept honest + caveated there).

The verdict consumes tidy per-(score, representation, scheme, rho, seed) metrics and is PURELY a
function of them -- it never re-tunes anything.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Cell labels (representation, scheme) -- must match scripts/run_unified_2x2.py
FEAT_SPLIT = ("feature", "split")
FEAT_MOND = ("feature", "Mondrian")
CPT_SPLIT = ("concept", "split")
CPT_MOND = ("concept", "Mondrian")

PRIMARY_SCORE = "APS"
R_MIN = 0.5                       # recovered-fraction threshold (pre-committed)


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


def _cell(records, score, rep, scheme, rho, key):
    """Per-seed values of ``key`` for one (score, representation, scheme, rho) cell, ordered by seed."""
    rows = [r for r in records
            if r["score"] == score and r["representation"] == rep and r["scheme"] == scheme
            and abs(r["test_corr"] - rho) < 1e-9]
    rows.sort(key=lambda r: r["seed"])
    return [r[key] for r in rows]


@dataclass
class RhoSubstitution:
    rho_test: float
    shifted: bool
    gap_feat_split: dict             # gap[feat,split]   across seeds
    gap_cpt_split: dict              # gap[cpt,split]    across seeds
    gap_feat_mond: dict              # gap[feat,Mondrian] across seeds
    gap_cpt_mond: dict               # gap[cpt,Mondrian]  across seeds
    d_repr: dict                     # paired gap[feat,split]-gap[cpt,split] across seeds (the test)
    mech_feat: dict                  # paired gap[feat,split]-gap[feat,Mondrian] (mechanism main effect)
    mech_cpt: dict                   # paired gap[cpt,split]-gap[cpt,Mondrian]
    R: float                         # recovered fraction (ratio of mean gap reductions)
    repr_significant: bool           # paired CI for d_repr excludes 0
    recovers: bool                   # R >= R_MIN AND repr_significant (per-rho GREEN ingredient)
    is_hardest: bool = False         # the largest shift (min shifted rho, e.g. rho_test=0.5)


@dataclass
class UnifiedVerdict:
    label: str
    green: bool
    score: str
    alpha: float
    rho_cal: float
    per_rho: list = field(default_factory=list)
    n_shifted: int = 0
    n_recovered: int = 0
    majority: bool = False                       # R>=R_MIN at a majority of shifted rho
    hardest_recovers: bool = False               # R>=R_MIN at the LARGEST shift (rho_test=0.5) [v3]
    sweep_mean_R: float = float("nan")
    sweep_mean_mech_feat: float = float("nan")   # honest mechanism main effect (feature)
    sweep_mean_mech_cpt: float = float("nan")
    rationale: str = ""
    fallback_claim: str = ("group-conditional calibration is the binding lever for worst-group "
                           "coverage; an invariant representation does not robustly substitute for "
                           "it under the strengthened criterion")


def unified_verdict(records: list, rho_cal: float, alpha: float = 0.1,
                    score: str = PRIMARY_SCORE) -> UnifiedVerdict:
    """Compute the pre-committed group-free-substitution verdict from tidy 2x2 records.

    ``records``: list of dicts with keys test_corr, score, representation, scheme, seed, worst_cov,
    cov_gap, marg_cov, mean_set_size. Returns a ``UnifiedVerdict``.
    """
    rhos = sorted({r["test_corr"] for r in records if r["score"] == score})
    shifted_rhos = [r for r in rhos if r < rho_cal - 1e-9]
    hardest_rho = min(shifted_rhos) if shifted_rhos else None      # largest shift, e.g. rho_test=0.5
    per_rho, R_means, mech_f_means, mech_c_means = [], [], [], []
    n_shifted = n_rec = 0
    hardest_recovers = False
    for rho in rhos:
        gfs = _cell(records, score, *FEAT_SPLIT, rho, "cov_gap")
        gcs = _cell(records, score, *CPT_SPLIT, rho, "cov_gap")
        gfm = _cell(records, score, *FEAT_MOND, rho, "cov_gap")
        gcm = _cell(records, score, *CPT_MOND, rho, "cov_gap")

        # paired across-seed differences (seeds aligned by sort order; all cells share the seed set)
        n = min(len(gfs), len(gcs), len(gfm), len(gcm))
        d_repr = _agg([gfs[i] - gcs[i] for i in range(n)])          # representation gap reduction
        mech_feat = _agg([gfs[i] - gfm[i] for i in range(n)])       # mechanism main effect, feature
        mech_cpt = _agg([gcs[i] - gcm[i] for i in range(n)])        # mechanism main effect, concept

        # R = ratio of MEAN reductions (stable; per-seed denominators can be ~0). Guard tiny denom.
        denom = mech_feat["mean"]
        R = float(d_repr["mean"] / denom) if (np.isfinite(denom) and abs(denom) > 1e-6) else float("nan")
        repr_significant = bool(d_repr["n"] > 0 and d_repr["lo"] > 0)   # CI excludes 0, concept tighter
        shifted = rho < rho_cal - 1e-9
        recovers = bool(shifted and np.isfinite(R) and R >= R_MIN and repr_significant)
        is_hardest = bool(hardest_rho is not None and abs(rho - hardest_rho) < 1e-9)
        if shifted:
            n_shifted += 1
            if np.isfinite(R):
                R_means.append(R)
            mech_f_means.append(mech_feat["mean"])
            mech_c_means.append(mech_cpt["mean"])
            if recovers:
                n_rec += 1
            if is_hardest:
                hardest_recovers = recovers
        per_rho.append(RhoSubstitution(
            rho_test=float(rho), shifted=shifted,
            gap_feat_split=_agg(gfs), gap_cpt_split=_agg(gcs),
            gap_feat_mond=_agg(gfm), gap_cpt_mond=_agg(gcm),
            d_repr=d_repr, mech_feat=mech_feat, mech_cpt=mech_cpt,
            R=R, repr_significant=repr_significant, recovers=recovers, is_hardest=is_hardest))

    sweep_R = float(np.mean(R_means)) if R_means else float("nan")
    sweep_mf = float(np.mean(mech_f_means)) if mech_f_means else float("nan")
    sweep_mc = float(np.mean(mech_c_means)) if mech_c_means else float("nan")
    majority = n_shifted > 0 and n_rec * 2 > n_shifted
    # v3 per-score GREEN: majority AND the LARGEST shift (rho_test=0.5) must recover. The v2 pass was
    # fragile precisely because it failed at the two largest shifts -- this closes that hole.
    green = bool(majority and hardest_recovers)
    hrho = hardest_rho if hardest_rho is not None else float("nan")
    if green:
        label = "GREEN (group-free substitution holds)"
        rationale = (
            f"An invariant concept score under pooled split recovers fraction R>={R_MIN} of the "
            f"Mondrian-mechanism gap reduction at a MAJORITY of shifted rho ({n_rec}/{n_shifted}) AND "
            f"at the LARGEST shift rho={hrho:g} (sweep-mean R={sweep_R:.2f}), with the paired "
            f"gap[feat,split]-gap[cpt,split] CI excluding 0. Honest mechanism main effect (reported, "
            f"not suppressed): Mondrian cuts the gap by {sweep_mf:+.3f} (feature) / {sweep_mc:+.3f} "
            f"(concept) on average.")
    else:
        why = []
        if not majority:
            why.append(f"only {n_rec}/{n_shifted} shifted rho recover")
        if not hardest_recovers:
            why.append(f"the largest shift rho={hrho:g} FAILS to recover")
        label = "FALLBACK (kill-switch)"
        rationale = (
            f"Group-free substitution does NOT hold under the strengthened criterion ("
            f"{'; '.join(why)}; sweep-mean R={sweep_R:.2f}). Falling back to: "
            f"\"{UnifiedVerdict.fallback_claim}\". Mechanism main effect: Mondrian cuts the gap by "
            f"{sweep_mf:+.3f} (feature) / {sweep_mc:+.3f} (concept) on average -- the binding lever.")
    return UnifiedVerdict(label=label, green=green, score=score, alpha=alpha, rho_cal=rho_cal,
                          per_rho=per_rho, n_shifted=n_shifted, n_recovered=n_rec,
                          majority=majority, hardest_recovers=hardest_recovers,
                          sweep_mean_R=sweep_R, sweep_mean_mech_feat=sweep_mf,
                          sweep_mean_mech_cpt=sweep_mc, rationale=rationale)


# ======================================================================================
# v3 COMBINED decision across score functions (GREEN requires >= 2 of 3)
# ======================================================================================
MIN_GREEN_SCORES = 2              # of {APS, RAPS, THR}


def combined_decision(verdicts_by_score: dict) -> dict:
    """Combine per-score verdicts into the v3 headline decision.

    v3 GREEN requires the per-score verdict to be GREEN for >= MIN_GREEN_SCORES of {APS,RAPS,THR}
    (robustness across score functions), where each per-score GREEN already requires the largest
    shift to recover. Otherwise FALLBACK. Returns a dict with the decision + which scores passed."""
    green_scores = sorted(s for s, v in verdicts_by_score.items() if v.green)
    n_green = len(green_scores)
    green = n_green >= MIN_GREEN_SCORES
    if green:
        rationale = (f"GREEN: group-free substitution holds for {n_green}/{len(verdicts_by_score)} "
                     f"score functions ({', '.join(green_scores)} >= {MIN_GREEN_SCORES} required), "
                     f"each including the largest shift. Robust across score functions.")
    else:
        rationale = (f"FALLBACK: GREEN for only {n_green}/{len(verdicts_by_score)} score functions "
                     f"({', '.join(green_scores) or 'none'}; >= {MIN_GREEN_SCORES} required). "
                     f"\"{UnifiedVerdict.fallback_claim}\".")
    return {"green": green, "label": "GREEN" if green else "FALLBACK (kill-switch)",
            "n_green_scores": n_green, "green_scores": green_scores,
            "min_required": MIN_GREEN_SCORES, "rationale": rationale}
