"""Pre-committed GREEN / RED verdict for the Spurious-Invariant Conformal de-risk.

THIS FILE ENCODES THE GATE BEFORE ANY NUMBERS ARE PRINTED. The hypothesis: under a shift in
spurious-correlation STRENGTH between calibration (rho_cal) and test (rho_test), split-conformal
in the shortcut-invariant CUB concept space (``STD-cpt``) keeps WORST-GROUP coverage >= 1-alpha
while being MORE EFFICIENT (smaller worst-group set size) than every valid baseline.

Methods (names are fixed; all evaluate the SAME test samples per rho, so comparisons are PAIRED):
  STD-f            split CP on the contaminated f-softmax score
  STD-cpt          split CP on the concept-probe score                  <- THE PROPOSED METHOD
  Mondrian-f       group-conditional CP on the f score
  Mondrian-cpt     group-conditional CP on the concept score
  RobustWasserstein  pooled TV-robust CP on the f score

Pre-committed criterion (alpha given; "worst group" = the group with the LOWEST coverage; the
efficiency comparison is on the fixed REFERENCE group g* = STD-f's worst-coverage group at that
rho, held common across methods so the bootstrap is paired):

  Consider only rho_test where STD-f worst-group coverage drops below 1-alpha (the shift actually
  breaks the contaminated baseline). GREEN iff at >= 1 such shifted rho BOTH:
    (i)  STD-cpt worst-group coverage CI lower bound >= 1-alpha, AND
    (ii) STD-cpt worst-group (g*) set size is strictly smaller than Mondrian-f, Mondrian-cpt AND
         RobustWasserstein -- each evaluated at MATCHED (>=1-alpha) worst-group coverage -- with
         the paired-bootstrap CI of (comparator_size - STD-cpt_size) excluding 0 (lower bound > 0).
  RED otherwise. Explicitly RED (NOT softened) if STD-cpt merely MATCHES a Mondrian variant on
  worst-group set size (size-diff CI contains 0) -> "valid but incremental vs Mondrian".
  Also RED-inconclusive if the shift never pushes STD-f below 1-alpha (regime not triggered).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

MINORITY_GROUPS = (1, 2)

STD_F = "STD-f"
STD_CPT = "STD-cpt"
MOND_F = "Mondrian-f"
MOND_CPT = "Mondrian-cpt"
ROBUST = "RobustWasserstein"
COMPARATORS = (MOND_F, MOND_CPT, ROBUST)


# --------------------------------------------------------------------------------------
# Per-sample -> grouped point metrics
# --------------------------------------------------------------------------------------
def _per_group_coverage(covered: np.ndarray, group: np.ndarray) -> dict:
    return {int(g): float(covered[group == g].mean()) for g in np.unique(group)}


def _per_group_size(size: np.ndarray, group: np.ndarray) -> dict:
    return {int(g): float(size[group == g].mean()) for g in np.unique(group)}


def _worst_group(covered: np.ndarray, group: np.ndarray) -> tuple[int, float]:
    """Group with the lowest coverage and that coverage."""
    pg = _per_group_coverage(covered, group)
    g_star = min(pg, key=pg.get)
    return int(g_star), float(pg[g_star])


def _minority_coverage(covered: np.ndarray, group: np.ndarray) -> float:
    m = np.isin(group, MINORITY_GROUPS)
    return float(covered[m].mean()) if m.any() else float("nan")


@dataclass
class MethodReport:
    name: str
    marginal_cov: float
    worst_group: int
    worst_group_cov: float
    worst_group_cov_lo: float          # bootstrap CI lower bound of worst-group coverage
    worst_group_cov_hi: float
    minority_cov: float
    avg_size: float
    ref_group_size: float              # avg set size on the SHARED reference group g*
    per_group_cov: dict = field(default_factory=dict)
    per_group_size: dict = field(default_factory=dict)


@dataclass
class SizeComparison:
    comparator: str
    qualifies: bool                    # comparator achieves >=1-alpha worst-group coverage
    delta: float                       # comparator_size - STDcpt_size on g* (>0 => STDcpt smaller)
    delta_lo: float
    delta_hi: float
    beaten: bool                       # qualifies AND delta_lo > 0


@dataclass
class RhoVerdict:
    rho_test: float
    stdf_undercovers: bool             # STD-f worst-group coverage < 1-alpha (the trigger)
    ref_group: int                     # g* = STD-f worst-coverage group (shared comparison group)
    reports: dict = field(default_factory=dict)        # name -> MethodReport
    stdcpt_wg_valid: bool = False      # criterion (i): STD-cpt WG cov CI-lower >= 1-alpha
    size_comparisons: list = field(default_factory=list)   # list[SizeComparison]
    green: bool = False
    rationale: str = ""


@dataclass
class ShiftCPVerdict:
    label: str                         # "GREEN" | "RED"
    rationale: str
    alpha: float
    rho_cal: float
    per_rho: list = field(default_factory=list)        # list[RhoVerdict]
    green_rhos: list = field(default_factory=list)


# --------------------------------------------------------------------------------------
# Bootstrap helpers (grouped coverage; paired size difference on a fixed group)
# --------------------------------------------------------------------------------------
def _bootstrap_worst_group_cov(covered, group, n_resamples, ci, seed):
    """Percentile-bootstrap CI of worst-group (min-over-groups) coverage."""
    rng = np.random.default_rng(seed)
    n = len(covered)
    groups = np.unique(group)
    stats = np.empty(n_resamples)
    for b in range(n_resamples):
        idx = rng.integers(0, n, n)
        cov_b, grp_b = covered[idx], group[idx]
        per = [cov_b[grp_b == g].mean() for g in groups if (grp_b == g).any()]
        stats[b] = min(per) if per else np.nan
    stats = stats[np.isfinite(stats)]
    a = 1.0 - ci
    lo = float(np.quantile(stats, a / 2)) if stats.size else float("nan")
    hi = float(np.quantile(stats, 1 - a / 2)) if stats.size else float("nan")
    return lo, hi


def _paired_size_delta_on_group(size_comp, size_cpt, mask, n_resamples, ci, seed):
    """Paired-bootstrap CI of mean(size_comp - size_cpt) over samples in ``mask`` (group g*)."""
    rng = np.random.default_rng(seed)
    sc = size_comp[mask].astype(float)
    sp = size_cpt[mask].astype(float)
    if sc.size == 0:
        return float("nan"), float("nan"), float("nan")
    point = float((sc - sp).mean())
    n = sc.size
    deltas = np.empty(n_resamples)
    for b in range(n_resamples):
        idx = rng.integers(0, n, n)
        deltas[b] = (sc[idx] - sp[idx]).mean()
    a = 1.0 - ci
    return point, float(np.quantile(deltas, a / 2)), float(np.quantile(deltas, 1 - a / 2))


# --------------------------------------------------------------------------------------
# Per-rho assembly + gate
# --------------------------------------------------------------------------------------
def _method_report(name, covered, size, group, ref_group, n_resamples, ci, seed):
    g_star_self, wg_cov = _worst_group(covered, group)
    lo, hi = _bootstrap_worst_group_cov(covered, group, n_resamples, ci, seed)
    ref_size = float(size[group == ref_group].mean()) if (group == ref_group).any() else float("nan")
    return MethodReport(
        name=name,
        marginal_cov=float(covered.mean()),
        worst_group=g_star_self,
        worst_group_cov=wg_cov,
        worst_group_cov_lo=lo,
        worst_group_cov_hi=hi,
        minority_cov=_minority_coverage(covered, group),
        avg_size=float(size.mean()),
        ref_group_size=ref_size,
        per_group_cov=_per_group_coverage(covered, group),
        per_group_size=_per_group_size(size, group),
    )


def evaluate_rho(rho_test, methods, alpha, n_resamples=1000, ci=0.95, seed=0) -> RhoVerdict:
    """Build the per-method reports + the gate decision for ONE rho_test.

    ``methods``: {name -> {"covered": (N,) int, "size": (N,) int, "group": (N,) int}}; every method
    must share the SAME ``group`` array (same test samples) so comparisons are paired.
    """
    target = 1.0 - alpha
    group = np.asarray(methods[STD_F]["group"])

    # reference group g* = STD-f's worst-coverage group (shared comparison group, held fixed)
    ref_group, stdf_wg_cov = _worst_group(np.asarray(methods[STD_F]["covered"]), group)
    stdf_under = stdf_wg_cov < target

    reports = {
        name: _method_report(name, np.asarray(m["covered"]), np.asarray(m["size"]),
                             np.asarray(m["group"]), ref_group, n_resamples, ci, seed)
        for name, m in methods.items()
    }

    # criterion (i): STD-cpt worst-group coverage CI lower bound >= 1-alpha
    cpt = reports[STD_CPT]
    stdcpt_wg_valid = np.isfinite(cpt.worst_group_cov_lo) and cpt.worst_group_cov_lo >= target

    # criterion (ii): STD-cpt strictly smaller set size on g* than each valid comparator
    ref_mask = group == ref_group
    size_cpt = np.asarray(methods[STD_CPT]["size"])
    comparisons = []
    for comp in COMPARATORS:
        crep = reports[comp]
        qualifies = np.isfinite(crep.worst_group_cov) and crep.worst_group_cov >= target
        delta, dlo, dhi = _paired_size_delta_on_group(
            np.asarray(methods[comp]["size"]), size_cpt, ref_mask, n_resamples, ci, seed)
        beaten = bool(qualifies and np.isfinite(dlo) and dlo > 0)
        comparisons.append(SizeComparison(comp, qualifies, delta, dlo, dhi, beaten))

    all_beaten = all(c.beaten for c in comparisons)
    green = bool(stdf_under and stdcpt_wg_valid and all_beaten)

    rv = RhoVerdict(rho_test=float(rho_test), stdf_undercovers=bool(stdf_under),
                    ref_group=int(ref_group), reports=reports,
                    stdcpt_wg_valid=bool(stdcpt_wg_valid), size_comparisons=comparisons,
                    green=green)
    rv.rationale = _rho_rationale(rv, alpha)
    return rv


def _rho_rationale(rv: RhoVerdict, alpha: float) -> str:
    if not rv.stdf_undercovers:
        return (f"rho_test={rv.rho_test:g}: STD-f worst-group coverage holds (>= {1-alpha:g}); "
                f"shift did not break the contaminated baseline here, so this rho cannot trigger "
                f"a win.")
    bits = [f"rho_test={rv.rho_test:g}: STD-f UNDER-COVERS worst group (trigger active, g*={rv.ref_group})."]
    cpt = rv.reports[STD_CPT]
    bits.append(f"STD-cpt WG cov={cpt.worst_group_cov:.3f} "
                f"[{cpt.worst_group_cov_lo:.3f},{cpt.worst_group_cov_hi:.3f}] "
                f"({'valid' if rv.stdcpt_wg_valid else 'FAILS'} (i): CI-lower vs {1-alpha:g}).")
    for c in rv.size_comparisons:
        if not c.qualifies:
            bits.append(f"vs {c.comparator}: comparator does NOT reach {1-alpha:g} WG cov "
                        f"(not matched).")
        else:
            verdict = "STD-cpt smaller" if c.beaten else "tie/larger"
            bits.append(f"vs {c.comparator}: size_diff={c.delta:+.3f}"
                        f"[{c.delta_lo:+.3f},{c.delta_hi:+.3f}] ({verdict}).")
    bits.append("GREEN here." if rv.green else "not GREEN here.")
    return " ".join(bits)


def shiftcp_verdict(rho_results, alpha, rho_cal, n_resamples=1000, ci=0.95, seed=0) -> ShiftCPVerdict:
    """Apply the pre-committed gate across the rho_test sweep.

    ``rho_results``: list of (rho_test, methods_dict) — see ``evaluate_rho``.
    """
    per_rho = [evaluate_rho(rho, methods, alpha, n_resamples, ci, seed)
               for rho, methods in rho_results]
    green_rhos = [rv.rho_test for rv in per_rho if rv.green]

    triggered = [rv for rv in per_rho if rv.stdf_undercovers]
    if green_rhos:
        label = "GREEN"
        why = (f"At shifted rho_test {green_rhos}, STD-cpt holds worst-group coverage >= {1-alpha:g} "
               f"(CI lower bound) AND produces strictly smaller worst-group sets than Mondrian-f, "
               f"Mondrian-cpt AND RobustWasserstein at matched coverage (paired-bootstrap size-diff "
               f"CIs exclude 0). The shortcut-invariant score is valid AND more efficient under the "
               f"correlation-strength shift — the idea is alive.")
    elif not triggered:
        label = "RED"
        why = ("REGIME NOT TRIGGERED: the correlation-strength shift never pushed STD-f below "
               f"{1-alpha:g} worst-group coverage at any rho_test, so there is no regime in which "
               f"to demonstrate an advantage. Inconclusive -> RED (not softened); the shift axis "
               f"needs to be made more severe before this idea can be de-risked.")
    else:
        # the shift DID break STD-f somewhere; diagnose why STD-cpt failed to clear the bar
        any_valid = any(rv.stdcpt_wg_valid for rv in triggered)
        # did STD-cpt only TIE a Mondrian variant on size where it was otherwise valid?
        incremental = any(
            rv.stdcpt_wg_valid and any(
                c.comparator in (MOND_F, MOND_CPT) and c.qualifies and not c.beaten
                and np.isfinite(c.delta_lo) and c.delta_lo <= 0 <= c.delta_hi
                for c in rv.size_comparisons)
            for rv in triggered)
        label = "RED"
        if not any_valid:
            why = (f"At every shifted rho where STD-f under-covers, STD-cpt FAILS worst-group "
                   f"coverage (CI lower bound < {1-alpha:g}). The invariant score is not even valid "
                   f"under the shift. Dead.")
        elif incremental:
            why = ("VALID BUT INCREMENTAL vs Mondrian: STD-cpt holds worst-group coverage under the "
                   "shift, but its worst-group set size only MATCHES a Mondrian variant (size-diff "
                   "CI contains 0) — no efficiency win over group-conditional CP. RED (not "
                   "softened): the proposed method buys nothing Mondrian doesn't already give.")
        else:
            why = ("No single shifted rho clears BOTH bars at once: STD-cpt is either invalid or "
                   "fails to strictly beat all three baselines on worst-group set size at matched "
                   "coverage. The efficiency-under-validity wall is not cleared. RED.")

    return ShiftCPVerdict(label=label, rationale=why, alpha=float(alpha), rho_cal=float(rho_cal),
                          per_rho=per_rho, green_rhos=green_rhos)
