"""Phase-1 metric aggregation + kill-switch GO/NO-GO verdict (Section 4).

Consumes a tidy per-sample table (one row per test sample) with columns:
  correct, is_minority, spurious_attr, conf_msp, and the signal columns.
Produces:
  - error-detection AUROC (overall / majority / minority) per signal, with bootstrap CIs;
  - minority selective-risk AURC per signal;
  - confident-but-wrong capture rate per signal at each budget;
  - contamination AUROC + MI per signal (P2);
  - paired-bootstrap tests of V_full vs trust and vs ensemble_disagree on the MINORITY group;
  - the pre-committed GO / NO-GO verdict.

Kill criterion (pre-committed, Section 4): on the minority group, V_full must achieve
error-detection AUROC strictly greater than BOTH trust AND ensemble_disagree (non-overlapping
95% CIs OR paired p<0.05), AND a lower minority AURC than both. Beating only MSP is NOT enough.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from . import metrics
from .bootstrap import (
    CI,
    PairedTest,
    bootstrap_ci,
    holm_correction,
    paired_bootstrap_test,
)

# The two baselines the kill-switch is gated on (NOT just conf_msp).
KILL_BASELINES = ["trust", "ensemble_disagree"]


@dataclass
class SignalReport:
    name: str
    auroc_overall: Optional[CI] = None
    auroc_majority: Optional[CI] = None
    auroc_minority: Optional[CI] = None
    aurc_overall: Optional[CI] = None
    aurc_minority: Optional[CI] = None
    contamination_auroc: float = float("nan")
    mutual_info: float = float("nan")
    capture_rate: dict = field(default_factory=dict)  # budget -> rate


@dataclass
class Phase1Verdict:
    decision: str                 # "GO" | "NO-GO"
    rationale: str
    auroc_tests: dict = field(default_factory=dict)   # baseline -> PairedTest (minority AUROC)
    aurc_tests: dict = field(default_factory=dict)     # baseline -> PairedTest (minority AURC)
    holm_auroc: dict = field(default_factory=dict)
    beats_auroc: dict = field(default_factory=dict)    # baseline -> bool
    beats_aurc: dict = field(default_factory=dict)     # baseline -> bool


def _maybe(arr, mask):
    return arr[mask] if mask is not None else arr


def evaluate_signals(
    df,
    signal_names,
    *,
    budgets=(0.1, 0.2, 0.3),
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
) -> dict[str, SignalReport]:
    """Compute the full per-signal metric suite. df is a pandas DataFrame of TEST rows."""
    correct = df["correct"].to_numpy().astype(int)
    minority = df["is_minority"].to_numpy().astype(bool)
    majority = ~minority
    conf = df["conf_msp"].to_numpy()
    spur = df["spurious_attr"].to_numpy() if "spurious_attr" in df else None
    y_true = df["y_true"].to_numpy() if "y_true" in df else None

    reports: dict[str, SignalReport] = {}
    for name in signal_names:
        if name not in df:
            continue
        sig = df[name].to_numpy().astype(float)
        rep = SignalReport(name=name)
        rep.auroc_overall = bootstrap_ci(
            metrics.error_detection_auroc, sig, correct, n_resamples=n_resamples, ci=ci, seed=seed
        )
        if majority.any():
            rep.auroc_majority = bootstrap_ci(
                metrics.error_detection_auroc, sig[majority], correct[majority],
                n_resamples=n_resamples, ci=ci, seed=seed,
            )
        if minority.any():
            rep.auroc_minority = bootstrap_ci(
                metrics.error_detection_auroc, sig[minority], correct[minority],
                n_resamples=n_resamples, ci=ci, seed=seed,
            )
            rep.aurc_minority = bootstrap_ci(
                metrics.aurc, sig[minority], correct[minority],
                n_resamples=n_resamples, ci=ci, seed=seed,
            )
        rep.aurc_overall = bootstrap_ci(
            metrics.aurc, sig, correct, n_resamples=n_resamples, ci=ci, seed=seed
        )
        if spur is not None:
            rep.contamination_auroc = metrics.contamination_auroc(sig, spur, y_true)
            rep.mutual_info = metrics.mutual_information(sig, spur, y_true=y_true)
        for b in budgets:
            rep.capture_rate[float(b)] = metrics.confident_but_wrong_capture_rate(
                sig, correct, conf, b
            )
        reports[name] = rep
    return reports


def kill_switch_verdict(
    df,
    *,
    primary="V_full",
    baselines=KILL_BASELINES,
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
    alpha_sig: float = 0.05,
) -> Phase1Verdict:
    """Apply the pre-committed Section-4 kill criterion on the minority group."""
    minority = df["is_minority"].to_numpy().astype(bool)
    correct = df["correct"].to_numpy().astype(int)

    if not minority.any() or primary not in df:
        return Phase1Verdict("NO-GO", f"missing minority group or {primary} signal")

    v = df[primary].to_numpy().astype(float)[minority]
    c = correct[minority]

    auroc_tests, aurc_tests = {}, {}
    beats_auroc, beats_aurc = {}, {}
    pvals = {}
    for b in baselines:
        if b not in df:
            beats_auroc[b] = False
            beats_aurc[b] = False
            continue
        bvals = df[b].to_numpy().astype(float)[minority]
        # AUROC: higher is better -> delta = AUROC(V) - AUROC(baseline) should be > 0
        t_auroc = paired_bootstrap_test(
            metrics.error_detection_auroc, (v, c), (bvals, c),
            primary, b, n_resamples=n_resamples, ci=ci, seed=seed,
        )
        # AURC: lower is better -> test on NEGATED aurc so "favored=primary" means V better
        neg_aurc = lambda s, cc: -metrics.aurc(s, cc)
        t_aurc = paired_bootstrap_test(
            neg_aurc, (v, c), (bvals, c),
            primary, b, n_resamples=n_resamples, ci=ci, seed=seed,
        )
        auroc_tests[b] = t_auroc
        aurc_tests[b] = t_aurc
        pvals[b] = t_auroc.p_value
        # "beats" on AUROC: CI on delta strictly positive OR paired p<alpha with positive point
        beats_auroc[b] = (t_auroc.delta_lo > 0) or (
            t_auroc.p_value < alpha_sig and t_auroc.delta > 0
        )
        beats_aurc[b] = t_aurc.delta > 0  # negated aurc delta>0 => V has lower AURC

    holm = holm_correction(pvals, alpha=alpha_sig) if pvals else {}

    all_beat_auroc = all(beats_auroc.get(b, False) for b in baselines)
    all_beat_aurc = all(beats_aurc.get(b, False) for b in baselines)
    decision = "GO" if (all_beat_auroc and all_beat_aurc) else "NO-GO"

    rationale_bits = []
    for b in baselines:
        ta = auroc_tests.get(b)
        if ta is not None:
            rationale_bits.append(
                f"{primary} vs {b}: minority AUROC delta={ta.delta:+.3f} "
                f"[{ta.delta_lo:+.3f},{ta.delta_hi:+.3f}] p={ta.p_value:.3f} "
                f"({'beats' if beats_auroc[b] else 'ties/loses'}); "
                f"AURC {'lower' if beats_aurc[b] else 'not lower'}"
            )
        else:
            rationale_bits.append(f"{primary} vs {b}: baseline missing")
    rationale = (
        f"Kill criterion {'MET' if decision == 'GO' else 'NOT met'}. "
        + " | ".join(rationale_bits)
    )

    return Phase1Verdict(
        decision=decision,
        rationale=rationale,
        auroc_tests=auroc_tests,
        aurc_tests=aurc_tests,
        holm_auroc=holm,
        beats_auroc=beats_auroc,
        beats_aurc=beats_aurc,
    )
