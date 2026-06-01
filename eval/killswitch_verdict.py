"""Pre-committed GREEN / RED kill-switch verdict for Arah 2 (counterfactual support gap).

Mirrors ``eval.phase1_eval.premise2_verdict`` in spirit, but tests the ONE hypothesis this gate
exists for: does the counterfactual support gap carry error information that BOTH a plain
concept-space trust score (``trust_concept`` — THE BAR) and a clean-support-only control
(``support_clean``) lack?

Decision is on the MINORITY group, on minority error-detection AUROC, with 1000-sample bootstrap
95% CIs and a PAIRED bootstrap test. "Beats" = the paired-delta 95% CI is strictly positive
(delta_lo > 0) — i.e. the two signals' (dependent) AUROCs are separated with the CI excluding 0.
This is the strict, pre-committed reading of "CIs non-overlapping" for paired samples; no p-value
fallback is used, so RED is never softened.

Pre-committed criterion (committed before any numbers are printed):
  GREEN (mechanism alive): at >= 1 dose, V_gap beats trust_concept AND V_gap beats support_clean.
    The support_clean clause is MANDATORY — it rules out the degeneracy where V_gap ~= support_clean
    (which happens if support_full is roughly constant, making -gap a relabelled support_clean).
  RED (dead): V_gap ties/loses trust_concept at every dose, OR V_gap never beats support_clean
    (so the gap adds nothing over clean-support). Any outcome that is not GREEN is RED — including
    the degenerate case where V_gap beats trust_concept at one dose and support_clean at a
    different dose but never BOTH at the same dose (no single configuration clears the wall).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from . import metrics
from .bootstrap import CI, PairedTest, bootstrap_ci, paired_bootstrap_test


@dataclass
class DoseVerdict:
    dose: int
    lam: float
    label: str                                  # "GREEN" | "RED"
    beats_trust_concept: bool
    beats_support_clean: bool
    vs_trust_concept: Optional[PairedTest] = None
    vs_support_clean: Optional[PairedTest] = None
    auroc_minority: dict = field(default_factory=dict)   # signal -> CI
    rationale: str = ""


@dataclass
class KillswitchVerdict:
    label: str                                  # "GREEN" | "RED"
    rationale: str
    per_dose: list = field(default_factory=list)        # list[DoseVerdict]
    green_doses: list = field(default_factory=list)     # doses where both bars cleared


def _beats(t: Optional[PairedTest]) -> bool:
    """Strict pre-committed rule: the paired-delta 95% CI is wholly positive (excludes 0)."""
    return bool(t is not None and np.isfinite(t.delta_lo) and t.delta_lo > 0)


def _dose_verdict(df, dose, lam, primary, bar, control, report_signals,
                  n_resamples, ci, seed) -> DoseVerdict:
    minority = df["is_minority"].to_numpy().astype(bool)
    correct = df["correct"].to_numpy().astype(int)
    if not minority.any() or primary not in df:
        return DoseVerdict(dose, lam, "RED", False, False,
                           rationale=f"missing minority group or {primary}")
    c = correct[minority]
    v = df[primary].to_numpy().astype(float)[minority]

    def _test(other):
        if other not in df or not df[other].notna().any():
            return None
        o = df[other].to_numpy().astype(float)[minority]
        return paired_bootstrap_test(metrics.error_detection_auroc, (v, c), (o, c),
                                     primary, other, n_resamples=n_resamples, ci=ci, seed=seed)

    t_bar = _test(bar)
    t_ctrl = _test(control)
    beats_bar = _beats(t_bar)
    beats_ctrl = _beats(t_ctrl)

    auroc_minority = {}
    for s in report_signals:
        if s in df and df[s].notna().any():
            sig = df[s].to_numpy().astype(float)[minority]
            auroc_minority[s] = bootstrap_ci(metrics.error_detection_auroc, sig, c,
                                             n_resamples=n_resamples, ci=ci, seed=seed)

    label = "GREEN" if (beats_bar and beats_ctrl) else "RED"
    bits = []
    for nm, t, b in ((bar, t_bar, beats_bar), (control, t_ctrl, beats_ctrl)):
        if t is None:
            bits.append(f"vs {nm}: missing")
        else:
            bits.append(f"vs {nm}: dAUROC={t.delta:+.3f}[{t.delta_lo:+.3f},{t.delta_hi:+.3f}] "
                        f"({'beats' if b else 'ties/loses'})")
    rationale = (f"dose={dose} lam={lam:g}: {label}. {primary} " + " | ".join(bits))
    return DoseVerdict(dose, lam, label, beats_bar, beats_ctrl, t_bar, t_ctrl,
                       auroc_minority, rationale)


def killswitch_verdict(
    dose_frames,
    *,
    primary: str = "V_gap",
    bar: str = "trust_concept",
    control: str = "support_clean",
    report_signals=("V_gap", "V_gap_pure", "support_clean", "V_comp", "trust_concept",
                    "trust", "ensemble_disagree", "conf_msp"),
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
) -> KillswitchVerdict:
    """Apply the pre-committed GREEN/RED criterion across all doses.

    ``dose_frames`` : list of {"dose": int, "lam": float, "df": DataFrame} where each df has
    columns is_minority, correct, and the signal columns. Returns the overall + per-dose verdict.
    """
    per_dose = [_dose_verdict(d["df"], d["dose"], d.get("lam", 0.0), primary, bar, control,
                              report_signals, n_resamples, ci, seed) for d in dose_frames]

    green_doses = [dv.dose for dv in per_dose if dv.label == "GREEN"]
    any_beats_bar = any(dv.beats_trust_concept for dv in per_dose)
    any_beats_ctrl = any(dv.beats_support_clean for dv in per_dose)

    if green_doses:
        label = "GREEN"
        why = (f"At dose(s) {green_doses}, {primary} beats BOTH {bar} (THE BAR) AND {control} "
               f"(the clean-support control) on minority AUROC with paired-delta CIs excluding 0. "
               f"The counterfactual support gap carries information neither distance-in-concept-"
               f"space nor clean-support alone has — the mechanism is alive.")
    else:
        label = "RED"
        if not any_beats_bar:
            why = (f"{primary} ties/loses {bar} at EVERY dose: no contribution over a plain "
                   f"concept-space trust score. Mechanism dead.")
        elif not any_beats_ctrl:
            why = (f"{primary} NEVER beats {control}: the gap adds nothing over clean-support "
                   f"(the V_gap ~= support_clean degeneracy). Mechanism dead.")
        else:
            why = (f"No SINGLE dose clears both bars at once: {primary} beats {bar} at some dose "
                   f"and {control} at some dose, but never simultaneously. The wall is not cleared "
                   f"at any configuration. Mechanism dead (RED, not softened).")

    return KillswitchVerdict(label=label, rationale=why, per_dose=per_dose,
                             green_doses=green_doses)
