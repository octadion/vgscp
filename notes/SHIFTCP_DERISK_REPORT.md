# SHIFT-CP DE-RISK — Spurious-Invariant Conformal under a correlation-strength shift

**Date:** 2026-06-02 · **Run mode:** smoke · **Seed:** 0 (single-seed gate) · **OVERALL VERDICT: RED**

> ⚠️ **SMOKE (synthetic) run** — fabricated shortcut f + clean CUB-like attrs. Validates the resample→calibrate→sweep→verdict PIPELINE end-to-end on CPU. The real GREEN/RED numbers come from the cache-reuse run.


Go/no-go gate for ONE hypothesis: under a shift in spurious-correlation STRENGTH between calibration (ρ_cal) and test (ρ_test), does split-conformal in the shortcut-invariant CUB concept space (**STD-cpt**, the proposed method) keep WORST-GROUP coverage ≥ 1−α while producing SMALLER worst-group sets than (a) standard split CP on the f-softmax score (**STD-f**), (b) group-conditional **Mondrian** CP on each score, and (c) a TV-robust CP baseline (**RobustWasserstein**)? If it only matches Mondrian (no efficiency win) or fails worst-group coverage, the idea is RED.

## Pre-committed criterion (committed in eval/shiftcp_verdict.py before any numbers)
Among shifted ρ_test where **STD-f worst-group coverage drops below 1−α**, GREEN iff at ≥1 such ρ BOTH: **(i)** STD-cpt worst-group coverage CI lower bound ≥ 1−α; **(ii)** STD-cpt worst-group set size (on the shared reference group g* = STD-f's worst-coverage group) is strictly smaller than Mondrian-f, Mondrian-cpt AND RobustWasserstein at matched (≥1−α) worst-group coverage (paired-bootstrap size-diff CI excludes 0). RED otherwise; explicitly RED if STD-cpt only MATCHES Mondrian on set size ("valid but incremental vs Mondrian"), and RED-inconclusive if the shift never breaks STD-f.

**OVERALL (α=0.1): RED.** At every shifted rho where STD-f under-covers, STD-cpt FAILS worst-group coverage (CI lower bound < 0.9). The invariant score is not even valid under the shift. Dead.

## Regime check (f) — is the shortcut real?
Cached f worst-group acc = 0.300 vs overall 0.831 (per-group {0: 1.0, 1: 0.3, 2: 0.33070866141732286, 3: 1.0}). A real background shortcut is the precondition for the f-score to be spurious-sensitive.

## Shift construction — group counts per ρ (ρ_cal=0.95, N_cal=700, N_test=800, score f=APS / concept=THR)
Correlation strength ρ = P(place = y) at balanced classes; group fractions g0=g3=ρ/2 (concordant), g1=g2=(1−ρ)/2 (minority). Test sets resample the test pool to each ρ at fixed N (with replacement within group). Calibration/test pools are disjoint.

| set | ρ target | ρ realized | g0 | g1 | g2 | g3 |
|---|---|---|---|---|---|---|
| calibration | 0.95 | 0.949 | 332 | 18 | 18 | 332 |
| test ρ=0.95 | 0.95 | 0.950 | 380 | 20 | 20 | 380 |
| test ρ=0.75 | 0.75 | 0.750 | 300 | 100 | 100 | 300 |
| test ρ=0.5 | 0.5 | 0.500 | 200 | 200 | 200 | 200 |
| test ρ=0.25 | 0.25 | 0.250 | 100 | 300 | 300 | 100 |

## α = 0.1  (coverage target 0.9) — **RED**

### ρ_test = 0.95 — TRIGGER (STD-f under-covers)

| method | marg cov | **worst-grp cov** [95% CI] | minority cov | avg size | WG(g*=1) size |
|---|---|---|---|---|---|
| STD-f | 0.877 | 0.250 [0.043,0.369] ⚠ | 0.250 | 0.911 | 0.950 |
| STD-cpt | 0.882 | 0.861 [0.750,0.888] ⚠ | 0.925 | 0.882 | 0.900 |
| Mondrian-f | 0.886 | 0.858 [0.829,0.892] ⚠ | 1.000 | 0.936 | 2.000 |
| Mondrian-cpt | 0.876 | 0.826 [0.750,0.858] ⚠ | 0.925 | 0.876 | 0.900 |
| RobustWasserstein | 1.000 | 1.000 [1.000,1.000] | 1.000 | 2.000 | 2.000 |

RobustWasserstein TV ball ε=0.213 → inflated α'=0.000.

Efficiency comparison on g* (worst group), at matched coverage:

| comparator | matched (≥1-α WG cov)? | size_diff (comp − STD-cpt) on g* [95% CI] | STD-cpt strictly smaller? |
|---|---|---|---|
| Mondrian-f | no | +1.100 [+1.000,+1.250] | no |
| Mondrian-cpt | no | +0.000 [+0.000,+0.000] | no |
| RobustWasserstein | yes | +1.100 [+1.000,+1.250] | **yes** |

_rho_test=0.95: STD-f UNDER-COVERS worst group (trigger active, g*=1). STD-cpt WG cov=0.861 [0.750,0.888] (FAILS (i): CI-lower vs 0.9). vs Mondrian-f: comparator does NOT reach 0.9 WG cov (not matched). vs Mondrian-cpt: comparator does NOT reach 0.9 WG cov (not matched). vs RobustWasserstein: size_diff=+1.100[+1.000,+1.250] (STD-cpt smaller). not GREEN here._

### ρ_test = 0.75 — TRIGGER (STD-f under-covers)

| method | marg cov | **worst-grp cov** [95% CI] | minority cov | avg size | WG(g*=1) size |
|---|---|---|---|---|---|
| STD-f | 0.790 | 0.290 [0.207,0.368] ⚠ | 0.385 | 0.926 | 0.910 |
| STD-cpt | 0.887 | 0.867 [0.801,0.889] ⚠ | 0.885 | 0.887 | 0.900 |
| Mondrian-f | 0.924 | 0.897 [0.860,0.914] ⚠ | 1.000 | 1.174 | 2.000 |
| Mondrian-cpt | 0.885 | 0.840 [0.762,0.863] ⚠ | 0.880 | 0.885 | 0.840 |
| RobustWasserstein | 1.000 | 1.000 [1.000,1.000] | 1.000 | 2.000 | 2.000 |

RobustWasserstein TV ball ε=0.256 → inflated α'=0.000.

Efficiency comparison on g* (worst group), at matched coverage:

| comparator | matched (≥1-α WG cov)? | size_diff (comp − STD-cpt) on g* [95% CI] | STD-cpt strictly smaller? |
|---|---|---|---|
| Mondrian-f | no | +1.100 [+1.045,+1.165] | no |
| Mondrian-cpt | no | -0.060 [-0.110,-0.020] | no |
| RobustWasserstein | yes | +1.100 [+1.045,+1.165] | **yes** |

_rho_test=0.75: STD-f UNDER-COVERS worst group (trigger active, g*=1). STD-cpt WG cov=0.867 [0.801,0.889] (FAILS (i): CI-lower vs 0.9). vs Mondrian-f: comparator does NOT reach 0.9 WG cov (not matched). vs Mondrian-cpt: comparator does NOT reach 0.9 WG cov (not matched). vs RobustWasserstein: size_diff=+1.100[+1.045,+1.165] (STD-cpt smaller). not GREEN here._

### ρ_test = 0.5 — TRIGGER (STD-f under-covers)

| method | marg cov | **worst-grp cov** [95% CI] | minority cov | avg size | WG(g*=1) size |
|---|---|---|---|---|---|
| STD-f | 0.635 | 0.295 [0.223,0.348] ⚠ | 0.350 | 0.925 | 0.930 |
| STD-cpt | 0.880 | 0.845 [0.800,0.883] ⚠ | 0.885 | 0.880 | 0.880 |
| Mondrian-f | 0.941 | 0.845 [0.797,0.894] ⚠ | 1.000 | 1.441 | 2.000 |
| Mondrian-cpt | 0.880 | 0.810 [0.760,0.846] ⚠ | 0.900 | 0.880 | 0.840 |
| RobustWasserstein | 1.000 | 1.000 [1.000,1.000] | 1.000 | 2.000 | 2.000 |

RobustWasserstein TV ball ε=0.344 → inflated α'=0.000.

Efficiency comparison on g* (worst group), at matched coverage:

| comparator | matched (≥1-α WG cov)? | size_diff (comp − STD-cpt) on g* [95% CI] | STD-cpt strictly smaller? |
|---|---|---|---|
| Mondrian-f | no | +1.120 [+1.080,+1.165] | no |
| Mondrian-cpt | no | -0.040 [-0.070,-0.015] | no |
| RobustWasserstein | yes | +1.120 [+1.080,+1.165] | **yes** |

_rho_test=0.5: STD-f UNDER-COVERS worst group (trigger active, g*=1). STD-cpt WG cov=0.845 [0.800,0.883] (FAILS (i): CI-lower vs 0.9). vs Mondrian-f: comparator does NOT reach 0.9 WG cov (not matched). vs Mondrian-cpt: comparator does NOT reach 0.9 WG cov (not matched). vs RobustWasserstein: size_diff=+1.120[+1.080,+1.165] (STD-cpt smaller). not GREEN here._

### ρ_test = 0.25 — TRIGGER (STD-f under-covers)

| method | marg cov | **worst-grp cov** [95% CI] | minority cov | avg size | WG(g*=1) size |
|---|---|---|---|---|---|
| STD-f | 0.479 | 0.273 [0.220,0.313] ⚠ | 0.332 | 0.907 | 0.903 |
| STD-cpt | 0.863 | 0.850 [0.766,0.868] ⚠ | 0.867 | 0.863 | 0.870 |
| Mondrian-f | 0.966 | 0.810 [0.724,0.887] ⚠ | 1.000 | 1.716 | 2.000 |
| Mondrian-cpt | 0.873 | 0.820 [0.748,0.863] ⚠ | 0.875 | 0.873 | 0.840 |
| RobustWasserstein | 1.000 | 1.000 [1.000,1.000] | 1.000 | 2.000 | 2.000 |

RobustWasserstein TV ball ε=0.474 → inflated α'=0.000.

Efficiency comparison on g* (worst group), at matched coverage:

| comparator | matched (≥1-α WG cov)? | size_diff (comp − STD-cpt) on g* [95% CI] | STD-cpt strictly smaller? |
|---|---|---|---|
| Mondrian-f | no | +1.130 [+1.093,+1.167] | no |
| Mondrian-cpt | no | -0.030 [-0.050,-0.017] | no |
| RobustWasserstein | yes | +1.130 [+1.093,+1.167] | **yes** |

_rho_test=0.25: STD-f UNDER-COVERS worst group (trigger active, g*=1). STD-cpt WG cov=0.850 [0.766,0.868] (FAILS (i): CI-lower vs 0.9). vs Mondrian-f: comparator does NOT reach 0.9 WG cov (not matched). vs Mondrian-cpt: comparator does NOT reach 0.9 WG cov (not matched). vs RobustWasserstein: size_diff=+1.130[+1.093,+1.167] (STD-cpt smaller). not GREEN here._

### Crux quantity (α=0.1) — score spurious-sensitivity vs coverage robustness (reported, NOT gated)
AUROC(score → spurious_attr) on calibration: f-score = 0.529, concept-score = 0.534. Mean worst-group coverage gap over triggered ρ: STD-f = 0.623, STD-cpt = 0.044. More spurious-sensitive score = **cpt**; larger coverage gap = **f**; rank-consistent = **False** (the more shortcut-entangled score degrades more under the shift, as the mechanism predicts).

## α = 0.2  (coverage target 0.8) — **RED**

### ρ_test = 0.95 — TRIGGER (STD-f under-covers)

| method | marg cov | **worst-grp cov** [95% CI] | minority cov | avg size | WG(g*=1) size |
|---|---|---|---|---|---|
| STD-f | 0.790 | 0.200 [0.041,0.294] ⚠ | 0.200 | 0.818 | 0.800 |
| STD-cpt | 0.770 | 0.550 [0.357,0.738] ⚠ | 0.675 | 0.770 | 0.800 |
| Mondrian-f | 0.799 | 0.763 [0.724,0.800] ⚠ | 1.000 | 0.849 | 2.000 |
| Mondrian-cpt | 0.776 | 0.600 [0.390,0.772] ⚠ | 0.775 | 0.776 | 0.600 |
| RobustWasserstein | 1.000 | 1.000 [1.000,1.000] | 1.000 | 2.000 | 2.000 |

RobustWasserstein TV ball ε=0.213 → inflated α'=0.000.

Efficiency comparison on g* (worst group), at matched coverage:

| comparator | matched (≥1-α WG cov)? | size_diff (comp − STD-cpt) on g* [95% CI] | STD-cpt strictly smaller? |
|---|---|---|---|
| Mondrian-f | no | +1.200 [+1.050,+1.400] | no |
| Mondrian-cpt | no | -0.200 [-0.400,-0.050] | no |
| RobustWasserstein | yes | +1.200 [+1.050,+1.400] | **yes** |

_rho_test=0.95: STD-f UNDER-COVERS worst group (trigger active, g*=1). STD-cpt WG cov=0.550 [0.357,0.738] (FAILS (i): CI-lower vs 0.8). vs Mondrian-f: comparator does NOT reach 0.8 WG cov (not matched). vs Mondrian-cpt: comparator does NOT reach 0.8 WG cov (not matched). vs RobustWasserstein: size_diff=+1.200[+1.050,+1.400] (STD-cpt smaller). not GREEN here._

### ρ_test = 0.75 — TRIGGER (STD-f under-covers)

| method | marg cov | **worst-grp cov** [95% CI] | minority cov | avg size | WG(g*=1) size |
|---|---|---|---|---|---|
| STD-f | 0.713 | 0.270 [0.193,0.346] ⚠ | 0.345 | 0.828 | 0.850 |
| STD-cpt | 0.770 | 0.740 [0.655,0.778] ⚠ | 0.765 | 0.770 | 0.790 |
| Mondrian-f | 0.841 | 0.757 [0.703,0.802] ⚠ | 1.000 | 1.091 | 2.000 |
| Mondrian-cpt | 0.779 | 0.710 [0.630,0.776] ⚠ | 0.790 | 0.779 | 0.710 |
| RobustWasserstein | 1.000 | 1.000 [1.000,1.000] | 1.000 | 2.000 | 2.000 |

RobustWasserstein TV ball ε=0.256 → inflated α'=0.000.

Efficiency comparison on g* (worst group), at matched coverage:

| comparator | matched (≥1-α WG cov)? | size_diff (comp − STD-cpt) on g* [95% CI] | STD-cpt strictly smaller? |
|---|---|---|---|
| Mondrian-f | no | +1.210 [+1.140,+1.295] | no |
| Mondrian-cpt | no | -0.080 [-0.125,-0.030] | no |
| RobustWasserstein | yes | +1.210 [+1.140,+1.295] | **yes** |

_rho_test=0.75: STD-f UNDER-COVERS worst group (trigger active, g*=1). STD-cpt WG cov=0.740 [0.655,0.778] (FAILS (i): CI-lower vs 0.8). vs Mondrian-f: comparator does NOT reach 0.8 WG cov (not matched). vs Mondrian-cpt: comparator does NOT reach 0.8 WG cov (not matched). vs RobustWasserstein: size_diff=+1.210[+1.140,+1.295] (STD-cpt smaller). not GREEN here._

### ρ_test = 0.5 — TRIGGER (STD-f under-covers)

| method | marg cov | **worst-grp cov** [95% CI] | minority cov | avg size | WG(g*=1) size |
|---|---|---|---|---|---|
| STD-f | 0.557 | 0.280 [0.217,0.337] ⚠ | 0.328 | 0.816 | 0.865 |
| STD-cpt | 0.756 | 0.735 [0.670,0.759] ⚠ | 0.757 | 0.756 | 0.760 |
| Mondrian-f | 0.875 | 0.740 [0.683,0.784] ⚠ | 1.000 | 1.375 | 2.000 |
| Mondrian-cpt | 0.781 | 0.715 [0.646,0.759] ⚠ | 0.802 | 0.781 | 0.715 |
| RobustWasserstein | 1.000 | 1.000 [1.000,1.000] | 1.000 | 2.000 | 2.000 |

RobustWasserstein TV ball ε=0.344 → inflated α'=0.000.

Efficiency comparison on g* (worst group), at matched coverage:

| comparator | matched (≥1-α WG cov)? | size_diff (comp − STD-cpt) on g* [95% CI] | STD-cpt strictly smaller? |
|---|---|---|---|
| Mondrian-f | no | +1.240 [+1.175,+1.295] | no |
| Mondrian-cpt | no | -0.045 [-0.070,-0.020] | no |
| RobustWasserstein | yes | +1.240 [+1.175,+1.295] | **yes** |

_rho_test=0.5: STD-f UNDER-COVERS worst group (trigger active, g*=1). STD-cpt WG cov=0.735 [0.670,0.759] (FAILS (i): CI-lower vs 0.8). vs Mondrian-f: comparator does NOT reach 0.8 WG cov (not matched). vs Mondrian-cpt: comparator does NOT reach 0.8 WG cov (not matched). vs RobustWasserstein: size_diff=+1.240[+1.175,+1.295] (STD-cpt smaller). not GREEN here._

### ρ_test = 0.25 — TRIGGER (STD-f under-covers)

| method | marg cov | **worst-grp cov** [95% CI] | minority cov | avg size | WG(g*=1) size |
|---|---|---|---|---|---|
| STD-f | 0.420 | 0.223 [0.179,0.267] ⚠ | 0.285 | 0.785 | 0.773 |
| STD-cpt | 0.739 | 0.720 [0.636,0.742] ⚠ | 0.738 | 0.739 | 0.757 |
| Mondrian-f | 0.943 | 0.760 [0.665,0.811] ⚠ | 1.000 | 1.692 | 2.000 |
| Mondrian-cpt | 0.764 | 0.680 [0.625,0.723] ⚠ | 0.772 | 0.764 | 0.680 |
| RobustWasserstein | 1.000 | 1.000 [1.000,1.000] | 1.000 | 2.000 | 2.000 |

RobustWasserstein TV ball ε=0.474 → inflated α'=0.000.

Efficiency comparison on g* (worst group), at matched coverage:

| comparator | matched (≥1-α WG cov)? | size_diff (comp − STD-cpt) on g* [95% CI] | STD-cpt strictly smaller? |
|---|---|---|---|
| Mondrian-f | no | +1.243 [+1.200,+1.293] | no |
| Mondrian-cpt | no | -0.077 [-0.112,-0.048] | no |
| RobustWasserstein | yes | +1.243 [+1.200,+1.293] | **yes** |

_rho_test=0.25: STD-f UNDER-COVERS worst group (trigger active, g*=1). STD-cpt WG cov=0.720 [0.636,0.742] (FAILS (i): CI-lower vs 0.8). vs Mondrian-f: comparator does NOT reach 0.8 WG cov (not matched). vs Mondrian-cpt: comparator does NOT reach 0.8 WG cov (not matched). vs RobustWasserstein: size_diff=+1.243[+1.200,+1.293] (STD-cpt smaller). not GREEN here._

### Crux quantity (α=0.2) — score spurious-sensitivity vs coverage robustness (reported, NOT gated)
AUROC(score → spurious_attr) on calibration: f-score = 0.529, concept-score = 0.534. Mean worst-group coverage gap over triggered ρ: STD-f = 0.557, STD-cpt = 0.114. More spurious-sensitive score = **cpt**; larger coverage gap = **f**; rank-consistent = **False** (the more shortcut-entangled score degrades more under the shift, as the mechanism predicts).

## Verdict
**OVERALL VERDICT (α=0.1): RED** (GREEN ρ_test: none).

> Single-seed de-risk. Significance is the bootstrap CIs only; no claim beyond them. **STOP — await human review before any verifier/flatness ablation, CelebA, multi-seed, figures, or paper machinery.**
