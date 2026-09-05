# Representation-level dissociation test

End-to-end fine-tuned backbones (ERM / GroupDRO / group-balanced), each carrying the full
head x calibration comparison. Answers ACML R1.2, R2.1, R3.1: the representation itself
varies here, not only the last layer. Intervals are two-stage cluster bootstraps over
fine-tune seeds (splits nested within seed), per R2.3.

## Manipulation check (READ FIRST)

Did the robust objectives actually produce more robust representations? If not, the
null coverage results below say nothing about representations and must not be read
as support for the thesis.

**celeba: WEAK — a robust objective beats the reference only at head(s) ['dfr'], NOT at the primary head 'erm' (best gain there -0.002). The primary lever therefore compares near-identical representations, so its null is weak evidence about representations and must be reported as such. Note the effect is small in absolute terms (best gain +0.019 < 0.05), so the representation axis was barely moved**

> The representation axis was barely moved on **celeba**: best gain +0.019 across heads, but only -0.002 at the primary head `erm`. Read this dataset's primary-lever null as weak evidence about representations, and say so in the write-up.

| head | representation | eval wg acc | Δ vs erm [95% CI] | more robust |
|---|---|---|---|---|
| dfr | erm (ref) | 0.860 | — | — |
| dfr | groupdro | 0.879 | +0.019 [+0.007, +0.025] | yes |
| dfr | reweight | 0.846 | -0.014 [-0.071, +0.021] | no |
| erm | erm (ref) | 0.410 | — | — |
| erm | groupdro | 0.408 | -0.002 [-0.021, +0.036] | no |
| erm | reweight | 0.396 | -0.014 [-0.043, +0.043] | no |
| groupdro_ll | erm (ref) | 0.788 | — | — |
| groupdro_ll | groupdro | 0.789 | +0.001 [-0.025, +0.046] | no |
| groupdro_ll | reweight | 0.794 | +0.006 [-0.039, +0.060] | no |

## celeba/APS

| representation | Mondrian spread (across heads) | equivalence @ margin | marginal spread |
|---|---|---|---|
| erm | 0.006 (hi 0.014) | EQUIVALENT (±0.030) | 0.077 |
| groupdro | 0.003 (hi 0.009) | EQUIVALENT (±0.030) | 0.076 |
| reweight | 0.006 (hi 0.017) | EQUIVALENT (±0.030) | 0.085 |

**Primary lever comparison — representation only** (head held fixed at `erm`, so the only axis that varies is the representation). Best marginal representation `['groupdro']` = 0.819; worst Mondrian representation `['erm']` = 0.882. Difference +0.063 [+0.041, +0.084] (cluster (unpaired), 3 seeds).

**Verdict: CALIBRATION LEVER DOMINATES (worst Mondrian beats best marginal, CI excludes 0)**

Same comparison with each head held fixed:

| head held fixed | best marginal | worst Mondrian | difference [CI] | verdict |
|---|---|---|---|---|
| dfr | 0.895 (reweight) | 0.885 (erm) | -0.010 [-0.020, -0.000] | **marginal WINS** |
| erm | 0.819 (groupdro) | 0.882 (erm) | +0.063 [+0.041, +0.084] | Mondrian dominates |
| groupdro_ll | 0.895 (groupdro) | 0.879 (erm) | -0.015 [-0.028, -0.003] | **marginal WINS** |

_Secondary (strictly harder) bound, maximising the marginal side jointly over representation x head: best marginal `['reweight', 'dfr']` = 0.895 vs worst Mondrian `['erm', 'groupdro_ll']` = 0.879, difference -0.016 [-0.027, -0.005] -> **marginal_wins**. This lets a robust **head** substitute for the representation, so it tests a different question than the reviewers asked -- but where it reads `marginal_wins`, the unconditional dominance claim fails and the paper must say so._


**Does a robust representation fix marginal calibration by itself?** (paired, vs the ERM representation, same head)

| comparison | Δ worst-group cov [95% CI] | significant |
|---|---|---|
| groupdro_vs_erm/dfr | -0.006 [-0.012, -0.001] | yes |
| groupdro_vs_erm/erm | +0.009 [-0.006, +0.025] | no |
| groupdro_vs_erm/groupdro_ll | +0.008 [+0.000, +0.017] | yes |
| reweight_vs_erm/dfr | +0.009 [+0.004, +0.014] | yes |
| reweight_vs_erm/erm | +0.000 [-0.026, +0.023] | no |
| reweight_vs_erm/groupdro_ll | -0.001 [-0.015, +0.013] | no |

## celeba/RAPS

| representation | Mondrian spread (across heads) | equivalence @ margin | marginal spread |
|---|---|---|---|
| erm | 0.006 (hi 0.014) | EQUIVALENT (±0.030) | 0.083 |
| groupdro | 0.002 (hi 0.009) | EQUIVALENT (±0.030) | 0.081 |
| reweight | 0.007 (hi 0.018) | EQUIVALENT (±0.030) | 0.090 |

**Primary lever comparison — representation only** (head held fixed at `erm`, so the only axis that varies is the representation). Best marginal representation `['groupdro']` = 0.812; worst Mondrian representation `['erm']` = 0.882. Difference +0.070 [+0.047, +0.093] (cluster (unpaired), 3 seeds).

**Verdict: CALIBRATION LEVER DOMINATES (worst Mondrian beats best marginal, CI excludes 0)**

Same comparison with each head held fixed:

| head held fixed | best marginal | worst Mondrian | difference [CI] | verdict |
|---|---|---|---|---|
| dfr | 0.895 (reweight) | 0.885 (erm) | -0.010 [-0.020, -0.000] | **marginal WINS** |
| erm | 0.812 (groupdro) | 0.882 (erm) | +0.070 [+0.047, +0.093] | Mondrian dominates |
| groupdro_ll | 0.893 (groupdro) | 0.879 (erm) | -0.014 [-0.028, -0.001] | **marginal WINS** |

_Secondary (strictly harder) bound, maximising the marginal side jointly over representation x head: best marginal `['reweight', 'dfr']` = 0.895 vs worst Mondrian `['erm', 'groupdro_ll']` = 0.879, difference -0.016 [-0.027, -0.005] -> **marginal_wins**. This lets a robust **head** substitute for the representation, so it tests a different question than the reviewers asked -- but where it reads `marginal_wins`, the unconditional dominance claim fails and the paper must say so._


**Does a robust representation fix marginal calibration by itself?** (paired, vs the ERM representation, same head)

| comparison | Δ worst-group cov [95% CI] | significant |
|---|---|---|
| groupdro_vs_erm/dfr | -0.007 [-0.013, -0.001] | yes |
| groupdro_vs_erm/erm | +0.008 [-0.008, +0.025] | no |
| groupdro_vs_erm/groupdro_ll | +0.008 [+0.000, +0.018] | yes |
| reweight_vs_erm/dfr | +0.008 [+0.002, +0.014] | yes |
| reweight_vs_erm/erm | +0.001 [-0.026, +0.025] | no |
| reweight_vs_erm/groupdro_ll | -0.001 [-0.016, +0.015] | no |

## celeba/THR

| representation | Mondrian spread (across heads) | equivalence @ margin | marginal spread |
|---|---|---|---|
| erm | 0.004 (hi 0.021) | EQUIVALENT (±0.030) | 0.089 |
| groupdro | 0.007 (hi 0.018) | EQUIVALENT (±0.030) | 0.082 |
| reweight | 0.001 (hi 0.017) | EQUIVALENT (±0.030) | 0.092 |

**Primary lever comparison — representation only** (head held fixed at `erm`, so the only axis that varies is the representation). Best marginal representation `['groupdro']` = 0.802; worst Mondrian representation `['erm']` = 0.879. Difference +0.076 [+0.042, +0.107] (cluster (unpaired), 3 seeds).

**Verdict: CALIBRATION LEVER DOMINATES (worst Mondrian beats best marginal, CI excludes 0)**

Same comparison with each head held fixed:

| head held fixed | best marginal | worst Mondrian | difference [CI] | verdict |
|---|---|---|---|---|
| dfr | 0.891 (reweight) | 0.878 (groupdro) | -0.013 [-0.032, +0.008] | indistinguishable |
| erm | 0.802 (groupdro) | 0.879 (erm) | +0.076 [+0.042, +0.107] | Mondrian dominates |
| groupdro_ll | 0.855 (groupdro) | 0.874 (erm) | +0.019 [-0.011, +0.046] | indistinguishable |

_Secondary (strictly harder) bound, maximising the marginal side jointly over representation x head: best marginal `['reweight', 'dfr']` = 0.891 vs worst Mondrian `['erm', 'groupdro_ll']` = 0.874, difference -0.016 [-0.039, +0.006] -> **indistinguishable**. This lets a robust **head** substitute for the representation, so it tests a different question than the reviewers asked -- but where it reads `marginal_wins`, the unconditional dominance claim fails and the paper must say so._


**Does a robust representation fix marginal calibration by itself?** (paired, vs the ERM representation, same head)

| comparison | Δ worst-group cov [95% CI] | significant |
|---|---|---|
| groupdro_vs_erm/dfr | -0.006 [-0.014, +0.001] | no |
| groupdro_vs_erm/erm | +0.001 [-0.021, +0.024] | no |
| groupdro_vs_erm/groupdro_ll | +0.008 [-0.013, +0.029] | no |
| reweight_vs_erm/dfr | +0.000 [-0.012, +0.012] | no |
| reweight_vs_erm/erm | -0.002 [-0.041, +0.037] | no |
| reweight_vs_erm/groupdro_ll | +0.003 [-0.022, +0.026] | no |

## Did the fine-tune change the representation? (eval worst-group accuracy)

**celeba** — worst-group accuracy, mean over fine-tune seeds

| representation | dfr | erm | groupdro_ll |
|---|---|---|---|
| erm | 0.860 | 0.410 | 0.788 |
| groupdro | 0.879 | 0.408 | 0.789 |
| reweight | 0.846 | 0.396 | 0.794 |

## Sub-target diagnostic (min-over-groups selection effect)

- mean worst-group coverage: **0.8819**
- mean *over-groups* coverage: **0.9034** (this is the quantity Mondrian targets; it should sit at 1-α)
- gap between them: **0.0214** — the min-over-k effect, not a validity failure

