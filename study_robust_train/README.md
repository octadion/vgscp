# study_robust_train — Pre-Registered Campaign v2 ("Conformal Burden")

Clean module for the v2 hardened study. Phase-0 passed on real Colab (DFR materially > ERM
worst-group; the published 0.86–0.92 reference is ERM-ResNet-specific, so CLIP ViT-B/32 landing
lower is a reference mismatch, not a break). The module now also contains the **H1/H2/H3 grid**
(cheap-first last-layer arms) — which **STOPS** before the optional heavy full-GroupDRO fine-tune
and any 3rd/4th dataset.

See [../AUDIT_study.md](../AUDIT_study.md) for the full diff of the v4 codebase against the spec's
§2 hard preconditions and the reuse/dead-code decisions that shaped this module.

## What Phase-0 does (spec §1)
- Models: **ERM** and **DFR (last-layer retrain)** only, on **Waterbirds** only, on frozen CLIP
  ViT-B/32 features (no large-model training).
- Reports, over **3 seeds**: each model's **worst-group accuracy** + a **sample cross-group
  conformity-score divergence (APS)** — Wasserstein-1 and KS between the worst group and the rest.
- Then **STOPS.** No calibration grid, no ρ-sweep, no H1/H2/H3.

## The human STOP gate (spec §1, HARD)
When the **real** Phase-0 runs on Colab, the researcher checks the worst-group accuracies against
the published reference:
- **DFR worst-group ≈ 0.86–0.92** (Waterbirds, last-layer DFR).
- **ERM worst-group ≈ 0.60–0.75** (much lower).

If **DFR ≈ ERM** or **near chance (~0.5)** → the pipeline is broken → write `BLOCKERS.md`, **STOP,
do not proceed to the grid.** The Colab notebook automates this check and halts itself on failure.

Note: this is distinct from `experiments.real_data.GATE_MIN_TOP1` (0.55), which is a *pipeline-sanity
floor* on the species head — a necessary-not-sufficient precondition, not the Phase-0 gate.

## What is reused vs. fresh (audit-verified)
- **Reused as-is** (audit-verified live, AUDIT_study.md §4–5):
  `conformal.scores` (APS/RAPS/THR), `conformal.split_conformal`, `conformal.group_robust`,
  `experiments.real_data` head helpers (`fit_species_head` = StandardScaler→multinomial LogReg,
  `assert_l2_normalized`, `head_probs`).
- **Fresh** (the one true capability gap, AUDIT_study.md §3): `divergence.py` — cross-group
  Wasserstein-1 / KS conformity-score divergence (did not exist anywhere in v1–v4).
- **Hardened** (AUDIT_study.md Gap A): `heads.assert_multinomial_safe` makes the multinomial
  guarantee explicit/version-safe.
- **Excluded from the import path** (AUDIT_study.md §5): all `signals/registry`-routed reliability
  signals, `models/verifier*`, the `ks_conformal/` package, and the legacy verdicts. None is
  reachable from this module.

## Files
| file | role |
|---|---|
| `heads.py` | ERM + DFR last-layer heads on frozen features (reuses the standardized probe + L2 guard) |
| `divergence.py` | **fresh** §4 metric: W1 + KS cross-group conformity-score divergence (APS/RAPS/THR), per score function |
| `metrics.py` | per-group / worst-group accuracy on the 4-group structure |
| `phase0.py` | Phase-0 orchestrator (ERM+DFR, seeds → worst-group acc + APS divergence) then STOP |
| **`methods.py`** | the 5 last-layer methods: erm, dfr, **afr** (group-label-free), **groupdro_ll** (numpy online GroupDRO), **balanced_subsample** |
| **`conformal_eval.py`** | per-(score, ρ, cal-split) eval: calibrate@ρ_cal=0.95, eval@ρ_test → worst-group cov / set size / cov gap / cross-group divergence |
| **`accuracy_matching.py`** | §5 confound control — **accuracy-matched** divergence (the H1 readout); raw kept separate, labeled "uncontrolled" |
| **`verdicts.py`** | H1 (matched-divergence CI excludes 0 on ≥2/3 scores), H2 (accuracy-vs-burden ranking inversion), H3 (ρ-survival) |
| **`grid.py`** | grid orchestrator (backbone × dataset × method × seed × score × ρ × split) + §2 worst-group gate + CSV/RESULTS_study.md emitters |
| **`features.py`** | frozen backbones: CLIP ViT-B/32 (reused) + **ERM ResNet-50** train+extract (Colab-only) |
| **`datasets.py`** | backbone-agnostic dataset→`GridData` adapter (Waterbirds + CelebA, no CUB coupling) |
| **`figures.py`** | spec §8 figures: accuracy-matched burden, shift curves, H2 ranking scatter |
| `synthetic.py` | synthetic frozen-feature + GridData generators for LOCAL logic validation only |
| `validate_synthetic.py` | Phase-0 machinery synthetic logic check; **claims no real numbers** |
| **`validate_grid.py`** | full H1/H2/H3 grid synthetic logic check (5 methods, scores, ρ, matching, verdicts, emitters); **claims no real numbers** |

Backbone naming: `resnet50_erm` (primary, literature-comparable), `clip_vitb32` (secondary). The
spec's "CLIP linear probe" baseline == (`erm` × `clip_vitb32`); the backbone axis covers it (no
redundant arm). The heavy full-GroupDRO fine-tune and 3rd/4th datasets are **not** here (post-checkpoint).

## Run
```bash
# LOCAL — logic validation on synthetic features (no torch/data; claims NO real numbers):
python -m study_robust_train.validate_synthetic        # Phase-0 machinery (exit 0 on PASS)
python -m study_robust_train.validate_grid             # full H1/H2/H3 grid chain (exit 0 on PASS)
python -m pytest tests/test_celeba_parse.py            # CelebA metadata parser

# REAL Phase-0 (Colab GPU): notebooks/phase0_robust_train.ipynb  (done — passed)
# REAL grid   (Colab GPU): notebooks/grid_robust_train.ipynb
#   build features (CLIP + ERM-ResNet-50) for Waterbirds[+CelebA] -> 5 last-layer arms
#   -> RESULTS_study.md (accuracy-matched H1 up front) + CSVs + figures -> STOP before heavy arms.
# REANALYZE existing results (Tasks A/B, NO retraining): notebooks/reanalyze_results.ipynb, or
#   python -c "from study_robust_train.grid import reanalyze; reanalyze('results/study/grid_records.csv')"
```

**H2/H3 reporting (post-Waterbirds-run hardening):** H3 is measured on **burden survival** — divergence
survival AND **set-size-disparity relocation** vs ρ — with coverage stability reported separately (flat
coverage + growing sets = "relocate, not remove"). H2 inversions are flagged **`inversion_real`** only
when the cov_gap difference CI excludes 0 (overlapping CIs → noise). CelebA §2 gate: DFR hard-floor 0.80,
soft-flag <0.85; ERM-ResNet trains on a documented random subsample to stay <5h.
</content>
