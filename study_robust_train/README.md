# study_robust_train — Pre-Registered Campaign v2 ("Conformal Burden")

Clean module for the v2 hardened study. **This turn delivers Phase-0 only.** The full H1/H2/H3
grid is intentionally **not** here — Phase-0-then-STOP is a hard discipline (spec §1): run the
minimal pilot, hand the numbers to the researcher, and **halt for human review** before any grid.

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
| `metrics.py` | per-group / worst-group accuracy on the Waterbirds 4-group structure |
| `phase0.py` | Phase-0 orchestrator (ERM+DFR, seeds → worst-group acc + APS divergence) then STOP |
| `synthetic.py` | synthetic frozen-feature generator for LOCAL logic validation only |
| `validate_synthetic.py` | runs the machinery on synthetic data, asserts the wiring; **claims no real numbers** |

## Run
```bash
# LOCAL — logic validation on synthetic features (no torch/data; claims NO real numbers):
python -m study_robust_train.validate_synthetic        # exits 0 on PASS

# REAL Phase-0 (Colab GPU): notebooks/phase0_robust_train.ipynb
#   fetch/cache Waterbirds frozen CLIP features -> train ERM + DFR (3 seeds)
#   -> print worst-group accuracies + sample APS divergence -> CHECK the DFR~0.86-0.92 /
#      ERM~0.6-0.75 gate -> STOP. It does NOT proceed to the grid.
```
</content>
