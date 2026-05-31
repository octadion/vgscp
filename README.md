# VG-SCP — Verifiability-Gated Selective Conformal Prediction under Spurious Correlation

Research codebase for the claim: **under shortcut learning, NCV concept-level *verifiability*
`V` is a better reliability signal than confidence / trust score / deep-ensemble disagreement
for selective prediction on the minority (conflict) group**, because `V` is computed in concept
space and adversarially stress-tested, so it is robust to the shortcut.

> **Status: PHASE 1 (kill-switch) only.** Phase 2+ is intentionally not built. See
> [`PHASE1_REPORT.md`](PHASE1_REPORT.md) for the GO / NO-GO verdict. Do not proceed past Phase 1
> without an explicit human "go".

## What Phase 1 establishes (the whole paper rests on this)

On CLEVR-Hans3, in the **minority/conflict group**, `V_full` must achieve
**error-detection AUROC strictly greater than `trust` AND `ensemble-disagreement`** (non-overlapping
95% bootstrap CIs or paired p<0.05) **and** a lower minority selective-risk AURC. The synthetic
Gaussian testbed must reproduce **P2 (separation)**. If `V` merely ties trust/ensemble → **NO-GO**,
and we write `PHASE1_NEGATIVE_REPORT.md`. Beating only MSP is *insufficient*.

## Method (precise definitions: `signals/`, `conformal/`)

Reliability signals (all "higher = more reliable"): MSP confidence, trust score (Jiang et al.),
deep-ensemble disagreement, MC-dropout, and **NCV verifiability**
`V_full = β·V_comp + (1−β)·V_sound`, where `V_comp = p_A(ŷ|S_M)` and
`V_sound = 1 − clip(R_adv,0,1)`. Primary method = verifiability-gated selective conformal
(x-measurable gate ⇒ valid retained marginal coverage). Secondary = verifier-aware nonconformity
score.

## Repository layout

```
configs/        YAML configs; perf.yaml is shared and FIXED across the whole suite (fairness)
data/           synthetic, clevr_hans3 loaders, waterbirds/celeba stubs; group labels
models/         base f, ensemble, mc_dropout, verifier_adapter.py (NCV wrapper)
signals/        confidence, trust, ensemble, mcdropout, ncv (V_comp/V_sound/V_full)
conformal/      THR/APS/RAPS scores, split-conformal, selective gate, verifier-aware, validity asserts
precompute/     PRECOMPUTE-ONCE, CACHE-EVERYTHING (logits/features/concepts/p_A)
eval/           metrics, bootstrap CIs, paired significance tests
theory/         synthetic Gaussian P1/P2 verification (pure numpy/scipy — runs on CPU, no torch)
vgscp_logging/  parquet per-sample logger, env/gpu/git capture, run manifest
viz/            make_figures.py (PDF), latex_tables.py
scripts/        run_phase1.py (kill-switch), reproduce.sh
tests/          conformal quantile + gate-validity unit tests
```

> **Naming note:** the logging package is `vgscp_logging/`, not `logging/`, to avoid shadowing
> Python's stdlib `logging` when the repo root is on `sys.path`. The spec's `logging/` role maps
> here.

## Design pillar: PRECOMPUTE-ONCE, CACHE-EVERYTHING (Section 13)

Each frozen model (f, concept extractor, ensemble members, MC-dropout passes) is run over **all**
splits **one time**, in large batches under `torch.inference_mode()` + autocast, and cached to
disk. Afterwards every signal, conformal variant, budget and ablation is a cheap vectorized op
over cached tensors. This gives (a) high GPU utilization during the one-time precompute,
(b) **identical inputs across all methods ⇒ fair comparison by construction**, (c) ablations in
seconds. The `perf` config (precision, batch policy via VRAM probe, AMP/TF32) is set once and
never changed mid-suite.

## Quickstart

```bash
# 1. Theory testbed — no GPU, no torch. Verifies P1/P2 and writes the theory figure.
python -m theory.run_synthetic --config configs/synthetic.yaml

# 2. CLEVR-Hans3 (needs a GPU box + dataset). Precompute caches, then compute everything.
export CLEVR_HANS3_ROOT=/path/to/CLEVR-Hans3
python -m scripts.run_phase1 --config configs/clevr_hans3.yaml

# 3. Regenerate all figures/tables from cached parquet logs.
python -m viz.make_figures --run results/runs/<run_id>
python -m viz.latex_tables --run results/runs/<run_id>
```

Reproduce Phase 1 end-to-end: `bash scripts/reproduce.sh`.

## Honesty constraints (baked into the code, do not violate)

- The contribution is **selective prediction / set trustworthiness**, NOT a new coverage guarantee.
  Conformal coverage is a validity check, not the headline.
- The advantage is **regime-specific**; we MAP the regime (shortcut-strength sweep), never hide it.
- The conformal gate is a function of `x` only — calibration labels are never used to select the
  gate or score hyperparameters. This is asserted in code (`conformal/validity.py`).
