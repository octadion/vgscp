# PREMISE-2 REPORT — does adversarial verifiability (Morgana) beat a plain concept probe?

**Date:** 2026-06-01 · **Status:** code complete, faithful PVG fixed + unit-tested, full
pipeline **smoke-verified on CPU**; **real Waterbirds/CUB/CLIP run handed off to Colab — numbers
PENDING.** Running `scripts/run_premise2.py` on Colab (GPU) overwrites this file with the real
GREEN/PARTIAL/NULL numbers and the auto-applied verdict. **STOP and await the filled-in numbers +
human review before any multi-seed paper sweep.**

---

## 0. The premise-2 question (the project's actual novelty)

Premise 1 is GREEN: CUB per-image attributes give minority `AUROC(attrs→y)=0.958` and contamination
`AUROC(attrs→place)=0.526` (≈ chance) — a strong, background-invariant concept space. But that
cleanliness is the catch: an attributes-only space has **no spurious concept** for the adversary to
exploit, so we expect `V_comp` to beat trust/ensemble via *bottleneck cleanliness* while Morgana
adds ~nothing. That would make the win "clean concepts", **not** the novel verifiability mechanism.

**Premise 2:** does the Prover–Verifier adversarial mechanism add value **over a plain probe on the
same concept space**, and over trust + ensemble, at flagging **minority** errors? This is only a
fair test in a concept space containing **both reliable and spurious concepts** — so the adversary
has something to expose.

**Honesty constraints honored:** contribution = selective prediction / set trustworthiness (NO new
coverage guarantee); a NULL/PARTIAL result is valid and reported plainly; the concept bank, masks,
and training were **not** tuned to manufacture a Morgana win.

## 1. The decisive design — a concept-space contrast (same Waterbirds `f`)

| Space | Concepts | Role | Expectation |
|---|---|---|---|
| **A. attributes_only** | 312 CUB per-image MTurk attributes (clean/causal) | cleanliness control | `V_comp` wins via bottleneck cleanliness; **Morgana idle** (`V_full ≈ V_comp`) |
| **B. mixed** | `[312 CUB attributes \| 16 frozen-CLIP scene/background concepts]` | **decisive test** | the CLIP scene concepts are the **spurious** concepts; Morgana can now expose them |
| (C. clip_only) | frozen-CLIP global embedding | known **RED** (probe) | minority `AUROC(concepts→y)≈0.49` — cited, not rerun |

The CLIP scene prompts are exactly the background/scene subset that scored ~0.97 AUROC for `place`
in the CLIP probe (fixed a-priori in `configs/premise2_waterbirds.yaml::clip.scene_concept_bank`).

## 2. Pre-committed premise-2 criterion (committed in code BEFORE the real numbers)

Enforced by `eval/phase1_eval.py::premise2_verdict`, on the Waterbirds **minority** group, with
paired bootstrap vs `V_full`:

- **NOVELTY-VALIDATED** iff, in the **mixed** space, `V_full` beats **`probe_concept` AND
  `trust_concept` AND `trust` AND `ensemble_disagree`** on minority error-detection AUROC
  (non-overlapping 95% CI **or** paired p<0.05) **AND** `V_full` > `V_comp` (Morgana adds value).
- **PARTIAL** if `V` beats the `f`-baselines but **not** the concept-space controls
  (`probe_concept`/`trust_concept`) ⇒ "clean concepts help", not adversarial verifiability; **or**
  `V` beats the controls but `V_full` ≯ `V_comp` (Morgana on/off shows no gap).
- **NULL** if `V` beats **no** concept-space control anywhere ⇒ no contribution beyond a plain probe.

## 3. The faithful PVG fix (the core delta — `models/verifier_adapter.py::ReimplNCV`)

**The old bug.** The previous reimpl trained Arthur on **random sparse masks** for both Merlin and
Morgana — contradictory targets on the same input distribution; greedy selection only happened at
inference. That is not a Prover–Verifier Game and cannot test premise 2.

**The fix** (reference: ZIB-IOL/merlin-arthur-classifiers; Turan et al. arXiv:2507.07532), over
**continuous concept vectors**:

- **Arthur** = MLP `A([concepts ⊙ mask | mask]) → logits` over `C` classes (+ a reject `⊥` head
  when Morgana is on).
- **Merlin** (cooperative) = greedy sparse subset **maximizing** `p_A(y_target | S)`.
- **Morgana** (adversarial) = greedy sparse subset **maximizing** `p_A(y' | S)` for the best wrong
  `y'`.
- **Training is the actual alternating game** (not random masks): every `prover_refresh` epochs the
  greedy Merlin/Morgana selections are recomputed from the **current** Arthur; Arthur is then
  updated to be **correct under Merlin's helpful set** (toward the TRUE label) and to **reject under
  Morgana's misleading set**. Concept inputs are standardized with **TRAIN-only** statistics fit in
  `train()` and reapplied in `predict()` (no leakage).
- **`morgana: on|off` switch** (`build_verifier`): OFF drops the adversarial branch entirely (no
  reject head, `S_A = S_M`, zero reject), so `V_full = V_comp` — the required ablation. A
  Morgana-OFF verifier is trained alongside the ON one and reported as `V_comp_moff`.
- Caches `p_A(·|S_M)`, `p_A(·|S_A)`, reject prob unchanged, so `signals/ncv.py`
  (`V_comp`, `R_adv`, `V_sound`, `V_full`) and `β`-tuning on D_learn work as-is.

**Unit test** (`tests/test_verifier_pvg.py`, runs on CPU): on a tiny synthetic concept space with a
planted spurious concept that flips on a 25% minority, after PVG training Arthur's
**accuracy-under-Merlin (1.00) > accuracy-under-Morgana (0.75)** — the game is wired correctly
(Merlin helps, Morgana bites exactly on the minority where the spurious concept misleads). The
Morgana-OFF ablation is verified to have no reject head and `S_A == S_M`.

## 4. What else was built / reused

**New:** `signals/concept_probe.py` (`probe_concept` = max-softmax confidence of a plain
concept→y probe trained TRAIN-only; `trust_concept` = Jiang et al. trust score IN the concept
space) wired into `signals/registry.py`; `eval/phase1_eval.py::premise2_verdict` (criterion §2, with
branch unit tests in `tests/test_premise2_verdict.py`); `configs/premise2_waterbirds.yaml`;
`scripts/run_premise2.py`.

**Reused unchanged:** `data/cub_attributes.py` (per-image attribute join, now over all 4 splits),
`models/concept_extractor_clip.py` (frozen CLIP + the scene prompts), `models/base_model.py` (ERM
ResNet-50), `precompute/` stage helpers, `signals/` (`conf_msp`, `trust`, `ensemble_disagree`,
`mcdropout`, `ncv`), `conformal/` (gated selective conformal + validity asserts), `eval/`
(`bootstrap`, `metrics`, `evaluate_signals`), `viz/`.

**Right-sized per the task:** short ERM (5 epochs), 3 ensemble members, 1 seed; concepts/verifier/
signals/conformal all CPU/vectorized after the one GPU precompute.

## 5. Smoke verification (executed here, CPU, synthetic — pipeline proof, NOT a premise-2 claim)

`python -m scripts.run_premise2 --smoke` fabricates a shortcut `f` (reads the background, so it is
confident-but-wrong on the minority) plus both concept spaces (clean core attributes; mixed adds
spurious place-tracking concepts), then runs the **entire** verifier→signals→verdict→conformal
pipeline. Result:

- **Verifier sanity matches theory:** Morgana-on completeness = 1.000 in both spaces; Morgana's
  `morgana_acc` = **0.960 in attributes_only (idle)** vs **0.698 in mixed (bites)** — the adversary
  exposes the spurious concepts exactly where they exist.
- **Selective-conformal validity holds:** 26/27 (score × budget) points pass retained marginal
  coverage ≥ 1−α; worst realized single-draw coverage ≈ 0.83 (finite-sample noise).
- **Verdict logic fires:** PARTIAL in both synthetic spaces (the clean synthetic saturates
  `trust_concept` at AUROC 1.00, so `V` cannot beat that control — the "clean concepts help"
  branch). All three verdict branches (NOVELTY-VALIDATED / PARTIAL / NULL) are covered by
  `tests/test_premise2_verdict.py`. **All 26 unit tests pass.**

The smoke proves the machinery and the verdict; the real premise-2 answer requires the real
concept spaces (below).

## 6. How to run the REAL gate (Colab, GPU once) — copy-paste

```python
from google.colab import drive; drive.mount('/content/drive')
%cd /content
!git clone https://github.com/octadion/vgscp.git 2>/dev/null || (cd vgscp && git pull)
%cd /content/vgscp
!pip -q install torch torchvision open_clip_torch==2.24.0 scikit-learn pandas pyarrow tqdm PyYAML

import os
os.environ['WATERBIRDS_ROOT'] = '/content/waterbirds'   # tarball cached on Drive; extracted local
os.environ['CUB_ROOT']        = '/content/cub'
# CUB download via Caltech URL needs a UA header (plain urllib gets 403). If the in-code urllib
# download 403s, pre-fetch once with wget -U and point cub.root at the extraction:
# !wget -U "Mozilla/5.0" -O /content/cub/CUB_200_2011.tgz \
#     https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz

!python -m scripts.run_premise2 --config configs/premise2_waterbirds.yaml --timestamp colabP2
# -> results/runs/premise2_waterbirds_colabP2/  (premise2_results.json, PREMISE2_REPORT.md,
#    logs/*.parquet, manifest.json). Then:
!python -m viz.latex_tables --run results/runs/premise2_waterbirds_colabP2
!python -m viz.make_figures  --run results/runs/premise2_waterbirds_colabP2
```

The run prints the per-space VERDICT and overwrites this `PREMISE2_REPORT.md` with the real numbers.

## 7. Results (PENDING — auto-filled by the Colab run)

| Concept space | minority AUROC: `V_full` / `probe_concept` / `trust_concept` / `trust` / `ensemble` | `V_full`−`V_comp` | Verdict |
|---|---|---|---|
| attributes_only | _pending_ | _pending_ | _pending (expect PARTIAL: Morgana idle)_ |
| **mixed** | _pending_ | _pending_ | _pending (decisive)_ |

Regime check (f): worst-group acc _pending_ (must be clearly below overall, ~0.4–0.7, not
degenerate). Verifier intrinsic completeness/soundness (morgana on & off): _pending_. Selective-
conformal confirmation on the winner: _pending (assert retained coverage ≥ 1−α)_.

## 8. Regime map (honesty) + what is intentionally NOT done

- **attributes_only**: clean/causal → expect `V_comp` wins via cleanliness, Morgana idle.
- **mixed**: the make-or-break setting; the CLIP scene concepts are the spurious concepts.
- **clip_only**: known RED (background-dominated, minority `AUROC(concepts→y)≈0.49`) — cited.

NOT done (per scope): no multi-seed paper sweep, no CelebA/CLEVR/Phase-3, no new coverage-guarantee
claim, no tuning of the concept bank/masks to win. **STOP — await the Colab numbers + human review.**
