# PROBE REPORT — CUB per-image attributes as the concept space on Waterbirds

**Date:** 2026-06-01 · **Status:** code complete + smoke-verified; **real run handed off to Colab
(CPU is fine) — numbers PENDING.** Do **not** build the premise-2 (Morgana) pipeline until the
human says "go" after reading the filled-in numbers. The probe script overwrites this file with
the real numbers + an auto-applied GREEN/AMBER/RED reading on the Colab run.

## Why this probe exists

Two prior concept-space attempts are dead ends:
- **CLEVR + GT scene graph = oracle** (verifier handed the causal rule → robust by construction).
- **Waterbirds + frozen CLIP = RED**: the global CLIP image embedding is dominated by the
  *background*, so the concept space is ~0.97 AUROC for `place` and carries no recoverable
  bird-type signal on the minority (minority AUROC(concepts→y) ≈ 0.49, ≈ 0.53 even after dropping
  background-correlated concepts).

The fix: use the dataset's own **per-image attribute annotations** as the concept space.
**CUB-200-2011** (which Waterbirds is built from) ships **312 MTurk attribute labels per image**
(e.g. `has_bill_shape::hooked`, `has_wing_color::black`). These describe the **bird**, are
independent of the pasted background, and therefore (hypothesis) carry minority bird-type signal
**and** are not background-contaminated — the canonical Concept Bottleneck Model setting (Koh et
al. 2020).

**This probe answers ONE question, then STOPS:** does the CUB per-image attribute concept space
have minority bird-type signal AND low background contamination — i.e. is **premise 1** solved on
Waterbirds? It does **not** test premise 2 (does Morgana/verifiability beat a plain probe).

## Honesty constraint — per-image attributes ONLY (no oracle)

- Concept vector for an image = its own 312 (presence) values from
  `attributes/image_attribute_labels.txt`.
- **FORBIDDEN:** `class_attribute_labels_continuous.txt` / any class-level / majority-vote
  attributes — identical for all images of a class ⇒ a deterministic function of the label ⇒
  injects the label into the concept space (the oracle trap). Per-image MTurk labels are noisy;
  that noise is what makes this an honest, imperfect concept space.
- Concepts are frozen annotations (no training, no finetuning). The bird **class** label is never
  used to build concepts.

## What was built (this turn) and verified

- `data/cub_attributes.py` — downloads CUB-200-2011, parses per-image attributes robustly
  (split on whitespace, read first 5 tokens, skip+count malformed lines — the official file has a
  known extra-token issue), and **joins to Waterbirds images** via the `species/filename.jpg`
  path suffix → an `(N, 312)` concept matrix aligned to a Waterbirds bundle split. **Asserts join
  coverage ≥ 99%** and reports unmatched examples (does not silently drop).
- `scripts/probe_waterbirds_cub.py` — the probe (mirrors the CLIP probe; reuses
  `probe_concept_usefulness` / `probe_contamination` from `scripts/probe_waterbirds_clip.py`
  unchanged), cache (`cub_concepts.npz` + `attribute_names.json`) + `probe_results.json` +
  manifest, prints the `==== WATERBIRDS CUB ... PROBE ====` summary, and auto-writes this report
  with the pre-committed reading. Has a `--smoke` synthetic self-test (no dataset).
- `configs/waterbirds_cub.yaml` — points at the Waterbirds root + CUB root, probe settings
  (`max_per_split: 1000`, binary `is_present`, standardize off by default).

**Verified here (no dataset):** `python -m scripts.probe_waterbirds_cub --smoke` passes — on
fabricated per-image binary attributes (some encode `y`, some encode `place`, rest noise) the
analysis recovers AUROC(attrs→y) ≈ 1.0 and AUROC(attrs→place) ≈ 0.85, and the verdict logic fires
correctly across GREEN/AMBER/RED branches. Only the real CUB/Waterbirds numbers remain.

## Reused as-is (no changes)

`data/waterbirds.py` (paths + `group_id` + `is_minority` + seeded disjoint splits, loaded with
`build_datasets=False` — no torch), `eval/metrics.py` (`auroc`), the probe analysis functions in
`scripts/probe_waterbirds_clip.py` (pure numpy/sklearn, concept-source-agnostic),
`ConceptStandardizer` from `models/concept_extractor_clip.py`, and `vgscp_logging/manifest.py`.

## The probe questions (filled from the Colab run)

| # | Question | Metric | How to read it |
|---|---|---|---|
| 1 | **Usefulness** | AUROC(attrs → bird type `y`), overall / majority / **minority** | attributes must carry **minority** bird signal (want ≥ ~0.65) |
| 2 | **Contamination** | AUROC(attrs → background `place`); per-attr ranking | want **clearly < ~0.8** (ideally ~0.5–0.65) vs CLIP's ~0.97 |

### Pre-committed reading rule (committed now, before numbers exist)

- **GREEN → premise 1 solved on real data; proceed to build the real pipeline + premise-2
  (Morgana) test.** Requires BOTH: minority `AUROC(attrs→y) ≥ ~0.65` **and** `AUROC(attrs→place)`
  clearly `< ~0.8` (ideally ~0.5–0.65). This is the exact contrast showing the annotation-based
  concept space fixes the CLIP background-domination.
- **AMBER** (minority ~0.6–0.65, or moderate contamination): proceed, but the premise-2 control
  baselines (plain attribute-probe) become decisive.
- **RED → this concept space also fails** if minority `AUROC(attrs→y) ≈ 0.5` (no bird signal even
  from annotations — double-check the join) **or** contamination ≈ CLIP's ~0.97 (would indicate a
  join bug).

> The probe is a gate on premise 1, not the paper. The real GO/NO-GO is premise 2 (does Morgana/V
> beat a plain attribute probe + trust + ensemble on the minority), tested only after GREEN.

## How to run it (Colab — CPU is fine, no GPU needed) — copy-paste

```python
from google.colab import drive; drive.mount('/content/drive')
%cd /content
!git clone https://github.com/octadion/vgscp.git 2>/dev/null || (cd vgscp && git pull)
%cd /content/vgscp
!pip -q install scikit-learn pandas pyarrow tqdm PyYAML

import yaml, os
WB  = '/content/waterbirds'            # reuse the local Waterbirds extraction (tarball cached on Drive)
CUB = '/content/cub'                   # CUB extracts here (local; only the 312-attr txt is read after)
os.makedirs(WB, exist_ok=True); os.makedirs(CUB, exist_ok=True)
cfg = yaml.safe_load(open('configs/waterbirds_cub.yaml'))
cfg['dataset']['root'] = WB
cfg['cub']['root'] = CUB
cfg['cub']['download'] = True
cfg['cub']['url'] = 'https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz'
cfg.setdefault('probe', {}); cfg['probe']['max_per_split'] = 1000
yaml.safe_dump(cfg, open('configs/waterbirds_cub_colab.yaml','w'), sort_keys=False)

!python -m scripts.probe_waterbirds_cub --config configs/waterbirds_cub_colab.yaml --timestamp colabCUB
# copy the cache to Drive so it survives the session:
import glob, shutil
npz = sorted(glob.glob('results/runs/*colabCUB*/cub_concepts.npz'))
if npz: shutil.copy(npz[-1], '/content/drive/MyDrive/vgscp/cub_concepts.npz')
```

Paste the printed `==== WATERBIRDS CUB PER-IMAGE ATTRIBUTE PROBE ====` summary back here. The
script also auto-fills this `PROBE_REPORT_CUB.md` with the real numbers + the GREEN/AMBER/RED read.

## Results (PENDING — fill from the Colab run)

| Metric | Value |
|---|---|
| join coverage (matched / total) | _pending (must be ≥ 0.99)_ |
| usefulness AUROC(attrs→y) overall / majority / **minority** | _pending_ |
| contamination AUROC(attrs→place), whole-vector | _pending_ |
| top contaminating attributes | _pending_ |
| reading | _pending: GREEN / AMBER / RED_ |

## What is intentionally NOT built (post-GREEN premise-2 step — needs "go")

The verifier, the CLIP-space controls, the conformal pipeline, and `run_phase1` changes are NOT
built here. Build the probe, run it, read GREEN/AMBER/RED, then STOP.

**STOP. Awaiting the Colab probe numbers + an explicit "go" before any premise-2 work.**
