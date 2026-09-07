# Revision log — ACML / Machine Learning, submission 62d20e3d-c3eb-4568-b964-7b09cf6f583b

Working record of every change made for the revision, written **as each edit is made** so the
point-by-point response can quote locations and text rather than reconstruct them afterwards.

**Anchors for the before/after document** (verified against `git log`, not recalled):

| commit | date | diff on `sn-article.tex` | what it is |
|---|---|---|---|
| `49e503c` | 2026-09-01 | +1034 / −0 | the manuscript **as submitted** — the "before" for everything |
| `e608533` | 2026-09-02 | +126 / −26 | first round of edits, made before the reviews were re-read |
| working tree | 2026-09-06/07 | uncommitted | title, abstract, `sec:repr`, softened §c1 claim |

`e608533` touched exactly three places, and nothing else in the manuscript:

1. line ~279 — the coverage-gap definition in §3 (internal consistency, see below);
2. lines ~310–319 — Proposition 1(ii) rewritten, +75/−39 (answers **R2.2**);
3. line ~972 — new appendix `app:subtarget`, +61 (answers **R1.1**).

There is only one `.tex` file in the repository, so every manuscript edit discussed anywhere in
this log is in `ACML_Journal___Robust_CP_Train_Study (1)/sn-article.tex`.

Each entry gives: the reviewer point it answers → what changed → where → the new text verbatim.

---

## R2.2 — "Wasserstein-1 or KS divergence alone generally does not imply a monotonic reduction in coverage at the pooled threshold. Please state sufficient assumptions or present the divergence–coverage link as an empirical hypothesis."

**Status: done** (in `e608533`, before the reviews were re-read; retained and verified)

**Where:** §3, Proposition 1(ii), Eqs. `eq:shortfall`/`eq:ksbound`, and the two Remarks that follow
(tex lines ~322–390).

**Change.** Proposition 1(ii) previously asserted that the coverage shortfall grows with the
divergence $D$. It was rewritten around an *exact identity* plus a bound that actually holds, and a
remark that states plainly what the divergences do and do not give.

**New text (verbatim).**

> **Remark (what the divergences do and do not give).** An earlier version asserted that the
> shortfall grows with the divergence $D$ of (eq:div); we thank a reviewer for observing that this
> does not follow. What is true: the shortfall is the cross-group CDF gap *at the threshold*,
> scaled by $(1-\pi)$, and $D_{\mathrm{KS}}$ bounds it because it is a supremum over $t$. $W_1$
> admits no such bound — it integrates the gap over the line while the shortfall depends on it at
> one point, so the gap can concentrate on a vanishing interval with $W_1 \to 0$ and the shortfall
> fixed — so we use $W_1$ only as a descriptive summary of heterogeneity. Neither divergence
> *orders* coverage without a direction assumption: $P_{\mathrm{wg}}$ stochastically *below* the
> pool has the same $W_1$ and $D_{\mathrm{KS}}$ as its mirror image above it but yields a coverage
> surplus. Monotonicity needs $P_{\mathrm{wg}} \succeq_{\mathrm{st}} P_{\neg\mathrm{wg}}$, which
> holds here by construction, the worst group being defined as the lowest-coverage one — an
> assumption about which group is worst, not a consequence of the divergence.

---

## Internal consistency (not raised by a reviewer, but a genuine error)

**Status: done** (in `e608533`)

**Where:** §3, definition of the coverage gap (tex line ~285).

**Change.** §3 defined the coverage gap in a way that contradicted both the implementation and
Appendix A. It now reads $(1-\alpha) - \min_g \mathrm{cov}_g$, matching the code and the appendix.
Flagged here because the response letter should disclose it rather than let a reader discover a
silent definitional change.

---

## R1.1 — "the authors state that $1-\alpha=0.90$ is the target throughout. However, their empirical values are around 0.84–0.89 … clearly explain the reason for this difference." (also R3, "Mondrian also does not always reach the target coverage")

**Status: partially done** — Appendix added in `e608533`; the decisive empirical number is new and
still to be written in.

**Where:** Appendix `app:subtarget` (+ `tab:subtarget`), and now also the abstract.

**Change so far.** A new appendix explains the level as a min-over-$k$ selection effect and reports
a simulation of an *exactly valid* Mondrian procedure showing the same sub-target minimum.

**Still to write.** The direct measurement, which settles it: Mondrian's **mean-over-groups**
coverage is $0.9024$ against a nominal $0.900$ across all eight backbone–dataset cells
(range $0.8995$–$0.9056$), while the *minimum* over four groups sits at $0.856$–$0.886$. The
observed shortfall $0.018$–$0.045$ is somewhat larger than the $1.03\,\mathrm{SD}$ that the
min-over-$k$ argument predicts ($0.013$–$0.030$); that gap must be reported, not smoothed over.

---

## R1.2 / R2.1 — "provide additional experiments where the representation itself is trained or fine-tuned" / "add a representation-changing robustness experiment if feasible; otherwise narrow the title, abstract and conclusions"

**Status: title and abstract done; the new results subsection is still to be written.**

**Where:** title; abstract, sentences 4–5 and the second paragraph.

**Change (title).**

- Before: *Worst-Group Conformal Reliability under Spurious Correlation Is a Calibration Problem,
  **Not a Representation Problem**: A Confound-Controlled Multi-Backbone Study*
- After: *Worst-Group Conformal Reliability under Spurious Correlation Is **More a Calibration
  Problem than a Representation Problem**: A Confound-Controlled Multi-Backbone Study*

The absolute negation is gone; the claim is now comparative, which is what the evidence supports
(cross-training spread $0.024$ under Mondrian against $0.358$ under marginal calibration).

**New text (abstract, verbatim).**

> We test that intuition on *both* levers. With the representation held fixed we run a
> confound-controlled grid over … ; separately, we move the representation *itself*, fine-tuning
> the backbone end-to-end under ERM, GroupDRO and reweighting.

> Fine-tuning the representation does not overturn this: with the head held fixed, calibration
> dominates in all six dataset–score cells.

**Change (new results subsection).** `\subsection{Moving the Representation Itself}`
(`sec:repr`, 66 lines incl. `tab:repr`), inserted immediately after the H(c1) result it
generalises. The manuscript as submitted did not mention this study at all.

It reports the end-to-end fine-tuning of the backbone under three objectives (ERM, GroupDRO,
reweighting), five seeds each, with the head held fixed at plain ERM so the representation is the
only thing that moves. It opens with a **manipulation check** — without evidence that the
intervention actually moved the representation, the experiment answers nothing:

> GroupDRO fine-tuning raises worst-group accuracy from $0.516$ to $0.738$ on Waterbirds ($+0.222$)
> and from $0.396$ to $0.470$ on CelebA ($+0.074$): the lever was pulled, and by the largest margin
> any intervention in this paper produces. Reweighting did not help ($0.466$ and $0.364$, both
> slightly below ERM). We report it regardless — a representation that fails to improve is still a
> representation that changed, and dropping it would be selecting on the outcome.

and concludes:

> The dissociation is therefore not an artefact of freezing the backbone. Even when the
> representation is genuinely improved, switching the calibration rule buys more worst-group
> coverage than switching the representation, and group-conditional calibration removes almost all
> of the difference between representations.

`tab:repr` gives the numbers: under marginal calibration the three representations span $0.249$ on
Waterbirds; under Mondrian they sit within $0.004$ of one another, all at $\approx 0.86$.

**Change (softening the absolute claim in §c1).** The submitted text ended the H(c1) result with a
sentence that asserted exactly what R3 objected to.

- Before: "**Worst-group coverage is delivered by the calibration mechanism; the representation
  does not provide it.**"
- After: "**Worst-group coverage here is governed by the calibration mechanism far more than by the
  backbone.** Whether that survives when the representation *itself* is retrained — rather than a
  new head fitted on frozen features — is the question of Section~\ref{sec:repr}."

Note the terminology correction embedded in it: the submitted paper used "representation" for the
*frozen backbone*, which it never varied. That conflation is part of what prompted R1.2 and R2.1.
Throughout the revision, **backbone** now means the frozen feature extractor, **representation**
means what the fine-tuning study actually changes, and **head** means the last-layer classifier.

---

## R1.3 — "Only two backbones are used, which limits the generalizability … results should be compared more clearly with those of other related studies."

**Status: backbone half done in the abstract; every table and the literature comparison still to do.**

**New text (abstract, verbatim).**

> … *four* frozen backbones spanning two architecture families and three pretraining regimes (an
> ERM-trained ResNet-50, CLIP ViT-B/32, DINOv2 ViT-B/14 and a supervised ViT-B/16) …

**Still to do.** Update `tab:c1`, `tab:c2`, `tab:h1`, `tab:h2`, `tab:predgroup` to four backbones;
add the literature comparison table.

---

## R3 — "the accuracy–set size correlation is weak in one of the four settings … provide statistical significance or uncertainty for the reported correlations" (also R2.6, "moderate broad empirical statements")

**Status: abstract done; tables still to do.**

**Change.** The submitted abstract reported the correlation as a headline finding
("correlation $-0.85$ to $-1.00$"). With cluster-bootstrap confidence intervals it is
uninformative in five of eight cells — worse than the reviewer supposed — and is now reported as a
direction, not an estimate.

**New text (abstract, verbatim).**

> Training buys *efficiency* rather than coverage, but conditional on Mondrian and as a tendency
> rather than a law: worst-group set size falls as base accuracy rises, yet with cluster-bootstrap
> confidence intervals the per-cell accuracy–size correlation is uninformative in five of eight
> cells, so we report a direction rather than an estimate.

---

## R2.5 — "The rho sweep mainly appears to change group mixture/correlation strength rather than the within-group data distribution, so the robustness claim should be scoped accordingly."

**Status: abstract done; §H3 text still to do.**

**New text (abstract, verbatim).**

> Under the correlation-strength sweep — which shifts the group mixture rather than the
> within-group distribution — Mondrian's worst-group coverage is flat to within $0.010$.

---

## Still to do (nothing below has been written yet)

| point | what is needed |
|---|---|
| R1.1 | write the mean-over-groups $0.9024$ measurement into §H3/`app:subtarget` |
| R1.2 / R2.1 | new results subsection: the fine-tuning (two-lever) study |
| R1.3 | four-backbone rows in every table; literature comparison table |
| R1.4 | more seeds — and disclose that the seed is inert for ERM and AFR (lbfgs ignores `random_state`; across-seed SD is exactly $0.000$) |
| R1.5 | add the Scientific African (2024) citation, doi 10.1016/j.sciaf.2024.e02135 |
| R2.3 | cluster-bootstrap CIs in the main tables (splits nested within seeds) |
| R2.4 | sensitivity analysis with the gated arms included; state that the floors were **pre-specified in code on 2026-06-11**, not preregistered in a registry |
| R2.5 | predicted-group mechanics: how predicted attributes set thresholds, what supervision is required, how probe error propagates; privacy/ethics of inferring the CelebA *Male* attribute |
| R2.6 | moderate "training buys efficiency" and "accuracy is a good proxy" in the body; exact calibration/test construction, per-group counts, hyperparameters, shift-robust procedure, seeds; archival code link with reproducible commands |
| R3 | correlation CIs in `tab:c2`; state the efficiency claim as conditional on Mondrian |
| — | disclose: mean set size below 1 on DINOv2/Waterbirds (empty sets are a legitimate outcome of standard split conformal); AFR's exclusion is not an artefact of an untuned $\gamma$ (oracle upper bound $0.327$–$0.477$ against a floor of $0.75$) |
| — | trim to the 20-page limit (includes references and appendices) |

---

## R1.3 (backbones, tables) + R1.1 — `tab:c1` and §c1 rewritten to eight cells

**Status: done**

**Where:** §c1 prose and `tab:c1`.

**Change.** `tab:c1` listed every method under every policy for **four** cells (45 lines). It now
reports **all eight** backbone--dataset cells in 30 lines: the same ERM model under both policies
with its lift, and the across-training spread under each policy. The per-method breakdown moves to
the released records. Rows are generated from `calibration_ablation_4bb.csv` by
`gen_tab_c1.py`, not hand-typed.

**New text (verbatim, §c1).**

> The dissociation is uniform across all eight cells. Under *marginal* calibration, worst-group
> coverage is governed by the training method: the across-training spread reaches $0.358$
> (CLIP/Waterbirds) and ERM is the worst arm in every cell, falling as low as $0.509$. Under
> *Mondrian* it is essentially flat across training methods everywhere — the spread never exceeds
> $0.024$, and is $\le 0.008$ in five of the eight cells … The two new self-supervised and
> supervised ViT backbones behave exactly like the two originally reported, which is the substance
> of the generalisation.

**R1.1 answered in the same section (verbatim).**

> Mondrian's level settles at $0.854$--$0.886$, below the nominal $0.90$. That is a property of the
> statistic rather than a validity failure, and we quantify it directly: the unweighted mean over
> groups, which is what group-conditional calibration actually targets, is $0.9024$ across the same
> eight cells (range $0.8995$--$0.9056$), while the *minimum* over four noisy per-group coverages
> sits $0.018$--$0.045$ below it.

**Numerical correction to disclose.** The submitted text gave the CLIP/CelebA Mondrian spread as
$0.004$. That came from differencing values already rounded to three decimals; from the unrounded
means it is $0.003$. Corrected.

**Handling of the two gated cells.** ERM fell below its pre-specified worst-group accuracy floor on
DINOv2/CelebA and ViT-B/16/CelebA, so it is absent from the gate-respecting analysis there. Its
values in those two rows are taken from the sensitivity run and marked $^{\dagger}$, and are
excluded from the spread — otherwise "spread" would silently mean different things in different
rows.

---

## R3 + R2.6 + R3.1 — `tab:c2` and §c2 rewritten; three claims withdrawn

**Status: done**

**Where:** §c2 prose and `tab:c2`.

**Change.** Eight cells instead of four, and the bare per-cell Pearson correlation now carries a
cluster-bootstrap CI. Three statements from the submitted version are **withdrawn in the text**,
each because a measurement contradicts it:

1. the correlation as a headline — five of eight CIs span essentially $[-1,+1]$;
2. "the absolute efficiency range is narrow" — the spread runs $0.036$ to $0.364$, and TOST
   rejects equivalence at a $0.10$ margin in six of eight cells;
3. the efficiency reading as a general claim — it is conditional on Mondrian.

**New text (verbatim).**

> The per-cell correlation, however, does not survive uncertainty quantification. Each is computed
> over the three to five retained methods of that cell, and with a cluster-bootstrap CI five of the
> eight span essentially $[-1,+1]$: the $-0.97$ and $-1.00$ we previously reported on the two
> original CelebA cells are, at $n=4$, not evidence of a deterministic relationship. … We therefore
> report the accuracy--efficiency link as a *direction* rather than an estimate, and only under
> Mondrian: under marginal calibration training still moves coverage, so "efficiency, not coverage"
> is conditional on the calibration policy being group-conditional and is not a general statement.

> The earlier characterisation of the efficiency range as "narrow" was not supported and is
> withdrawn.

**Disclosure added (verbatim).**

> On DINOv2/Waterbirds the mean worst-group set size is *below* one ($0.926$--$0.964$ across
> scores), which means a small fraction of prediction sets are empty. That is a legitimate outcome
> of standard split conformal with a highly accurate model — the calibration quantile can fall
> below every class score — and coverage there remains valid under Mondrian ($0.858$--$0.881$). We
> report it rather than clipping sets to a minimum size of one, which would hide it.

---

## Defect found in the submitted manuscript (not raised by a reviewer)

`tab:predgroup` declares eight columns (`llcccccc`) but every row supplies seven cells, so it
renders with a stray empty column. Being fixed as part of rewriting that table to four backbones.

---

## R2.3 — "the current interpolation uses relatively few independent training seeds, and several robust models show no overlap in accuracy with ERM … otherwise, limit the conclusion to matched settings and use a hierarchical uncertainty analysis that respects repeated calibration splits within the same trained model"

**Status: done. This is the largest retraction in the revision.**

**Where:** §h1 prose and `tab:h1`.

**What we found when we looked.** The submitted analysis pooled each arm's
$(\text{accuracy}, D)$ points across seeds **and** calibration splits. Measured separately, ERM's
per-seed base accuracy is **identical to five decimals** in every cell — its last-layer solver is
deterministic, so the training seed does not move it — while its across-split range is $0.02$–$0.03$.
So the entire accuracy axis it contributed was *which evaluation rows were drawn*, not *which model
was trained*. Matching on it compares ERM on unlucky draws against a robust arm on lucky ones. The
accompanying bootstrap over pooled points also treated fifty nested rows as fifty independent draws,
which is exactly what R2.3 objects to.

**Recomputed at model level** (one accuracy per seed, two-stage cluster bootstrap resampling seeds
then splits within them): ERM's accuracy support is a single **point**, and overlapping support
exists in **one of eight cells** — GroupDRO-LL on ResNet-50/Waterbirds, whose per-seed accuracies
bracket ERM's $0.9399$. There the reduction is real and **replicates** the submitted estimate:

| score | $\Delta(a^\star)$ | cluster CI |
|---|---|---|
| APS | $+0.084$ | $[+0.076, +0.095]$ |
| RAPS | $+0.084$ | $[+0.074, +0.094]$ |
| THR | $+0.150$ | $[+0.146, +0.158]$ |

**Withdrawn.** The two other matched values in the submitted `tab:h1` — GroupDRO-LL on
CLIP/Waterbirds ($+0.100$) and balanced subsampling on ResNet-50/Waterbirds ($+0.059$) — rested on
the split-level spread. At model level neither arm's accuracy support reaches ERM's.

**New text (verbatim).**

> Recomputed at model level, with one accuracy per training seed and a two-stage cluster bootstrap
> that resamples seeds and then splits within them, ERM's accuracy support is a single *point* and
> overlapping support exists in exactly one of the eight cells … The two further matched values
> reported in the submitted version … rested on the split-level spread and are withdrawn: at model
> level neither arm's accuracy support reaches ERM's.

> We therefore limit H1 to that one matched setting, as the reviewer asks, and draw no
> accuracy-matched conclusion elsewhere. The reason the comparison is unavailable is itself
> informative and is the more robust reading.

---

## R2.6 / R3 — `tab:h2` and §h2, inversion re-tested with cluster intervals

**Status: done.** Eight cells; the inversion is called real only when a paired cluster bootstrap of
the $\covgap$ difference excludes zero.

**Result, and it favours the paper.** Point-estimate inversions occur in **5 of 8** cells but clear
the interval in **1 of 8** (ResNet-50/CelebA, $+0.014$). The submitted version claimed 2 of 4 real.
So the inversion is *rarer* than reported and H2 is stronger; the claim is nonetheless stated as a
tendency, not a rule.

---

## Page-budget cuts (ACML's 20-page limit includes references and appendices)

**Status: done, 59 lines recovered; no dangling references and no unused labels afterwards.**

| cut | lines | why |
|---|---|---|
| `fig:c1` deleted | −13 | showed marginal-vs-Mondrian for two cells; `tab:c1` now shows all eight in less space |
| §audit compressed, `fig:audit` deleted | −12 | no reviewer raised H(audit); the finding and its numbers stay as one paragraph, AUROC updated to the four-backbone range |
| `app:h1`, `app:h2` deleted | −20 | rewriting §h1 and §h2 left both with **zero** references |
| `app:c1`, `app:shiftrobust`, `app:h3`, `app:predgroup` merged into `app:supp` | −13 | each was mostly a pointer to the released records; references redirected |

No result was removed by any of these.

---

## Introduction rewritten — it had become false

**Status: done.**

The revisions turned several contribution-list claims into misstatements, and an introduction that
contradicts its own results sections is worse than one that overclaims. Corrected:

| before | after |
|---|---|
| "tracks base accuracy in three of four cells (corr $\in[-1.00,-0.85]$)" | seven of eight by direction; five of eight CIs uninformative, so reported as a direction |
| "16 of 18 method$\times$cell combinations" | 30 of 34 arms |
| "$\auroc\in[0.95,0.99]$" | $[0.945, 0.999]$ |
| "the inversion … appears in only 2 of 4 cells" | CI-separated in 1 of 8 |
| "The representation does not deliver worst-group coverage; the calibration mechanism does." | "governed by the calibration *mechanism* far more than by the representation … not that the representation is irrelevant, but that changing it, even by retraining it, buys much less worst-group coverage" |
| (absent) | a new contribution item for the fine-tuning study — the answer to R1.2 and R2.1 |

Every numeral in the new introduction was checked to appear at least twice in the manuscript, so no
claim exists only in the introduction.

---

## Self-contradictions removed (15 places)

**Status: done.** After the title, four tables and five sections changed, the same claims were
still standing in the abstract, introduction, hypothesis list, figure caption, discussion,
limitations, conclusion, appendices and back matter. A revision that softens the title while
leaving the absolute claim in five other places has not answered R1.2, R2.1 or R3.

An audit found them, and it needed two passes: the first matched line by line and
case-sensitively, so it missed the §c1 subsection title ("**Not** a Representation One") and the
H(c1) hypothesis statement, whose clause is split by a line wrap.

| where | before | after |
|---|---|---|
| §c1 title | "Coverage Is a Calibration Property, Not a Representation One" | "Calibration Governs Worst-Group Coverage Far More than Training Does" |
| `fig:overview` caption | "is a property of the calibration mechanism, not of the representation" | "is governed far more by the calibration mechanism than by the representation" |
| H(c1) statement | "the calibration policy, not the representation, controls worst-group coverage" | "the calibration policy governs worst-group coverage far more than the representation does" |
| Discussion | "delivered by the calibration policy"; "The lever is the calibration mechanism, not the score representation" | "governed chiefly by"; "The dominant lever is ... rather than ... — not the only one, but by a wide margin" |
| Conclusion | "is a calibration property, not a representation one" | "is governed far more by the calibration policy than by the representation — even when the representation is retrained end-to-end" |
| Related work | "to show the representation does not substitute for the mechanism" | "to measure how little the representation substitutes for the mechanism" |
| §exp | "Two frozen backbones" | four, listed with dimension and pretraining regime |
| Limitations | "two frozen backbones ... all four cells" | "four frozen backbones plus one fine-tuned end-to-end ... all eight frozen cells and all six cells of the fine-tuning study" |
| Conclusion | grid 2x2x5x3x3 | 4x2x5x3x3 |
| Related work | AUROC [0.95, 0.99] | [0.945, 0.999] |
| §predgroup | "16 of 18" | "30 of 34" |
| `app:subtarget` | mean over groups 0.906, minimum 0.868, gap 0.038 | measured: 0.9024, 0.8728, 0.0295 |
| back matter, `app:hyper` | old CSV filenames (4 places) | the four released files with record counts |

---

## R1.4 — "increase the number of independent training seeds"

**Status: done, with a disclosure the reviewer did not ask for but needs.**

Five seeds now, up from three. But two of the five arms are **deterministic**: ERM and AFR fit
with L-BFGS, whose solver ignores the seed, so their across-seed SD is exactly 0.000 and their
base accuracy is identical to five decimals in every cell. A new paragraph in §exp says so:

> Running more seeds therefore adds replicates for DFR, balanced subsampling and GroupDRO-LL
> (across-seed SD 0.003--0.027 in worst-group accuracy) but not for ERM or AFR. We report this
> rather than present zero variance as stability.

It is also why every interval in the revision is a cluster bootstrap that resamples seeds and then
splits within them — the ten splits inside a seed share one trained model.

---

## R2.5 — the four things asked about predicted groups, and the rho-sweep scoping

**Status: done.** §predgroup gains three new paragraphs and §h3 one.

- **How a predicted attribute sets a threshold.** The label is never predicted; only the attribute
  is. Stratum ghat = 2y + ahat; calibration points binned by ghat; each bin's
  ceil((1-alpha)(n+1))-th smallest score is its threshold; coverage is tallied against the **true**
  group so a mis-assignment is penalised rather than hidden.
- **What supervision is required.** The attribute on the *training* split only — not on
  calibration, never at test. The reduction is from "attribute labels on calibration and test" to
  "attribute labels on training data only": a real weakening, not a removal.
- **How probe error propagates.** Asymmetrically. The minority stratum carries the *largest*
  threshold, so a minority point misplaced into a majority stratum is judged against a tighter
  threshold and under-covered, while the reverse error only wastes efficiency. That predicts where
  the failures fall, and it matches: they concentrate where the probe is weakest (Waterbirds,
  AUROC 0.945-0.976) and on ERM.
- **Privacy and ethics.** Written as a constraint on use, not a disclaimer. Two points stated
  plainly: CelebA's Male field is a third-party binary annotation, not self-identified gender, so a
  study treating it as gender has erred before any ethical question arises; and inferring it at
  test time assigns a sensitive label to someone who has not provided it, wrongly for 2-5% of
  people even at the AUROC we measure. The recommendation is scoped to attributes already lawfully
  held or to an operator auditing their own system, and explicitly ruled out where the inferred
  label would be stored, disclosed, or used to route individual outcomes.
- **rho-sweep scoping (§h3).** The sweep changes the group mixture while leaving the
  class-conditional feature distribution within each group untouched, so it is a group-prior shift
  and **not** a general covariate shift. The stability claim is scoped to that.

---

## R2.6 — moderate the claims, and the reproducibility detail

**Status: done** (the claim-moderation is recorded under R3/`tab:c2` above).

`app:hyper` now gives the exact calibration/test construction (disjoint reservoirs from a seeded
permutation, resampled within their own group indices, truncated to a common size, realised rho
logged, separate uniform streams for the randomised scores), per-group counts for both datasets,
every head hyperparameter including the fine-tuning optimisers, the shift-robust threshold in
closed form, the seed sets, and the measured run times.

**One gap the authors must close:** the archival code entry is a flagged placeholder —
*"[To be replaced before publication with the archived DOI of the tagged release.]"* A DOI must be
minted (e.g. via Zenodo from a GitHub release tag) before submission. It is left visible rather
than invented.

---

## R1.5 — the requested citation

**Status: done**, and placed where it is substantively relevant rather than dropped into a list:
the revision now fine-tunes backbones end-to-end, so how much of a pretrained network to unfreeze
is a choice we make.

> How much of a pretrained network to unfreeze is itself consequential, and staged or partial
> unfreezing schedules change what a fine-tuned representation encodes
> [elmehdi2024unfreezing]; we unfreeze the whole network, so the representation lever is
> pulled as far as each objective allows.

---

## R1.3 (second half) — "compared more clearly with those of other related studies"

**Status: done.** New `tab:lit` plus three paragraphs before §Results.

All published values come from **one** table (Qiu et al. 2023, Table 1, ResNet-50) rather than
being assembled across papers. That mattered: a web search returned DFR on CelebA as "92%", which
is wrong — the source table gives 0.883.

The comparison is unflattering and is presented as such. Our ResNet-50 arms are 10-24 points
below the published values on Waterbirds:

| method | ours (WB) | published | ours (CelebA) | published |
|---|---|---|---|---|
| ERM | 0.510 | 0.726 | 0.416 | 0.472 |
| DFR | 0.812 | 0.929 | 0.851 | 0.883 |
| AFR | 0.686 | 0.904 | 0.434 | 0.820 |
| GroupDRO (last-layer here) | 0.654 | 0.914 | 0.757 | 0.889 |

Three protocol differences are stated: our arms refit only the last layer on cached features
(where the published GroupDRO and AFR train the network), our ResNet-50 uses a deliberately light
recipe and a 30k CelebA training subsample, and worst-group accuracy is read on the pooled
evaluation domain rather than the benchmark test split. DFR — the one arm whose protocol matches —
is the closest (0.851 vs 0.883).

The defence given in the text is the true one: every result is a **within-cell** contrast on the
same posteriors, so a weaker backbone lowers all arms in a cell together and leaves the calibration
contrast intact. The four backbones we compare span a 0.20-0.95 range of worst-group accuracy
and the dissociation is the same across all of them.

---

## Still outstanding

| item | blocker |
|---|---|
| `tab:predgroup` rewrite to four backbones (also fixes the 8-declared/7-supplied column defect in the submitted version) | needs `predicted_group_mondrian_4bb.csv` downloaded locally |
| cheap tightening: Related Work, Discussion+Conclusion merge (~60 lines) | none |
| point-by-point response letter | this log is its source |
| archival DOI for the code | authors must mint it |

---

## `tab:predgroup` rewritten to four backbones — and a defect fixed

**Status: done.**

The submitted table declared eight columns (`llcccccc`) but supplied seven cells per row, so it
rendered with a stray empty column. Found by a column-count check rather than by reading it, and
fixed as part of the rewrite. All nine tables in the manuscript now pass that check.

Listing all 34 arms would take a page, so the table is now per cell: probe AUROC, the median and
maximum $|\cov(a)-\cov(c)|$ over the cell's retained arms, how many clear the $0.02$ bar, and the
arm responsible for the maximum. $40 \to 27$ lines while covering eight cells instead of four.

**Headline: 30 of 34 arms are deployable**, matching the abstract.

---

## A claim I got wrong twice, and the correction

**This is worth recording because the error was mine, not the data's.**

I wrote — in the new R2.5 mechanics paragraph and again in the results prose — that the
predicted-group failures "concentrate where the probe is weakest", and that the fourth failure was
ERM. The generated numbers refute both:

| cell | probe AUROC | deployable |
|---|---|---|
| ResNet-50 / Waterbirds | **0.945** (lowest) | **5/5, no failure** |
| CLIP / CelebA | **0.999** (highest) | **3/4, GroupDRO-LL fails at 0.025** |

So probe accuracy does **not** order the failures, and the fourth failure is GroupDRO-LL, not ERM.
The asymmetry mechanism is still sound — a misplaced minority point is judged against a tighter
threshold, so under-coverage is the costly direction — but AUROC is not its predictor.

Both places were corrected, not just the one a reader would hit first. The text now says:

> The mechanism is sound but its magnitude is *not* ordered by probe accuracy, and we were careful
> to check rather than assume. Across our eight cells the probe's AUROC does not predict which arms
> fail: the cell with the *lowest* AUROC (0.945, ResNet-50/Waterbirds) has no failures at all,
> while a cell with AUROC = 0.999 (CLIP/CelebA) has one. What does travel with the penalty is the
> arm: the two largest failures are ERM, whose minority points sit closest to the decision boundary
> ... So the probe's AUROC is necessary to report alongside any coverage claim that relies on it,
> but it is not sufficient.

and the four failures are now named individually (ERM on CLIP/Waterbirds 0.051 and
DINOv2/Waterbirds 0.024, AFR on ViT-B/16/Waterbirds 0.022, GroupDRO-LL on CLIP/CelebA 0.025), with
the note that three of four exceed the bar only marginally.

---

## Manuscript state

1,402 lines, roughly 27 pages against the 20 that applied at submission. The supervising author
has confirmed the increase is acceptable and expected after addressing reviewers; the ACML call
for papers states 20 pages including references and appendices but is silent on revisions.

An aggressive-cut plan is held in reserve should the editorial office object: removing
`fig:overview` and `fig:shift`, merging §h1 with §h2, folding `tab:subtarget` into prose, and
merging Discussion with Conclusion recovers roughly 160 lines and would bring it back under 21
pages. Nothing in that plan removes a result; it removes two figures and merges sections.

**Still to do:** the cheap tightening (Related Work, Discussion+Conclusion merge, ~60 lines), the
point-by-point response letter, and the archival DOI the authors must mint.

---

## Figures — three problems, found because the author asked

**Status: done.** The author asked whether the figures still matched the new data. They did not, and
checking turned up more than staleness.

1. **All PNGs were stale.** Every file in `figures/` is dated 1 September, generated from the
   two-backbone June data — before the grid (4 Sep), the calibration ablation (6 Sep) and the
   predicted-group study (6 Sep). Publishing a figure whose underlying data is not the data in the
   tables is wrong even when the curves look similar.
2. **The figure and the text described different calibration policies.** `fig:shift` plotted
   `grid_records`, which contains only `marginal_split`, while the §h3 text it illustrates is about
   **Mondrian** being flat and sub-target. This predates the revision.
3. **Twelve PNGs are now unreferenced** and must not be uploaded with the submission. Only
   `shift_mondrian_vs_marginal.png` is referenced.

**Replacement.** One figure, two panels (Waterbirds | CelebA), worst-group coverage against
$\rhotest$ under Mondrian (solid) and marginal split (dashed), for all four backbones, from
`calibration_ablation_4bb.csv`. It supports §c1 and §h3 at once and covers eight cells instead of
two. The plotted values are printed by the generating script so the figure can be checked against
the tables.

The caption states what the earlier one did not:

> Mondrian is flat to within 0.010 over the whole sweep in every cell and sits just below the
> nominal 0.90 — the min-over-groups effect of Appendix B, not a shift collapse. On Waterbirds the
> two policies are widely separated and the marginal curves fan out by backbone; on CelebA they
> nearly coincide, because ERM's marginal coverage there was never far from target to begin with.
> Note the sweep varies the *group mixture* and not the within-group distribution, so this is
> stability under group-prior shift, not under general covariate shift.

The CelebA panel showing the two policies nearly coinciding is not flattering, and is kept: it is
what the data says and it already appears in the §c1 text.

**Figure count.** The manuscript now has two figures: `fig:overview` (a TikZ schematic, no external
file, still accurate) and `fig:shift`. `fig:c1` was cut as redundant with the eight-cell `tab:c1`,
and `fig:audit` with the compressed §audit.

---

## A silent corruption, and the scan that caught it

While fixing the "Pre-registered hypotheses" heading, a bash here-document stripped one backslash
from `\\ref`, after which Python read `\r` as a carriage return and wrote `Appendix~<CR>ef{app:excl}`
into the manuscript. No error, no failed assertion — just a broken cross-reference.

Repaired, and the whole file was then scanned for the same damage from earlier here-document edits:
zero stray control characters, no LaTeX commands with an eaten backslash, braces balanced (773/773),
math-mode delimiters even (1056), no dangling `\ref`. That was the only casualty.

All subsequent edits to the manuscript were made through script files rather than here-documents.

---

## Response letter

`RESPONSE_LETTER.md` written: a summary-of-changes table followed by point-by-point replies to all
thirteen reviewer comments, each quoting the comment, stating what changed and where, and quoting
the new text. It closes with the three changes made without being asked (the Section 3 definitional
inconsistency, the table column defect, and the empty-set disclosure).

It states the retractions plainly rather than framing them as clarifications, because two of the
reviewers' comments identified real errors and a response letter that obscures that is worse than
one that owns it.

---

## Audit of everything data-derived — prompted by the author asking whether the figures still matched

Checking the figures turned up more stale content than the figures themselves. Everything below was
carried over from the two-backbone June run and had not been rechecked against the new data.

### Three claims in §h3 that the new runs contradict

| claim as submitted | measured on the four-backbone data | outcome |
|---|---|---|
| "Mondrian worst-group coverage is **2–4× more stable** across calibration splits than marginal (SD 0.011–0.036 vs **0.030–0.138**)" | Mondrian SD 0.011–0.034 (correct); marginal SD **0.009–0.042**, ratio **0.6–1.8×**, and in **two of eight cells Mondrian is the less stable of the two** | **withdrawn** |
| "worst-group coverage varies by <0.05 over the whole sweep **for every method**" | holds under Mondrian (max range 0.024) and marginal (0.037); **fails under shift-robust (0.347)**, which inflates its level with the observed shift by construction | **scoped to two policies** |
| CelebA shortfall "0.016–0.020" | 0.016–**0.023** | corrected |
| Appendix B: "bracket the 0.01–0.05 we observe" | 0.016–0.044 | corrected |

What survives is the argument the paragraph exists to make, and it did not need the withdrawn
comparison: the shortfall co-scales with the per-group quantile noise — Waterbirds SD 0.023–0.034
with shortfall 0.030–0.044, CelebA SD 0.011–0.018 with shortfall 0.016–0.023. The withdrawal is
stated in the text rather than silently dropped:

> (An earlier version also claimed Mondrian was 2–4× more split-stable than marginal calibration. On
> the four-backbone data the two are comparable — marginal's SD is 0.009–0.042 — and in two of the
> eight cells Mondrian is the less stable of the two, so we withdraw that comparison. Mondrian's
> advantage here is in the *level* it attains, not in its variance across splits.)

### Figures

Every PNG in `figures/` was dated 1 September, i.e. generated from the two-backbone June data,
before the grid (4 Sep), the calibration ablation (6 Sep) and the predicted-group study (6 Sep).
Worse, `fig:shift` plotted `grid_records`, which holds only `marginal_split`, while the §h3 text it
illustrated is about **Mondrian** — figure and text were describing different calibration policies.

Both figures were rebuilt from current data in the house style of the `figures4papers` repository
(Arial/Helvetica stack, font.size 13–14, `axes.linewidth` 2, right and top spines removed, palette
anchor blue `#0F4D92`, comparator red `#B64342`, teal `#42949E`, violet `#9A4D8E`, neutral
`#767676`, dpi 300, `tight_layout(pad=1)`):

- **`shift_mondrian_vs_marginal.png`** — worst-group coverage against ρ_test under both policies,
  four backbones, two panels. Supports §c1 and §h3 at once and covers eight cells instead of two.
- **`repr_lever.png`** (new, `fig:repr`) — the fine-tuning study, which is the revision's most
  important new evidence and previously had only a six-row table. Grey (worst-group accuracy) and
  red (marginal coverage) move together across the three representations; blue (Mondrian) does not
  move at all.

Both scripts print every plotted value so the figures can be checked against the tables; they match.

**Twelve PNGs are now unreferenced and must not be uploaded with the submission.** Only
`repr_lever.png` and `shift_mondrian_vs_marginal.png` are used.

### Final validation

1,429 lines. Nine tables, every column count correct; braces balanced globally (780/780); math-mode
delimiters even (1076); no dangling `\ref`; no stray control characters; two figures, both
referenced, both from current data.

---

## Figures rebuilt to the figures4papers house style — six in total

The first pass applied `design-theory.md` by hand. Reading `api.md` and `common-patterns.md`
properly showed three conventions that had been missed and that matter:

1. **Vector export.** `finalize_figure` writes PDF alongside PNG, with `pdf.fonttype = 42` and
   `svg.fonttype = 'none'` so text stays selectable. A 300-dpi raster is the usual reason a figure
   looks soft in a compiled paper; every figure now has a vector twin and the manuscript includes
   them extensionless so pdflatex picks the PDF.
2. **Frameless legends**, per `apply_publication_style`.
3. **Semantic colour mapping** — blue for the key mechanism, red for the baseline it is contrasted
   with, neutral grey for background quantities. The shift figure had been using colour for
   *backbone* and line style for *policy*, which buried the contrast the figure exists to show. It
   now uses blue for Mondrian and red for marginal with a shaded envelope across the four
   backbones, and the envelope width is the paper's central claim in one number: **0.014 against
   0.121** on Waterbirds at the calibration point.

The spec is a set of conventions to implement, not a package, so it is implemented as
`study_robust_train/figstyle.py` — in the repository, so the archived code reproduces the figures
as well as the tables (R2.6).

### The six figures, and what each is for

| figure | content | answers |
|---|---|---|
| `fig:overview` | two-lever schematic (TikZ, in the manuscript) | — |
| **`fig:c1`** *(new)* | dumbbell: the same ERM model under both policies, all eight cells, lift annotated | **R1.3** — backbone generality, visible at a glance |
| **`fig:h1`** *(new)* | scatter of (base accuracy, divergence); on CelebA the robust arms sit at 0.81–0.91 against ERM's 0.68–0.70, with no overlap | **R2.3** — the hardest point to grasp from a table |
| `fig:repr` | grouped bars: accuracy and marginal coverage move together across the three fine-tuned representations; Mondrian does not move | R1.2 / R2.1 |
| `fig:shift` | coverage vs ρ under both policies, envelope across backbones | R2.5 |
| **`fig:subtarget`** *(new)* | mean-over-groups sits on 0.90 in all eight cells while the reported minimum sits below | **R1.1** — makes the sub-target answer immediate |

Three of the six are new and each maps to a reviewer point that is far easier to see than to read.

### One design choice reversed on inspection

`fig:h1` first shaded the accuracy range ERM occupies **pooled across the four backbones**. That
overstates the available matches, because matching is *within* a cell — a robust arm falling inside
the pooled band need not overlap its own cell's ERM. The band was removed and replaced with a ring
around the single (cell, arm) pair in the whole grid whose per-seed accuracy support actually
brackets its own ERM, which is the finding rather than a suggestion of one.

### Stale files

Eleven PNGs from the 1 September two-backbone run remain in `figures/` and **must not be uploaded**:
`burden_*`, `c1_clip_*`, `c1_resnet_*`, `h2_*`, `recoverability_tradeoff`, `shift_clip_waterbirds`,
`shift_resnet_celeba`. Only the five current figures (PDF, with PNG previews) are referenced.

### Manuscript state

1,471 lines, roughly 28 pages. Nine tables all passing the column check, six figures all
referenced, braces balanced globally (801/801), math-mode delimiters even (1108), no dangling
`\ref`, no stray control characters.

---

## The main figure rebuilt with real data and real images; the subtarget figure rebuilt twice

Two complaints drove this batch: the figures looked alike, and the main method figure was thin.
Both were fair.

### The schematic is gone

`fig:overview` was a TikZ cartoon: two little axes with five invented dots that scatter under
marginal calibration and sit flat under Mondrian. Nothing in it was measured, and `fig:c1` now
shows that same dissociation on real data across all eight cells — so the schematic was both
redundant and the weaker of the two. The one sentence in the introduction that cited it as evidence
for the dissociation now cites `fig:c1`, which is where the evidence actually is.

Its slot is taken by `teaser_method`, a three-panel figure that does what no other figure in the
paper does.

**(a) The confound, on real images.** Four Waterbirds photographs, one per group, with each
group's share of the calibration pool at ρ = 0.95 — 47.5 / 2.5 / 2.5 / 47.5, minority groups
outlined in red. The paper studies a spurious correlation between habitat background and species
label and, until now, showed no images at all; a reader could not see what the confound *is*.

**(b) The mechanism, fitted to a measured cell rather than drawn freehand.** Two conformity-score
distributions, with the worst group's shift solved numerically so that the *pooled* 90% quantile
reproduces CLIP/Waterbirds ERM's measured marginal worst-group coverage of 0.509:

```
shift=1.386  q_pool=1.409  cov_wg(marginal)=0.509 (target 0.509)  q_wg=2.668
```

The shaded area is therefore the actual shortfall of Eq. (7) in Proposition 1(ii), 0.391, at the
actual π = 0.05 — not an illustrative sketch that happens to look like the identity.

**(c) What the grid varies**, so the design is legible without reading Section 4.

### The images had to be inspected one by one

Twelve images were downloaded from the `arubique/waterbirds` mirror on the Hugging Face
datasets-server (three per group; the splits are named by group, so each example comes from the
right stratum by construction rather than by filtering metadata we would have to trust). **One of
the twelve carries a `fotolibra.com` stock-photo watermark** — Waterbirds composites birds onto
Places backgrounds, and Places contains watermarked stock. It is `waterbird_on_land_2.jpg`, and it
is excluded; the four used were chosen by eye. Any figure built from this dataset needs that check,
and an automated pick would have put a watermark in the paper's opening figure.

### The subtarget figure: two wrong versions before a right one

**Version 1 (dumbbell).** Mean over groups against minimum, one pair per cell. The content was
correct, but it is the same visual idiom as `fig:c1` — which is precisely what made the figure set
look repetitive.

**Version 2 (shortfall against `n_cal_worst_group`, on a simulated curve).** This one was *wrong*,
not merely repetitive, and it is worth recording why. Within a dataset every cell has the same
group construction — Waterbirds is always [1245, 66, 66, 1245] — so the predicted shortfall is a
single number, not a position on a curve. The column `n_cal_worst_group` records only *which* group
happened to come out worst, flipping between 66 and 1245 across cells whose prediction is
identical. The points would have been spread along an axis carrying no information about them, and
the resulting scatter would have looked like a dose–response relationship that does not exist.

**Version 3 (`subtarget_law_vs_observed`), kept.** Since the construction is fixed per dataset, the
exactly-valid law is a *distribution*. The figure compares distributions on the coverage axis: the
observed mean over groups, the observed minimum, and the simulated law of that minimum at our own
per-group calibration and test counts.

| | mean over groups | observed min | exactly-valid E[min] | residual | runs inside its 95% range |
|---|---|---|---|---|---|
| Waterbirds (600 runs) | 0.9042 | 0.8660 | 0.8695 | −0.0035 | 98% |
| CelebA (410 runs) | 0.9007 | 0.8800 | 0.8847 | −0.0047 | 95% |

This is a materially stronger answer to R1.1 than the letter previously gave. The closed-form
1.03·SD figure covers only the min-over-*k* selection effect (0.013–0.030 against an observed
0.018–0.045); simulating the full Mondrian law also captures the second effect — each threshold
estimated from that group's own small sample — and the residual falls to 0.004–0.005. The residual
is still reported rather than described as an exact match.

The observed spread is wider than the predicted band by construction: the simulation varies only
the calibration and test draw, while the runs also vary over backbones, arms and seeds. The caption
says so, since the agreement to be read is in location, not width.

The cell-level numbers already in the manuscript were re-verified against the CSV and are unchanged
— mean over groups 0.9024, range 0.8995–0.9056, gaps 0.018–0.045 — so the caption deliberately
avoids restating a mean under a different aggregation that would appear to contradict them.

### Manuscript state

1,458 lines. Six figures, all referenced, all from the four-backbone data; no TikZ remaining; no
dangling `\ref` or `\eqref`; braces balanced (784/784); math delimiters even; no stray control
characters. Not compiled here — there is no TeX installation on this machine.

### Stale files, updated list

**Sixteen** files in `figures/` are now superseded and must not be uploaded: the eleven PNGs from
the 1 September two-backbone run (`burden_*`, `c1_clip_*`, `c1_resnet_*`, `h2_*`,
`recoverability_tradeoff`, `shift_clip_waterbirds`, `shift_resnet_celeba`) plus the two dead
subtarget attempts in both formats (`subtarget_mean_vs_min.{pdf,png}`,
`subtarget_shortfall_vs_ncal.{pdf,png}`). Deletion is deferred by request; the six current figures
are the only ones the manuscript includes.

---

## The main figure redesigned, and a house-style defect found in `figstyle.py`

The teaser was rejected twice — first as untidy with unequal panel heights, then as inelegant, with
colliding labels and too many boxes. Both rounds traced to the same mistake: `design-theory.md` had
been applied as a colour palette and an export policy, while the *layout grammar* of the
repository's own teaser (`assets/VIGIL_teaser.png`, `figure_VIGIL/plot_concept.py`) had not been
read at all.

### The font was wrong in every figure

`figstyle.py` declared the family as `("DejaVu Sans", "Helvetica", "Arial", "sans-serif")`. DejaVu
Sans is matplotlib's *own* default and is the single strongest signal that a figure is an unstyled
matplotlib plot; `design-theory.md` specifies Helvetica, with Arial as the portable stand-in. Arial
is installed here, so the default is now `("Arial", "Helvetica", "DejaVu Sans", "sans-serif")` and
**all six figures were regenerated**. Every printed number was identical across the rebuild, which
is the check that the change was typographic and nothing else.

### What the house teaser actually does, and what was being done instead

| convention | what the rejected version did |
|---|---|
| Panel label is a bare lowercase letter, set large and light, outside the content | `(a)` glued to a long parenthesised title |
| The panel's *message* is a short claim above the panel | descriptive titles: "the mechanism", "what the study varies" |
| Fills are `alpha≈0.12` under a `lw≈2.2` line | `alpha` of 0.30, 0.22 and 0.55 — muddy, and forcing white-on-red labels that vanished |
| Blue marks the contribution and nothing else | blue used decoratively |

### Three changes that removed the untidiness rather than patching it

1. **Structure is typography, not enclosures.** The claim banners were rounded boxes; they are now
   bold headings with a hairline rule. The "what we vary" panel was a five-box flow chart — the
   boxiest thing in the figure, and it only restated Section 4 — and is gone. Box count went from
   about thirteen to four, and the four remaining are photographs.
2. **Nothing is positioned by eye any more.** In panel (b) the two thresholds label themselves
   *outward* from their own lines, one right-aligned and one left-aligned, so the labels cannot
   meet however close the thresholds are; the curve labels sit over their own flanks rather than
   centred on peaks only $1.39$ apart. In panel (c) the values sit inside their bars and the lift
   is a span drawn in the empty strip between two bar rows. Four separate collisions were fixed
   this way, each verified by looking at the render rather than by reasoning about it.
3. **Panel (a) now shows the group structure instead of describing it**, laid out as the $2\times2$
   that defines the groups — species by background, with shared row and column headers, the
   off-diagonal cells outlined red, and one line of text carrying the pool shares that had been
   four separate captions. Images are centre-cropped to the cell aspect rather than stretched to
   it; `aspect="auto"` was distorting the birds, which in a figure about telling two species apart
   is the one distortion that cannot be allowed.

### Panel (c) is new, and it is the strongest thing in the figure

The old third panel was a diagram. The new one is three measured bars from the CLIP/Waterbirds ERM
cell, and it states the paper in one glance:

| | coverage |
|---|---|
| pooled coverage — looks perfectly healthy | **0.892** |
| worst group, marginal split | **0.509** |
| worst group, Mondrian — same scores, new rule | **0.854** |

The $+0.345$ lift agrees with the CLIP/Waterbirds row of `fig:c1`, which is the internal
consistency check worth having between a teaser and the result it previews.

### Two defects in the figure's own claims, caught while rebuilding

- The flow chart's scale line said **"2 correlation strengths"**. There are six: $\rhotest$ sweeps
  $0.50$ to $0.95$ with $\rhocal$ fixed at $0.95$. The line is gone with the panel, but the claim
  would have been wrong in print.
- LaTeX had leaked into matplotlib text — `1{,}010` and `backbone--dataset` render literally
  outside TeX. Both were visible in the draft render.

### Manuscript state

1,462 lines. Caption for `fig:overview` rewritten, since the figure lost a panel and gained a
measured one; it now names the cell and the averaging. Six figures, all referenced, all present as
PDF; nine tables passing the column check; braces 785/785; no dangling `\ref` or `\eqref`; no stray
control characters. Still not compiled here — no TeX on this machine.

---

## A fourth panel, and a mislabelled quantity caught in the teaser

The three-panel teaser was judged closer but still short of a top-venue figure. Two things were
missing, one of them substantive.

### The figure was showing only the favourable half of the trade

Panels (a)-(c) said: the confound exists, pooled calibration under-covers the rare group, and
re-calibrating recovers $+0.345$ of coverage for free. *For free* was never measured. The same rows
of the same CSV carry the efficiency side, and it is not free:

| | worst-group set size | mean set size |
|---|---|---|
| marginal split | 1.080 | 0.984 |
| Mondrian | 1.319 | 1.001 |

So the worst group's sets grow by $0.239$ while the mean grows by $0.017$. That is a genuinely
better story than "free" — the burden is **moved onto the group that was previously under-served,
and is close to free in aggregate** — and it connects the teaser to the conserved-burden framing the
paper builds on. It is now panel (d), *The cost is local, not global*. Leaving it out would have
been showing the favourable half.

### A quantity was mislabelled, in the figure and then in the caption

The shaded wedge in panel (b) was labelled *"0.391 of the worst group falls outside"*. It is not.
Checked directly:

```
mass covered below the pooled threshold  = 0.5091
mass ABOVE the pooled threshold          = 0.4909   <- what "falls outside" means
mass between the two thresholds (wedge)  = 0.3909   <- what is actually shaded
```

The wedge is the shortfall **from the $0.90$ target**, not the mass outside the set. It now reads
"0.391 short of the 0.90 target", and the caption says the same.

The caption also now explains why panel (b)'s $0.391$ and panel (c)'s $+0.345$ differ: Mondrian
lands at $0.854$ rather than $0.900$, which is the min-over-groups selection effect of Appendix B.
Two adjacent panels carrying two nearly-equal numbers invite the suspicion of an inconsistency, so
the difference is stated rather than left to be inferred.

### Design changes in this pass

- **Panel (a)** carries a share badge on every cell (2.5% red, 47.5% grey) so the number is tied to
  the image rather than to a sentence underneath; the caption drops to a single claim.
- **A monospace cell tag** (`CLIP · Waterbirds · ERM · APS`) marks panel (b) as fitted to a real
  cell rather than drawn. Monospace accents for technical tokens are a house-style device from
  `figure_VIGIL/plot_concept.py`.
- **Tick and cross marks were tried and removed.** They collided with the lift arrow, and a tick
  beside $0.854$ would assert that Mondrian meets the $0.90$ target when it does not. The dashed
  target line carries that honestly without a verdict glyph.
- Four collisions fixed by construction rather than by nudging: the cell tag was crossing the
  Mondrian threshold line (which rises to $0.96$ of the axis) and then the "worst group" label;
  panel (c)'s axis name was falling into panel (d)'s heading, fixed by growing the canvas to
  $5.6$ in rather than by shrinking type.

### What still limits this figure

The strongest thing a conformal paper can show is a **real prediction set on a real image** —
the same bird under both policies, where marginal returns a set that excludes the true label and
Mondrian returns one that contains it. That needs per-image conformity scores, and the local
`results/` tree holds only aggregate CSVs; the features and models live on Colab. It is not
something to fabricate, so it is recorded here as the one available improvement that this machine
cannot make.

### Manuscript state

1,469 lines. Six figures, all referenced and present as PDF; nine tables passing the column check;
braces 788/788; no dangling `\ref` or `\eqref`; no stray control characters. Not compiled here.

---

## Response letter and change summary rebuilt as LaTeX, and two errors found on the way

The two companion documents are now `response-to-reviewers.tex` and `before-after.tex`, both
standalone (plain `article`, no Springer class needed) and both alongside the manuscript.

### Every reviewer point now has the same four parts

The Markdown letter stated where each point was addressed, but inconsistently: only 4 of the 13
points carried an explicit marker, and the other 9 named a section inside a sentence. The LaTeX
letter gives all 13 an identical structure -- the reviewer's words, what we did, a
**Where addressed** line, and the revised text quoted verbatim behind a rule so it cannot be
mistaken for commentary. Counts: 13 point sections, 13 reviewer quotes, 13 `Where addressed` lines,
8 verbatim manuscript quotations.

### Two numeric errors, found by checking the letter against the CSVs

Earlier consistency passes compared the letter to the manuscript. Comparing it to the *data*
instead turned up two claims that neither document had caught.

1. **The CelebA accuracy range was wrong in the letter.** R2.3 said the robust arms reach
   `0.83-0.91`. Per-cell means run **0.8089 to 0.9123**, so the range is `0.81-0.91` -- which is
   what the manuscript's `fig:h1` caption already said. The letter was the wrong one of the two.
2. **The seed claim was unqualified.** R1.4 said seeds were "increased from three to five". That
   holds for the main grid (`train_seed` 0-4) and the fine-tuning study (`ft_seed` 0-4), but the
   calibration-policy ablation ran three (0-2) -- and the ablation is where most of the revision's
   tables come from. The manuscript states the design as `>=3 training seeds x >=10 calibration
   splits` for exactly this reason. Both letters now say the same.

The stale page count ("roughly 27") was also corrected to about 23, matching the last actual
compile. Both fixes were applied to the Markdown too, so the two versions cannot drift.

### Before/after facts were extracted, not recalled

Every count in `before-after.tex` comes from diffing the submitted source (commit `49e503c`)
against the working tree:

| | submitted | revised |
|---|---|---|
| source lines | 1,034 | 1,469 |
| tables | 6 | 9 |
| figures | 4 | 6 (none carried over) |
| numbered equations | 3 | 5 |
| main sections | 17 | 13 |
| appendices | 8 | 4 |
| unique citations | 25 | 26 |
| TikZ schematics | 1 | 0 |

Two of these needed reporting rather than glossing. **Appendices went from eight to four** --
consolidated, not deleted: the eight per-hypothesis appendices were folded into four longer ones,
two of which are new or substantially expanded. **Sections went from 17 to 13** for the same
reason. A summary that showed only growth would have misdescribed the revision.

The document also has explicit sections for the four withdrawn claims and the two retracted
matched-divergence values, so an editor meets them in the summary rather than discovering them.

### Citation audit

26 unique `\cite` keys against 26 bib entries: nothing dangling, nothing unused. The net change is
one entry, the reference R1.5 asked for. Both new documents pass a structural check -- balanced
braces, matched `begin`/`end`, no undefined environment or macro, no non-ASCII.

---

## The real-prediction-set panel needs an export, not a re-run

Asked whether the missing qualitative panel requires re-running anything. It does not, and the
reason is worth writing down because it decides how cheap the panel is.

### The pipeline already computes it and throws it away

`conformal_eval.evaluate` builds a full `membership` matrix -- for every test point, which classes
are in its prediction set -- and then reduces it to coverage and mean set size. The per-example
information the panel needs is one line above the aggregation, discarded because no caller wanted
it. So the panel needs an *export*, not an experiment:

| | needed? |
|---|---|
| Re-extract features | **No** -- the Drive caches from the 4-backbone runs are reused |
| Re-train end-to-end | **No** |
| Refit the last-layer head | Yes, but it is seconds on cached features |
| Re-run the grid | **No** -- one cell, one split, two policies |

### Why the export reproduces the paper exactly

Every stochastic step inside `evaluate` is seeded off `split_seed`: the cal/test split
(`split_pool`), both rho resamplings (`split_seed*2+1`, `*2+2`), and the two uniform streams that
randomise APS (`split_seed*7+1`, `*7+2`). The ERM head fits with L-BFGS, whose solver ignores
`random_state`. Re-deriving a cell therefore gives back the same numbers.

Rather than trust that, `export_example_sets` looks up the stored CSV row for the same cell and
**asserts** that seven aggregates match it -- coverage, mean-group coverage, both set sizes,
`base_top1`, `n_eval`. If the panel would disagree with Table 1, the export raises instead of
writing a file.

### What was added

- `conformal_eval.evaluate(..., return_examples=True)` -- an opt-in hook attaching the arrays the
  function already has: test indices, labels, groups, the membership matrix, the scores, and the
  thresholds actually used. Off by default, so the grid and the CSV schema are untouched.
- `study_robust_train/export_example_sets.py` -- refits the head, evaluates both policies, verifies
  against the CSV, then picks examples by *what they demonstrate*: `rescued` (marginal drops a
  worst-group point, Mondrian keeps it), `widened` (both cover, Mondrian by returning a larger
  set), `unchanged` (a majority point both handle identically). Copies the images and writes a
  manifest.

It lives in the repository, so the archived code reproduces this figure along with the tables
(R2.6).

### Tested locally before it ever runs on Colab

Two test scripts, on synthetic L2-normalised features, since the standing requirement is that a
Colab run must not have to be repeated for a bug found afterwards. All pass:

1. The hook is inert when off (no extra key), the aggregates are bit-identical with and without it,
   and the exported arrays reconstruct five of the record's own aggregates. The synthetic Mondrian
   thresholds also show the mechanism plainly -- minority groups get `0.941`/`0.932` against the
   majorities' `0.803`/`0.823`.
2. The export's happy path produces all three cases, and the guards fire: a corrupted CSV value is
   **rejected**, a wrong `paths_eval` length is **rejected**, and misaligned labels are
   **rejected**.

The synthetic run already produced the panel's story verbatim: true class `waterbird`, marginal
returns `{landbird}` -- excluding the truth -- and Mondrian returns `{landbird, waterbird}`.

### Two bugs caught by writing the tests

- The first test version passed un-normalised synthetic features and `assert_l2_normalized` stopped
  it. That guard exists because an un-normalised linear probe under-fits badly; good to see it fire.
- Pool index to image correspondence was the one assumption in the file that nothing checked, and a
  silent re-ordering inside `build_griddata` would have paired the wrong image with the right
  prediction set -- invisible in the output. `y_eval`/`g_eval` are now optional arguments that turn
  it into an assertion, and the usage snippet passes them.

---

## Table 7 (`tab:h2`) narrowed to fit the text block

It was the widest table in the paper -- seven columns, three of them left-aligned text -- and it
overran the layout. It is already `\footnotesize`, so the width came out of redundancy instead of
type size. **No cell value changed**; all 45 numeric values are identical, asserted rather than
eyeballed.

| what | before | after |
|---|---|---|
| row label | `ResNet-50 / Waterbirds` | `ResNet-50`, with the dataset as a spanning group header |
| inversion column | `no (CI incl.\ 0, $+0.011$)` | `no ($+0.011$)`, with the meaning stated once in the caption |
| arm names | `Bal.\ subsample` | `Bal.\ sub.` |
| `\tabcolsep` | 6pt (default) | 3.5pt, scoped to this table |

The dataset was being repeated in all eight row labels even though the rows are already grouped by
dataset with a `\midrule` between the blocks, so that column was paying for information the
structure already carried.

Summed over columns, the widest rendered cell per column goes from **100 to 72 characters (28%
narrower)**, and `\tabcolsep` recovers a further 40pt (~0.56 in) across the eight gutters.

### Two measurement mistakes made while checking this

Worth recording because both would have produced a confident wrong answer.

1. **The first verification compared against `git HEAD`.** HEAD still holds the *two-backbone*
   submitted paper, whose `tab:h2` has entirely different numbers, so the diff reported eighteen
   values "lost" and twenty-four "gained" when nothing had changed. The baseline has to be the
   pre-edit working tree, not the last commit -- the revision has never been committed.
2. **The first width measurement counted source characters**, which reported a 10% saving for a
   change that removes a whole column's worth of text. LaTeX markup costs characters and no width
   (`\textbf{}`, `\,`). Measuring the sum over columns of the widest *rendered* cell gives 28%.

An earlier check also tripped on `3.5` from `\setlength{\tabcolsep}{3.5pt}` being counted as a cell
value; the comparison now runs over data rows only.

### Manuscript state

1,474 lines. Nine tables all passing the column check, six figures all referenced and present as
PDF, braces balanced, no dangling `\ref`/`\eqref`, no stray control characters.

---

## Table 8 narrowed too, and the Colab export notebook prepared

### `tab:predgroup` (Table 8)

Same three sources of width as Table 7, same treatment, **no cell value changed**: the dataset was
repeated in all eight row labels despite the rows already being grouped by dataset with a
`\midrule`; the header cells `median gap` and `max gap` were wider than any number beneath them;
and `\tabcolsep` sat at the 6pt default.

| | before | after |
|---|---|---|
| row label | `ResNet-50 / Waterbirds` | `ResNet-50`, dataset as a spanning group header |
| gap columns | `median gap`, `max gap` | `median`, `max` under a `\cmidrule`-spanned `gap` |
| size | `\small` | `\footnotesize`, matching the neighbouring `tab:h2` |
| `\tabcolsep` | 6pt | 3.5pt |

Summed widest-cell-per-column: **64 to 45 characters, 30% narrower**, plus 35pt from `\tabcolsep`
and the size change. Verified three ways -- 24 decimal values, 8 `deploy.` counts (`3/3`, `3/4`,
`4/4`, `4/5`x3, `5/5`) and the four arm names all identical.

Manuscript: 1,479 lines, nine tables passing the column check, six figures, braces 812/812, no
dangling references.

### `notebooks/export_example_sets.ipynb`

Seventeen cells, generated by script so the JSON is valid by construction; every code cell was
parsed with `ast` after writing. The setup cells deliberately mirror
`notebooks/calibration_ablation.ipynb` -- same clone, same Drive cache symlinks, same `cfg_for` --
so it hits the feature caches the paper's numbers came from rather than extracting new ones.
Waterbirds and one backbone only: the panel needs a single cell and building the other seven would
cost time for nothing.

The one thing that could have gone wrong silently is index-to-image alignment, so the notebook
passes `y_eval`/`g_eval` recovered from the bundle, which makes the export assert that the labels
carried alongside the paths agree with the labels carried alongside the features.

It also refuses to run without the ablation CSV: the export verifies itself against the stored row
for the same cell, and cell 3 searches five plausible locations before prompting for an upload.

Signatures were checked against the notebook's calls after generation.
