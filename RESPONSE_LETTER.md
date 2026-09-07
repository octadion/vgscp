# Response to Reviewers

**Manuscript:** Worst-Group Conformal Reliability under Spurious Correlation Is More a Calibration
Problem than a Representation Problem: A Confound-Controlled Multi-Backbone Study

**Submission ID:** 62d20e3d-c3eb-4568-b964-7b09cf6f583b — ACML 2026 collection, *Machine Learning*

**Authors:** One Octadion, Novanto Yudistira, Yuta Nakashima

---

We thank the three reviewers and the editor for reviews that were specific enough to act on. Two
of the comments identified genuine errors in our analysis, and acting on them changed several
reported numbers; we say so explicitly below rather than presenting the revision as a
clarification. All reviewer requests have been implemented, including the two that required new
experiments.

The revision rests on new runs: a four-backbone grid (36,000 records), a four-backbone calibration
ablation across three calibration policies (64,800), a four-backbone predicted-group study
(10,800), and an end-to-end fine-tuning study in which the representation itself is retrained
(4,320). Every table in the paper has been recomputed from these.

## Summary of the principal changes

| # | change | prompted by |
|---|---|---|
| 1 | **New experiment:** the backbone is fine-tuned end-to-end under three objectives, so the representation itself moves rather than only the head. New Section 5.2 and Table 4. | R1.2, R2.1 |
| 2 | **Backbones extended from two to four**, spanning two architecture families and three pretraining regimes. Every results table now reports eight backbone–dataset cells. | R1.3 |
| 3 | **Title narrowed** from an absolute dichotomy to a comparative claim, and the same softening applied in fifteen further places. | R1.2, R2.1, R3 |
| 4 | **Proposition 1(ii) rewritten** around an exact identity and a bound that holds, with an explicit remark on what the divergences do and do not give. | R2.2 |
| 5 | **Uncertainty quantification replaced** with a two-stage cluster bootstrap that respects calibration splits nested within trained models. | R2.3 |
| 6 | **Four claims withdrawn** and two matched-divergence values retracted, because the corrected analysis does not support them. | R1.3, R2.3, R2.6, R3 |
| 7 | **Sub-target coverage explained with a direct measurement:** mean-over-groups coverage is 0.9024 against a nominal 0.900. | R1.1, R3 |
| 8 | **Sensitivity analysis over the excluded runs**, and the exclusion rules described as pre-specified rather than preregistered. | R2.4 |
| 9 | **Predicted-group mechanics, supervision, error propagation and ethics** written out in full; the ρ-sweep claim scoped to group-prior shift. | R2.5 |
| 10 | **Reproducibility appendix expanded**; literature comparison table added; requested citation added. | R1.3, R1.5, R2.6 |

The revised manuscript runs to about 23 pages against the 20 that applied at submission. We would rather report the
requested experiments and analyses in full than compress them; we are glad to shorten if the
editorial office requires it.

---

# Reviewer 1

## R1.1 — the target is 0.90 but the empirical values are 0.84–0.89

> "In Section 3 … the authors state that (1−α=0.90) is the target throughout. However, their
> empirical values are around 0.84–0.89, which are below the stated 0.90 target. The authors should
> clearly explain the reason for this difference."

The reviewer is right that the paper reported a sub-target number without an adequate explanation.
The cause is that we report a **minimum over four noisy per-group coverages**, which is a
downward-biased statistic even when every group is individually valid. We now demonstrate this with
a direct measurement rather than an argument.

**Where:** Section 5.1 (end), Section 5.7, and Appendix B.

> Mondrian's level settles at 0.854–0.886, below the nominal 0.90. That is a property of the
> statistic rather than a validity failure, and we quantify it directly: the unweighted mean over
> groups, which is what group-conditional calibration actually targets, is **0.9024** across the
> same eight cells (range 0.8995–0.9056), while the *minimum* over four noisy per-group coverages
> sits 0.018–0.045 below it.

Appendix B additionally simulates an *exactly valid* Mondrian procedure with no model and no data,
which reproduces the same sub-target minimum at our calibration sizes, and a new figure there puts
our runs against that law directly:

> The blue line is where an *exactly valid* Mondrian procedure puts that minimum, simulated at our
> own per-group calibration and test counts with no model and no data: **0.8695 against an observed
> 0.8660 on Waterbirds and 0.8847 against 0.8800 on CelebA**, with 98% and 95% of runs inside its
> central 95% range.

The two effects are worth separating. The closed-form 1.03·SD figure covers only the
min-over-*k* selection effect and predicts 0.013–0.030 against an observed 0.018–0.045, so on its
own it accounts for the level's direction and order of magnitude but not every point of it.
Simulating the full Mondrian law additionally captures the second effect — each threshold is
estimated from that group's own small calibration sample — and then the residual is
**0.004–0.005 of coverage**. We still report the residual rather than claim an exact match.

## R1.2 — the backbones are frozen, so why the title claim?

> "The authors freeze the neural network backbones … Therefore, why do the authors claim in the
> title … ? The authors should clarify this claim or provide additional experiments where the
> representation itself is trained or fine-tuned."

We have done both. This was the most substantial change to the paper.

**New experiment (Section 5.2, Table 4).** We fine-tune the backbone **end-to-end** under three
objectives — plain ERM, GroupDRO and group reweighting — with five seeds each, holding the head
fixed at plain ERM so the representation is the only thing that moves. The section opens with a
**manipulation check**, because a representation-level experiment answers nothing unless the
intervention actually moved the representation:

> GroupDRO fine-tuning raises worst-group accuracy from 0.516 to 0.738 on Waterbirds (+0.222) and
> from 0.396 to 0.470 on CelebA (+0.074): the lever was pulled, and by the largest margin any
> intervention in this paper produces. Reweighting did not help (0.466 and 0.364, both slightly
> below ERM). We report it regardless — a representation that fails to improve is still a
> representation that changed, and dropping it would be selecting on the outcome.

The dissociation survives. Under marginal calibration the three representations span **0.249** of
worst-group coverage on Waterbirds; under Mondrian they sit within **0.004** of one another.

**Title narrowed.** "…Is a Calibration Problem, **Not a Representation Problem**" has become "…Is
**More a Calibration Problem than a Representation Problem**". We also traced the same absolute
claim through the rest of the paper and softened it in fifteen further places — the figure caption,
the hypothesis statement, the Section 5.1 heading, the discussion, the conclusion and the related
work — since softening only the title would not have answered the objection.

We also corrected a terminological conflation the reviewer's question exposed: the submitted paper
used "representation" for the frozen backbone, which it never varied. Throughout the revision,
**backbone** means the frozen feature extractor, **representation** means what the fine-tuning
study changes, and **head** means the last-layer classifier.

## R1.3 — only two backbones; compare more clearly with related work

> "Only two backbones are used, which limits the generalizability of the results. In addition, the
> experimental results should be compared more clearly with those of other related studies."

**Backbones: two → four,** chosen to span two architecture families and three pretraining regimes —
an ERM-trained ResNet-50 (*d*=2048), CLIP ViT-B/32 (512, image–text contrastive), DINOv2 ViT-B/14
(768, self-supervised, never label-trained), and a supervised ImageNet ViT-B/16 (768). Every
results table now reports all eight backbone–dataset cells. The substantive finding is that the two
added backbones behave like the two originally reported: the Mondrian cross-training spread never
exceeds 0.024 in any cell.

**Literature comparison: new Table 2** and three paragraphs before Section 5. The comparison is
unflattering and we present it as such — our ResNet-50 arms are 10–24 points below published
worst-group accuracies on Waterbirds — and we state the three protocol differences that explain it:
our arms refit only the last layer on cached features (where the published GroupDRO and AFR train
the network), our ResNet-50 uses a deliberately light recipe and a 30,000-image CelebA training
subsample, and worst-group accuracy is read on our pooled evaluation domain rather than the
benchmark test split. DFR, the one arm whose protocol matches its published counterpart, is
correspondingly the closest (0.851 against 0.883 on CelebA).

We also explain why this does not bear on the paper's claims: every result is a **within-cell**
contrast computed on the same posteriors, so a weaker backbone lowers all arms in a cell together
and leaves the calibration contrast intact. The four backbones span a 0.20–0.95 range of
worst-group accuracy and the dissociation is unchanged across all of them.

## R1.4 — more independent training seeds

> "The authors should increase the number of independent training seeds to ensure the
> reproducibility and stability of the reported results."

The main grid and the end-to-end fine-tuning study now run **five** training seeds; the
calibration-policy ablation runs three, and the manuscript states the design as ≥3 training seeds ×
≥10 calibration splits rather than quoting a single figure that would not hold everywhere. In
raising the count we found something we must disclose, because presenting it as stability would be
misleading: **two of the five arms are deterministic.** ERM and AFR fit with L-BFGS, whose solver
ignores the random seed, so their across-seed standard deviation is exactly 0.000 and their base
accuracy is identical to five decimals in every cell.

**Where:** Section 4, new paragraph.

> Running more seeds therefore adds replicates for DFR, balanced subsampling and GroupDRO-LL
> (across-seed SD 0.003–0.027 in worst-group accuracy) but not for ERM or AFR. We report this
> rather than present zero variance as stability.

This is also why every interval in the revision is a cluster bootstrap resampling seeds and then
splits within them: the ten calibration splits inside a seed share one trained model.

## R1.5 — add the requested reference

> "The authors must add the following recent study to the references: 'New unfreezing strategy of
> transfer learning in satellite imagery…', Scientific African, 2024."

Added, and placed where it is substantively relevant rather than appended to a list. Because the
revision now fine-tunes backbones end-to-end, how much of a pretrained network to unfreeze is a
choice we make:

> How much of a pretrained network to unfreeze is itself consequential, and staged or partial
> unfreezing schedules change what a fine-tuned representation encodes [citation]; we unfreeze the
> whole network, so the representation lever is pulled as far as each objective allows and this is
> the strongest test of whether calibration still dominates that our setting admits.

---

# Reviewer 2

## R2.1 — scope of the main claim

> "All training interventions modify only the last-layer classifier while the backbone remains
> frozen. Please add a representation-changing robustness experiment if feasible; otherwise, narrow
> the title, abstract, and conclusions…"

We have done both, as described under R1.2: the new Section 5.2 retrains the representation
end-to-end, and the title, abstract and conclusions have been narrowed regardless. The claim is now
comparative throughout, and the abstract states the scope in its fourth sentence — "We test that
intuition on *both* levers" — so a reader encounters it before the results.

## R2.2 — Proposition 1(ii)

> "Wasserstein-1 or KS divergence alone generally does not imply a monotonic reduction in coverage
> at the pooled threshold. Please state sufficient assumptions for such a relationship or present
> the divergence–coverage link as an empirical hypothesis."

The reviewer is correct and the earlier claim did not follow. Proposition 1(ii) is now built on an
**exact identity** rather than an assertion about divergences:

  (1−α) − F_wg(q̂) = (1−π)·[F_¬wg(q̂) − F_wg(q̂)]

with the bound (1−α) − F_wg(q̂) ≤ (1−π)·D_KS following immediately. A remark then states plainly
what each divergence does and does not give:

> W₁ admits no such bound — it integrates the gap over the line while the shortfall depends on it
> at one point, so the gap can concentrate on a vanishing interval with W₁ → 0 and the shortfall
> fixed — so we use W₁ only as a descriptive summary of heterogeneity. Neither divergence *orders*
> coverage without a direction assumption: P_wg stochastically *below* the pool has the same W₁ and
> D_KS as its mirror image above it but yields a coverage surplus. Monotonicity needs
> P_wg ⪰_st P_¬wg, which holds here by construction, the worst group being defined as the
> lowest-coverage one — an assumption about which group is worst, not a consequence of the
> divergence.

All four statements were verified numerically before being written.

## R2.3 — the accuracy-matched analysis

> "The current interpolation uses relatively few independent training seeds, and several robust
> models show no overlap in accuracy with ERM. More controlled overlap and independent runs would
> strengthen the analysis; otherwise, limit the conclusion to matched settings and use a
> hierarchical uncertainty analysis that respects repeated calibration splits within the same
> trained model."

This comment led to the largest correction in the revision, and the reviewer's suspicion was better
founded than we realised.

**What we found.** The submitted analysis pooled each arm's (accuracy, divergence) points across
seeds **and** calibration splits. Measured separately, ERM's per-seed base accuracy is identical to
five decimals in every cell — its solver is deterministic — while its across-split range is
0.02–0.03. So the entire accuracy axis ERM contributed was *which evaluation rows were drawn*, not
*which model was trained*. Matching on it compares ERM on unlucky draws against a robust arm on
lucky ones. The accompanying bootstrap over pooled points also treated fifty nested rows as fifty
independent draws, which is exactly the hierarchical-uncertainty problem the reviewer raises.

**What we did.** We recomputed at model level (one accuracy per training seed) with a two-stage
cluster bootstrap that resamples seeds and then splits within them. ERM's accuracy support is then a
single **point**, and overlapping support exists in **one of the eight cells**: GroupDRO-LL on
ResNet-50/Waterbirds, whose per-seed accuracies bracket ERM's 0.9399. There the reduction is real
and **replicates** the earlier estimate — Δ_APS = +0.084 [+0.076, +0.095], Δ_RAPS = +0.084
[+0.074, +0.094], Δ_THR = +0.150 [+0.146, +0.158].

**What we withdraw.** The two other matched values in the submitted Table 3 — GroupDRO-LL on
CLIP/Waterbirds (+0.100) and balanced subsampling on ResNet-50/Waterbirds (+0.059) — rested on the
split-level spread. At model level neither arm's accuracy support reaches ERM's, and both are now
reported as unmatched.

Following the reviewer's instruction, we **limit H3 to that one matched setting** and draw no
accuracy-matched conclusion elsewhere. We also note that the reason the comparison is unavailable is
itself the more robust reading: on CelebA the robust arms raise base accuracy to 0.81–0.91 against
ERM's 0.68–0.70, so no common operating point exists at all.

## R2.4 — impact of the excluded runs, and preregistration status

> "The failed AFR runs on CelebA and the excluded GroupDRO seed are informative for comparing
> training interventions. Please include them in a sensitivity analysis and clarify whether the
> exclusion rules were preregistered or only pre-specified."

**Status of the rules.** They were **pre-specified, not preregistered.** The per-method accuracy
floors were fixed in the codebase on 11 June 2026, three months before any run reported here, so
they are pre-specified with respect to every number in the paper — but they were never lodged in a
public registry, and we do not claim otherwise. We have also corrected the paper's own heading,
which read "Pre-registered hypotheses" and now reads "Pre-specified hypotheses" with the
distinction stated.

**Sensitivity analysis (Appendix C).** Every excluded arm was evaluated and its records retained, so
this needed no re-run. Including them, the dissociation is unaffected: the Mondrian cross-training
spread rises only to 0.028 against 0.024. **One thing does change, and we report it against
ourselves:** the pre-specified C1 criterion — Mondrian within 0.02 of target *and* marginal
significantly below target for *every* method — is met in exactly one of eight cells, and that
single cell **flips to failing** when the gated AFR arm is included. We therefore rest the claim on
the spread magnitudes rather than on the pass/fail flag.

**AFR's collapse is not an untuned hyperparameter.** Since excluding a method on a hyperparameter we
chose would be unfair, we investigated. The mechanism is a class-prior inversion: at the default
γ=2 the reweighting drives the head to predict the positive class for ≈86% of examples where the
true rate is ≈15%, so a *majority* group becomes the worst group. We then computed an **oracle
upper bound** — the best γ chosen with knowledge of the evaluation set, which no selection rule
could beat — of 0.327, 0.427, 0.445 and 0.477 across the four CelebA backbones, all far below the
0.75 floor. AFR's exclusion is a property of the method in this regime, not of our choice of γ.

## R2.5 — shift and predicted-group deployment claims

> "The rho sweep mainly appears to change group mixture/correlation strength rather than the
> within-group data distribution, so the robustness claim should be scoped accordingly … For
> predicted-group Mondrian, describe exactly how predicted attributes determine thresholds, what
> supervision is required, how probe errors affect coverage, and discuss the privacy or ethical
> implications of inferring the CelebA Male attribute."

All five requests implemented.

**ρ-sweep scoping (Section 5.7).** The reviewer's reading is correct.

> The ρ sweep changes the *group mixture* of the evaluation distribution … while the
> class-conditional feature distribution within each of the four groups is untouched. It is
> therefore a group-prior shift, a special case of label-and-attribute shift, and *not* a general
> covariate shift … Our stability claim is scoped accordingly.

**Mechanics.** The label is never predicted; only the attribute is. The stratum is ĝ = 2y + â;
calibration points are binned by ĝ, each bin's ⌈(1−α)(n+1)⌉-th smallest score becomes its
threshold, and a test point is admitted if its score is at most its own bin's threshold. Coverage is
always tallied against the **true** group, so a mis-assignment is penalised rather than hidden.

**Supervision required.** The attribute on the *training* split only — not on the calibration
split, never at test. The reduction is from "attribute labels on calibration *and* test data" to
"attribute labels on training data only": a meaningful weakening of the requirement, not its
removal.

**Probe error.** It propagates asymmetrically: the minority stratum carries the *largest* threshold,
so a minority point misplaced into a majority stratum is judged against a tighter threshold and
under-covered, while the reverse error only wastes efficiency. We checked whether probe accuracy
predicts the penalty and it does **not** — the cell with the lowest AUROC (0.945) has no failures,
while a cell at AUROC 0.999 has one. What travels with the penalty is the arm's own
group-dependence: the two largest failures are ERM. We say so rather than offer AUROC as a
sufficient diagnostic.

**Privacy and ethics.** Written as a constraint on use rather than a disclaimer. Two points are
stated plainly: CelebA's *Male* field is a third-party binary annotation and **not self-identified
gender**, so a study that treats it as gender has erred before any ethical question arises; and
inferring such an attribute at test time assigns a sensitive label to someone who has not provided
it, wrongly for 2–5% of people even at the AUROC we measure. The recommendation is scoped to
settings where the attribute is already lawfully held, or where the inference stays inside an audit
an operator runs on their own system, and it is explicitly ruled out where the inferred label would
be stored, disclosed, or used to route individual outcomes. We note that special-category data
regimes may prohibit the inference outright regardless of accuracy.

## R2.6 — moderate the claims; improve reproducibility

> "Claims such as 'training buys efficiency' and 'accuracy is a good proxy' should be presented as
> empirical tendencies … Please also provide the exact calibration/test construction, per-group
> counts, training hyperparameters, the shift-robust procedure, random seeds, and a link to
> archival code with reproducible commands."

**Moderation.** Four statements are now withdrawn rather than softened, because measurement
contradicts them:

1. The per-cell accuracy–set-size correlation, reported bare in the submitted version, is
   **uninformative in five of eight cells** once a cluster-bootstrap CI is attached — each rests on
   only three to five methods. We now report a *direction*, not an estimate.
2. "The absolute efficiency range is narrow" is **withdrawn**. The spread runs 0.036 to 0.364, and a
   two-one-sided-test equivalence check rejects equivalence at a 0.10 margin in six of eight cells.
3. "Training buys efficiency, not coverage" is now stated as **conditional on Mondrian**: under
   marginal calibration training still moves coverage, so it is not a general claim.
4. "Mondrian is 2–4× more split-stable than marginal calibration" is **withdrawn**. It held on the
   two backbones of the submitted version but not on four. **Where:** Section 5.7.

   > An earlier version also claimed Mondrian was 2–4× more split-stable than marginal calibration.
   > On the four-backbone data the two are comparable — marginal's SD is 0.009–0.042 — and in two of
   > the eight cells Mondrian is the less stable of the two, so we withdraw that comparison.
   > Mondrian's advantage here is in the *level* it attains, not in its variance across splits.

   This one was found only because the reviewers asked for more backbones (R1.3), which is a fair
   illustration of why the request mattered.

"Accuracy is a good proxy for conformal burden" is likewise a tendency: point-estimate inversions
occur in five of eight cells but clear their cluster-bootstrap interval in only one, by 0.014 of
coverage.

**Reproducibility (Appendix D).** Now gives the exact calibration/test construction — disjoint
reservoirs from a seeded permutation, resampled within their own group indices, truncated to a
common size, realised ρ logged per draw, separate uniform streams for the randomised scores — the
per-group counts for both datasets, every head hyperparameter including the fine-tuning optimisers,
the shift-robust threshold in closed form, the seed sets, and the measured run times. We also
report that the worst group's Mondrian calibration count never falls below 50 (minimum 66 on
Waterbirds, 373 on CelebA), so the pooled-quantile fallback is never triggered by these runs.

**Archival code.** A DOI will be minted from a tagged release and inserted before publication; the
manuscript currently carries a visible placeholder rather than a provisional link.

---

# Reviewer 3

## R3.1 — the calibration-versus-representation claim is too strong; provide CIs or equivalence tests

> "The claim that coverage is a calibration property rather than a representation property seems
> too strong … Mondrian also does not always reach the target coverage. The authors should narrow
> the claim and provide stronger statistical tests, such as confidence intervals or equivalence
> tests."

We agree on all three counts.

**Narrowed.** The title is now comparative, and the same softening was applied in fifteen further
places rather than the title alone. Section 5.1's heading, for instance, has changed from "Coverage
Is a Calibration Property, Not a Representation One" to "Calibration Governs Worst-Group Coverage
Far More than Training Does".

**Confidence intervals.** Every point estimate in the main tables now carries a two-stage cluster
bootstrap interval that resamples training seeds and then calibration splits within them.

**Equivalence test.** Added: a two-one-sided-test check of whether the across-method spread in
worst-group set size lies inside a stated margin. It **rejects** equivalence at 0.10 in six of eight
cells, which is why we withdrew the description of the efficiency range as narrow.

**Mondrian not reaching target.** Addressed as under R1.1: the mean over groups is 0.9024 against a
nominal 0.900, so the sub-target *minimum* is a min-over-groups selection effect. We continue to
report the minimum, because it is the operationally relevant number, but now always alongside the
mean so the two are not confused.

## R3.2 — "training buys efficiency" is specific to Mondrian; the correlation is weak in one setting

> "The claim that training buys efficiency rather than coverage also seems specific to Mondrian
> calibration … Moreover, the accuracy-set size correlation is weak in one of the four settings. The
> authors should state this claim as conditional on Mondrian calibration and provide statistical
> significance or uncertainty for the reported correlations."

Both implemented, and the reviewer's concern proved larger than stated.

The claim is now **explicitly conditional on Mondrian** in the abstract, the introduction,
Section 5.4 and the discussion. And with cluster-bootstrap intervals, the correlation is
uninformative not in one of four settings but in **five of eight** — including the −0.97 and −1.00
we previously highlighted, which at *n* = 4 methods are not evidence of a deterministic
relationship. Table 5 now prints each interval, italicised where it spans essentially [−1, +1], and
the text reports a direction rather than an estimate.

---

# Changes we made without being asked

Three, all disclosed so that no reader discovers a silent alteration.

1. **A definitional inconsistency in Section 3.** The coverage gap was defined in a way that
   contradicted both the implementation and Appendix B. It now reads (1−α) − min_g cov_g,
   matching both.
2. **A rendering defect in the predicted-group table.** It declared eight columns and supplied
   seven, leaving a stray empty column. Fixed during the rewrite; all nine tables now pass a
   column-count check.
3. **An empty-set disclosure.** On DINOv2/Waterbirds the mean worst-group set size is *below one*
   (0.926–0.964 across scores), meaning a small fraction of prediction sets are empty. This is a
   legitimate outcome of standard split conformal with a highly accurate model, and coverage there
   remains valid, but a reader expecting sets of size ≥ 1 would otherwise be confused. We report it
   rather than clipping sets to a minimum of one, which would hide it.

---

We are grateful for the care the reviewers took. The revision is materially more honest than the
submission: two analyses were corrected, four claims withdrawn, and one pre-specified criterion
reported as failing in seven of eight cells. We believe the paper's central finding is stronger for
it, since it now rests on eight backbone–dataset cells and on a study in which the representation
itself is retrained, rather than on the absolute dichotomy the reviewers rightly questioned.
