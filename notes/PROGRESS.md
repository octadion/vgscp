# Progress — paper reconstruction

Status against your concerns and the 41 findings in `audit-manuskrip.md`.
Updated after each editing pass. Numbers are measured, not estimated.

**Last updated:** #28/#31/#35/#36 closed and verified from disk — 39 of 41 done

---

## Measured state

| | now | target | source |
|---|---|---|---|
| Main body | **7,676 words** | ~6,300 | audit #25 |
| Appendix | 3,103 words **+ 671 table rows + 3 derivations + 2 figures** | tables not prose | audit #26 |
| Body sections | 8 | — | |
| Results subsections | **5** (was 8) | 4–5 | audit #25 |
| Tables | **17** (was 9) | — | |
| Figures | 7 (5 body, 2 appendix) | — | |
| Reviewer points still addressed | **19/19**, only reproducibility detail in appendix | all | hard constraint |

---

## Your concerns

| # | Concern | Status |
|---|---|---|
| 1 | No "AI report" phrases — `.csv`, code, reviewer codes | **done** — 0 `.csv`, 0 reviewer codes, 0 hypothesis codes in body, 0 codebase/date sentences |
| 2 | Appendix must expand, not just describe | **done** — 8 generated tables (671 rows), 3 derivations, proof moved there |
| 3 | Main body too long | **partial** — 7,676 words (from 8,913, -14%) |
| 4 | Natural language, not model-ish | **done** — 8 X-not-Y left of 23, all load-bearing; voice tail cleared |
| 5 | Presentation must be followable | **needs your read** — I can't judge continuity from inside the edits |
| 6 | All reviewers stay addressed | **done and re-checked after every pass** |

---

## A. Pipeline residue visible to the reader

- [x] **1** `gate` in three places → excluded / exclusion rule
- [x] **2** `robust arms` baked into Figure 4's legend → regenerated
- [x] **3** `arm→method` find-replace broke sentences ("Of the 200 methods"); **run** now defined once and used
- [x] **4** H1–H6 labels in four captions and eight sentences → **0 in body**
- [x] **5** Appendix D still described **two** backbones after the study grew to four
- [x] **6** "ImageNet-V2 weights" → torchvision `IMAGENET1K_V2` checkpoint
- [ ] **7** DOI placeholder — **deferred at your request** (Zenodo later)
- [x] **8** broken citation `(elm 2024)` — authors added; paired with Kumar et al. 2022 so it does real work

## B. Intellectual, not cosmetic

- [x] **9** Cresswell contradiction confronted — our data reproduces it (disparity up in **7/8** settings)
- [x] **10** D_KS promoted to headline divergence; W₁ demoted to descriptive
- [x] **11** mechanism made load-bearing — **adapted**: Eq (3) is an identity (marginal coverage is within 0.006 of 0.90 everywhere), so the testable claim is Eq (4); bound holds 8/8 and predicts the gain at **r = 0.93**. New Figure 5.
- [x] **12** `burden` named two quantities → **0 uses**
- [x] **13** §6.5 heading/content mismatch → merged into §5.4

## C. Voice and sentences

- [x] **14** X-not-Y rhythm: **23 → 9**, the 9 remaining load-bearing
- [x] **15** defensive clauses — most gone, a few remain
- [x] **16** process narration — split-level reasoning to appendix; "organized around", "omitted as unnecessary", the two post-proposition remarks all cut
- [x] **17** paragraph closers — cross-reference closers removed from the results subsections
- [x] **18** "Question? Two-word answer."
- [x] **19** meta-sentences ("The unifying thesis…", "characterization rather than a method")
- [x] **20** academic words — `deflationary`, `co-scaling`, `operative`, `re-composit`, `invariant-ization` all 0
- [x] **21** `lever` ×7 → 1 in prose
- [x] **22** AUROC range — 5 prose mentions -> 1 (rest are table values)
- [x] **23** results paragraphs open with object+finding (§5.1 rewritten)
- [x] **24** self-cancelling limitation ("though the consistency…") removed

## D. Structure and proportion

- [ ] **25** body length — 8,913 → 7,730; still above the ~6,300 target
- [x] **26** appendix contains results instead of describing them
- [x] **27** moves: §4 folded in, §5.4+5.5 merged, §6.6–6.8 merged, Problem Setup 1,322→1,206, Experimental Setup 1,034→742, Table 1 to appendix
- [x] **28** Figure 3 moved to the appendix; Table 4 and §5.2 stay in the body
- [x] **29** captions — 1,466 → ~950 words; "Bars begin at zero" removed
- [x] **30** Introduction — 706 → 426 words, related-work dump out, 4 contributions with 1 number each, mechanism named
- [x] **31** concept-bottleneck dismissal sentence removed

## E. Citations

- [x] **32** Romano et al. 2020 (origin of equalized coverage) added
- [x] **33** Idrissi et al. 2022 for balanced subsampling
- [x] **34** Jones venue corrected to ICLR 2021 + lineage sentence after Prop 1(ii)
- [x] **35** Vovk 2003, Ding 2023, Kumar 2022, Guo 2017 (distinguishes the two senses of "calibration"), Lei & Wasserman 2014
- [x] **36** arXiv years swept; one entry dated 2026 with a July-2025 id corrected

## F. Things I hadn't asked about

- [x] **37** ethics declaration now points to §5.5
- [x] **38** "≥3 seeds" replaced with the exact counts per study
- [x] **39** 89/51 no longer restated in full in the introduction
- [x] **40** title shortened to a claim — **note:** I used *"Calibration Beats Robust Training for Worst-Group Coverage"*; the audit prefers *"Group-Conditional Calibration, Not Robust Training, Governs Worst-Group Coverage"*. Worth your call.
- [x] **41** abstract — audit judged the arc correct; left at 173 words

---

## Next

1. **Update the response letter's "Where addressed" pointers** to the new section numbers — the structure has now stopped moving, so this is the remaining editorial task
2. Optional further length reduction, if you want the body nearer 6,300 than 7,730

## Blocked on you

- Zenodo DOI (#7) — deferred
- Title wording (#40) — my choice vs the audit's
- A deep read for narrative continuity, which I can't do from inside the edits
- Optional: per-group coverage for all four groups needs a cheap Colab re-run; would let a reader verify the min-over-groups argument instead of trusting it


---

## Note on a near-miss

`finish_items.py` printed "ok" for five edits and then raised on the last one, before its single
write at the end — so none of them were saved. A follow-up one-liner then added a `\citep` for a
bibliography entry that had never been written, leaving a citation that would have compiled to "?"
in the PDF. The structural checker did not catch it because it validated `ef` against `\label`
and never checked `\cite` against the bibliography.

Both are fixed: the edits were redone and verified by reading the files back rather than trusting
memory, and the checker now validates citations in both directions (30 cites, 30 entries, none
dangling, none unused).


---

# Round 3 — the second audit's findings (2026-09-08)

All closed. Verified by `check_tex_final.py` (structure, citations both ways, column counts),
`tabwidth.py` (every table measured against the 372pt text block), `numguard.py` (nothing lost),
`reviewer_safety.py` (19/19 points still addressed) and `final_sweep.py` (no report artefacts, no
stale vocabulary, no contradictory numbers).

| # | Finding | What was done |
|---|---------|---------------|
| 1 | Table 2 had neither intervals nor a test, though R1.4 asked for seeds and R3.1 for CIs | Cluster-bootstrap intervals on the ERM lift and the across-method spread; TOST finds the per-group spread equivalent to zero within 0.05 in **8/8** settings; the reason for three seeds is now stated |
| 2 | "which group is worst … holds here by construction" pointed at an appendix that never argued it | The claim was **false** — the worst group moves with the training method. Replaced with what the records show, and Appendix A gained a subsection plus Table 8: pinned on the minority group in 85–100% of ERM/shared runs, but 15–38% under per-group thresholds, near the 25% of a four-way tie |
| 3 | §5.1 heading "Decides" was absolute | Now "The Calibration Rule Moves Coverage More Than Training Does" |
| 4 | §5.3 said two intervals exclude zero and one points the other way | **Three** exclude zero, all negative; nothing points the other way. Corrected against Table 4 |
| 5 | Duplicated sentence in Limitations | Removed |
| 6 | Jones comparison sat inside the proposition environment | Moved out, after the proof pointer |
| 7 | Coverage band 0.85–0.89 excluded a kept run at 0.841 | Now 0.84–0.89 in all three places |
| 8 | Table 5's accuracy ranges vs per-seed matching looked self-contradictory | Caption now says ranges are method means while overlap is assessed per seed at model level |
| 9 | Pre-specification sentence was in §3, away from the exclusion rules | Moved beside them in §4; duplicated seed counts dropped |
| 10 | The dagger in Tables 2 and 5 pointed at prose | Appendix E gained Table 18 with those exact values (0.813/0.885, 0.804/0.886) |
| 11 | R2.4's seed exclusions were reported only bundled with method exclusions | Separated: restoring the three seeds leaves the max spread at 0.024, largest shift under 0.001 |
| 12 | Per-group calibration/test counts (R2.6) were only a minimum | Table 19 now prints them: 2,622 and 14,936 per half, splitting 1245/66/66/1245 and 7095/373/373/7095 |
| 13 | Figure 4 (fine-tuning) sat in the appendix while §5.2 cited it | Moved into §5.2, beside Table 3 |
| 14 | Three appendix grid tables were at 93% of the text block | Padding tightened to 84% |
| 15 | Three citations had been dropped in the Related Work trim | Restored with their bib entries — the paragraph's own heading still promised them |
| 16 | before-after.tex described a manuscript that no longer existed | Rewritten: counts remeasured, appendix contents tabulated, numbering corrected |
| 17 | Response letter pointed at the pre-restructure numbering | All 22 cross-references remapped by content; R3.1 now reports the coverage equivalence test |
| 18 | "fixed in the codebase on 11 June 2026" survived in the letter | Rephrased to keep the checkable claim without the development log |
| 19 | arm/cell vocabulary survived in the letter | Purged, matching the paper |

**Kept against the audit's suggestion, with reasons:** the top-*k* projection ablation (three lines,
and it is a counterfactual supporting the thesis, not padding) and §3's matched-accuracy definition
(§5.4 applies it without restating it, so it is not duplicated, and R2.3 is about exactly it).

**Still open:** Zenodo DOI (deferred); a compile to check float placement, which needs LaTeX.


---

# Round 4 — the third audit (2026-09-09)

Every disputed number was recomputed from the records before editing. The audit's arithmetic was
right in every case it raised, and in three places the truth was worse than it reported.

## Factual errors corrected in the manuscript

| Claim | Was | Is |
|---|---|---|
| ERM yields the largest worst-group sets (Contribution 3) | "every setting where it is retained" | **four of six** — AFR is larger on ResNet-50/WB (1.217 vs 1.138), GroupDRO-LL on DINOv2/WB (0.962 vs 0.927) |
| Most accurate method gives smallest sets | seven of eight | **six of eight** |
| ERM's raw divergence | 0.14–0.22 | **0.05–0.22** (DINOv2/WB is 0.053) |
| Mondrian split-to-split SD | 0.023–0.034 / 0.011–0.018 | **0.021–0.036 / 0.011–0.019** |
| Marginal split-to-split SD | 0.009–0.042, Mondrian less stable in **two** settings | **0.007–0.043**, less stable in **five** |
| Empty sets, DINOv2/WB | 0.926–0.964 "across scores" | **0.888–0.962**; the variation is across methods |
| Coverage there | 0.858–0.881 | **0.857–0.882** |
| Set-size disparity rises | seven of eight, 0.080–0.355 → 0.140–0.608 | **all eight**, 0.080–0.339 → **0.168**–0.608 |
| Median predicted-group gap | 0.001–0.013 | **0.001–0.012** |
| Two largest predicted-group failures | both ERM | ERM, then **GroupDRO-LL** |
| Mondrian band, score-agnostic | 0.84–0.89 | **0.83–0.89** (RAPS reaches 0.838); 0.84–0.89 kept where scoped to APS |
| Auditing paragraph coverage | 0.854–0.886 | **0.84–0.89** (0.854–0.886 was the ERM row of Table 2, not a range over settings) |
| Waterbirds accuracy trade | 4–10 points | **0.1 above ERM to 14.9 below** |
| GroupDRO fine-tuning's +0.222 | "largest margin any intervention produces" | **not the largest** — DFR gives +0.302 and +0.435 at the last layer (Table 1) |
| Soft-band count | twelve methods | twelve **runs** |
| Appendix B prose | grid tables "give the mean over groups and set-size disparity" | they do not; prose corrected and **Table B10 added** to carry both |
| Shift-robust rule | "over-covers (0.84–0.96)" | over-covers for robust methods, but ERM stays at **0.653/0.777/0.798**; range is 0.65–0.97 |

## Claims scoped rather than corrected

- **r = 0.93 (Contribution 2).** The right panel of Figure 2 really is an ordering of coverage by
  divergence *across models* — the thing §3 says no divergence licenses. The bound is still applied
  run by run (left panel); the correlation is now presented as an association over eight aggregated
  points falling in two dataset clusters, with no interval claimed.
- **The 0.05 equivalence margin** was chosen after seeing the spreads and now says so.
- **Table 2's caption** now states three seeds and that the resampling is coarse.
- **Appendix E's seed restoration** now says the three seeds sit in a setting whose spread is 0.003,
  so they could not have moved the maximum — arithmetic, not evidence.
- **Appendix A** no longer says the argmin is "settled by sampling noise" (15% vs 38% is not a tie),
  and states that its two columns are different populations.

## Cross-document

Letter: masthead and R1.2 carried the *old* title; R2.1 quoted a sentence absent from the abstract;
R3.1 claimed every main table has intervals (four do not); R1.3's literature gap was 10–24 against
Table 1's **12–26**; two quoted blocks had drifted; appendix table numbers did not match the
compiled labels (A1, B2–B10, C11, E12, F13); page count was 23 against **36**. All corrected, and
R2.4 now reports the seed/method separation the paper makes.

before/after: table inventory, row count, appendix labels, stability numbers, coverage range and
the equivalence-testing row all corrected.

## Still open

Zenodo DOI. Float placement in Appendix B (three tables drift past the start of Appendix C) needs a
compile to confirm and fix. The `\big[...\big]` extraction artefact the audit saw is most likely a
PDF text-layer issue, not a source defect.

**Added after the round-4 write-up:** Table B7 (fine-tuning) was printing the objective under a
"Dataset" heading with two permanently blank columns, and the objectives as raw record values
(`erm`, `groupdro`, `reweight`). Rebuilt as five columns with the objectives named. Every
structural check had passed it, because the column *count* was right — so `blankcol.py` now flags
any column blank in every data row.

## Round 4, closing items (2026-09-09)

- **Appendix tables audited row by row.** Every cell of the grids, divergences, sweep, disparity and
  variance tables was re-derived from the records along a code path independent of the generator —
  812 cells, all matching. `audit_appendix.py` asserts it found rows, so a check cannot pass by
  reading nothing.
- **Trivial prediction sets disclosed.** Under THR with shift-robust calibration, CelebA collapses:
  mean worst-group set size 2.000 at coverage 1.000 in seven (setting, method) cells, above 1.9 in
  twelve. With two classes that is the set containing both, so the coverage is the trivial
  predictor, not a result. Appendix B now says so, and the shift-robust recommendation is scoped to
  APS and RAPS — which the paper claimed but had not actually done.
- **Float drift fixed at source** with `placeins[section]`, so an appendix's tables cannot render
  after the next appendix has begun. Needs a compile to confirm.
- **Contribution 3 reframed.** It read "six of eight … four of six … The sign is consistent", which
  contradicted its own figures. It now leads with what survived — the efficiency spread is
  0.036–0.364 and equivalence is *rejected* at 0.10 in six of eight settings — and demotes the
  accuracy ordering to a tendency, noting the correlations are uninformative in five of eight.

Body 7,650 words; appendix 3,814. Remaining: Zenodo DOI, and a compile to confirm float placement.

---

# Round 5 — the fourth audit (2026-09-09)

Every disputed figure recomputed from the records first. Two of the audit's findings did **not**
survive that check and were deliberately not applied:

- **Table 6 is correct.** The audit compared it against Table B2 (three-seed calibration
  comparison); Table 6 draws on the five-seed main grid, where DFR's coverage gap is 0.0503 and
  balanced subsampling's 0.0390. The defect was an unlabelled source, now stated in the caption.
- **Table 3's 0.249 and Table B7's [0.000] are correct.** 0.7333 − 0.4845 = 0.2488; the across-seed
  SD is 0.00016. Both are rounding, now disclosed in Table 2's caption.

## The serious one: CelebA group indexing — my error, from round 3

Table A1 labelled its second column "minority group" and put g3 for CelebA. `n_cal_worst_group`
gives that group **7095** — 47.5% of the calibration set. I derived the label from the modal argmin
and named it "minority" without checking its size. Consequences, all now fixed:

- §3's mechanism sentence ("the shortfall falls hardest on the minority group … π is small") is
  scoped to Waterbirds, where the worst group holds 2.5%. On CelebA compositing to ρ=0.95 leaves it
  holding 47.5%, so the small-π reading does not apply there.
- Table A1 rebuilt: it now prints the group's **share of the calibration set** beside its index,
  and the two columns no longer force one group to serve two populations.
- **Figure 2's bound was computed with a single π = 0.05 for both datasets**, inflating CelebA's
  bound by ~1.9×. It now reads π per run from the records. The bound still holds 8/8; tightness is
  0.59–0.93 (Waterbirds) and 0.43–0.58 (CelebA).
- R2.2's answer in the letter reports the correction, since R2.2 is the comment that asked for
  these assumptions to be stated.

## Other corrections

Body: §5.1's Mondrian band was Table 2's ERM row (0.854–0.886 → 0.856–0.884); "straddle ERM without
reaching it" contradicted "0.1 point above"; Limitations still said set sizes lie in [1,2] after the
empty-set disclosure; shift-robust does not lift every robust method above target (four cells
below); Appendix F's ε note scoped to APS/RAPS; the Discussion's "order of magnitude" had no
supporting number and is gone; "almost always" → six of eight; Waterbirds Mondrian level
0.85 → 0.87; H1→H3 mislabel; a cross-reference pointing into its own section.

Appendix: Table C11's caption misdescribed its own columns three ways (counts summing to 502, mean
"at nominal 0.90" when it is 0.915, "shrinks" when row 3 rises); "bracket" → "span the same scale";
H1 now reported as **split** — its near-target clause is not met on the paper's own C1 test, which
is stated rather than left to Appendix E; C1's definition added to Appendix D; H2's base-versus-
worst-group accuracy substitution noted; the 0.85 soft reference explained.

Captions: Table 2 (post-hoc margin, the binding CLIP/Waterbirds row at 0.042 against 0.05, and the
rounding disclosure); Table 6 (source and calibration rule); Table 5 (the reverse-direction
caveat); E12 (scoped to Table 2); B4 (degenerate-cell warning); B9 (two columns no longer share a
heading); B7 (brackets promised only where they appear).

Floats: `\FloatBarrier` before each results subsection and the Discussion, since `placeins[section]`
does not reach subsections and four of six body tables were landing one or two subsections late.

Contributions: each item now states its kind — finding, mechanism, finding, methodological note —
rather than presenting four claims of unequal strength as equal. Contribution 1 gains the
lift-versus-spread ratio (smallest lift 0.059 is 2.45× the largest spread 0.024).

Companions: empty-set range, predicted-group failures, table count, page count (36 → ~39),
literature gap, Mondrian band, post-hoc margin — all were corrected in the manuscript and not in
the letter or summary. Now synced.

Body 8,142 words; appendix 4,284. Remaining: Zenodo DOI, and a compile to confirm float placement.

---

# Round 6 — the fifth audit (2026-09-09)

The audit closed with "if 1(a), 1(b), 1(c) and the CelebA determination are settled, this is
submittable." All four are settled.

## The one that mattered: compositing draws with replacement

The audit spotted that Appendix F reports CelebA's reweighting split as 4,650/3,895/1,331/**81** —
g3 is 0.81% of the natural data — while Table A1 and F13 give that group **7,095** calibration rows.
`experiments/shift_resampler.py` settles it: `resample_to_rho` draws `replace=True` by default,
"so an extreme rho stays reachable from a finite pool", and `conformal_eval` does not override it.
So CelebA's g3 reaches 7,095 rows from a few hundred distinct images.

This does not touch the central comparison — both calibration rules are evaluated on the same
composited distribution — but three things were wrong and are now fixed:

- Appendix F now says the draw is **with replacement**, and that g3's nominal 7,095 overstates the
  information behind it.
- Appendix C's "19× imbalance in threshold precision" was a Waterbirds statement presented as
  general. On CelebA the imbalance runs the *other* way once unique images are counted.
- Appendix F's reassurance about "the worst group's Mondrian calibration count" named a fixed worst
  group that Appendix A says does not exist, and quoted a nominal count. Same for Table F13's
  caption.

Appendix C now also states which rule's worst group it simulates — the distinction I flagged as
least clearly stated.

## §5.4 claimed what its own correction removed

Titled "Most of the Apparent Transfer Is Accuracy" and asserting "It is." — while the Discussion
says the controlled estimate is not smaller in the one setting where it exists, and undefined in
the other seven. Retitled to "The Apparent Transfer Cannot Be Separated from Accuracy"; the
assertion and intro item 4 now say the reduction is unidentified, which is what the data show.

## Other corrections

Abstract asserted the exact H1 clause Appendix D adjudicates as failed ("returns to near the
target"); §5.5's ≥0.88 bar was an average over methods presented as a per-method claim (CelebA
clears it in 7 of 14 kept cells); the two shortfall ranges 0.018–0.045 and 0.016–0.044 are
different quantities, both correct, now labelled; the score-agnostic band is 0.83–0.89; Limitations
echoed the withdrawn "efficiency range is narrow"; Table 1's caption promised three protocol
differences where §4 gives two; 0.9024 − 0.8728 = 0.0296 not 0.0295; Table 1's dagger implied our
own runs use no group annotations.

## Hypotheses

All six now adjudicated in one table (Table 8) — predicted, what the data returned **with the
number**, verdict. Two came back split. Deliberately *not* a "where reported" index column.

## Structure

r = 0.93 cut from intro item 2, which is now a mechanism check and nothing else. The 0.10
equivalence margin is declared post-hoc like the 0.05. Table 2's 250-word caption trimmed, its
margin discussion moved into §5.1 where it also fixes a disclosure asymmetry. Dataset panel labels
in the long tables can no longer be orphaned from their rows.

Companions: before/after's stale Table 17/19 and Figure 7 labels, table counts, and the letter's
page breakdown (~40, 22/16/3).

Body 8,254 words; appendix 4,582. Remaining: Zenodo DOI; one Colab query for g3's exact unique
support; a compile.

---

# Round 7 — the Colab run (2026-09-09)

**The overwritten CSV is safe.** `results/calibration_ablation_4bb.csv` was replaced during the
run, and `results/` is gitignored so there is no history to recover from — but all 812 appendix
cells still verify against the current file, so it is numerically identical for everything the
paper uses. Do not overwrite it again.

**Unique support, measured and folded in.** Averaged over the ten calibration splits:

| | nominal | distinct | ratio |
|---|---|---|---|
| CelebA g3 (blond, male) | 7,095 | **140** | **50.8×** |
| CelebA g0 | 7,095 | 4,426 | 1.6× |
| Waterbirds g3 | 1,245 | **294** | **4.2×** |
| minority groups | 66 / 373 | 59–361 | ~1.0× |

The fifth audit estimated "roughly 120" for CelebA g3; measured 140, and the pool half contains
only 140, so every image is used about fifty times. **Waterbirds has the same structure in mild
form**, which no audit had noticed — the paper had treated it as the clean case. Table F13 gained a
"distinct images" column; Appendix C and F now carry the measured figures instead of "a few
hundred". The sharpened point: on CelebA the least precisely estimated threshold belongs to the
group the composited counts make look best supported.

**The assertion failure is not a defect in the run.** In-run consistency was exact — 2,400 rows,
max coverage difference 0.00e+00, zero argmin mismatches — so the recomputed per-group minimum
equals `evaluate()`'s own `worst_group_cov` everywhere. What differs is the *released* CSV, by up
to 1.29e-02 per row, while the per-setting means still match to <6e-4 (which is why the appendix
tables pass). Per-row differences that cancel in the mean point at either a cold feature cache
(ResNet-50 is trained in-domain, so its features are stochastic) or a changed head fit.

`notebooks/diagnose_cell.py` separates the two: ERM and AFR fit with L-BFGS, which ignores the
seed, so if *they* move the features differ and head fitting is not the cause. `pg` is still in the
Colab kernel, so it costs seconds rather than another 40 minutes.

The notebook itself now **saves before it checks** — an assertion that discards a 40-minute run is
the wrong shape — and reports the released-CSV comparison rather than dying on it.

Outstanding: the per-group coverage table (needs the diagnostic cell to write the CSV); Zenodo DOI;
a compile.

## Round 7b — the Colab results are in

**The discrepancy is confined to one method.** Diagnosed locally from both CSVs: ERM, DFR,
GroupDRO-LL and balanced subsampling reproduce the released records **exactly** (0.00e+00 on all
1,920 rows). Only AFR differs — 36 of 480 runs, at most 0.013, on ResNet-50 and CLIP over
Waterbirds.

My own diagnostic probe was misleading: it grouped ERM and AFR as "the deterministic solvers", so
AFR moving made it print "features changed". ERM alone is the clean test, and ERM is exact — so the
features are reproducible. AFR is the only two-stage fit here: a first head's converged
probabilities become the second's sample weights, so a last-bit difference is amplified. Disclosed
in Appendix F; no reported number changes.

**Per-group coverage is now measured, and it confirms the paper exactly.** Table B10 replaces the
paragraph that used to report its own absence:

- Under per-group thresholds every group sits at **0.886–0.917** — individually near target.
- Per-setting mean over groups **0.8995–0.9056**, averaging **0.9024** — the exact figure §5.1
  already quoted, now measured rather than simulated.
- Mean-minus-minimum **0.018–0.045** — again the range already stated.
- Half the group-cells fall below 0.90 and half above: the minimum is low because it is a minimum.

Under a shared threshold the same table shows the dissociation group by group (Waterbirds g2 at
0.735–0.866 against g0 at 0.908–0.931), and on CelebA the lowest group is g3 — an aligned 47.5%
group — while the nominal minorities are *over*-covered at up to 0.953. That is the group-indexing
correction visible directly in the coverages.

The recomputed minimum reproduces the published worst-group coverage on all 2,400 runs, so these
are the same runs at finer grain.

22 tables now (8 body, 14 appendix). Remaining: Zenodo DOI, and a compile.

---

# Round 8 — closing the loop (2026-09-09)

All thirteen items from the fifth audit, fixed at **+13 words net** on 13,432 (0.1%).

**Five were my own bugs, all introduced while fixing earlier audits.** Recorded because the pattern
is the point:

| bug | how it happened |
|---|---|
| Table 2's caption never got trimmed | my "move to §5.1" used an anchor that lives *in the caption*, so the text moved caption→caption and left a pointer to a discussion never written |
| Appendix C: "the minimum falls on a minority group" | written while fixing the group indexing; Table B2 puts it on g3, an aligned group, in all four Waterbirds settings |
| "exactly on all 2,400 runs" vs AFR's 36 | Appendix B written before the AFR result, Appendix F after, never reconciled |
| "a mean of the minima of 0.856–0.884" | that is the range; the mean is 0.8728 |
| "half below 0.90 and half above" | 13 below, 15 above, 4 exactly at it |

**The tooling gap that let them through.** Every mechanical check I had verified *numbers* — 812
appendix cells, cross-document values, column counts, table widths. None could see two paragraphs
disagreeing, which is the class that produced all five. `contradict.py` now asserts the invariants
each one violated, with the reason attached, so a future edit that reintroduces one fails loudly.

**The other eight:** Appendix C's 19× scoped to nominal count (5–12× by distinct images); a clause
noting Figure C1 and Table C12 simulate at nominal counts and why that is conservative; Table F15's
caption naming its columns instead of counting from the right; the Discussion aligned with §5.4;
H5 scoped to Mondrian; AFR's "third decimal" corrected to "third, and occasionally the second",
with the re-run's scope stated; §5.3's AFR exception given its reproduction context; and the
companions' table numbers (7 body / 15 appendix), labels (C12, D13, E14, F15, B2–B11), page
breakdown (45: 22/20/3) and two missing content entries.

**Paid for by removing duplication, not by cutting evidence.** Appendix C re-derived 0.9024 and
0.8728 as if establishing them — those are Table B2's numbers now, so it cites them; §5.1 carried
two framings of one gap; Appendix F restated the six counts Table F15 prints; the per-group
paragraph said three things twice.

Body 8,252 words (−2), appendix 5,180 (+15). All checks pass: structure, 812 appendix cells, table
widths, no blank columns, cross-document values, contradictions, **19/19 reviewer points**, and six
of eight quotation blocks verbatim with two differing only by the stated convention.

Next: page optimisation (targets to be agreed first), then one final audit on the submission
candidate.

---

# Round 9 — page optimisation, items 1–3 (2026-09-09)

**1. The RAPS grid became a cross-score summary.** 34 rows supported one sentence — that the
dissociation reproduces under RAPS and THR. Table B3 now states it for all three scores in eight
rows: per-group spreads run **0.001–0.029** against **0.002–0.478** for the shared rule, and are
the smaller of the two in **22 of 24 pairs**. Both exceptions are ViT-B/16 on CelebA, where the
shared threshold already leaves nothing to recover — stated rather than buried.

That is better evidence than the grid it replaces: the grid made a reader take maxima and minima
across 34 rows to see the claim; the summary is the claim. The APS grid stays (every body table
draws on it) and the THR grid stays (the degenerate-cell disclosure needs its individual
1.000/2.000 cells). Full RAPS cells remain in the released records. All 48 cells of the new table
are verified against those records by `audit_appendix.py`, which now reads which grids exist rather
than assuming three.

**2. The ten longest captions.** Captions had reached 2,159 words across 29 — outside every word
count I had been watching, which is how they grew unnoticed. Now **2,008**, with the longest down
from 164 to 132. What came out was argument, never data and never a definition needed to parse a
column: the three-cluster caveat (in §4), the GroupDRO-LL bracketing case (in §5.4), the small-π
reading (in §3 and Appendix A), the CelebA g3 example (in the paragraph below its table).

One removal did **not** survive elsewhere and was put back: the reason the smallest simulated pool
over-covers at 0.915 appears nowhere else, so a reader meeting an "exactly valid" procedure at
0.915 would have had no explanation. Ten words restored.

**3. The repetition sweep found almost nothing, and I did not manufacture cuts.** The 12 apparent
mentions of 0.024 are mostly different quantities — across-method spread, variation over the sweep,
table cells, a predicted-group gap. "Six of eight" appears in the introduction, §5.3 and the
verdict table, which are three different jobs. The genuine duplication had already gone with the
batch-6 trim.

Body 8,187 words (−65), appendix 5,151 (−29), captions 1,508 (−151). All checks pass, including
19/19 reviewer points and the contradiction invariants.
