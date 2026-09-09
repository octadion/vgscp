# Second audit: is every reviewer point still answered, and is anything in the wrong place?

I will attach the compiled manuscript (`paper.pdf`). The first audit was about writing and
structure and its findings are largely implemented. **This audit is about one risk only: whether
the revision still answers the reviewers, and whether moving material to the appendix has put
anything where it should not be.**

A wrong call here costs a rejection, not a comment, so I would rather you be conservative and tell
me something is risky than reassure me.

## What changed since the reviews

The paper was substantially restructured. Sections were merged, a section was folded into another,
and a good deal moved to the appendix. The main body is now 7,676 words in eight sections; the
appendix has six sections, 671 rows of generated tables, and three derivations.

**Current structure**

```
1 Introduction
2 Related Work
3 Problem Setup and Notation
4 Experimental Setup
5 Results
  5.1 The Calibration Rule Decides Worst-Group Coverage
  5.2 Retraining the Representation
  5.3 What Training Buys Instead: Smaller Sets
  5.4 Most of the Apparent Transfer Is Accuracy
  5.5 When Deployment Differs from Calibration
6 Discussion   7 Limitations   8 Conclusion

App A Derivations
App B Full Results
App C Why Mondrian Sits Below Target: The Min-over-Groups Effect
App D Pre-Specified Hypotheses
App E Excluded and Flagged Methods, and Their Influence
App F Procedures, Hyperparameters, and Statistics
```

Body: Figures 1–5, Tables 1–7. Appendix: Figures 6–7, Tables 8–17.

## Task 1 — verify each reviewer point

Below is each comment **verbatim**, then where I claim it is now answered. For each one tell me:

- **answered / partly answered / not answered** in the manuscript as it now stands
- if partly or not, what specifically is missing
- whether a reviewer re-reading this revision would recognise their point as having been taken
  seriously, or would think it had been deflected

Do not take my "where" claims on trust — check them against the PDF.

---

**R1.1** — *"In Section 3 … the authors state that (1−α=0.90) is the target throughout. However,
their empirical values are around 0.84–0.89, which are below the stated 0.90 target. The authors
should clearly explain the reason for this difference."*

Claimed: §5.1 (final paragraph, the mean over groups sits at 0.9024 while the minimum sits
0.018–0.045 below it), Appendix C in full, Figure 7, Tables 8 and 17.

---

**R1.2** — *"The authors freeze the neural network backbones … Therefore, why do the authors claim
in the title …? The authors should clarify this claim or provide additional experiments where the
representation itself is trained or fine-tuned."*

Claimed: §5.2 with Table 3 (end-to-end fine-tuning under three objectives); title now comparative;
abstract; §6; §8; §3 terminology. **Figure 6, the bar chart for this study, was moved to Appendix
B** — the table and the section text stayed in the body.

---

**R1.3** — *"Only two backbones are used, which limits the generalizability of the results. In
addition, the experimental results should be compared more clearly with those of other related
studies."*

Claimed: §4 and every results table (four backbones, eight backbone–dataset settings); Table 1 and
one paragraph in §4 for the literature comparison. The three protocol differences behind the gap
moved to Appendix F.

---

**R1.4** — *"The authors should increase the number of independent training seeds to ensure the
reproducibility and stability of the reported results."*

Claimed: §4, paragraph "What the seed does not move" — five seeds for the main grid and the
fine-tuning study, three for the calibration comparison, plus the disclosure that ERM and AFR are
deterministic because L-BFGS ignores the seed. Table 17 gives the seed-vs-split variance
decomposition.

---

**R1.5** — *"The authors must add the following recent study to the references: 'New unfreezing
strategy of transfer learning in satellite imagery…', Scientific African, 2024."*

Claimed: §5.2, in the fine-tuning protocol, cited alongside Kumar et al. (2022) for the claim that
how much of a network you unfreeze changes what the representation encodes.

---

**R2.1** — *"All training interventions modify only the last-layer classifier while the backbone
remains frozen. Please add a representation-changing robustness experiment if feasible; otherwise,
narrow the title, abstract, and conclusions…"*

Claimed: both. §5.2 is the experiment; the title, abstract and conclusions are all comparative now.

---

**R2.2** — *"Wasserstein-1 or KS divergence alone generally does not imply a monotonic reduction in
coverage at the pooled threshold. Please state sufficient assumptions for such a relationship or
present the divergence–coverage link as an empirical hypothesis."*

Claimed: §3, Proposition 1(ii) rebuilt on an exact identity (Eq. 3) with a KS bound (Eq. 4), plus a
remark on what each divergence does and does not give. **The proof moved to Appendix A**, along
with the construction showing W₁ carries no bound.

---

**R2.3** — *"The current interpolation uses relatively few independent training seeds, and several
robust models show no overlap in accuracy with ERM. More controlled overlap and independent runs
would strengthen the analysis; otherwise, limit the conclusion to matched settings and use a
hierarchical uncertainty analysis that respects repeated calibration splits within the same trained
model."*

Claimed: §5.4 — matching recomputed at model level, two-stage cluster bootstrap throughout, the one
matched setting reported with all three intervals, two previously-matched values withdrawn. Table 5
and Figure 4 in the body. **The explanation of why split-level matching misleads moved to
Appendix F.**

---

**R2.4** — *"The failed AFR runs on CelebA and the excluded GroupDRO seed are informative for
comparing training interventions. Please include them in a sensitivity analysis and clarify whether
the exclusion rules were preregistered or only pre-specified."*

Claimed: §4 carries one sentence — the hypotheses and accuracy floors were fixed before the
experiments and not publicly registered, so pre-specified rather than preregistered. **The six
hypotheses themselves moved to Appendix D**, and the sensitivity analysis plus the AFR diagnosis
are in Appendix E.

---

**R2.5** — *"The rho sweep mainly appears to change group mixture/correlation strength rather than
the within-group data distribution, so the robustness claim should be scoped accordingly … For
predicted-group Mondrian, describe exactly how predicted attributes determine thresholds, what
supervision is required, how probe errors affect coverage, and discuss the privacy or ethical
implications of inferring the CelebA Male attribute."*

Claimed: all five in §5.5 — the sweep scoped to group-prior shift, the stratum construction, the
supervision actually required, the asymmetry of probe error, and a closing paragraph on the ethics.
The Declarations section now points at §5.5 rather than saying ethics approval is not applicable.

---

**R2.6** — *"Claims such as 'training buys efficiency' and 'accuracy is a good proxy' should be
presented as empirical tendencies … Please also provide the exact calibration/test construction,
per-group counts, training hyperparameters, the shift-robust procedure, random seeds, and a link to
archival code with reproducible commands."*

Claimed: moderation in §5.3 and §5.4 (four claims withdrawn outright); reproducibility in
Appendix F, including the exact split construction, per-group counts, all hyperparameters for four
backbones, the shift-robust threshold in closed form, and the seed sets. **The archival DOI is
still a visible placeholder** — it will be minted from a tagged release.

---

**R3.1** — *"The claim that coverage is a calibration property rather than a representation
property seems too strong … Mondrian also does not always reach the target coverage. The authors
should narrow the claim and provide stronger statistical tests, such as confidence intervals or
equivalence tests."*

Claimed: title and §5.1 heading comparative; cluster-bootstrap intervals in every main table; a
two-one-sided-test equivalence check in §5.3 that rejects equivalence at 0.10 in six of eight
settings; the sub-target level handled as under R1.1.

---

**R3.2** — *"The claim that training buys efficiency rather than coverage also seems specific to
Mondrian calibration … Moreover, the accuracy-set size correlation is weak in one of the four
settings. The authors should state this claim as conditional on Mondrian calibration and provide
statistical significance or uncertainty for the reported correlations."*

Claimed: conditioning stated in the abstract, §1, §5.3 and §6; Table 4 prints every interval,
italicised where it spans essentially [−1, +1] — which is five of eight, not one of four.

---

## Task 2 — judge the placements

These are the moves I am least sure about. For each: **safe, or bring it back?**

1. **Proposition 1's proof → Appendix A.** R2.2 asked for sufficient assumptions to be stated. The
   statement, identity, bound and remark are in §3; only the proof moved.
2. **The six pre-specified hypotheses → Appendix D**, with one sentence left in §4 recording that
   they were fixed in advance and not registered. R2.4 is partly about this distinction.
3. **Figure 6 (the fine-tuning bar chart) → Appendix B**, with Table 3 and §5.2 kept in the body.
   R1.2 and R2.1 both asked for this experiment.
4. **Table 9 (dataset split sizes) → Appendix F.**
5. **Why split-level accuracy matching misleads → Appendix F.** R2.3 is about exactly this
   methodological point, though the corrected analysis is in the body.

Also: **is there anything still in the body that could safely move out, that I have not moved?**

## Task 3 — length

The main body is 7,676 words. The first audit suggested ~6,300, but that target came from
conference papers with a hard page limit, and Springer *Machine Learning* has none.

My position is that further cutting now removes material that answers reviewers, and that the risk
of over-trimming exceeds the benefit. **Do you agree, or is there still slack I am not seeing?** If
there is, name the passages.

## What I do not need

A general writing critique — the first audit covered that and its findings are implemented. Focus
on reviewer coverage, placement risk, and whether any cut went too far.

## Output

1. A table: point → answered / partly / not → what is missing, if anything.
2. For each of the five placements: safe or bring back, with one sentence of reasoning.
3. Your verdict on length.
4. **The single thing most likely to make a reviewer feel their point was not taken seriously.**
