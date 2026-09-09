# Fourth audit: verify the corrections, and judge two calls I made myself

I will attach the compiled manuscript (`paper.pdf`), the response letter
(`response-to-reviewers.pdf`) and the before/after summary (`before-after.pdf`).

The third audit was devastating and correct. It found sixteen wrong numbers in the manuscript —
including one inside the paper's own Contribution 3 — plus seven cross-document contradictions.
Every disputed figure was recomputed from the released records before editing, and its arithmetic
held in every case; in three places the truth was worse than it reported. All of it is implemented.

**Do not re-audit prose, structure, length, reviewer coverage or placement.** Audits 1 and 2 closed
those. Two further things are also already done, and re-doing them wastes the pass:

- **Every numeric cell in the appendix tables has been re-derived from the records** along a code
  path independent of the generator — 812 cells across the three grids, the divergences, the sweep,
  the disparity table and the variance table. They match. So do not recompute them. What that check
  *cannot* see is whether a column heading correctly describes the column it sits over, or whether
  a caption describes the table it belongs to, because the script was told the intended mapping.
  **Those are still worth your eyes** — two captions were wrong on exactly that before.
- **Contribution 3 has already been reframed** (see Target 3), so judge the reframing rather than
  the version the third audit criticised.

---

## Target 1 — did the corrections land, and did any of them break something?

For each figure below, check it as it now appears in the PDF, and check that nothing elsewhere
still carries the old value. A half-applied correction is worse than none, because the paper then
contradicts itself.

| Claim | now reads |
|---|---|
| Most accurate method gives smallest worst-group sets | six of eight |
| ERM gives largest worst-group sets | four of six retained settings; AFR (1.217 vs 1.138) and GroupDRO-LL (0.962 vs 0.927) named as exceptions |
| ERM's raw divergence | 0.05–0.22 |
| Mondrian split-to-split SD | 0.021–0.036 (WB) / 0.011–0.019 (CelebA) |
| Marginal split-to-split SD; Mondrian less stable in | 0.007–0.043; five of eight |
| Empty sets, DINOv2/Waterbirds | 0.888–0.962, across methods and scores |
| Coverage there | 0.857–0.882 |
| Set-size disparity rises | all eight settings, 0.080–0.339 → 0.168–0.608 |
| Median predicted-group gap | 0.001–0.012 |
| Largest predicted-group failures | ERM, then GroupDRO-LL |
| Mondrian band | 0.83–0.89 where score-agnostic; 0.84–0.89 only where scoped to APS |
| Auditing paragraph coverage | 0.84–0.89 |
| Waterbirds accuracy trade | 0.1 point above ERM to 14.9 below |
| GroupDRO fine-tuning's +0.222 | no longer called the largest in the paper; DFR's last-layer gains acknowledged |
| Soft-band count | twelve runs, not methods |
| Shift-robust rule | over-covers for robust methods (0.84–0.96) but ERM stays at 0.653/0.777/0.798; range 0.65–0.97 |
| Literature gap (letter) | 12–26 points |
| Page count (letter) | 36 |

Also check the narrowings for consistency — does the paper elsewhere still assert the strong form?

- The r = 0.93 correlation is now presented as an association across eight aggregated points in two
  dataset clusters, explicitly **not** underwritten by the bound, because it orders coverage by
  divergence across models. Do §5.1, Contribution 2, the abstract and the conclusion all agree?
- The 0.05 equivalence margin is declared post-hoc.
- Table 2's caption states three seeds and calls the resampling coarse.
- Appendix E says the three restored seeds could not have moved the maximum — arithmetic, not
  evidence.
- Appendix A no longer says the argmin is settled by sampling noise, and states that its two
  columns are different populations.

## Target 2 — four changes made since the third audit, which nobody has reviewed

These are new. Be adversarial: assume I may have overcorrected, or introduced a fresh contradiction.

1. **Trivial prediction sets, disclosed in Appendix B.** Under THR with shift-robust calibration,
   CelebA collapses: mean worst-group set size 2.000 at coverage 1.000 in seven (setting, method)
   cells, above 1.9 in twelve. With two classes that is the set containing both, so the coverage is
   the trivial predictor. Is the disclosure adequate and in the right place, or does it belong in
   the body? Does any claim elsewhere in the paper still quietly rely on those shift-robust CelebA
   numbers? **Check the sweep and stability claims especially.**
2. **The shift-robust recommendation is now scoped to APS and RAPS.** Previously the paper said it
   was scoped and it was not. Is it consistently scoped now, including in the Discussion, the
   Limitations and the Conclusion?
3. **Table B7 (fine-tuning) was rebuilt.** It had been printing the objective under a "Dataset"
   heading with two permanently blank columns and raw values (`erm`, `groupdro`, `reweight`). Every
   structural check passed it because the column *count* was right. Does it read correctly now, and
   is there any other table with the same class of defect — a heading that does not match its
   column, or a value in the wrong column?
4. **`placeins[section]` was added** so an appendix's floats cannot render after the next appendix
   begins. The third audit found three Appendix B tables doing exactly that. Is it fixed, and did
   the barrier introduce large white gaps, a stranded page, or a table now far from its discussion?

## Target 3 — judge the Contribution 3 reframing

After the third audit, Contribution 3 read "six of eight … four of six … The sign is consistent",
which contradicted its own two figures. I rewrote it to lead with what survived: the worst group's
set size differs between methods by 0.036–0.364 and a two-one-sided-test check **rejects**
equivalence at a 0.10 margin in six of eight settings — a positive result rather than a failure to
reject — with the accuracy ordering demoted to a tendency and the correlations noted as
uninformative in five of eight.

- Is that the right call, or did I trade a false claim for a vague one?
- Does the new Contribution 3 match what §5.3 actually shows?
- **Does the paper still support its title, "Calibration Beats Robust Training for Worst-Group
  Coverage"?** Contribution 1 (Table 2: spread 0.358 → 0.024, equivalence to zero in 8/8, holding
  under end-to-end fine-tuning) looks to me like the one strong finding, with Contributions 2 and 3
  now materially weaker. If the honest structure is one headline result plus two supporting
  observations, say so — I would rather restructure than defend three contributions of unequal
  strength.

## Target 4 — the rendered document

- Any `??`, `[?]`, overfull lines into the margin, or broken spacing?
- Do the three `longtable` grids break sensibly with headers repeating?
- The dagger markers in Tables 1 and 2 abutted the next column's value in the text layer. Does that
  render correctly?
- Page count and overall visual soundness.

## Output

1. **Corrections that did not land, or that broke something else** — the specific sentence.
2. **Your verdict on each of the four changes in Target 2** — sound, overreaching, or wrong.
3. **Your answer on Target 3**, with a recommendation I can act on.
4. **Rendering problems** visible in the PDF.
5. **The single thing most likely to be caught by a reviewer who checks arithmetic.**
