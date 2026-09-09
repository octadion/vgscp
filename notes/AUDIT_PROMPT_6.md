# Sixth audit: verification pass

I will attach the compiled manuscript (`paper.pdf`), the response letter
(`response-to-reviewers.pdf`) and the before/after summary (`before-after.pdf`).

**This is a verification pass, not a hunt.** The fifth audit closed with: *"If items 1(a), 1(b),
1(c) and the CelebA determination in section 2 are settled, this is submittable."* All four are
settled. What I need is confirmation that the settling was done correctly and introduced nothing
new — and, if so, a clear verdict that the paper can go.

I would rather read "no findings in this section" than a defect manufactured to fill the output. If
the paper is ready, say so.

---

## What is already verified mechanically — do not re-derive

- **Every numeric cell in the appendix tables**, recomputed from the released records along a code
  path independent of the generator: 812 cells. All match.
- **Cross-document consistency**: superseded values asserted absent from all four source files, the
  title asserted identical in three.
- **Structure**: balanced braces and math in all four files, no dangling reference, no citation
  without a bibliography entry and none unused, every table's columns matching its rows, no column
  blank in every row, every table measured against the 372pt text block.
- **All 19 reviewer points** still test as addressed.
- **Quotation fidelity**: six of eight blue-ruled blocks in the letter are verbatim; two differ only
  by an expanded citation and dropped cross-references, which the letter states as its convention.

## Target 1 — the four items the fifth audit said must be settled

**1(a) §5.4.** Was titled "Most of the Apparent Transfer Is Accuracy" and asserted "It is." while
the Discussion said the controlled estimate is not smaller in the one setting where it exists.
Now titled "The Apparent Transfer Cannot Be Separated from Accuracy"; the assertion reads "It
is---and in seven of eight settings it cannot be told apart at all"; intro item 4 says "cannot be
separated from an accuracy effect."

*Check*: do §5.4, the Discussion, intro item 4 and the response letter's R2.3 answer now say the
same thing? Is the weaker claim actually supported by Table 5 and Figure 5?

**1(b) Table F13 and Appendix F** no longer name a fixed Mondrian worst group. *Check the two
captions against each other and against Appendix A.*

**1(c) The abstract** said coverage "returns to near the target under every training method" — the
exact clause Appendix D scores as failed. Now "most of that gap closes under every training
method." *Check the abstract, introduction, Discussion and Conclusion together: is the strong form
gone from all four, or has it survived in one of them?*

**Section 2, the CelebA determination.** Resolved from the source, not the tables:
`resample_to_rho` draws **with replacement** by default — the resampler's own docstring says this is
what makes an extreme ρ reachable from a finite pool — and `conformal_eval` does not override it.
So CelebA's g3 (blond-and-male, under 1% of the natural data) reaches its nominal 7,095 calibration
rows from a few hundred distinct images. Your case 1. Three consequences were written up:

- Appendix F now discloses the with-replacement draw and says the nominal count overstates the
  information behind it.
- Appendix C's "19× imbalance in threshold precision" is now scoped: on Waterbirds the 2.5% groups
  are the imprecise ones; on CelebA the imbalance runs the other way once unique images are counted.
- Appendix C now states which rule's worst group it simulates.

*Check*: is the disclosure adequate and in the right place, and does any surviving sentence still
treat a composited count as a sample size? **I could not compute g3's exact unique support here —
the raw attribute file is not available locally — so the text states the mechanism and the natural
share the paper already reports rather than inventing a count. Is that acceptable as written, or
does it need the exact number before submission?**

## Target 2 — what changed since, which nobody has read

- **Table 8, the hypothesis verdicts.** All six pre-specified hypotheses are now adjudicated in one
  table: predicted, what the data returned with the number, verdict. Two came back split. I
  deliberately did *not* include a "where reported" column — an index of pointers reads as a
  project artefact rather than a result. *Is each verdict right, and does it match what Section 5
  reports?*
- **Intro item 2** lost the r = 0.93 sentence and is now a mechanism check only, per your
  recommendation. *Does item 2 now read as a check rather than a result?*
- **Table 2's caption** was trimmed from ~250 words; the equivalence-margin discussion moved into
  §5.1, which also puts the 8/8-becomes-7/8-at-0.04 fragility in the body rather than only a
  caption. **The 0.10 set-size margin is now declared post-hoc** alongside the 0.05.
- **Panel labels in the long tables** can no longer be orphaned from their rows.
- **Companions**: before/after's stale Table 17/19 and Figure 7 labels corrected; page breakdown in
  the letter now ~40 pages as 22 body, 16 appendix, 3 references.

## Target 3 — the rendered document

The fifth audit found page 11 half empty with Table 2 landing on page 12, and attributed it to
Table 2's caption being too tall to fit under Figure 3. The caption is now about a third of its
former length.

- **Does Table 2 now print inside §5.1?** This is the one placement that has never been right.
- Is page 11 still half empty, and did trimming free enough space?
- Table 5's placement relative to its citation.
- Do the long tables' dataset panel labels now stay with their rows across page breaks?
- The new Table 8 is the first in the paper to use `p{}` columns and needs `\usepackage{array}`,
  which was added. Does it render, fit the text block, and break sensibly if it must?
- Any `??`, `[?]`, overfull lines, or broken spacing. Page count.

## Target 4 — the verdict

Body 8,254 words, appendix 4,582. Springer *Machine Learning* has no page limit.

1. Are items 1(a), 1(b), 1(c) and the CelebA determination correctly settled?
2. Did settling them break anything?
3. Is any of the added disclosure now so long that the claim is lost inside it? Name passages.
4. **Is this submittable?** If yes, say so plainly. If not, give me the shortest list that would
   make it so.

## Output

Short. One line per section where you find nothing. For anything you do find: the two sentences
that disagree, or the number and where it should be. Then the verdict in Target 4.
