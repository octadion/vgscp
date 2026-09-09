# Fifth audit: adversarial read before submission

I will attach the compiled manuscript (`paper.pdf`), the response letter
(`response-to-reviewers.pdf`) and the before/after summary (`before-after.pdf`).

Four audits have run. The first two covered writing, structure, reviewer coverage and placement and
are closed. The third found sixteen wrong numbers in the manuscript. The fourth found that those
corrections had not reached the companion documents, that several sentences had been broken by the
corrections around them, and one structural error about CelebA's group indexing that turned out to
run deeper than it could see from the tables alone.

**This is the last pass before submission.** Assume the obvious defects are gone and go looking for
the ones that survive four rounds of checking.

---

## What has already been verified — do not spend the pass re-doing it

Re-deriving these would consume the audit and find nothing.

- **Every numeric cell in the appendix tables** has been recomputed from the released records along
  a code path independent of the generator: 812 cells across the three grids, the divergences, the
  sweep, the disparity table and the variance table. All match.
- **Cross-document numeric consistency** is now checked mechanically. Seventeen superseded values
  are asserted absent from all four source files, and the title is asserted identical in three.
- **Structural checks pass**: balanced braces and math delimiters in all four files, no dangling
  reference, no citation without a bibliography entry and none unused, every table's column count
  matching its rows, no column blank in every data row, every table measured against the 372pt text
  block (widest 89.5%).
- **All 19 reviewer points** still test as addressed.

## What the fourth audit got wrong — so you do not repeat it

Two of its findings did not survive checking against the records, and I did not apply them. If you
reach the same conclusions, you are reading the same trap:

- **Table 6 is correct.** It draws on the five-seed main grid, not the three-seed calibration
  comparison behind Appendix B, so it is not expected to reproduce from Table B2. DFR's coverage
  gap there is 0.0503 and balanced subsampling's 0.0390. The caption now says which experiment it
  comes from.
- **Table 3's 0.249 and Table B7's [0.000] are correct.** 0.7333 − 0.4845 = 0.2488, and the
  across-seed SD is 0.00016, not zero. Table 2's caption now discloses that values are computed
  before rounding.

## Target 1 — the CelebA group-indexing correction, which touched five places

The fourth audit derived a contradiction from Table B10's mean-over-groups and could not resolve it
without the records. The records resolve it against the paper: on CelebA the group that comes out
worst under a shared threshold holds **47.5%** of the calibration set (7095 of 14936), not 2.5%.
Compositing to ρ = 0.95 makes the label-attribute-aligned cells the large ones, so the group is a
minority in the *class* but not in the group prior.

Everything downstream was corrected:

- §3's mechanism sentence is scoped — the (1 − π) reading now covers Waterbirds and is explicitly
  said not to cover CelebA.
- Table A1 was rebuilt: it prints the worst group's **share of the calibration set** beside its
  index, and its two columns no longer force one group index to serve two populations.
- **Figure 2's bound was being computed with a single π = 0.05 for both datasets**, inflating
  CelebA's bound by roughly 1.9×. It now reads π per run from the records. The bound still holds in
  all eight settings; tightness is 0.59–0.93 on Waterbirds and 0.43–0.58 on CelebA.
- The response letter reports the correction under R2.2, the comment that asked for these
  assumptions.

**Check**: is the scoping consistent everywhere, or does some sentence still describe both datasets
as if the small-π story covered them? Does Appendix C's simulation — which uses 373/7095 per-group
counts for CelebA — simulate the right group, given that the *shared-threshold* minimum lands on a
7095 group while the *Mondrian* minimum lands on a 373 group in most runs? That distinction is the
one I am least sure I have stated clearly.

## Target 2 — sentences the corrections may have broken

Every correction risks leaving the sentence around it inconsistent; the fourth audit found six such
cases and this round changed roughly forty passages. Read for **internal contradiction** rather
than for wrong values:

- A range whose stated bounds are contradicted by a clause in the same sentence.
- A claim scoped in one place and unscoped three paragraphs later.
- A caption describing a column the table no longer has, or omitting one it gained.
- A hedge that has been added in the body but not in the abstract, the introduction, the discussion
  or the conclusion — these four are where the strong form of a claim survives longest.

Specific places worth your attention, all rewritten this round: §5.1's opening and its
bound-tightness sentence; §5.4's accuracy-straddle sentence; §7's |Y| = 2 clause; Appendix B's
shift-robust paragraph and its degenerate-set disclosure; Appendix C's simulation caption; Appendix
D's hypotheses; Appendix F's ε note; Table 2's, 5's, 6's, B4's, B7's, B9's and E12's captions.

## Target 3 — the pre-specification apparatus

Appendix D now reports **H1 as split**: its "near target for every training method" clause is not
met on the paper's own C1 test (one setting of eight), while its training-dependence clause is
confirmed in all eight. C1's definition moved into Appendix D, and H2's base-versus-worst-group
accuracy substitution is noted.

**Check**: is every one of the six hypotheses now adjudicated somewhere a reader can find, and does
each verdict match what Section 5 actually reports? This apparatus is where a sceptical reviewer
looks for selective scoring, and the paper takes credit for reporting against itself.

## Target 4 — the four items in the introduction

Following the fourth audit's argument that four contributions of unequal strength read as padding,
each item now declares its kind: **finding**, **mechanism**, **finding**, **methodological note**.
The numbering was kept because renumbering would ripple into both companion documents. Contribution
1 gained the ratio the audit suggested: the smallest lift from re-calibrating (0.059) is 2.45× the
largest across-method spread (0.024).

**Check**: does each item's body match its declared kind? Is the "mechanism" item now clearly not
claiming to be an independent empirical result? And — the question I most want answered — **would
you still recommend collapsing to two contributions plus a methods note, or does the labelling
solve it?** If labelling is not enough, say so plainly; I will do the restructure.

## Target 5 — the rendered document

`\FloatBarrier` was added before each results subsection and before the Discussion, because
`placeins[section]` does not reach subsections and four of six body tables were landing one or two
subsections after the text that owns them.

- Does every body table and figure now appear within the subsection that discusses it?
- Did the barriers create stranded pages, large white gaps, or a table pushed to a page of its own?
- The appendix floats were already fixed; confirm they still sit inside their own appendices now
  that a table has been added to Appendix B.
- Any `??`, `[?]`, overfull lines into the margin, or broken spacing? Table B5 and B10 have the
  longest headers; the dagger markers in Tables 1 and 2 sit at column boundaries.
- Page count.

## Target 6 — length

The body is now 8,142 words and the appendix 4,284, up about 500 words this round, all of it
disclosure the fourth audit asked for. Springer *Machine Learning* has no page limit.

**Is any of the added hedging now excessive** — that is, does the paper spend so long qualifying a
claim that a reader loses the claim? If so, name the passages. I would rather cut hedging than
cut evidence.

## Output

1. **Internal contradictions** — the specific two sentences that disagree.
2. **Anything still wrong about the CelebA group indexing**, including Appendix C.
3. **Hypothesis adjudication gaps**.
4. **Your answer on Target 4**, with a recommendation I can act on.
5. **Rendering and float problems**.
6. **The single thing most likely to be caught by a reviewer**, whatever kind of thing it is.

If you find nothing in a section, say so — a clean verdict is useful now, and I would rather hear
"this is submittable" than have a defect manufactured to fill the output.
