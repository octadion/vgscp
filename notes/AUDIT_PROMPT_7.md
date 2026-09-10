# Seventh audit: verifying a length reduction

I will attach the compiled manuscript (`paper.pdf`), the response letter
(`response-to-reviewers.pdf`) and the before/after summary (`before-after.pdf`).

The science was settled at the sixth audit, which found no numeric errors. Since then the paper was
cut from 42 pages to 36 — four appendix tables removed, roughly 700 words of prose removed, and
float placement changed. **Nothing about the analysis was meant to change.** What I need to know is
whether the cutting broke anything.

**Read this as a verification pass, not a hunt.** Five audits have already run. Each one added
roughly 450 words to the appendix, because a finding phrased as "the paper should also state X"
always looks safer than no finding. That is the failure mode I am asking you to avoid. If a section
is clean, write "no findings" and move on. If the paper is submittable, say so plainly.

---

## Hard constraints on your findings

These are not preferences. A finding that violates one of them is not usable.

1. **No finding may make the paper longer.** The paper is over its target length. If something is
   wrong, the fix is to correct or delete the sentence that is wrong — never to append a
   clarification, a caveat, a disclosure sentence, or a "we note that". If your only available fix
   adds words, say so explicitly and let me decide.
2. **Every finding needs evidence I can check.** Give the page or section, quote the text, and for a
   numeric claim show the recomputation. "This seems inconsistent" is not a finding.
3. **No style, voice or preference findings.** Not word choice, not paragraph order, not "would read
   better as". One exception, below, and it runs the other way.
4. **No findings that ask for new experiments, new baselines, new datasets or new related work.** The
   revision is due 16 September and the compute is not available.
5. **Distinguish "wrong" from "differently arguable".** Only the first is a finding. If the paper
   makes a defensible choice you would have made differently, that is not a defect.

## What is verified mechanically — do not re-derive these

Recomputed from the released records along a code path independent of the generator that wrote the
tables, plus structural checks. All currently pass:

- **Six appendix tables cell by cell**: the cross-score spread, the divergences, the fine-tuning
  study including its across-seed standard deviations, the predicted-group table, the variance
  decomposition, and per-group coverage including its set-size disparity column.
- **One prose bound**: the claim that no setting departs from its ρ=0.95 worst-group coverage by
  more than 0.019 across the sweep.
- **Structure**: balanced braces and math in all files, no dangling reference, no citation without a
  bibliography entry and none unused, every table's declared columns matching its rows, no column
  blank in every data row, every table measured against the 372pt text block.
- **Cross-document consistency**: superseded values asserted absent from all four source files, the
  title asserted identical in three, the appendix table numbers derived from the manuscript rather
  than listed by hand, and the before/after table's source-line and equation counts derived from the
  manuscript.
- **All 19 reviewer points** still test as addressed, and the placement-sensitive ones test as
  present in the body rather than exiled to an appendix.
- **Quotation fidelity**: six of eight blue-ruled blocks in the letter are verbatim; three differ
  only by an expanded citation and dropped cross-references, which the letter states as its
  convention.

## Target 1 — the tables nothing has ever machine-verified

This is the most valuable thing you can do and the reason I am asking for a seventh pass.

The mechanical audit covers the appendix. **It has never covered the body tables.** Tables 1–7 were
checked by hand only, and Table 2 carries the paper's central claim. Also unverified: the
worst-group-identity table in Appendix A, the excluded-baseline table in Appendix E, and the split
sizes in Appendix F.

*Check*: recompute what you can of Tables 2, 3, 4, 5, 6 and 7 from the released records and tell me
about any cell that does not match. Say which cells you could not recompute and why. A mismatch here
is a real finding; report it however small.

Two things will otherwise look like errors and are not:

- **Table 6 comes from the five-seed main grid**, not the three-seed calibration comparison behind
  Appendix B, so it is not expected to reproduce from Appendix B's tables. An earlier audit called
  this an error on exactly this misunderstanding.
- **Lifts and spreads are computed before rounding**, so they can differ by 0.001 from the
  difference of the printed columns.

## Target 2 — did the cutting remove something load-bearing

Four appendix tables were removed: the two per-method calibration grids, the correlation-strength
sweep, and a mean-coverage-and-disparity table. Two tables lost their RAPS rows and one lost two
Wasserstein columns. Roughly 700 words of prose went, most of it appendix passages that restated
what Sections 3 and 4 already said.

*Check*: does any surviving sentence now cite evidence that is no longer in the paper? Does any
claim now rest on numbers the reader cannot see? Is there a cross-reference to a table that no
longer exists, or a description of the appendix's contents that overstates what it now holds?

The reasoning behind the removals was: a table earns space if a claim requires reading its numbers,
and belongs in the released records otherwise. Tell me if you think a specific removal crossed that
line — but note that the released records contain every removed cell.

## Target 3 — the register constraint, and it runs toward deletion

The paper must not read like a machine-written report. Specifically banned: category labels on
claims (`Finding:`, `Mechanism:`, `Methodological note:`), internal artefacts (file names, script
names, reviewer codes, dated engineering notes), index tables whose column is "where reported", and
self-narrating phrases ("we say so rather than", "it is worth being clear why", "we note this
rather than").

*Check*: flag any surviving instance in the manuscript or the two companion documents. This is the
one place I want style findings, and the fix is always deletion.

## Explicitly out of scope

Do not report on: the choice of two datasets; the choice of four backbones; the use of last-layer
methods rather than full training; the absence of a theoretical convergence result; the page count;
the figure aesthetics; the bibliography style; the decision to report a minimum over groups rather
than a mean; the phrasing of the title. All were settled in earlier rounds.

## Facts that are correct and have tripped up earlier readers

Do not report these as defects:

- **CelebA's worst group is not a minority group.** Compositing to ρ = 0.95 gives the two
  label-attribute-aligned cells 47.5% each, so the group that comes out worst holds 47.5% of the
  calibration set. The (1−π) reading of the shortfall identity describes Waterbirds, not CelebA, and
  the paper says so.
- **The compositing draws with replacement.** CelebA's g3 reaches a nominal 7,095 calibration rows
  from 140 distinct images. Nominal counts are not sample sizes.
- **AFR does not reproduce bit-exactly** — 36 of 480 runs, at most 0.013. It is the only two-stage
  fit, and this is disclosed in Appendix F.
- **Mean worst-group set size is below one on DINOv2/Waterbirds.** Empty sets are a legitimate
  outcome of split conformal with an accurate model; the paper discloses this rather than clipping.
- **One pre-specified criterion fails in seven of eight settings.** This is reported, not hidden.
- **Appendix F's split construction, hyperparameters and seed sets are quoted almost verbatim in the
  response letter.** Do not propose rewording them.

## Output I want

For each of the three targets: either "no findings", or a numbered list. For each finding give the
location, the quoted text, why it is wrong, and a fix that does not add words. Sort by severity and
tell me which findings you consider submission-blocking — I expect that list to be empty, and if it
is, say so.

Close with one sentence: can this be submitted as it stands?
