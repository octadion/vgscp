# Handover — ACML revision, what is done and what is left

Written 2026-09-09 at the end of a long session, for whoever picks this up next. Revision is due
**16 September 2026**. Submission `62d20e3d-c3eb-4568-b964-7b09cf6f583b`, ACML 2026 collection,
Springer *Machine Learning*.

Read the two "How not to" sections before touching the paper. They are the whole point of this
document.

---

## Where things stand

The science is finished and checked. What is left is length and voice.

| | now | target |
|---|---|---|
| pages | 43 (last compile, before the float fix) | **~30** |
| body | 8,164 words | ~6,500 |
| appendix | 5,151 words | ~3,500 |
| captions | 1,508 words | ~1,000 |
| tables | 22 (7 body, 15 appendix) | ~15 |
| figures | 7 | 6? |

**All 19 reviewer points are addressed** and `tools/reviewer_safety.py` verifies it. Every number in
the appendix tables has been re-derived from `results/*.csv` along an independent path. There are no
known internal contradictions. Two commits back is a clean state if anything goes wrong.

---

## How not to make it worse — the register

The author's first instruction, repeated at least four times, was that the paper must not read like
an AI report. It has drifted that way twice, both times because a suggestion from an AI reviewer was
taken over the author's standing constraint. **Do not do this.**

Banned outright:

- Category labels on claims: `\textbf{Finding:}`, `\textbf{Mechanism:}`, `\textbf{Methodological
  note:}`. Four of these were added and then removed. A sentence should carry its own weight; if a
  claim is a check rather than a result, the sentence says so in words.
- Internal artefacts: `.csv` filenames, script names, reviewer codes (`R2.4`) anywhere in the
  manuscript, dated engineering notes ("fixed in the codebase on 11 June").
- Index tables whose column is "where reported". A table of hypothesis *verdicts with numbers* is
  fine; a table of pointers is a project artefact.
- Unusual or high-register vocabulary. Earlier drafts used *arm*, *cell*, *gate*, *verdict*,
  *burden*, *pool* as jargon; these were purged. A **run** is one (backbone, dataset, method, seed)
  tuple; a **setting** is one backbone--dataset pair.

`tools/contradict.py` and the sweep in `notes/PROGRESS.md` record what has already been cleaned.

## How not to make it worse — the length

The appendix grew 35% across four review rounds. The mechanism was always the same: a criticism
arrives, and adding a disclosure sentence feels safer than rewriting the one that was wrong.

**Rule: fix by correcting, not by adding.** New text is justified only when a reviewer asked for it.
If a round of edits ends with more words than it started, something went wrong.

Measure with `python tools/sizes.py` before and after.

---

## What was just changed, and why the page count should already be lower

Not yet recompiled, so the saving is unverified:

- **Five `\FloatBarrier` calls removed.** They were added so tables would stay in the subsection
  that cites them. That is stricter than any journal asks, and it backfired: a barrier flushes
  pending floats, and one that cannot fit goes to a page of its own. Table 6 taking a whole page was
  this.
- **Float parameters loosened** (`\topfraction` 0.9, `\textfraction` 0.07, `\floatpagefraction`
  0.75, `totalnumber` 5). The class defaults assume few floats; with 22 tables they exile large ones
  to float-only pages. This is typesetting, not content — expect 3–5 pages back for nothing.
- **`placeins[section]` kept.** Keeping an appendix's tables inside their own appendix is worth
  enforcing; keeping a table inside its subsection is not.

**Compile first.** Decide what to cut against the real page count, not the one inflated by white
space.

---

## What to cut, in order

Measured, not guessed. `python tools/sizes.py` and `tools/optimise_targets.py` produce these.

1. **Appendix tables, 15 → ~9.** Several exist only because one review round asked for them. The
   rule to apply: a table earns *body* space if a body claim requires reading its numbers, *appendix*
   space if a claim cites it, and otherwise belongs only in the released records. Candidates:
   `tab:pgfull` (24 rows, three conditions × three scores — could be one score), `tab:reprfull`
   (18 rows, same), `tab:div` (8 rows, W1 columns enter no argument).
2. **Body prose.** §5.5 is 1,296 words and bundles three topics; §3 is 1,336. Both can lose a third
   without losing content. R2.5 lives in §5.5 with five sub-answers — all five must survive.
3. **Captions, 1,508 words.** A caption says what the table shows and defines its notation. Anything
   arguing why it matters belongs in the text, where it usually already is. Check before deleting:
   one trim removed the only explanation of why the smallest simulated pool over-covers, and it had
   to go back.
4. **Figures.** Seven. Figure 2's right panel was demoted to "an association" — ask whether it still
   earns half a figure.

Do not touch Appendix F's split construction, hyperparameters or seed sets: that is R2.6, quoted
almost verbatim in the response letter.

---

## The companions

`response-to-reviewers.tex` and `before-after.tex` must track the manuscript. Three things bite:

- **Eight blue-ruled blocks quote the paper verbatim.** Change a quoted passage and the block goes
  stale. `tools/check_quotes.py` scores each; blocks 3, 5 and 6 sit below 1.00 by design (a citation
  expanded to author-year, cross-references dropped), and the letter states that convention.
- **Table numbers.** LaTeX numbers appendix tables sequentially with the section letter: currently
  A1, B2–B11, C12, D13, E14, F15. Add or remove an appendix table and every later label shifts. The
  companions cite these by number.
- **Corrections reached the manuscript and not the letter for three rounds running**, because every
  check was per-document. `tools/crossdoc.py` now asserts superseded values are absent from all four
  files.

---

## Tools

`tools/README.md` explains each. Run them all after any edit:

```sh
cd "ACML_Journal___Robust_CP_Train_Study (1)" && python ../tools/check_tex_final.py
cd tools && for f in audit_appendix contradict reviewer_safety crossdoc tabwidth \
                     blankcol check_quotes; do printf "%-18s " $f; python $f.py | tail -1; done
```

`appendix-tables.tex` is generated:

```sh
cd tools && python gen_appendix_tables.py && python fix_grid_tables.py
```

**No LaTeX locally.** Table widths are checked analytically against the 372pt text block
(`tabwidth.py`); float placement and page count need the author to compile.

**No datasets locally** either — torch is broken and the data is Colab-only. `results/*.csv` holds
every number the paper reports. `notebooks/per_group_coverage.ipynb` is the pattern to follow if a
re-run is ever needed; it copies its setup cells verbatim from the notebook that produced the
released records, because hand-transcribing them silently breaks reproduction.

---

## Things a reader might trip over

Recorded because each cost a round to find:

- **CelebA's worst group is not a minority group.** Compositing to ρ = 0.95 gives the two
  label-attribute-aligned cells 47.5% each, so the group that comes out worst holds 47.5% of the
  calibration set. The (1−π) reading of Eq. (3) describes Waterbirds and not CelebA, and the paper
  now says so. Do not "simplify" this back.
- **The compositing draws with replacement.** CelebA's g3 reaches a nominal 7,095 calibration rows
  from 140 distinct images. Nominal counts are not sample sizes.
- **AFR does not reproduce bit-exactly** — 36 of 480 runs, at most 0.013. It is the only two-stage
  fit. ERM shares its features and solver and is exact, so this is not a feature-cache problem.
- **Table 6 comes from the five-seed main grid**, not the three-seed calibration comparison behind
  Appendix B, so it is not expected to reproduce from Table B4. One audit called it an error on
  exactly this misunderstanding.
- **Lifts and spreads are computed before rounding**, so they can differ by 0.001 from the
  difference of the printed columns. Table 2's caption says so.

---

## Open

- Zenodo DOI — a visible placeholder, to be minted from a tagged release.
- One compile to confirm the float fix and the real page count.
- The author will proofread once the length is right.
