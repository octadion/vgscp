# Third audit: do the numbers hold, and do the three documents agree?

I will attach the compiled manuscript (`paper.pdf`), the response letter
(`response-to-reviewers.pdf`) and the before/after summary (`before-after.pdf`).

The first two audits are closed. Audit 1 covered writing and structure; audit 2 covered reviewer
coverage and whether material had been moved somewhere it should not be. Both sets of findings are
implemented. **Do not re-audit those axes.** If you find yourself commenting on prose style or on
whether a reviewer point is addressed, you are in the wrong audit — unless a *number* or an
*internal contradiction* is what leads you there.

This audit has three targets, in order of how much I expect them to yield.

---

## Target 1 — numerical integrity (highest priority)

Both real defects the previous audit found were of one kind: a number in the body disagreed with
the same number in the appendix, because the script that generated the appendix table filtered the
records differently from the script that generated the body table. Neither was visible from
reading; both were visible from comparing.

So: **cross-check every number that appears more than once.** Concretely,

- A value quoted in the abstract, introduction, discussion or conclusion against the table it comes
  from. Ranges are the common failure: a band like "$0.84$--$0.89$" must contain every value it
  claims to span, including the smallest and largest cells in the relevant table.
- A value in a section's prose against the table in that same section.
- A value in a body table against the corresponding appendix table. Appendix B holds the full grid;
  the body tables are filtered views of it. Where the body holds a variable fixed (a single score,
  a single calibration rule, a single head) the appendix row should reproduce the body number
  exactly, not approximately.
- Counts that are stated in several places: how many runs, how many excluded, how many settings,
  how many methods, how many seeds, how many records.
- A number in a figure caption against the same number in the text.
- Percentages and counts that should sum or reconcile (per-group counts against the split total;
  excluded plus retained against the total).

For each mismatch, tell me **both values, where each appears, and which one you believe**, if you
can tell. If you cannot tell which is right, say so — I can check against the records.

Also flag any number that is stated with more precision than it can support, and any place where a
"range across X" is quoted without X being identifiable.

## Target 2 — the claims that changed in this round, which no one has checked

I made several substantive changes recently on my own analysis, not at a reviewer's request. They
have not been reviewed by anyone. **Be adversarial about these specifically**, and assume I may
have talked myself into something.

1. **Appendix A, "Which group is worst" (Table 8).** An earlier draft asserted that the stochastic
   ordering needed for a divergence-to-coverage argument "holds here by construction". I checked it
   against the records, found it false, withdrew it, and replaced it with a measured claim: the
   worst group is the minority group in 85–100% of ERM runs under a shared threshold, but only
   15–38% under per-group thresholds, near the 25% a four-way tie would give.
   - Is the new claim actually established by what is shown, or does the table show something
     narrower than the text says?
   - The paper argues nothing needs a fixed worst group, because the identity and the KS bound are
     applied per run to whichever group is worst in that run. **Is that true throughout?** Check
     Section 5.1 and the mechanism figure in particular: an $r=0.93$ correlation computed across
     eight settings, each using its own worst group, may or may not be free of the assumption.
   - Does withdrawing this weaken any other claim that quietly relied on it?

2. **Table 2's equivalence test.** I report that the across-method spread in worst-group coverage
   under per-group thresholds is equivalent to zero within a $0.05$ margin in 8/8 settings.
   - Is a $0.05$ margin defensible, or is it wide enough to make the test uninformative? The
     observed spreads are $0.003$–$0.024$, so the margin is 2–15× the effect.
   - Is "equivalent to zero" the right framing, or is it close to asserting a null?
   - Table 2's intervals come from three training seeds. Is a cluster bootstrap over three clusters
     honest, and is the paper sufficiently clear that it is three?

3. **The separated sensitivity analysis (Appendix E).** I now report that restoring the three
   excluded *seeds* leaves the maximum spread unchanged at $0.024$, separately from restoring the
   excluded *methods*, which moves it to $0.028$. Does the separation actually answer R2.4, or does
   it look like slicing the analysis until a number stops moving?

4. **The Mondrian band, now $0.84$–$0.89$.** I widened it from $0.85$–$0.89$ because a retained run
   sits at $0.841$. Check that $0.84$–$0.89$ is now correct everywhere it appears, and that no
   other range in the paper has the same defect.

## Target 3 — do the three documents agree with each other?

This has never been audited, and I found one live contradiction myself: the response letter was
quoting, as the paper's own words, a sentence the paper no longer contains — in fact a claim the
paper now explicitly denies. I fixed that one. **Assume there are others.**

- The letter sets off quotations from the manuscript with a blue rule. **Check each against the
  PDF.** They should be verbatim, except that citations are expanded to author-year and internal
  cross-references are dropped, which the letter states as its convention. Anything else that
  differs is a defect.
- The letter's "Where addressed" lines give section, table and figure numbers. **Follow each one
  into the paper.** Does the named location contain what the letter says it contains?
- The before/after document gives an inventory (counts of tables, figures, equations, sections,
  appendices, citations) and a table of what each appendix holds. Check those against the PDF.
- Do the three documents describe the same paper? Same title, same section names, same headline
  numbers, same claims withdrawn.

---

## What I also cannot check myself, if you can see it in the PDF

I have no LaTeX locally, so the following were verified analytically rather than by compiling, and
I would value a second look at the rendered document:

- **Float placement.** Does any table or figure land far from the text that discusses it, or after
  the section that owns it?
- **Table overflow.** I measured every table against the 372pt text block using Times metrics; the
  widest came out at 89.5%. Does anything actually run into the margin?
- **Page breaks in the three long appendix tables**, which are `longtable`s. Do their headers
  repeat correctly, and do they break sensibly?
- Any `??` or `[?]` from an unresolved reference or citation.
- Page count, and whether anything looks visually broken.

## What I do not need

Prose criticism, suggestions to restructure, or a verdict on length — all settled. And please do
not tell me a reviewer point is unaddressed unless a *number* or a *cross-document contradiction*
is your evidence; audit 2 covered coverage and I have its findings implemented.

## Output

1. **A table of numerical mismatches**: value A (where), value B (where), which is wrong if you can
   tell. Empty is a fine answer if you find none, but say how hard you looked.
2. **Your verdict on each of the four changed claims** in Target 2 — sound, overstated, or wrong —
   with the specific sentence at fault.
3. **A list of cross-document contradictions**, quotation drift included.
4. **Rendering defects** visible in the PDF.
5. **The single thing most likely to be caught by a reviewer who checks arithmetic.**
