# Kickoff for the next session

Paste the block between the rules as your first message. Nothing else is needed.

---

```
Read HANDOVER.md first, then tools/README.md. Both are in this repo and written for you.

This is a Springer journal revision due 16 September. The science is finished and verified:
all 19 reviewer points addressed, every appendix number re-derived from results/*.csv, no
internal contradictions. What is left is length and voice.

It compiles to 42 pages. Target ~30. The body ends around page 22, so the appendix is
roughly 20 pages and that is where nearly all of the reduction has to come from.

One thing already tried, so you do not repeat it: the previous session removed five
\FloatBarrier calls and loosened \topfraction / \textfraction / \floatpagefraction. That
recovered one page. So the white space is NOT mainly a float-parameter problem, and there
is no more to win from tuning them.

Your job, in this order.

1. Fix appendix table placement properly. Table B8 still sits in white space and it is not
   alone. Appendix tables are read in sequence and referenced by number -- they have no
   reason to float at all. Try \usepackage{float} and [H] placement for the appendix tables
   so they appear exactly where written, and check what that does to the page count before
   going further. Note the three grid tables are longtable, which does not float, so they
   need different handling.

2. Cut the appendix from 15 tables to about 8, and from 5,151 words to about 2,500. This is
   the bulk of the 12 pages. HANDOVER.md names candidates and gives the rule: a table earns
   body space if a body claim requires reading its numbers, appendix space if a claim cites
   it, and otherwise belongs only in the released records. Several tables exist because one
   review round asked for them, not because a reviewer or a reader needs them.

   Do not touch Appendix F's split construction, hyperparameters or seed sets. That is
   reviewer point R2.6 and it is quoted almost verbatim in the response letter.

3. Tighten the body prose. Section 5.5 is 1,296 words and Section 3 is 1,336; both can lose
   a third. R2.5's five sub-answers live in 5.5 and all five must survive.

4. Trim the captions from 1,508 words. A caption says what the table shows and defines its
   notation. Argument belongs in the text, where it usually already is -- but check before
   deleting: one earlier trim removed the only explanation of a surprising number and had to
   be put back.

Three rules I need you to hold.

Fix by correcting, not by adding. The appendix grew 35% across four review rounds because
answering a criticism by adding a disclosure sentence always felt safer than rewriting the
sentence that was wrong. Run `python tools/sizes.py` before and after every batch and tell
me both numbers. If a batch ends longer than it started, undo it.

Do not let the paper read like a report. No category labels on claims ("Finding:",
"Mechanism:", "Methodological note:"), no filenames, no reviewer codes in the manuscript, no
dated engineering notes. HANDOVER.md lists what was purged. This constraint has been broken
twice, both times by taking an AI reviewer's structural advice over it.

Verify after every batch, not at the end:

    cd "ACML_Journal___Robust_CP_Train_Study (1)" && python ../tools/check_tex_final.py
    cd tools && for f in audit_appendix contradict reviewer_safety crossdoc tabwidth \
                         blankcol check_quotes; do printf "%-18s " $f; python $f.py | tail -1; done

reviewer_safety.py fails if a reviewer point stops being addressed; contradict.py fails if
one paragraph contradicts another. Both have caught real damage. If either fails, the batch
is wrong.

Two things that will bite when you remove appendix tables. Removing one shifts every later
table number, and both companion documents cite them by number (currently A1, B2-B11, C12,
D13, E14, F15). And response-to-reviewers.tex quotes the manuscript verbatim in eight
places, so changing a quoted passage makes the quote stale -- tools/check_quotes.py scores
them.

appendix-tables.tex is generated, not hand-written:
    cd tools && python gen_appendix_tables.py && python fix_grid_tables.py

I cannot compile. Tell me when you want a page count and I will run it and report back.
Work in batches. Do not ask me to commission another external audit -- five have run, the
last found no numeric errors, and each added roughly 450 words to the appendix.
```

---

## Notes for you, not for the prompt

**Where the 12 pages have to come from.** Body ends around page 22, appendix runs 22–42. Trimming
the body helps the reader but barely moves the count. The appendix is the target: 15 tables and
5,151 words in roughly 20 pages. Halving it lands you near 30.

**Why the float fix only gave one page.** Loosening the parameters lets LaTeX *fit* more per page,
but it will still defer a float it cannot place and flush it later. In an appendix that is mostly
tables, the fix is to stop them floating at all — `[H]` places a table exactly where it is written.
That is the first thing to try, and it costs nothing.

**Give it the page count when it asks.** Two or three compiles across the session is normal; the cut
decisions depend on real numbers.

**Watch the word count each batch.** It should fall every time. That single number is the best
signal that the session is doing what you asked rather than drifting.

**Two checkers are the safety net:** `reviewer_safety.py` and `contradict.py`. If a session reports a
batch done without running them, ask.

**Then one compile and your proofread.** Not another audit.
