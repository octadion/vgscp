# Kickoff for the next session

Paste the block below as the first message. Everything it needs is in the repo.

**Before you paste it: compile and note the page count.** The float fix in the last commit is
untested, and it may already have recovered 3–5 pages. Cutting content against an inflated number
would remove things that did not need removing.

---

```
Read HANDOVER.md first, then tools/README.md. Both are in this repo and written for you.

Short version: this is a journal revision due 16 September. The science is finished and
verified -- all 19 reviewer points are addressed, every appendix number is re-derived from
results/*.csv, no internal contradictions remain. What is left is length and voice.

The paper compiles to N pages. [REPLACE N WITH THE NUMBER YOU JUST MEASURED.] Target ~30.

Your job, in this order:

1. Cut the appendix from 15 tables to about 9. HANDOVER.md names the candidates and gives
   the rule for deciding. Do not touch Appendix F's split construction, hyperparameters or
   seed sets -- that is reviewer point R2.6.

2. Tighten the body prose. Section 5.5 is 1,296 words and Section 3 is 1,336; both can lose
   a third. R2.5's five sub-answers live in 5.5 and all five must survive.

3. Trim the captions from 1,508 words. A caption says what the table shows and defines its
   notation; argument belongs in the text.

Three hard rules.

Fix by correcting, not by adding. The appendix grew 35% across four review rounds because
adding a disclosure always felt safer than rewriting the sentence that was wrong. Run
`python tools/sizes.py` before and after every batch. If a batch ends longer than it
started, undo it.

Do not let the paper read like a report. No category labels on claims ("Finding:",
"Methodological note:"), no filenames, no reviewer codes in the manuscript, no dated
engineering notes. HANDOVER.md has the full list and the words that were purged. This
constraint has been broken twice by taking an AI reviewer's advice over it; do not be the
third.

Verify after every batch, not at the end:

    cd "ACML_Journal___Robust_CP_Train_Study (1)" && python ../tools/check_tex_final.py
    cd tools && for f in audit_appendix contradict reviewer_safety crossdoc tabwidth \
                         blankcol check_quotes; do printf "%-18s " $f; python $f.py | tail -1; done

reviewer_safety.py fails if a reviewer point stops being addressed. contradict.py fails if a
paragraph contradicts another. Both have caught real damage. If either fails, the batch is
wrong -- fix it before continuing.

Keep the two companion documents in step: response-to-reviewers.tex quotes the manuscript
verbatim in eight places, and removing an appendix table shifts every later table number,
which both companions cite.

I cannot compile here, so tell me when you want a page count and I will run it. Work in
batches and report the word count each time. Do not ask me to commission another external
audit -- that loop is what caused the bloat.
```

---

## Practical notes

**Same directory.** Start the new session in `c:\jagr\vgscp` so the tools resolve.

**Give it the page count.** That one number decides how much has to go. If the compile comes back
near 36–38, the cut list in HANDOVER.md is roughly right. If it is still 43, the float fix did not
take and that is the first thing to investigate — not more cutting.

**Expect it in batches.** Ask for a word count after each. The number should fall every time; if it
does not, something is being added again.

**Two checkers are the safety net.** `reviewer_safety.py` and `contradict.py`. Everything else is
useful; those two are what stop real damage. If a session tells you a batch is done but has not run
them, ask.

**One compile at the end**, then your proofread. Not another audit — you have had five, the last
found no numeric errors, and each one added roughly 450 words to the appendix.

## If you would rather not start over

Staying here also works. The downside is this conversation carries a lot of superseded decisions and
my own accumulated errors, and I will tend to follow the old track. The upside is nothing gets lost
in translation. Your call — the handover is written either way, and the tools are in the repo now
regardless.
