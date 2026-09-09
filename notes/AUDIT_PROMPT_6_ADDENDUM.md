# Addendum — two things the prompt gets wrong about the PDF you have

Send this as a follow-up message in the same session, before reading the reply.

I sent the prompt before finishing two changes, so two of its statements are stale. The PDF is
correct; the prompt is not. Please treat the PDF as ground truth and read these two items as part
of Target 2 ("changes nobody has read"), because nobody has.

## 1. The CelebA unique support is measured, not estimated

The prompt asks: *"I could not compute g3's exact unique support here — the raw attribute file is
not available locally — so the text states the mechanism and the natural share the paper already
reports rather than inventing a count. Is that acceptable as written, or does it need the exact
number before submission?"*

**Disregard that question.** It has been measured, on the released splits, averaged over the ten
calibration splits:

| | nominal rows | distinct images | ratio |
|---|---|---|---|
| CelebA g3 (blond, male) | 7,095 | **140** | 50.8× |
| CelebA g0 | 7,095 | 4,426 | 1.6× |
| CelebA g1 / g2 | 373 | 361 / 340 | ~1.0× |
| Waterbirds g3 | 1,245 | **294** | 4.2× |
| Waterbirds g0 | 1,245 | 726 | 1.7× |
| Waterbirds g1 / g2 | 66 | 64 / 59 | ~1.0× |

The calibration half contains only 140 g3 images on CelebA, so every one is used and each appears
about fifty times. Table F13 gained a "distinct images" column and Appendices C and F now carry
these figures.

**Worth checking:** Waterbirds turns out to have the same structure in mild form (4.2× on g3), which
no audit had raised and which the paper had implicitly treated as the clean case. Is it now
described accurately, and is the sharpened claim — that on CelebA the least precisely estimated
threshold belongs to the group the composited counts make look best supported — correct as written?

## 2. Appendix B no longer reports its own absence — it has the table

The prompt describes Appendix B as stating that a per-group coverage table "would require
re-running the evaluation". That re-run has happened. **Table B10 is new and no audit has seen it.**

The evaluation was re-run at the headline configuration on the released splits; the minimum over
the four per-group coverages reproduces the published worst-group coverage on all 2,400 runs, so
these are the same runs at finer grain. What it shows:

- Under per-group thresholds every group sits at **0.886–0.917** — individually near target.
- Per-setting mean over groups **0.8995–0.9056**, averaging **0.9024** — the exact figure §5.1
  already quoted, now measured rather than simulated.
- Mean minus minimum **0.018–0.045** — again the range already stated.
- Half the group-cells fall below 0.90 and half above.
- Under a shared threshold: Waterbirds g2 at 0.735–0.866 against g0 at 0.908–0.931. On CelebA
  the minimum falls on one of the two aligned 47.5% groups in all four settings — g3 under
  ResNet-50 and CLIP, g0 under DINOv2 and ViT-B/16 — while g2, a nominal minority at 2.5%, is the
  *best*-covered group in every one of them, reaching 0.953.

  I had first written "on CelebA the lowest group is g3", which is true in only two of the four
  settings; I caught it against the table and corrected it. Worth confirming I have not made the
  same kind of error elsewhere in that paragraph.

**Worth checking:** does the new text overclaim? The measured numbers agree with what the paper had
already asserted, which is either genuine confirmation or a sign I fitted the description to the
table. Please read it adversarially. Also: does Table B10 make Appendix C's simulation redundant,
or do the two now say the same thing twice?

## 3. One reproducibility fact, also new

Re-running the evaluation reproduced the released records **exactly** for ERM, DFR, GroupDRO-LL and
balanced subsampling, and **not** for AFR: 36 of its 480 runs differ, by at most 0.013, on
ResNet-50 and CLIP over Waterbirds. AFR is the only two-stage fit here — a first head's converged
probabilities become the second's sample weights — so a last-bit difference is amplified. ERM
shares AFR's features and solver and is exact, so the features are reproducible and the effect is
specific to that second stage. A paragraph in Appendix F says this. No reported number changes.

**Worth checking:** is that disclosure proportionate, or does it invite more doubt than the fact
warrants? And is AFR load-bearing anywhere it should not be — §5.3 now names AFR as the method
producing the largest sets on ResNet-50/Waterbirds.

## Everything else in the prompt stands

Targets 1, 3, 4, 5 and 6 are unaffected. The verdict I want is still the one in Target 4: is this
submittable, and if not, the shortest list that makes it so.
