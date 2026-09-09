# Audit request: is this a research paper, or does it still read like a generated report?

You are auditing a manuscript under revision for the ACML 2026 collection of Springer's
*Machine Learning*. I will attach four PDFs:

1. **`paper.pdf`** — our manuscript, the thing being audited.
2. **`2010.14134.pdf`** — Jones et al., *Selective Classification Can Magnify Disparities Across
   Groups*, ICLR 2021.
3. **`2204.02937.pdf`** — Kirichenko et al., *Last Layer Re-Training is Sufficient for Robustness
   to Spurious Correlations*, ICLR 2023.
4. **`2410.01888.pdf`** — Cresswell et al., *Conformal Prediction Sets Can Cause Disparate
   Impact*, ICLR 2025.

Papers 2–4 are the standard I am trying to reach. They are close to my topic and they are written
the way I want mine written. **Treat them as the reference for style, structure and proportion, not
as papers to imitate sentence by sentence.** Where my paper differs from all three in some habit,
say so and name the habit.

## The core problem, stated plainly

The manuscript was revised with heavy AI assistance and it reads like it. It has the texture of a
*report*: it narrates procedure instead of telling the story of a piece of research, it defends
choices nobody has challenged yet, it recites numbers in prose that already sit in tables, and it
uses vocabulary from our own codebase in front of the reader. Several rounds of fixing have
removed some of this, and I no longer trust that the remainder has been found.

Some examples of what has already been caught and removed, to calibrate you on the *kind* of thing
I mean — do not report these back, find what is left:

- "The per-method worst-group accuracy floors were fixed in the codebase on 11 June 2026" — an
  engineering diary entry inside a research paper.
- "We report this rather than present zero variance as stability" — arguing with an imaginary
  critic in the middle of a result.
- Naming released `.csv` files in the body of the paper.
- Private pipeline words used as if they were English: *verdict*, *quality gate*, *gated out*,
  *arm* (for a training method), *cell* (for a backbone–dataset pair).
- Reviewer codes visible in the paper itself: "the sensitivity analysis R2.4 asks for".
- An abstract of 370 words that recited every result in the paper.

## What I want audited

Be specific and quote the text. A finding I cannot locate is not useful.

### 1. Voice and sentence-level writing
- Sentences that read as generated rather than written: over-hedging, stacked qualifiers,
  formulaic parallelism, "not only … but also", elaborate em-dash constructions, three-clause
  sentences that could be two short ones.
- Words used because they sound academic rather than because they are the ordinary word.
- Any place we narrate our own process ("we then computed", "this required no re-run", "we were
  careful to") instead of reporting a finding.
- Any place we defend a decision pre-emptively.
- Repetition: a word or construction leaned on far too often. (I already know *settings* appears
  about 47 times. Find the others.)
- Compare directly with how the three reference papers open a results paragraph, introduce a
  number, and state a limitation.

### 2. Structure and proportion
- **Main body vs appendix.** All three references have roughly 10–13 pages of main text and a
  large, substantive appendix. Mine is longer in the body and thin in the appendix. Say concretely
  which sections or subsections should move, and what should be promoted into the appendix to make
  it worth reading.
- Section order and whether any section earns its place.
- Whether the introduction does what an introduction in those three papers does.
- Whether the related work is positioned or merely listed.
- Whether the paper has a single organising idea that everything hangs from, the way Jones hangs
  everything on the margin distribution. If it does not, say what the candidate is.

### 3. The appendix specifically
In the reference papers the appendix carries proofs, derivations, ablations, extra figures and
tables — real content. Mine is largely prose that *describes* results held in released files.
Tell me what an appendix at that standard would contain given the material listed below, and what
is missing.

### 4. Figures, tables, notation
- Are six figures and nine tables the right number, and is each one earning its space?
- Do captions do their job without re-arguing the text?
- Table design: layout, width, whether any table is really an appendix table.
- Notation: is it introduced where needed, consistent, and no heavier than necessary?
- Is the formal content (one proposition, an identity and a bound) integrated the way Jones
  integrates his theory, or bolted on?

### 5. Title, abstract, framing
- Current title: **"Calibration Beats Robust Training for Worst-Group Coverage"**. Does it work? Is
  it accurate, and does it pull?
- The abstract is ~173 words. Does it follow the arc those three use — one belief, one pivot, two
  findings, one recommendation?

### 6. Citations
- Anything obviously missing for this literature.
- Anything cited in a list rather than used.
- Citation-to-claim mismatches you can spot.

### 7. Anything I have not thought to ask
Please add it.

## Hard constraint: the reviewers must stay addressed

This is a revision. Every point below must remain visibly addressed in the manuscript. **If any of
your recommendations would weaken or hide one of these, say so explicitly and propose an
alternative.** Where a point is currently addressed in the main body, moving it to an appendix is
acceptable *provided* it stays findable — tell me if you think a specific move goes too far.

Current locations:

| Point | What the reviewer asked | Where it is now |
|---|---|---|
| R1.1 | Target is 0.90 but empirical values are 0.84–0.89 — explain | §6.1 (end), §6.6, Appendix B, Figure 6, Table 9 |
| R1.2 | Backbones are frozen — justify the title claim or run a representation experiment | §6.2 + Table 4 + Figure 3 (new end-to-end fine-tuning study); title; abstract; §6.1 heading; §7; §9; §3 terminology |
| R1.3 | Only two backbones; compare with related work | §5 and every results table (now four backbones, eight backbone–dataset pairs); Table 2 and the paragraphs before §6 |
| R1.4 | More independent training seeds | §5, paragraph on seeds and the two deterministic methods; Appendix D |
| R1.5 | Add a specific reference on unfreezing strategy | §6.2, in the fine-tuning protocol, used substantively |
| R2.1 | Narrow the claim, or add a representation-changing experiment | §6.2; title, abstract, §7, §9 |
| R2.2 | W1/KS do not imply monotonic coverage reduction — state assumptions | §3, Proposition 1(ii) rewritten as an exact identity plus a KS bound, and the remark after it |
| R2.3 | Accuracy-matched analysis is weak; use hierarchical uncertainty | §6.4 rewritten at model level; Table 6; Figure 4; cluster bootstrap used throughout; Appendix D |
| R2.4 | Include excluded runs in a sensitivity analysis; preregistered or pre-specified? | §5 (floors, pre-specified not preregistered); Appendix C (sensitivity analysis + the AFR diagnosis) |
| R2.5 | Scope the ρ-sweep claim; explain predicted-group mechanics, supervision, probe error, ethics | §6.6 (scoping); §6.8 (mechanics, supervision, probe error); §6.8 final paragraph (ethics) |
| R2.6 | Present claims as tendencies; give full reproducibility detail | §6.3, §6.5, §6.6 (moderation, four claims withdrawn); Appendix D |
| R3.1 | Claim too strong; give CIs or equivalence tests | Title and §6.1 heading (comparative); cluster-bootstrap CIs in all main tables; two-one-sided-test equivalence check in §6.3; §6.1 and Appendix B for the sub-target level |
| R3.2 | "Training buys efficiency" is Mondrian-specific; give uncertainty on correlations | Abstract, §1, §6.3, §7 (conditional on Mondrian); Table 5 prints each interval, italicised where uninformative |

## The paper in one paragraph, so you can judge whether it lands

Under a spurious correlation, we ask whether making a model *accurate* on its worst group also
makes its conformal prediction sets *reliable* for that group. It does not. Worst-group coverage is
governed far more by the calibration rule — one threshold shared by all groups, versus one
threshold per group — than by how the model was trained, and this holds across four frozen
backbones, two benchmarks, five last-layer training methods and three conformity scores, and
survives fine-tuning the representation end-to-end. What training does buy is smaller prediction
sets, and only once calibration is group-conditional.

## Material available for the appendix

About 116,000 conformal evaluations: the main grid, a calibration comparison across three policies
(one shared threshold, per-group, and a shift-robust variant), a predicted-group study, and the
fine-tuning study. Per-method numbers exist for all three conformity scores, six correlation
strengths, three seeds or five depending on the study, and ten calibration splits, along with
per-group coverage means, set sizes, set-size disparity and cross-group divergences.

## Output I want

A numbered, prioritised list of concrete findings. For each one:

- **where** — section, or a quoted phrase I can search for
- **what is wrong** — in one sentence
- **why it reads as a report** — or which reference paper does it differently, and how
- **the fix** — concrete enough to act on; rewrite the sentence if it is a sentence
- **severity** — blocking / worth fixing / minor

End with:

1. The five changes that would most raise this toward the standard of the three reference papers.
2. An honest verdict: if you read this cold, would you think a person wrote it or a model did, and
   what specifically gives it away.
3. A proposed main-body/appendix split, section by section, with a target page count.

Do not soften. I would rather read a hard list than a polite one.
