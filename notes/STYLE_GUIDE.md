# Writing voice for the ACML revision

Read this before rewriting any prose in `sn-article.tex`. It is a description of a taste, taken from
reading an accepted paper closely, not a set of word-count targets.

**Reference paper.** Kivimäki et al., *Performance Estimation in Binary Classification Using
Calibrated Confidence*, Machine Learning 115:67 (ACML 2025 journal track, 22 pages;
`acml-journal-2025-acceptedpapers/s10994-025-06970-3.pdf`). Closest to our topic: calibration,
uncertainty, binary classification, a theorem plus experiments, proofs in an appendix. Secondary:
Seyedsalehi et al., *Gender Bias Mitigation in Dense Neural Rankers*, 115:93 (34 pages, protected
attribute, long appendix) — used for how limitations and scope are stated.

---

## The taste in one paragraph

The reference paper **explains**; ours **argues**. It walks the reader through the work as if
across a table: here is the problem, here is what we did, here is what we saw, here is what it
means. It trusts the reader. When something is weak it says so once, plainly, and moves on. Our
draft defends every sentence as it goes — each claim arrives already wrapped in its qualification,
its contrast with what it is not, its number, and its pointer to the appendix. Every one of those
is honest. Together they make the paper read like an audit report. The fix is not to remove the
honesty but to give each piece its own sentence, or its own place.

## Principles

**1. One idea per sentence. A qualification is a sentence of its own.**
The reference never nests a caveat inside a claim. It states the claim, then the caveat.

> "Although CBPE comes with relatively strong theoretical guarantees, it also has its limitations.
> First and foremost, the theoretical guarantees hinge on the assumption of perfect calibration,
> which is not achievable in any real-life scenario. However, our experiments show that in most
> cases, CBPE yields consistent estimates…"

Connectives are the plain ones: *However, Thus, Similarly, Also, Finally, In contrast,
Furthermore.* No em-dash asides, almost no semicolons.

**2. Tell the reader where you are going.**
Sections open with a short roadmap, and the text uses a guiding "we".

> "In this section, we present the necessary background information. We begin with calibration,
> then briefly discuss different types of distributional shifts."
>
> "Let us start with the positive predictions." · "We will now show how to apply a shortcut…"

**3. After a formal statement, say what it means — once, in words.**

> "In effect, this definition means that when a model is calibrated, any of its confidence scores
> can be interpreted as a probability…"
>
> "The significance of this theorem is that it guarantees that the point estimates … reflect how
> the model operates over the whole underlying target distribution."

**4. Report a result as: what is shown → the pattern → one anchoring number → what to do.**
The figure or table holds the numbers. The text names the pattern.

> "The results are shown in Figure 1. The figure shows that the approximation error decreases
> rapidly, with the mean absolute error already below 0.001 at window size 100. … Based on these
> findings, we recommend deriving the full distributions … for small monitoring window sizes…"

**5. Admit things plainly, including the unglamorous ones.**
No meta-commentary about the admission ("we note this rather than…", "it is worth being clear").
Just the fact, sometimes with a human word.

> "Unfortunately, the API is not well maintained, which made accessing most of the datasets
> somewhat difficult." · "…but interestingly less so for recall and F1."

**6. State a choice and its reason briefly. Do not litigate it.**

> "In our implementation, we set the value of the metric to be 0 in these cases for simplicity."

**7. Headings name topics, not findings.**
*Background · Confidence Calibration · Estimating the Confusion Matrix · Shortcuts for Point
Estimates · Quality of the Confidence Intervals.* Across all 27 accepted papers, 1 heading in 557
is a sentence. The finding belongs in the first sentence under the heading.

**8. The appendix holds proofs, details and extra results; the text points to it only to hand
those over.**
Checked across all 27 papers: 13 put appendices after the conclusion, 6 put the same kind of
material (proofs included) in a separate Supplementary Material file, and 8 have neither. Of the
13, four never mention their appendix in the body; the others do so between one and eight times.
Every reference hands over something concrete — a proof, dataset or configuration details, full or
additional results — usually in a short sentence of its own, often to a sub-appendix:

> "All the proofs of the theorems can be found in Appendix A." · "Details on these datasets are in
> Appendix B.3." · "See Appendix A.2 for the proof." · "…for which the proof is given in Appendix
> A." · "The Bayesian optimization procedure and its search space are detailed in Appendix B
> (Supplementary Material)."

None uses the appendix to carry the argument or the caveat of a claim made in the body. Appendix
titles name their content: *Appendix A: Proofs · Appendix B: Experimental Details · Appendix C:
Additional Ablation Studies.* The appendix itself opens with a two-line roadmap ("We start with the
proof of Theorem 1 … Then, we move on to prove Lemma 2").

**9. Captions describe what is plotted. The text interprets it.**

> "Fig. 2 The fraction of times the actual value of a metric was within the predicted confidence
> interval over 10,000 trials"

Column definitions can live in the text that introduces the table ("we report class imbalance …
as 'Bias', the Adaptive Expected Calibration Error … as 'ACE' … All values are given as
percentage units").

**10. Limitations are enumerated, then turned into future work.** (secondary reference)

> "This study is bounded by four interrelated limitations. First, … Second, … Third, … Fourth, …
> Future work can address these limitations in several ways."

Scope can be one plain sentence, even in the abstract: "The findings are empirical and scoped to
binary gender bias in English on the evaluated datasets and models, without claims of broader
generality."

**11. The conclusion is short and ends modestly.**
Around a hundred words: what was presented, what it enables, one forward-looking line. "We hope
our work lays the groundwork for future research on unsupervised performance estimation."

---

## Before and after — our own sentences

These show the voice, not final text. Every fact and every reviewer-required phrase survives.

**Section 5.1, the equivalence margin** (two em-dashes, a semicolon, "rather than", three numbers
in one sentence):

> *Before.* That margin was not pre-specified---we chose it after seeing the spreads, as a
> judgement about what coverage difference would be immaterial in deployment---and it is wide
> relative to the spreads themselves, so the test bounds the effect rather than resolving it;
> CLIP/Waterbirds comes closest to binding, at an upper bound of $0.042$, so the result would be
> seven of eight at a margin of $0.04$.
>
> *After.* That margin was not pre-specified. We chose it after seeing the spreads, as the largest
> coverage difference we would consider immaterial in deployment. It is also wide compared with the
> spreads themselves, so the test bounds the effect but does not resolve it. At a margin of $0.04$
> the result would be seven of eight, since CLIP/Waterbirds reaches an upper bound of $0.042$.

**Section 5.1, opening of the mechanism paragraph** (announces a distinction before making it):

> *Before.* How large that gain will be is, to a first approximation, visible before any
> re-calibration is done---though the two halves of that statement have different standing, and
> we separate them.
>
> *After.* The size of the gain can be anticipated before re-calibrating. Two results bear on this,
> and they differ in strength.

**Section 5.5, stability under the sweep** (result, caveat and failure mode in one run):

> *Before.* Shift-robust calibration inflates its level with the observed calibration-to-test
> distance by construction, so its coverage rises as $\rhotest$ falls---by up to $0.347$---which is
> intended behaviour under APS and RAPS rather than instability, while under THR on CelebA the same
> rule degenerates, returning the set of both labels and a coverage of $1.000$ that means nothing.
>
> *After.* Shift-robust calibration behaves differently by design. Its threshold grows with the
> calibration-to-test distance, so its coverage rises as $\rhotest$ falls, by up to $0.347$. Under
> THR on CelebA it degenerates instead: it returns both labels, and its coverage of $1.000$ says
> nothing about calibration.

**Limitations, opening** (one sentence carrying four limitations and a parenthetical):

> *Before.* Our conclusions are empirical and scoped to two vision benchmarks, four frozen backbones
> plus one fine-tuned end-to-end, and binary spurious attributes; a third dataset and a multi-class
> task would tell us whether the dissociation is specific to $|\mathcal{Y}|=2$, where a set can hold
> at most two labels and the efficiency axis is correspondingly compressed---bounded above by two,
> and, as Section~\ref{sec:c2} reports, not bounded below by one.
>
> *After.* This study has five main limitations. First, the evidence comes from two vision
> benchmarks with binary labels and binary spurious attributes. With two classes a set holds at most
> two labels, so the efficiency results may not carry over to multi-class tasks; a third dataset and
> a multi-class task would test this. Second, …

**Headings.**

| before | after |
|---|---|
| What Training Buys Instead: Smaller Sets | Set Size under Group-Conditional Calibration |
| The Apparent Transfer Cannot Be Separated from Accuracy | Divergence at Matched Accuracy |
| *What the seed does not move.* | *Deterministic solvers.* |
| *Which runs clear the floor.* | *Exclusions.* |
| *When inferring the attribute is not acceptable.* | *Ethical scope.* |
| *What is true by construction, and what is not.* | *A dissociation result.* |

The Section 5.1 heading, "The Calibration Rule Moves Coverage More Than Training Does", is quoted in
the response letter as evidence that the claims were softened (R1.2). Renaming it means updating
the letter and before-after together, or leaving it as the one exception.

**Appendix references.**

> *Before.* …is quantified in Section~\ref{sec:h3} and Appendix~\ref{app:subtarget}, where the
> unweighted mean over groups sits at the nominal $0.90$.
>
> *After.* …and the unweighted mean over groups sits at the nominal $0.90$ (Appendix C).

Keep a reference where something is deferred (a proof, a full table, a procedure). Drop it where it
only says "there is more about this elsewhere".

**Captions.** Keep what is plotted and the notation a reader needs to read it. Move the argument
to the paragraph that cites the figure. Check first: one caption once held the only explanation of
a surprising number, and cutting it had to be undone.

---

## What does not change

- Every reviewer point stays answered, and in the same place (`tools/reviewer_safety.py`).
- Phrases other checks require stay, reworded around if needed (`tools/contradict.py`: e.g. "That
  margin was not pre-specified").
- Eight passages are quoted verbatim in the response letter. Rewriting one means rewriting its
  quotation too (`tools/check_quotes.py`).
- The register bans in `HANDOVER.md` still hold: no category labels, no filenames, no reviewer codes
  in the manuscript.

## How to tell you have drifted back

Symptoms, not targets: em-dashes in running prose; a semicolon joining a claim to its caveat; "rather
than", "which is why", "we therefore" in consecutive paragraphs; a heading that is a sentence; three
numbers in one sentence; a caption longer than the paragraph that cites it; "(Appendix X)" after a
sentence that defers nothing.
