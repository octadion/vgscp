# Question for whoever built Table 5 (`tab:h1`): which population produced its robust ranges?

This is a verification question, not a bug report. I could not reproduce two columns and I would
rather ask than guess, because guessing wrong here means changing correct numbers.

## Context

Every other numeric table in the paper now recomputes exactly from `results/*.csv` along a code
path independent of whatever generated it: Tables 2, 3, 4, 6, 7 in the body, A1 and E10 in the
appendix, and the six generated appendix tables. Table 5 is the one exception, and only two of its
columns.

## What reproduces in Table 5

`$D_{\mathrm{ERM}}$` and `acc$_{\mathrm{ERM}}$` match to three decimals in all six settings where
ERM is not excluded, using:

```python
rows = [r for r in csv.DictReader(open("results/calibration_ablation_4bb.csv"))
        if r["score"] == "APS" and r["rho_test"] == "0.95"
        and r["calibration"] == "marginal_split" and r["method"] == "erm"]
D   = mean(float(r["div_wasserstein1"]) for r in rows)   # matches col 2
acc = mean(float(r["base_top1"])        for r in rows)   # matches col 3
```

So the score, the correlation strength, the calibration rule, the divergence metric and the
accuracy field are all confirmed correct.

## What does not reproduce

`$D$ (rob.)` and `acc (rob.)` — the min–max over the robust methods' means. Same filter as above,
`method in {dfr, afr, balanced_subsample, groupdro_ll}`, `gate_status != "excluded"`, taking
`min` and `max` over the per-method means. **20 of 32 range endpoints match; 12 do not.**

| setting | column | printed | recomputed | diff |
|---|---|---|---|---|
| Waterbirds/ResNet-50 | D max | 0.110 | 0.1090 | 0.0010 |
| Waterbirds/CLIP | D min | 0.036 | 0.0336 | 0.0024 |
| CelebA/ResNet-50 | D max | 0.031 | 0.0320 | 0.0010 |
| CelebA/ResNet-50 | acc min | 0.810 | 0.8089 | 0.0011 |
| CelebA/ResNet-50 | acc max | 0.859 | 0.8618 | 0.0018 |
| CelebA/CLIP | acc min | 0.900 | 0.9021 | 0.0021 |
| CelebA/CLIP | acc max | 0.913 | 0.9123 | 0.0007 |
| CelebA/DINOv2 | D min | 0.014 | 0.0146 | 0.0006 |
| CelebA/DINOv2 | acc min | 0.830 | 0.8289 | 0.0011 |
| CelebA/DINOv2 | acc max | 0.876 | 0.8769 | 0.0009 |
| CelebA/ViT-B/16 | D min | 0.022 | 0.0204 | 0.0016 |
| CelebA/ViT-B/16 | acc min | 0.833 | 0.8337 | 0.0007 |

Every difference is at most 0.0025 and the direction is mixed, so it is not a rounding convention
I can invert.

## What I ruled out

- **Gate filter.** `kept` only, `!= "excluded"`, and no filter all score 20/32 or worse.
  `!= "excluded"` is confirmed correct by one cell: CelebA/ViT-B/16's printed `acc min` of 0.833 is
  DFR's mean, and DFR on CelebA is *flagged*, not kept — so flagged runs are in.
- **Averaging order.** Flat mean over rows and mean-of-per-seed-means both give 20/32.
- **Score and correlation strength.** Pooling scores or ρ values drops the match to 1/32 or 0/32,
  so the caption's "(APS, ρ_cal = 0.95)" is right. (For contrast, Table A1's two percentage columns
  only reproduce when pooled over *all* scores and *all* ρ — which is correct for that table and
  which I initially got wrong, so a per-table population difference is clearly normal here.)
- **Calibration rule.** Including Mondrian and shift-robust rows drops it to 10/32.

## What I could not find

The code path that emits these two columns. `study_robust_train/accuracy_matching.py` has
`raw_divergence()`, which returns exactly a mean divergence and mean base accuracy per method, but
its `_filter()` applies no `gate_status` filter and no calibration filter, so the population is set
by whoever calls it, and I could not identify the caller. Nothing under `tools/` generates
`tab:h1`.

## What I am asking

1. **Which script or notebook produced the `D (rob.)` and `acc (rob.)` columns**, and what
   population did it pass to `raw_divergence()` (or whatever it used)?
2. Is there a reason those two columns would legitimately come from a different set of runs than
   the ERM columns beside them — a different seed set, an earlier snapshot of
   `calibration_ablation_4bb.csv`, a per-split rather than per-run aggregation?
3. If the answer is "no reason, they are stale", the fix is to regenerate the twelve endpoints and
   nothing else. Can you produce the corrected values, or the script that does?

## Constraints on the answer

- **Do not redesign the table or the analysis.** The paper is in revision, due 16 September, and
  Table 5's role is to show that divergence and accuracy are confounded. Its argument rests on the
  `matched` column (`unm.` / GroupDRO-LL / `no ref.`), which is verified and not in question.
- **Do not change any other table.** The other ten are verified against the records; if your
  reconstruction disagrees with one of them, that is a signal your population is wrong, not theirs.
- **Do not add prose to the paper.** If a caveat seems needed, say so and let the author decide.
- If the twelve differences are within the noise of a legitimate re-aggregation and you would leave
  the table alone, say that plainly — that is a perfectly good answer and it is the one I expect.
