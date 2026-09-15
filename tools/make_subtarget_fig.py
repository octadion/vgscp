"""Redesign fig:subtarget so it is neither a second dumbbell nor a misleading x-axis.

Two earlier attempts were wrong in different ways.

  1. Dumbbell (mean over groups vs minimum, one pair per cell). Correct content, but the same
     visual idiom as fig:c1, which is what made the figure set look repetitive.
  2. Shortfall against ``n_cal_worst_group`` on a simulated curve. This one was *wrong*, not just
     repetitive. Within a dataset every cell has the same group construction -- Waterbirds is
     always [1245, 66, 66, 1245] -- so that column does not index a position on the curve. It only
     records which group happened to come out worst, flipping between 66 and 1245 for cells whose
     predicted shortfall is identical. Points would have been spread along an axis carrying no
     information about them.

The construction is fixed per dataset, so the exactly-valid law is a *distribution*, not a curve.
This version compares distributions on the coverage axis directly: what the mean over groups does,
what the minimum over groups does, and where an exactly valid Mondrian procedure with our own
calibration and test counts puts that minimum. If the observed minimum sits in the predicted band
while the mean sits on 0.90, the sub-target level is the statistic, not under-coverage.

The observed spread is wider than the predicted band by construction: the simulation varies only
the calibration and test draw, while the observed records also vary over training arms, backbones
and seeds. The comparison that matters is location, and the caption says so.
"""
import sys

sys.path.insert(0, r"c:\jagr\vgscp")
import numpy as np

from study_robust_train.calibration_ablation import records_from_csv
from study_robust_train.figstyle import (PALETTE, FigureStyle, apply_publication_style,
                                         create_subplots, finalize_figure)
from study_robust_train.stats import group_counts_at_rho, simulate_min_coverage

FIG = r"c:\jagr\vgscp\ACML_Journal___Robust_CP_Train_Study (1)\figures"
BLUE, RED, GREY = PALETTE["blue_main"], PALETTE["red_strong"], "#767676"
apply_publication_style(FigureStyle(font_size=14, axes_linewidth=1.8))
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

TARGET, ALPHA, RHO = 0.90, 0.1, 0.95
A = records_from_csv(r"c:\jagr\vgscp\results\calibration_ablation_4bb.csv")
KEEP = [r for r in A if r["gate_status"] != "excluded" and r["calibration"] == "mondrian"
        and r["score"] == "APS" and r["rho_test"] == RHO]

fig, axes = create_subplots(1, 2, figsize=(12.4, 4.3), sharey=False)
for ax, (ds, nice) in zip(axes, [("waterbirds", "Waterbirds"), ("celeba", "CelebA")]):
    sel = [r for r in KEEP if r["dataset"] == ds]
    means = np.array([float(r["mean_group_cov"]) for r in sel])
    mins = np.array([float(r["worst_group_cov"]) for r in sel])

    # The calibration and test pools are the same size here (a 50/50 split of the release), so the
    # realised eval count fixes both sides of the simulation.
    n_eval = int(float({r["n_eval"] for r in sel}.pop()))
    counts = [int(c) for c in group_counts_at_rho(n_eval, RHO, 4)]
    sim = simulate_min_coverage(counts, counts, k_groups=4, alpha=ALPHA, policy="mondrian",
                                n_draws=3000, seed=0)   # matches Appendix C's stated draw count

    ax.axvspan(sim["min_lo"], sim["min_hi"], color=BLUE, alpha=0.10, lw=0, zorder=0)
    ax.axvline(sim["expected_min"], color=BLUE, lw=2.4, ls="-", zorder=4)
    ax.axvline(TARGET, color="k", lw=2.0, ls="--", zorder=4)

    bins = np.linspace(min(mins.min(), sim["min_lo"]) - 0.01, max(means.max(), TARGET) + 0.01, 46)
    ax.hist(means, bins=bins, color=GREY, alpha=0.55, zorder=2)
    ax.hist(mins, bins=bins, color=RED, alpha=0.70, zorder=3)

    # Two lines: on one line the two panel titles ran into each other across the gutter.
    ax.set_title(f"{nice}\ncalibration counts {counts[1]} / {counts[0]} per group",
                 fontsize=12.5, fontweight="bold", pad=8, linespacing=1.25)
    ax.set_xlabel("coverage")
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    print(f"{nice:11s} n={len(sel):4d}  mean-over-groups {means.mean():.4f}  "
          f"min-over-groups {mins.mean():.4f}  diprediksi {sim['expected_min']:.4f} "
          f"[{sim['min_lo']:.4f}, {sim['min_hi']:.4f}]  "
          f"selisih {mins.mean() - sim['expected_min']:+.4f}  "
          f"di dalam pita: {100 * np.mean((mins >= sim['min_lo']) & (mins <= sim['min_hi'])):.0f}%")

axes[0].set_ylabel("runs")
axes[0].legend(handles=[
    Patch(facecolor=GREY, alpha=0.55, label="observed mean over groups"),
    Patch(facecolor=RED, alpha=0.70, label=r"observed $\min_g$ (reported)"),
    Line2D([], [], color=BLUE, lw=2.4, label=r"exactly valid Mondrian: $\mathbb{E}[\min_g]$"),
    Patch(facecolor=BLUE, alpha=0.10, label="its central $95\\%$ range"),
    Line2D([], [], color="k", lw=2.0, ls="--", label=r"nominal $0.90$")],
    loc="upper left", fontsize=10, labelspacing=0.45)
print("\n->", finalize_figure(fig, FIG + r"\subtarget_law_vs_observed", formats=["pdf", "png"]))
