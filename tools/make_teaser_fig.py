"""The main method figure: the confound, why pooled calibration fails, what it costs, and what the
fix costs in return.

Design follows the layout grammar of the repository's own teaser (`assets/VIGIL_teaser.png`,
`figure_VIGIL/plot_concept.py`): bare lowercase panel letters set large and light, the panel's
*message* as a short claim heading over a hairline rule rather than a descriptive title, light
fills (`alpha` ~0.12) under `lw` ~2.2 lines, blue reserved for the contribution and nothing else,
and a monospace tag naming the cell so a reader can see the numbers are measured rather than drawn.

Four panels, which are the paper's argument in order:

  (a) the confound, on real images, laid out as the 2x2 that *defines* the groups;
  (b) the mechanism, fitted to a real cell, so the wedge is Eq. (7) at the measured value;
  (c) what marginal calibration costs the rare group, and what a per-group threshold recovers;
  (d) what that recovery costs in efficiency -- the honest other half. The worst group's sets grow
      from 1.080 to 1.278, and the *mean* set size grows with them, 0.984 -> 1.214: the burden is
      not removed but moved onto the group that was previously under-served, and it is not free in
      aggregate either. Leaving (d) out would have shown only the favourable half of the trade.
      (Both numbers moved when the Mondrian set construction was corrected to be label-conditional;
      every value in this figure is read from the records, so re-running it is the only fix needed.)

Two labelling errors from earlier drafts, both worth recording:

  - The wedge was labelled "0.391 of the worst group falls outside". It is not: the mass above the
    pooled threshold is 1 - 0.509 = 0.491. The wedge, bounded by the two thresholds, is the
    shortfall *from the 0.90 target*, 0.391. It is now labelled as that.
  - Panel (c)'s realised lift is +0.345, not the wedge's 0.391, because Mondrian lands at 0.854
    rather than 0.900 -- the min-over-groups finite-sample effect of Appendix B. The two panels
    therefore carry different numbers on purpose.

Numbers come from the CLIP/Waterbirds ERM cell under APS at rho_cal = rho_test = 0.95, averaged
over 3 training seeds x 10 calibration splits (30 runs per policy).

One of the twelve downloaded images carries a fotolibra.com stock watermark -- Waterbirds
composites birds onto Places backgrounds, and Places contains watermarked stock -- so the four used
here were checked by eye.
"""
import csv
import os
import sys

sys.path.insert(0, r"c:\jagr\vgscp")
import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

from study_robust_train.figstyle import (PALETTE, FigureStyle, apply_publication_style,
                                         finalize_figure)

apply_publication_style(FigureStyle(font_size=13, axes_linewidth=1.6))
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

BLUE, BLUE2, RED = PALETTE["blue_main"], PALETTE["blue_secondary"], PALETTE["red_strong"]
RED_DARK, BLUE_DARK = "#7E2E2D", "#0A3466"
GREY, NEUTRAL = "#767676", PALETTE["neutral"]
IMG = r"c:\jagr\vgscp\results\sample_images"
OUT = r"c:\jagr\vgscp\ACML_Journal___Robust_CP_Train_Study (1)\figures\teaser_method"
# The two glyphs are taken from DejaVu Sans explicitly; Arial carries the text but not always
# U+2713/U+2717, and a missing-glyph box in the teaser would be the worst place to find that out.
GLYPH = "DejaVu Sans"

# ---------------------------------------------------------------- measured numbers
CELL = dict(backbone="clip_vitb32", dataset="waterbirds", method="erm", score="APS",
            rho_test="0.95")
rows = list(csv.DictReader(open(r"c:\jagr\vgscp\results\calibration_ablation_4bb.csv")))


def cell_mean(policy, col):
    sel = [r for r in rows if all(r[k] == v for k, v in CELL.items())
           and r["calibration"] == policy and r["gate_status"] == "kept"]
    assert sel, f"tidak ada baris untuk {policy}"
    return float(np.mean([float(r[col]) for r in sel])), len(sel)


POOLED, n_runs = cell_mean("marginal_split", "marginal_cov")
WG_MARG, _ = cell_mean("marginal_split", "worst_group_cov")
WG_MOND, _ = cell_mean("mondrian", "worst_group_cov")
SET_MARG, _ = cell_mean("marginal_split", "worst_group_set_size")
SET_MOND, _ = cell_mean("mondrian", "worst_group_set_size")
MEAN_MARG, _ = cell_mean("marginal_split", "mean_set_size")
MEAN_MOND, _ = cell_mean("mondrian", "mean_set_size")
TARGET, PI = 0.90, 0.05
print(f"sel ({n_runs} run/kebijakan): pooled {POOLED:.4f} | wg {WG_MARG:.4f} -> {WG_MOND:.4f} | "
      f"wg set {SET_MARG:.4f} -> {SET_MOND:.4f} | mean set {MEAN_MARG:.4f} -> {MEAN_MOND:.4f}")

# rows = species, columns = background; the off-diagonal cells are the minority groups
GRID = [["landbird_on_land_1.jpg", "landbird_on_water_2.jpg"],
        ["waterbird_on_land_1.jpg", "waterbird_on_water_0.jpg"]]
ROWLAB, COLLAB = ["landbird", "waterbird"], ["on land", "on water"]

# ══════════════════════════════════════════════════════════════════════════════ layout
# Taller than the three-panel version: column 3 now stacks two panels, and each needs room for
# ticks and an axis name. At 4.9in the coverage panel's axis name was landing in panel (d)'s
# heading.
fig = plt.figure(figsize=(15.6, 5.6))
gs = GridSpec(1, 3, width_ratios=[0.84, 1.22, 0.82], wspace=0.235, figure=fig,
              left=0.050, right=0.988, top=0.760, bottom=0.150)
axa = fig.add_subplot(gs[0])
axa.set_axis_off()
axb = fig.add_subplot(gs[1])
col3 = gs[2].subgridspec(2, 1, height_ratios=[1.18, 0.82], hspace=1.05)
axc = fig.add_subplot(col3[0])
axd = fig.add_subplot(col3[1])

# ══════════════════════════════════════════════════════════ (a) the confound as a 2x2
pa = axa.get_position()
FIGW, FIGH = fig.get_size_inches()
# Equal gutters in inches, not figure fractions: the panel is far wider than tall, so one
# fraction gives horizontal and vertical gaps of visibly different size.
GUT_IN = 0.15
GUT_X, GUT_Y = GUT_IN / FIGW, GUT_IN / FIGH
CELL_W = (pa.width - GUT_X) / 2
CELL_H = (pa.height - GUT_Y) / 2
CELL_ASPECT = (CELL_W * FIGW) / (CELL_H * FIGH)


def center_crop(arr, aspect):
    """Crop to the cell's aspect ratio instead of stretching to it. `aspect="auto"` fills the cell
    but distorts the bird, which in a figure about telling two species apart is the one distortion
    that is not acceptable."""
    h, w = arr.shape[:2]
    if w / h > aspect:
        nw = int(round(h * aspect))
        return arr[:, (w - nw) // 2:(w - nw) // 2 + nw]
    nh = int(round(w / aspect))
    return arr[(h - nh) // 2:(h - nh) // 2 + nh]


for i in range(2):
    for j in range(2):
        ax = fig.add_axes([pa.x0 + j * (CELL_W + GUT_X),
                           pa.y1 - (i + 1) * CELL_H - i * GUT_Y, CELL_W, CELL_H])
        ax.imshow(center_crop(mpimg.imread(os.path.join(IMG, GRID[i][j])), CELL_ASPECT))
        ax.set_xticks([])
        ax.set_yticks([])
        # One rectangle, not four spines: spines are drawn per side and at this size rendered
        # unevenly, leaving borders that looked broken on two of the four images.
        for sp in ax.spines.values():
            sp.set_visible(False)
        minority = i != j
        ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, fill=False,
                               edgecolor=RED if minority else "#C4C4C4",
                               linewidth=3.0 if minority else 1.1, zorder=10, clip_on=False))
        # Every cell carries its own share, so the reader ties the number to the image rather than
        # to a sentence underneath, and the caption can then be a single line.
        ax.text(0.042, 0.95, "2.5%" if minority else "47.5%", transform=ax.transAxes,
                ha="left", va="top", fontsize=10.5, color="white", fontweight="bold", zorder=11,
                bbox=dict(boxstyle="square,pad=0.22", facecolor=RED if minority else "#6E6E6E",
                          edgecolor="none"))
for j in range(2):
    fig.text(pa.x0 + j * (CELL_W + GUT_X) + CELL_W / 2, pa.y1 + 0.020, COLLAB[j],
             ha="center", va="bottom", fontsize=12, color="0.25")
for i in range(2):
    fig.text(pa.x0 - 0.012, pa.y1 - (i + 0.5) * CELL_H - i * GUT_Y, ROWLAB[i],
             ha="center", va="center", fontsize=12, color="0.25", rotation=90)
fig.text(pa.x0, pa.y0 - 0.060,
         "The background predicts the label for 95% of the data.\n"
         "Red marks the two groups where it does not.",
         ha="left", va="top", fontsize=11, color="0.35", linespacing=1.55)

# ══════════════════════════════════════════════════════ (b) the mechanism, fitted to the cell
sd = 1.0


def wg_cov_at(shift):
    q = brentq(lambda t: PI * norm.cdf(t, shift, sd)
               + (1 - PI) * norm.cdf(t, 0.0, sd) - TARGET, -8, 12)
    return norm.cdf(q, shift, sd), q


shift = brentq(lambda s: wg_cov_at(s)[0] - WG_MARG, 0.01, 6.0)
cov_m, q_pool = wg_cov_at(shift)
q_wg = norm.ppf(TARGET, shift, sd)
print(f"panel b: shift={shift:.3f} q_pool={q_pool:.3f} cov={cov_m:.4f} q_wg={q_wg:.3f} "
      f"wedge={norm.cdf(q_wg, shift, sd) - cov_m:.4f}")

x = np.linspace(-3.2, shift + 3.9, 900)
top = norm.pdf(0, 0, sd)
for mu, col, lw in [(0.0, GREY, 2.0), (shift, RED, 2.6)]:
    axb.fill_between(x, 0, norm.pdf(x, mu, sd), color=col, alpha=0.12, lw=0)
    axb.plot(x, norm.pdf(x, mu, sd), color=col, lw=lw)
mask = (x >= q_pool) & (x <= q_wg)
axb.fill_between(x[mask], 0, norm.pdf(x[mask], shift, sd), color=RED, alpha=0.34, lw=0)
axb.axvline(q_pool, color="#111111", lw=1.8, ymax=0.80, zorder=4)
axb.axvline(q_wg, color=BLUE, ls="--", lw=2.0, ymax=0.80, zorder=4)

# Curve labels sit over their own flanks, pushed apart: the peaks are only 1.39 apart, so a label
# centred over each peak overlapped its neighbour.
axb.text(-1.95, top * 1.05, "everyone else", color="0.35", fontsize=11.5, ha="center")
axb.text(shift + 0.50, top * 1.05, "worst group", color=RED, fontsize=11.5, ha="center")
axb.text(0.42, top * 0.155, f"covered\n{cov_m:.3f}", color=RED_DARK, fontsize=11,
         ha="center", va="center", linespacing=1.45)
# NOT "falls outside": the mass above the pooled threshold is 0.491. This wedge, bounded by the
# two thresholds, is the shortfall from the 0.90 target.
axb.text(q_wg + 0.30, top * 0.50,
         f"{TARGET - cov_m:.3f}\nshort of the\n0.90 target", color=RED, fontsize=11.5,
         ha="left", va="center", linespacing=1.5, fontweight="bold")

# Threshold labels are anchored to run *away* from each other; as tick labels they overlapped,
# the two thresholds being only 1.26 apart.
axb.text(q_pool - 0.10, -top * 0.035, "pooled threshold\n(marginal)", ha="right", va="top",
         fontsize=11.2, color="#111111", linespacing=1.4, clip_on=False)
axb.text(q_wg + 0.10, -top * 0.035, "its own threshold\n(Mondrian)", ha="left", va="top",
         fontsize=11.2, color=BLUE, linespacing=1.4, clip_on=False)
axb.text(x[0], -top * 0.035, "conformity score", ha="left", va="top", fontsize=11.5,
         color="0.45", clip_on=False)
# Above everything: the Mondrian threshold line rises to 0.96*top, so a tag at 0.92 crossed it,
# and at 1.05 it would meet the "worst group" label.
axb.text(x[-1], top * 1.14, "CLIP · Waterbirds · ERM · APS", ha="right", va="center",
         fontsize=10.2, color="0.55", family="monospace")
axb.set_xticks([])
axb.set_yticks([])
axb.set_ylim(0, top * 1.20)
axb.set_xlim(x[0], x[-1])
axb.spines["left"].set_visible(False)

# ══════════════════════════════════════════════════════ (c) coverage, measured
# No tick/cross marks. They collided with the lift arrow and the target line, and a tick beside
# 0.854 would assert that Mondrian meets the 0.90 target, which it does not -- it sits below by the
# min-over-groups selection effect of Appendix B. The target line carries that honestly on its own.
cbars = [("pooled", POOLED, NEUTRAL, "#8A8A8A", "#333333"),
         ("worst · marginal", WG_MARG, RED, RED_DARK, "white"),
         ("worst · Mondrian", WG_MOND, BLUE, BLUE_DARK, "white")]
ypos = np.arange(len(cbars))[::-1]
for yp, (lab, val, face, edge, ink) in zip(ypos, cbars):
    axc.barh(yp, val, height=0.50, color=face, edgecolor=edge, linewidth=1.2, zorder=3)
    axc.text(val - 0.022, yp, f"{val:.3f}", va="center", ha="right", fontsize=12.2,
             fontweight="bold", color=ink, zorder=4)
axc.axvline(TARGET, color="#111111", ls="--", lw=1.5, zorder=5)
axc.text(TARGET - 0.014, ypos[0] + 0.36, "target 0.90", ha="right", va="bottom", fontsize=10.8,
         color="#111111")
mid = float(np.mean([ypos[1], ypos[2]]))
axc.annotate("", xy=(WG_MOND, mid), xytext=(WG_MARG, mid),
             arrowprops=dict(arrowstyle="<|-|>", lw=1.6, color=BLUE2), zorder=6)
axc.text((WG_MARG + WG_MOND) / 2, mid + 0.06, f"+{WG_MOND - WG_MARG:.3f}", ha="center",
         va="bottom", fontsize=12.2, color=BLUE2, fontweight="bold", zorder=7)
axc.set_yticks(ypos)
axc.set_yticklabels([b[0] for b in cbars], fontsize=11)
axc.set_xlim(0, 1.02)
axc.set_xticks([0, 0.5, 0.9])
axc.set_ylim(-0.52, len(cbars) - 0.32)
axc.spines["left"].set_visible(False)
axc.tick_params(axis="y", length=0)
axc.text(0.0, -0.30, "worst-group coverage", transform=axc.transAxes, ha="left", va="top",
         fontsize=11.2, color="0.45")

# ══════════════════════════════════════════════════════ (d) what the fix costs
dbars = [("worst · marginal", SET_MARG, RED, RED_DARK), ("worst · Mondrian", SET_MOND, BLUE,
                                                         BLUE_DARK)]
dpos = np.arange(len(dbars))[::-1]
for yp, (lab, val, face, edge) in zip(dpos, dbars):
    axd.barh(yp, val, height=0.46, color=face, edgecolor=edge, linewidth=1.2, zorder=3)
    axd.text(val - 0.030, yp, f"{val:.3f}", va="center", ha="right", fontsize=12.2,
             fontweight="bold", color="white", zorder=4)
axd.annotate("", xy=(SET_MOND, 1.62), xytext=(SET_MARG, 1.62),
             arrowprops=dict(arrowstyle="<|-|>", lw=1.6, color=BLUE2), zorder=6)
axd.text((SET_MARG + SET_MOND) / 2, 1.72, f"+{SET_MOND - SET_MARG:.3f}", ha="center",
         va="bottom", fontsize=12.2, color=BLUE2, fontweight="bold", zorder=7)
axd.set_yticks(dpos)
axd.set_yticklabels([b[0] for b in dbars], fontsize=11)
axd.set_xlim(0, 1.62)
axd.set_xticks([0, 0.5, 1.0, 1.5])
axd.set_ylim(-0.50, 2.20)
axd.spines["left"].set_visible(False)
axd.tick_params(axis="y", length=0)
axd.text(0.0, -0.38, "worst-group set size", transform=axd.transAxes, ha="left", va="top",
         fontsize=11.2, color="0.45")
axd.text(0.0, -0.74, f"mean set size follows: {MEAN_MARG:.3f} → {MEAN_MOND:.3f}",
         transform=axd.transAxes, ha="left", va="top", fontsize=10.8, color="0.35")

# ══════════════════════════════════════════════════════ headings: bold claim + hairline rule
def heading(ax, letter, text, col, y):
    p = ax.get_position()
    fig.text(p.x0, y, letter, fontsize=19, va="center", ha="left", color="0.15")
    fig.text(p.x0 + 0.0185, y, text, fontsize=12.4, va="center", ha="left", color=col,
             fontweight="bold")
    fig.add_artist(Line2D([p.x0, p.x1], [y - 0.052, y - 0.052], color="#D0D0D0", lw=1.0))


heading(axa, "a", "Background co-varies with the species label", "0.15", 0.945)
heading(axb, "b", "One pooled threshold under-covers the rare group", "0.15", 0.945)
heading(axc, "c", "Calibration alone recovers it", BLUE, 0.945)
heading(axd, "d", "And it is paid in set size", BLUE, axd.get_position().y1 + 0.088)

print("->", finalize_figure(fig, OUT, formats=["pdf", "png"], pad=0.06))
