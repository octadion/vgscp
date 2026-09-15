"""Build every figure in the paper from current data, in the figures4papers house style.

Five matplotlib figures (a sixth, fig:overview, is drawn by make_teaser_fig.py and the seventh,
fig:subtarget, by make_subtarget_fig.py). Three are new,
and each of the three exists because a reviewer point is far easier to see than to read:

  fig:c1        the ERM lift and the across-training spread in all eight cells   -> R1.3
  fig:h1        why accuracy matching is unavailable: the (accuracy, divergence)
                clouds of ERM and the robust methods barely overlap                 -> R2.3
  fig:subtarget mean-over-groups sits at the nominal level while the minimum
                does not                                                          -> R1.1

Semantic colour mapping is applied consistently across all five: blue = Mondrian, the mechanism
the paper is about; red = marginal split, the baseline it is contrasted with; neutral grey =
background quantities such as accuracy. In the shift figure this replaces an earlier version that
used colour for backbone and line style for policy -- which buried the contrast the figure exists
to show.

Everything is exported as PDF (for the manuscript) and PNG (to look at). Every plotted value is
printed so the figures can be checked against the tables.
"""
import sys

sys.path.insert(0, r"c:\jagr\vgscp")
import numpy as np

from study_robust_train.calibration_ablation import records_from_csv
from study_robust_train.figstyle import (PALETTE, FigureStyle, annotate_bars,
                                         apply_publication_style, create_subplots,
                                         finalize_figure, make_grouped_bar)
from study_robust_train.grid import records_from_csv as grid_records_from_csv
from study_robust_train.representation import records_from_representation_csv

FIG = r"c:\jagr\vgscp\ACML_Journal___Robust_CP_Train_Study (1)\figures"
BLUE, RED, GREY = PALETTE["blue_main"], PALETTE["red_strong"], "#767676"
apply_publication_style(FigureStyle(font_size=15, axes_linewidth=2.0))
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

A = records_from_csv(r"c:\jagr\vgscp\results\calibration_ablation_4bb.csv")
AK = [r for r in A if r["gate_status"] != "excluded" and r["score"] == "APS"]
G = [r for r in grid_records_from_csv(r"c:\jagr\vgscp\results\grid_records.csv")
     if r.get("gate_status") != "excluded" and r["score"] == "APS" and r["rho_test"] == 0.95]
RR = records_from_representation_csv(r"c:\jagr\vgscp\results\representation_records.csv")

NAME = {"resnet50_erm": "ResNet-50", "clip_vitb32": "CLIP", "dinov2_vitb14": "DINOv2",
        "vit_b16_in1k": "ViT-B/16"}
BBS = ["resnet50_erm", "clip_vitb32", "dinov2_vitb14", "vit_b16_in1k"]
CELLS = [(bb, ds) for ds in ("waterbirds", "celeba") for bb in BBS]
RHOS = [0.95, 0.9, 0.8, 0.7, 0.6, 0.5]


def cov(bb, ds, cal, rho=0.95, method=None, gated_ok=False):
    src = A if gated_ok else AK
    v = [r["worst_group_cov"] for r in src
         if r["backbone"] == bb and r["dataset"] == ds and r["calibration"] == cal
         and r["score"] == "APS" and r["rho_test"] == rho
         and (method is None or r["method"] == method)]
    return float(np.mean(v)) if v else float("nan")


# ══════════════════════════════════════════════════════ fig:c1 — the headline, all eight cells
print("=== fig:c1 ===")
fig, ax = create_subplots(1, 1, figsize=(11.5, 4.4))
ax = ax[0]
y = np.arange(len(CELLS))[::-1].astype(float)
for i, (bb, ds) in enumerate(CELLS):
    yy = y[i]
    # ERM is gated out on two CelebA cells; fall back to the sensitivity records there, and
    # mark those rows with a dagger rather than letting them look like ordinary values.
    gated = not any(r["method"] == "erm" and r["backbone"] == bb and r["dataset"] == ds
                    for r in AK)
    m = cov(bb, ds, "marginal_split", method="erm", gated_ok=gated)
    d = cov(bb, ds, "mondrian", method="erm", gated_ok=gated)
    ax.plot([m, d], [yy, yy], color="0.72", lw=3.0, zorder=1, solid_capstyle="round")
    ax.scatter([m], [yy], s=110, color=RED, edgecolor="k", lw=1.4, zorder=3)
    ax.scatter([d], [yy], s=110, color=BLUE, edgecolor="k", lw=1.4, zorder=3)
    ax.annotate(f"+{d - m:.3f}", xy=((m + d) / 2, yy), xytext=(0, 9),
                textcoords="offset points", ha="center", fontsize=10.5, color="0.35")
    print(f"  {NAME[bb]:10s}/{ds:10s} marginal={m:.3f} mondrian={d:.3f} lift=+{d-m:.3f}"
          + ("  (ERM excluded; sensitivity value)" if gated else ""))

ax.axvline(0.90, color="k", ls=":", lw=1.6)
ax.text(0.902, len(CELLS) - 0.35, r"target $1-\alpha$", fontsize=11, color="0.25", va="top")
ax.set_yticks(y)
ax.set_yticklabels([f"{NAME[b]} / {'Waterbirds' if d == 'waterbirds' else 'CelebA'}"
                    + ("$^{\\dagger}$" if (b, d) in [("dinov2_vitb14", "celeba"),
                                                     ("vit_b16_in1k", "celeba")] else "")
                    for b, d in CELLS])
ax.set_xlabel("worst-group coverage of the same ERM model")
ax.set_xlim(0.47, 0.95)
ax.legend(handles=[Line2D([], [], marker="o", ls="", ms=10, mfc=RED, mec="k",
                          label="marginal split"),
                   Line2D([], [], marker="o", ls="", ms=10, mfc=BLUE, mec="k",
                          label="Mondrian")],
          loc="lower left", ncol=1)
ax.set_ylim(-0.7, len(CELLS) - 0.3)
print("  ->", finalize_figure(fig, FIG + r"\c1_erm_lift", formats=["pdf", "png"]))

# ══════════════════════════════════════════════════════ fig:h1 — the accuracy confound
print("\n=== fig:h1 ===")
fig, axes = create_subplots(1, 2, figsize=(11.5, 4.0))
for ax, ds, dsn in zip(axes, ("waterbirds", "celeba"), ("Waterbirds", "CelebA")):
    for bb in BBS:
        for m in sorted({r["method"] for r in G if r["backbone"] == bb and r["dataset"] == ds}):
            rr = [r for r in G if r["backbone"] == bb and r["dataset"] == ds and r["method"] == m]
            a = float(np.mean([r["base_top1"] for r in rr]))
            d = float(np.mean([r["div_wasserstein1"] for r in rr]))
            is_erm = m == "erm"
            ax.scatter([a], [d], s=170 if is_erm else 90,
                       color=RED if is_erm else BLUE,
                       marker="D" if is_erm else "o",
                       edgecolor="k", lw=1.4, zorder=3 if is_erm else 2,
                       alpha=1.0 if is_erm else 0.75)
    # No pooled ERM band. Matching is WITHIN a cell, so a band spanning ERM's accuracy across
    # all four backbones would suggest overlaps that do not exist for any single comparison.
    # Instead, ring the one (cell, arm) pair in the whole grid whose per-seed accuracy support
    # actually brackets its own cell's ERM -- which is the finding.
    if ds == "waterbirds":
        rr = [r for r in G if r["backbone"] == "resnet50_erm" and r["dataset"] == ds
              and r["method"] == "groupdro_ll"]
        ax.scatter([float(np.mean([r["base_top1"] for r in rr]))],
                   [float(np.mean([r["div_wasserstein1"] for r in rr]))],
                   s=340, facecolors="none", edgecolors="k", lw=2.2, zorder=4)
        ax.annotate("the only matched pair\n(GroupDRO-LL, ResNet-50)",
                    xy=(float(np.mean([r["base_top1"] for r in rr])),
                        float(np.mean([r["div_wasserstein1"] for r in rr]))),
                    xytext=(-172, -6), textcoords="offset points", fontsize=10.5,
                    ha="left", va="center", color="0.25",
                    arrowprops=dict(arrowstyle="-", color="0.45", lw=1.4))
    ax.set_title(dsn)
    ax.set_xlabel("base top-1 accuracy")
axes[0].set_ylabel(r"cross-group divergence $W_1$")
axes[0].legend(handles=[Line2D([], [], marker="D", ls="", ms=10, mfc=RED, mec="k",
                               label="ERM (reference)"),
                        Line2D([], [], marker="o", ls="", ms=9, mfc=BLUE, mec="k",
                               label="robust methods")],
               loc="upper left")
print("  ->", finalize_figure(fig, FIG + r"\h1_accuracy_confound", formats=["pdf", "png"]))

# ══════════════════════════════════════════════════════ fig:subtarget — mean vs min over groups
print("\n=== fig:subtarget ===")
fig, ax = create_subplots(1, 1, figsize=(11.0, 3.9))
ax = ax[0]
x = np.arange(len(CELLS), dtype=float)
mg, wg = [], []
for bb, ds in CELLS:
    sel = [r for r in AK if r["backbone"] == bb and r["dataset"] == ds
           and r["calibration"] == "mondrian" and r["score"] == "APS" and r["rho_test"] == 0.95]
    mg.append(float(np.mean([float(r["mean_group_cov"]) for r in sel])))
    wg.append(float(np.mean([r["worst_group_cov"] for r in sel])))
    print(f"  {NAME[bb]:10s}/{ds:10s} mean-over-groups={mg[-1]:.4f}  min={wg[-1]:.4f}  "
          f"gap={mg[-1]-wg[-1]:.4f}")
ax.vlines(x, wg, mg, color="0.72", lw=3.0, zorder=1)
ax.scatter(x, mg, s=115, color=BLUE, edgecolor="k", lw=1.4, zorder=3, label="mean over groups")
ax.scatter(x, wg, s=115, color=GREY, edgecolor="k", lw=1.4, zorder=3,
           label=r"minimum over groups (reported)")
ax.axhline(0.90, color="k", ls=":", lw=1.6)
ax.text(len(CELLS) - 0.45, 0.9018, r"target $1-\alpha$", fontsize=11, color="0.25",
        ha="right", va="bottom")
ax.set_xticks(x)
ax.set_xticklabels([f"{NAME[b]}\n{'WB' if d == 'waterbirds' else 'CelebA'}" for b, d in CELLS],
                   fontsize=11)
ax.set_ylabel("coverage under Mondrian")
ax.set_ylim(0.845, 0.918)
ax.legend(loc="lower left", ncol=2)
print(f"  mean of the means {np.mean(mg):.4f} | mean of the minima {np.mean(wg):.4f}")
print("  ->", finalize_figure(fig, FIG + r"\subtarget_mean_vs_min", formats=["pdf", "png"]))

# ══════════════════════════════════════════════════════ fig:repr — the representation lever
print("\n=== fig:repr ===")
REPS = [("erm", "ERM"), ("groupdro", "GroupDRO"), ("reweight", "Reweighting")]
fig, axes = create_subplots(1, 2, figsize=(11.0, 4.2), sharey=True)
for ax, ds, dsn in zip(axes, ("waterbirds", "celeba"), ("Waterbirds", "CelebA")):
    def ser(field, cal=None):
        return [float(np.mean([r[field] for r in RR if r["dataset"] == ds
                               and r["representation"] == rep and r["head"] == "erm"
                               and r["score"] == "APS"
                               and (cal is None or r["calibration"] == cal)]))
                for rep, _ in REPS]

    acc, marg, mond = ser("worst_group_acc"), ser("worst_group_cov", "marginal_split"), \
        ser("worst_group_cov", "mondrian")
    print(f"  {dsn}: acc {[f'{v:.3f}' for v in acc]} | marginal {[f'{v:.3f}' for v in marg]} "
          f"(spread {max(marg)-min(marg):.3f}) | mondrian {[f'{v:.3f}' for v in mond]} "
          f"(spread {max(mond)-min(mond):.3f})")
    bars = make_grouped_bar(ax, [n for _, n in REPS], [acc, marg, mond],
                            ["worst-group accuracy", "coverage, marginal", "coverage, Mondrian"],
                            ylabel="", colors=[GREY, RED, BLUE])
    annotate_bars(ax, bars, fmt="{:.3f}", fontsize=10, padding=3, color=BLUE)
    ax.axhline(0.90, color="k", ls=":", lw=1.6)
    ax.set_title(dsn)
    ax.set_ylim(0, 1.16)          # bars start at zero; headroom is for the labels
axes[0].set_ylabel("worst-group coverage / accuracy")
axes[0].text(-0.45, 1.075, r"dotted line: target $1-\alpha=0.90$", fontsize=11, color="0.25")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.05))
print("  ->", finalize_figure(fig, FIG + r"\repr_lever", formats=["pdf", "png"]))

# ══════════════════════════════════════════════════════ fig:shift — semantic recolour
print("\n=== fig:shift ===")
fig, axes = create_subplots(1, 2, figsize=(11.5, 3.9), sharey=True)
for ax, ds, dsn in zip(axes, ("waterbirds", "celeba"), ("Waterbirds", "CelebA")):
    for cal, c in (("mondrian", BLUE), ("marginal_split", RED)):
        curves = []
        for bb in BBS:
            ys = [cov(bb, ds, cal, rho) for rho in RHOS]
            curves.append(ys)
            ax.plot(RHOS, ys, "-", marker="o", ms=4.2, lw=2.0, color=c, alpha=0.85)
        st = np.vstack(curves)
        ax.fill_between(RHOS, st.min(axis=0), st.max(axis=0), color=c, alpha=0.16, linewidth=0)
        print(f"  {dsn:11s} {cal:15s} envelope at rho=0.95: "
              f"[{st[:, 0].min():.3f}, {st[:, 0].max():.3f}]  width {np.ptp(st[:, 0]):.3f}")
    ax.axhline(0.90, color="k", ls=":", lw=1.6)
    ax.set_xlabel(r"$\rho_{\mathrm{test}}$")
    ax.set_title(dsn)
    ax.set_xticks(RHOS)
    ax.set_xticklabels([f"{r:g}" for r in RHOS])
    ax.invert_xaxis()
axes[0].set_ylabel("worst-group coverage")
axes[0].set_ylim(0.69, 0.925)
axes[0].text(0.955, 0.905, r"target $1-\alpha$", fontsize=11, color="0.25", va="bottom")
axes[1].legend(handles=[Line2D([], [], color=BLUE, lw=2.4, marker="o", ms=4.2,
                               label="Mondrian (4 backbones)"),
                        Line2D([], [], color=RED, lw=2.4, marker="o", ms=4.2,
                               label="marginal split (4 backbones)")],
               loc="lower left")
print("  ->", finalize_figure(fig, FIG + r"\shift_mondrian_vs_marginal", formats=["pdf", "png"]))
