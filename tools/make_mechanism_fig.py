"""The mechanism figure: the KS bound holds, and it predicts how much re-calibrating will help.

The audit asked for Eq (3)'s right-hand side to be computed per setting and shown to predict the
observed lift. Checking that first showed it cannot carry the claim: under marginal calibration the
pooled threshold is chosen so pooled coverage equals 1-alpha, and it lands within 0.006 of 0.90 in
every setting, so "(1-alpha) - F_wg" and "marginal_cov - F_wg" are the same number. Eq (3) is an
identity. Presenting it as a prediction would be dressing up a definition.

What is not tautological is the bound that follows from it, Eq (4): the shortfall is at most
(1-pi) * D_KS. D_KS is measured from the scores under marginal calibration, before Mondrian is run,
and nothing forces the inequality to be tight or even close.

  Left panel  -- the bound holds in all eight settings; shortfall/bound is 0.59-0.93 on Waterbirds
                 and 0.43-0.58 on CelebA, so it is tight on the first and loose on the second.
  Right panel -- D_KS, measured before re-calibrating, predicts how much re-calibrating gains
                 (r = 0.93). That is the sense in which we know what controls the effect.
"""
import csv
import sys
from collections import defaultdict

sys.path.insert(0, r"c:\jagr\vgscp")
import numpy as np

from study_robust_train.figstyle import (PALETTE, FigureStyle, apply_publication_style,
                                         create_subplots, finalize_figure)

apply_publication_style(FigureStyle(font_size=14, axes_linewidth=1.8))
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BLUE, RED, GREY = PALETTE["blue_main"], PALETTE["red_strong"], "#767676"
OUT = r"c:\jagr\vgscp\ACML_Journal___Robust_CP_Train_Study (1)\figures\mechanism_ks"
TARGET = 0.90  # pi is read per run from the records, not assumed: see below

rows = [r for r in csv.DictReader(
    open(r"c:\jagr\vgscp\results\calibration_ablation_4bb.csv"))
    if r["score"] == "APS" and r["rho_test"] == "0.95" and r["gate_status"] == "kept"]
pair = defaultdict(dict)
for r in rows:
    pair[(r["backbone"], r["dataset"], r["method"], r["train_seed"],
          r["split_seed"])][r["calibration"]] = r

agg = defaultdict(list)
for k, d in pair.items():
    if "marginal_split" in d and "mondrian" in d:
        m, mo = d["marginal_split"], d["mondrian"]
        # pi is the worst group's share of the calibration half, which differs by dataset:
        # 2.5% on Waterbirds but 47.5% on CelebA, where compositing makes the group that comes
        # out worst one of the two large ones. A single constant here inflates CelebA's bound.
        agg[(k[0], k[1])].append((float(m["worst_group_cov"]), float(mo["worst_group_cov"]),
                                  float(m["div_ks_stat"]),
                                  int(m["n_cal_worst_group"]) / int(m["n_eval"])))

NAME = {"resnet50_erm": "ResNet-50", "clip_vitb32": "CLIP",
        "dinov2_vitb14": "DINOv2", "vit_b16_in1k": "ViT-B/16"}
short, bound, ks, lift, isw, lab = [], [], [], [], [], []
for (bb, ds) in sorted(agg):
    a = np.array(agg[(bb, ds)])
    wgm, wgmo, k_, pi_ = a.mean(axis=0)
    short.append(TARGET - wgm)
    bound.append((1 - pi_) * k_)
    ks.append(k_)
    lift.append(wgmo - wgm)
    isw.append(ds == "waterbirds")
    lab.append(f"{NAME[bb]}/{'WB' if ds == 'waterbirds' else 'CelebA'}")
short, bound, ks, lift = map(np.array, (short, bound, ks, lift))

print(f"pelanggaran batas: {(short > bound + 1e-9).sum()}/8")
print(f"ketat: shortfall/bound = {(short / bound).min():.2f}--{(short / bound).max():.2f}")
r_ks = np.corrcoef(ks, lift)[0, 1]
print(f"korelasi(D_KS, lift) = {r_ks:.3f}")
for _w, _n in ((True, "waterbirds"), (False, "celeba")):
    _m = np.array(isw) == _w
    print(f"  {_n:12s} rasio {(short[_m]/bound[_m]).min():.2f}-{(short[_m]/bound[_m]).max():.2f}")

fig, ax = create_subplots(1, 2, figsize=(11.6, 4.5))

# ---- (a) the bound holds
a0 = ax[0]
hi = max(bound.max(), short.max()) * 1.12
a0.plot([0, hi], [0, hi], color=GREY, lw=1.6, ls="--", zorder=1)
a0.fill_between([0, hi], [0, hi], [hi, hi], color=GREY, alpha=0.07, lw=0)
for x, y, w in zip(bound, short, isw):
    a0.scatter([x], [y], s=120, color=BLUE if w else RED, edgecolor="k", lw=1.3,
               marker="o" if w else "s", zorder=3)
a0.text(hi * 0.30, hi * 0.88, "no point can lie\nabove this line", ha="left", va="top",
        fontsize=11, color="0.45")
a0.set_xlabel(r"$(1-\pi)\,D_{\mathrm{KS}}$, the bound")
a0.set_ylabel("observed shortfall")
a0.set_xlim(0, hi)
a0.set_ylim(0, hi)
a0.set_title("The bound holds in every setting", fontsize=13, fontweight="bold", pad=8)

# ---- (b) it predicts the gain from re-calibrating
a1 = ax[1]
c = np.polyfit(ks, lift, 1)
xx = np.linspace(ks.min() * 0.9, ks.max() * 1.05, 50)
a1.plot(xx, np.polyval(c, xx), color=GREY, lw=1.8, zorder=1)
for x, y, w in zip(ks, lift, isw):
    a1.scatter([x], [y], s=120, color=BLUE if w else RED, edgecolor="k", lw=1.3,
               marker="o" if w else "s", zorder=3)
a1.text(0.04, 0.93, f"$r = {r_ks:.2f}$", transform=a1.transAxes, fontsize=14,
        fontweight="bold", va="top")
a1.set_xlabel(r"$D_{\mathrm{KS}}$ under one shared threshold")
a1.set_ylabel("gain from a per-group threshold")
a1.set_title("Divergence tracks the gain across settings", fontsize=13, fontweight="bold", pad=8)
a1.axhline(0, color="0.75", lw=1.0, zorder=0)

a0.legend(handles=[Line2D([], [], marker="o", ls="", ms=9, mfc=BLUE, mec="k",
                          label="Waterbirds"),
                   Line2D([], [], marker="s", ls="", ms=9, mfc=RED, mec="k",
                          label="CelebA")], loc="upper left")
print("->", finalize_figure(fig, OUT, formats=["pdf", "png"]))
