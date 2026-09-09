"""Build the per-group coverage table Appendix B currently reports the absence of.

The paper explains the sub-target Mondrian level as a min-over-groups selection effect: each group
is individually near target, and the minimum of four noisy coverages sits below their mean. Until
now a reader could check that only against the simulation in Appendix C, because the records stored
the mean, the minimum and the range but not the four groups separately.

These are the four, measured on the released splits. Kept methods only, matching every other table;
gate status is joined from the ablation records rather than re-derived.
"""
import csv
from collections import defaultdict

import numpy as np

PG = r"c:\jagr\vgscp\results\per_group_coverage.csv"
AB = r"c:\jagr\vgscp\results\calibration_ablation_4bb.csv"
OUT = (r"C:\Users\jayan\AppData\Local\Temp\claude\c--jagr-vgscp"
       r"\9ec1a810-3e09-41f2-b79c-798a80d2e27c\scratchpad\pergroup_table.tex")
NAME = {"resnet50_erm": "ResNet-50", "clip_vitb32": "CLIP",
        "dinov2_vitb14": "DINOv2", "vit_b16_in1k": "ViT-B/16"}
BB = list(NAME)
KEY = ("backbone", "dataset", "method", "train_seed", "calibration", "split_seed")

gate = {}
for r in csv.DictReader(open(AB)):
    if r["score"] == "APS" and float(r["rho_test"]) == 0.95:
        gate[tuple(r[k] for k in KEY)] = r["gate_status"]

rows = [r for r in csv.DictReader(open(PG))
        if gate.get(tuple(r[k] for k in KEY)) not in (None, "excluded")]
print(f"baris disimpan: {len(rows)} dari 2400")

L = [r"\begin{tabular}{@{}ll" + "c" * 4 + r"cc@{}}", r"\toprule",
     r" & & \multicolumn{4}{c}{coverage by group} & & \\",
     r"\cmidrule(lr){3-6}",
     r"Setting & rule & $g_0$ & $g_1$ & $g_2$ & $g_3$ & mean & min \\", r"\midrule"]

print(f"\n{'setting':22s} {'rule':10s} "
      + " ".join(f'g{g}'.rjust(7) for g in range(4)) + "   mean     min    gap")
for ds in ("waterbirds", "celeba"):
    L.append(r"\multicolumn{8}{@{}l}{\emph{"
             + ("Waterbirds" if ds == "waterbirds" else "CelebA") + r"}} \\")
    for bb in BB:
        first = True
        for pol, lab in (("marginal_split", "shared"), ("mondrian", "per-group")):
            sel = [r for r in rows if r["backbone"] == bb and r["dataset"] == ds
                   and r["calibration"] == pol]
            if not sel:
                continue
            g = [np.mean([float(r[f"cov_g{k}"]) for r in sel]) for k in range(4)]
            mn = np.mean([float(r["cov_min_recomputed"]) for r in sel])
            mean = float(np.mean(g))
            L.append(" & ".join([NAME[bb] if first else "", lab]
                                + [f"{x:.3f}" for x in g]
                                + [f"{mean:.3f}", f"{mn:.3f}"]) + r" \\")
            print(f"{NAME[bb]+'/'+ds:22s} {lab:10s} "
                  + " ".join(f"{x:7.3f}" for x in g)
                  + f"  {mean:.3f}  {mn:.3f}  {mean-mn:+.3f}")
            first = False
L += [r"\botrule", r"\end{tabular}"]
open(OUT, "w").write("\n".join(L))
print(f"\n-> {OUT}")

# the headline the appendix needs: every group near target under per-group thresholds
mon = [r for r in rows if r["calibration"] == "mondrian"]
allg = [float(r[f"cov_g{k}"]) for r in mon for k in range(4)]
per_setting_mean, per_setting_min = [], []
for ds in ("waterbirds", "celeba"):
    for bb in BB:
        sel = [r for r in mon if r["backbone"] == bb and r["dataset"] == ds]
        if not sel:
            continue
        per_setting_mean.append(np.mean([float(r[f"cov_g{k}"]) for r in sel for k in range(4)]))
        per_setting_min.append(np.mean([float(r["cov_min_recomputed"]) for r in sel]))
print(f"\nper-grup di bawah Mondrian, seluruh sel   : {min(allg):.3f}-{max(allg):.3f}")
print(f"rerata atas grup, per setting            : "
      f"{min(per_setting_mean):.4f}-{max(per_setting_mean):.4f}")
print(f"rerata minimum, per setting              : "
      f"{min(per_setting_min):.4f}-{max(per_setting_min):.4f}")
gaps = [a - b for a, b in zip(per_setting_mean, per_setting_min)]
print(f"selisih rerata-minus-minimum             : {min(gaps):.3f}-{max(gaps):.3f}")

# how often is an individual group actually below target, as opposed to the minimum being below it
below = np.mean([x < 0.90 for x in allg])
print(f"cakupan per-grup di bawah 0.90           : {100*below:.1f}% dari sel-grup")
