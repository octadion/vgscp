"""Generate the appendix tables from the released records.

The appendix currently describes results and points at files. Reviewers do not open CSVs, so those
results are effectively absent. Everything below already exists in the 116,000 evaluations; this
puts it on the page.

Six tables, written to a file the manuscript inputs:

  G1  the full calibration comparison: every setting x method x policy, for each conformity score
  G2  cross-group divergences, D_KS and W1, per setting and score
  G3  the correlation-strength sweep: worst-group coverage at each rho_test under both rules
  G4  the fine-tuning study in full, with across-seed spread
  G5  predicted groups: all three conditions, all three scores
  G6  variance decomposition: across-seed against across-split standard deviation

One table the audit asked for cannot be built: per-group coverage for all four groups. The records
store the mean over groups, the worst group's coverage and the range, but not each group's
coverage separately, so producing it would need the evaluation re-run rather than re-aggregated.
That is noted in the text rather than quietly skipped.
"""
import csv
import io
from collections import defaultdict

import numpy as np

R = r"c:\jagr\vgscp\results"
OUT = r"c:\jagr\vgscp\ACML_Journal___Robust_CP_Train_Study (1)\appendix-tables.tex"

NAME = {"resnet50_erm": "ResNet-50", "clip_vitb32": "CLIP",
        "dinov2_vitb14": "DINOv2", "vit_b16_in1k": "ViT-B/16"}
MNAME = {"erm": "ERM", "dfr": "DFR", "afr": "AFR",
         "balanced_subsample": "Bal.\\ sub.", "groupdro_ll": "GroupDRO-LL"}
PNAME = {"marginal_split": "shared", "mondrian": "per-group", "shift_robust": "shift-robust"}
BB = ["resnet50_erm", "clip_vitb32", "dinov2_vitb14", "vit_b16_in1k"]
DS = ["waterbirds", "celeba"]
SC = ["APS", "RAPS", "THR"]
MS = ["erm", "dfr", "afr", "balanced_subsample", "groupdro_ll"]

abl = list(csv.DictReader(open(f"{R}/calibration_ablation_4bb.csv")))
rep = list(csv.DictReader(open(f"{R}/representation_records.csv")))
pg = list(csv.DictReader(open(f"{R}/predicted_group_mondrian_4bb.csv")))
out = []


def w(rows, col):
    return np.mean([float(r[col]) for r in rows]) if rows else float("nan")


def f(x, d=3):
    return "---" if np.isnan(x) else f"{x:.{d}f}"


def head(title, label, spec, cols, small="footnotesize"):
    out.append(r"\begin{table}[htbp]" "\n" r"\centering")
    out.append(f"\\caption{{{title}}}")
    out.append(f"\\label{{{label}}}")
    out.append(f"\\{small}")
    out.append(r"\setlength{\tabcolsep}{3.5pt}")
    out.append(f"\\begin{{tabular}}{{{spec}}}")
    out.append(r"\toprule")
    out.append(" & ".join(cols) + r" \\")
    out.append(r"\midrule")


def foot():
    out.append(r"\botrule")
    out.append(r"\end{tabular}")
    out.append(r"\end{table}" "\n")


# ───────────────────────────────────────────────────────── G1 full calibration comparison
for sc in SC:
    head(f"Calibration comparison, {sc} at $\\rhocal=\\rhotest=0.95$. Worst-group coverage, "
         f"mean over groups, mean set size, worst-group set size and set-size disparity, "
         f"for every training method under each calibration rule. Excluded runs are omitted.",
         f"tab:grid{sc}", "llccccc",
         ["Setting", "Method", "rule", "wg cov", "mean cov", "wg size", "disparity"])
    for ds in DS:
        out.append(r"\multicolumn{7}{l}{\emph{" + ("Waterbirds" if ds == "waterbirds"
                                                   else "CelebA") + r"}} \\")
        for bb in BB:
            for m in MS:
                sel = [r for r in abl if r["backbone"] == bb and r["dataset"] == ds
                       and r["method"] == m and r["score"] == sc
                       and r["rho_test"] == "0.95" and r["gate_status"] != "excluded"]
                if not sel:
                    continue
                first = True
                for pol in ("marginal_split", "mondrian", "shift_robust"):
                    p = [r for r in sel if r["calibration"] == pol]
                    if not p:
                        continue
                    out.append(" & ".join([
                        NAME[bb] if first else "", MNAME[m] if first else "", PNAME[pol],
                        f(w(p, "worst_group_cov")), f(w(p, "mean_group_cov")),
                        f(w(p, "worst_group_set_size")), f(w(p, "set_size_disparity"))]) + r" \\")
                    first = False
    foot()

# ───────────────────────────────────────────────────────── G2 divergences
head(r"Cross-group conformity-score divergences under one shared threshold "
     r"($\rhocal=\rhotest=0.95$), averaged over the kept training methods. "
     r"$D_{\mathrm{KS}}$ is the quantity bounded in Eq.~\eqref{eq:ksbound}; $\Wone$ is "
     r"reported as a descriptive summary and enters no bound.",
     "tab:div", "l" + "cc" * 3,
     ["Setting"] + [c for sc in SC for c in
                    (f"$D_{{\\mathrm{{KS}}}}$ {sc}", f"$\\Wone$ {sc}")])
for ds in DS:
    out.append(r"\multicolumn{7}{l}{\emph{" + ("Waterbirds" if ds == "waterbirds"
                                               else "CelebA") + r"}} \\")
    for bb in BB:
        cells = []
        for sc in SC:
            sel = [r for r in abl if r["backbone"] == bb and r["dataset"] == ds
                   and r["score"] == sc and r["rho_test"] == "0.95"
                   and r["calibration"] == "marginal_split" and r["gate_status"] != "excluded"]
            cells += [f(w(sel, "div_ks_stat")), f(w(sel, "div_wasserstein1"))]
        out.append(" & ".join([NAME[bb]] + cells) + r" \\")
foot()

# ───────────────────────────────────────────────────────── G3 rho sweep
RHOS = ["0.95", "0.9", "0.8", "0.7", "0.6", "0.5"]
head(r"Worst-group coverage across the correlation-strength sweep (APS, calibration held "
     r"at $\rhocal=0.95$), averaged over the kept training methods. Each pair of rows is one "
     r"setting under the two calibration rules.",
     "tab:sweep", "ll" + "c" * len(RHOS),
     ["Setting", "rule"] + [f"$\\rhotest={r}$" for r in RHOS])
for ds in DS:
    out.append(r"\multicolumn{8}{l}{\emph{" + ("Waterbirds" if ds == "waterbirds"
                                               else "CelebA") + r"}} \\")
    for bb in BB:
        for pol in ("marginal_split", "mondrian"):
            cells = []
            for rt in RHOS:
                sel = [r for r in abl if r["backbone"] == bb and r["dataset"] == ds
                       and r["score"] == "APS" and r["rho_test"] == rt
                       and r["calibration"] == pol and r["gate_status"] != "excluded"]
                cells.append(f(w(sel, "worst_group_cov")))
            out.append(" & ".join([NAME[bb] if pol == "marginal_split" else "",
                                   PNAME[pol]] + cells) + r" \\")
foot()

# ───────────────────────────────────────────────────────── G4 fine-tuning in full
head(r"The fine-tuning study in full: the ResNet-50 retrained end-to-end under three "
     r"objectives with the head held fixed at plain ERM, for all three conformity scores "
     r"($\rhocal=\rhotest=0.95$). Values are means over five fine-tuning seeds; the two "
     r"coverage columns carry the across-seed standard deviation in brackets, and worst-group "
     r"accuracy is a per-objective mean that does not vary with the conformity score.",
     "tab:reprfull", "ll" + "ccc",
     ["Objective", "Score", "wg acc", "wg cov shared", "wg cov per-group"])
# The dataset is already a section header inside the table, so a Dataset column would be blank in
# every row. Objectives are named as the body's Table 3 names them, not as the raw record values.
RPNAME = {"erm": "ERM", "groupdro": "GroupDRO", "reweight": "Reweighting"}
reps = sorted({r["representation"] for r in rep})
for ds in DS:
    out.append(r"\multicolumn{5}{l}{\emph{" + ("Waterbirds" if ds == "waterbirds"
                                               else "CelebA") + r"}} \\")
    for rp in reps:
        for sc in SC:
            # Hold the head at plain ERM, exactly as Table 3 in the body does. Without this
            # the table averages over all three heads and disagrees with the body.
            base = [r for r in rep if r["dataset"] == ds and r["representation"] == rp
                    and r["score"] == sc and r["rho_test"] == "0.95"
                    and r["head"] == "erm"]
            if not base:
                continue
            acc = w(base, "worst_group_acc")
            cells = []
            for pol in ("marginal_split", "mondrian"):
                p = [r for r in base if r["calibration"] == pol]
                per_seed = [np.mean([float(x["worst_group_cov"]) for x in p
                                     if x["ft_seed"] == s])
                            for s in sorted({r["ft_seed"] for r in p})]
                cells.append(f"{np.mean(per_seed):.3f}\\,[{np.std(per_seed):.3f}]"
                             if per_seed else "---")
            out.append(" & ".join([RPNAME.get(rp, rp) if sc == SC[0] else "", sc, f(acc)]
                                  + cells) + r" \\")
foot()

# ───────────────────────────────────────────────────────── G5 predicted groups in full
COND = sorted({r["condition"] for r in pg})
head(r"Predicted-group calibration in full: worst-group coverage under all three "
     r"conditions and all three scores ($\rhocal=\rhotest=0.95$), averaged over the kept "
     r"training methods. Coverage is always scored against the true groups.",
     "tab:pgfull", "ll" + "c" * (len(COND) * 1) + "c",
     ["Setting", "Score"] + [c.replace("_", " ") for c in COND] + ["probe AUROC"])
for ds in DS:
    out.append(r"\multicolumn{" + str(2 + len(COND) + 1) + r"}{l}{\emph{"
               + ("Waterbirds" if ds == "waterbirds" else "CelebA") + r"}} \\")
    for bb in BB:
        for sc in SC:
            cells = []
            for c in COND:
                sel = [r for r in pg if r["backbone"] == bb and r["dataset"] == ds
                       and r["score"] == sc and r["condition"] == c
                       and r["gate_status"] != "excluded"]
                cells.append(f(w(sel, "worst_group_cov")))
            au = [r for r in pg if r["backbone"] == bb and r["dataset"] == ds
                  and r["gate_status"] != "excluded"]
            out.append(" & ".join([NAME[bb] if sc == SC[0] else "", sc] + cells
                                  + [f(w(au, "probe_auroc"))]) + r" \\")
foot()

# ───────────────────────────────────────────────────────── G6 variance decomposition
head(r"Where the variance sits, and how the two rules compare on it. For each setting, the standard deviation of worst-group "
     r"coverage across training seeds and across calibration splits (APS, per-group "
     r"thresholds for the first two columns, $\rhocal=\rhotest=0.95$), averaged over the "
     r"kept training methods; the last column repeats the split-to-split figure under one "
     r"shared threshold, so the two rules can be compared. "
     r"ERM and AFR are omitted: their solver ignores the seed, so their across-seed "
     r"standard deviation is exactly zero.",
     "tab:variance", "lccc",
     ["Setting", "SD across seeds", "SD across splits", "SD across splits"])
# Two columns sharing a heading is resolved only by the sub-line below, which does not visibly
# cover the first of them; spell the rule out in the header instead.
# The shared-threshold column exists because Section 5.5 compares the two rules' split-to-split
# stability and withdraws an earlier claim that Mondrian was the steadier of the two. Without it
# the marginal numbers appear in the prose and in no table, so no reader can check the comparison.
out[-2] = " & ".join(["Setting", "SD across seeds",
                          "SD across splits", "SD across splits"]) + r" \\"
out.insert(len(out) - 1, r" & (per-group) & (per-group) & (shared) \\")
for ds in DS:
    out.append(r"\multicolumn{4}{l}{\emph{" + ("Waterbirds" if ds == "waterbirds"
                                               else "CelebA") + r"}} \\")
    for bb in BB:
        seed_sd, split_sd, marg_sd = [], [], []
        for m in ("dfr", "balanced_subsample", "groupdro_ll"):
            for pol, sink in (("mondrian", split_sd), ("marginal_split", marg_sd)):
                sel = [r for r in abl if r["backbone"] == bb and r["dataset"] == ds
                       and r["method"] == m and r["score"] == "APS" and r["rho_test"] == "0.95"
                       and r["calibration"] == pol and r["gate_status"] != "excluded"]
                if not sel:
                    continue
                by_seed = defaultdict(list)
                for r_ in sel:
                    by_seed[r_["train_seed"]].append(float(r_["worst_group_cov"]))
                if pol == "mondrian":
                    means = [np.mean(v) for v in by_seed.values()]
                    if len(means) > 1:
                        seed_sd.append(np.std(means))
                sink.append(np.mean([np.std(v) for v in by_seed.values()]))
        out.append(" & ".join([NAME[bb], f(np.mean(seed_sd), 4) if seed_sd else "---",
                               f(np.mean(split_sd), 4) if split_sd else "---",
                               f(np.mean(marg_sd), 4) if marg_sd else "---"]) + r" \\")
foot()

# ───────────────────────────────────────────────────────── G7 mean coverage and disparity
# The reshaped grid tables carry worst-group coverage and set size only, so these two quantities
# -- one of which the Discussion quotes -- would otherwise appear in no table.
head(r"Mean coverage over groups and set-size disparity (APS, $\rhocal=\rhotest=0.95$), "
     r"averaged over the kept training methods. Disparity is the spread in mean set size "
     r"across groups; it is the cost the Discussion prices against the coverage gained by "
     r"giving each group its own threshold.",
     "tab:disparity", "l" + "cc" * 2,
     ["Setting", "mean cov (shared)", "mean cov (per-group)",
      "disparity (shared)", "disparity (per-group)"])
for ds in DS:
    out.append(r"\multicolumn{5}{l}{\emph{" + ("Waterbirds" if ds == "waterbirds"
                                               else "CelebA") + r"}} \\")
    for bb in BB:
        cells = []
        for col in ("mean_group_cov", "set_size_disparity"):
            for pol in ("marginal_split", "mondrian"):
                sel = [r for r in abl if r["backbone"] == bb and r["dataset"] == ds
                       and r["score"] == "APS" and r["rho_test"] == "0.95"
                       and r["calibration"] == pol and r["gate_status"] != "excluded"]
                cells.append(f(w(sel, col)))
        out.append(" & ".join([NAME[bb]] + cells) + r" \\")
foot()

io.open(OUT, "w", encoding="utf-8", newline="").write("\n".join(out) + "\n")
n_tab = sum(1 for l in out if l.startswith(r"\begin{table}"))
n_row = sum(1 for l in out if l.endswith(r"\\") and "multicolumn" not in l
            and "toprule" not in l)
print(f"-> {OUT}")
print(f"   {n_tab} tabel, ~{n_row} baris data, {len(out)} baris LaTeX")
