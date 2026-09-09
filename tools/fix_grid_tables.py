"""Reshape the three grid tables and let them break across pages.

They were 97 data rows each -- one row per (setting, method, calibration rule) -- inside a `table`
environment, which cannot break across a page. So they ran off the bottom.

Two fixes. The rules move from rows to column groups, which turns 97 rows into 40: one row per
(setting, method), with worst-group coverage and worst-group set size under each of the three
rules. That is both shorter and easier to read, since the comparison the table exists to support
now sits side by side on one line instead of spread over three. And the environment becomes
`longtable`, so if a table still runs past a page it breaks with its header repeated rather than
overflowing.
"""
import csv
import io
import re

import numpy as np

R = r"c:\jagr\vgscp\results"
TEX = r"c:\jagr\vgscp\ACML_Journal___Robust_CP_Train_Study (1)\sn-article.tex"
OUT = r"c:\jagr\vgscp\ACML_Journal___Robust_CP_Train_Study (1)\appendix-tables.tex"

# ---- preamble needs longtable
s = io.open(TEX, encoding="utf-8", newline="").read()
if "longtable" not in s:
    s = s.replace("\\usepackage{booktabs}%", "\\usepackage{booktabs}%\n\\usepackage{longtable}%", 1)
    io.open(TEX, "w", encoding="utf-8", newline="").write(s)
    print("ok  \\usepackage{longtable} ditambahkan")

NAME = {"resnet50_erm": "ResNet-50", "clip_vitb32": "CLIP",
        "dinov2_vitb14": "DINOv2", "vit_b16_in1k": "ViT-B/16"}
MNAME = {"erm": "ERM", "dfr": "DFR", "afr": "AFR",
         "balanced_subsample": "Bal.\\ sub.", "groupdro_ll": "GroupDRO-LL"}
BB = ["resnet50_erm", "clip_vitb32", "dinov2_vitb14", "vit_b16_in1k"]
DS = ["waterbirds", "celeba"]
# RAPS is summarised rather than tabulated in full: see the cross-score spread table
SC = ["APS", "THR"]
SC_ALL = ["APS", "RAPS", "THR"]
MS = ["erm", "dfr", "afr", "balanced_subsample", "groupdro_ll"]
POL = ["marginal_split", "mondrian", "shift_robust"]

abl = list(csv.DictReader(open(f"{R}/calibration_ablation_4bb.csv")))


def m(rows, col):
    return np.mean([float(r[col]) for r in rows]) if rows else float("nan")


def f(x):
    return "---" if np.isnan(x) else f"{x:.3f}"


new_blocks = []
for sc in SC:
    L = []
    # Eight columns of numbers at the default 6pt padding come to 93% of the text block, which is
    # inside the margin but leaves nothing for the difference between these Times metrics and the
    # class's actual face. Tightening the padding alone takes it to 84%; the group keeps the change
    # from leaking into the tables that follow, and the caption stays at its normal size.
    L.append(r"\begingroup\setlength{\tabcolsep}{4pt}")
    L.append(r"\begin{longtable}{@{}ll" + "cc" * 3 + r"@{}}")
    warn = (r" Under the shift-robust rule this table contains degenerate cells: seven "
            r"(setting, method) pairs sit at a mean worst-group set size of $2.000$ with "
            r"coverage $1.000$, which with two classes is the set holding both labels and "
            r"covers by construction. See the discussion in this appendix before reading that "
            r"column as a result." if sc == "THR" else "")
    L.append(r"\caption{Calibration comparison, " + sc + r" at $\rhocal=\rhotest=0.95$: "
             r"worst-group coverage (cov) and worst-group set size (size) under each of the "
             r"three calibration rules, for every training method. Means over the kept runs; "
             r"methods excluded by their accuracy floor are omitted rather than shown."
             + warn + r"}\label{tab:grid" + sc
             + r"} \\")
    L.append(r"\toprule")
    hdr = (r" & & \multicolumn{2}{c}{shared} & \multicolumn{2}{c}{per-group} & "
           r"\multicolumn{2}{c}{shift-robust} \\")
    sub = (r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}" "\n"
           r"Setting & Method & cov & size & cov & size & cov & size \\")
    L += [hdr, sub, r"\midrule", r"\endfirsthead",
          r"\toprule", hdr, sub, r"\midrule", r"\endhead",
          r"\bottomrule", r"\endfoot"]
    for ds in DS:
        # A panel label can end a page while its rows begin the next one, leaving a reader
        # mid-table unable to tell which dataset a row belongs to. \nopagebreak keeps the label
        # attached to the rows that follow it.
        L.append(r"\multicolumn{8}{@{}l}{\emph{"
                 + ("Waterbirds" if ds == "waterbirds" else "CelebA")
                 + r"}} \\*" "\n" r"\nopagebreak")
        for bb in BB:
            first = True
            for meth in MS:
                cells = []
                any_row = False
                for pol in POL:
                    sel = [r for r in abl if r["backbone"] == bb and r["dataset"] == ds
                           and r["method"] == meth and r["score"] == sc
                           and r["rho_test"] == "0.95" and r["calibration"] == pol
                           and r["gate_status"] != "excluded"]
                    if sel:
                        any_row = True
                    cells += [f(m(sel, "worst_group_cov")),
                              f(m(sel, "worst_group_set_size"))]
                if not any_row:
                    continue
                L.append(" & ".join([NAME[bb] if first else "", MNAME[meth]] + cells) + r" \\")
                first = False
    L.append(r"\end{longtable}\endgroup" "\n")
    new_blocks.append("\n".join(L))


# ── cross-score spread: the claim the RAPS grid used to carry, for all three scores ──────────
MS_ALL = MS
sp_rows = []
for ds in DS:
    sp_rows.append(r"\multicolumn{7}{@{}l}{\emph{"
                   + ("Waterbirds" if ds == "waterbirds" else "CelebA") + r"}} \\")
    for bb in BB:
        cells = []
        for sc in SC_ALL:
            for pol in ("marginal_split", "mondrian"):
                per = []
                for meth in MS_ALL:
                    sel = [r for r in abl if r["backbone"] == bb and r["dataset"] == ds
                           and r["method"] == meth and r["score"] == sc
                           and r["rho_test"] == "0.95" and r["calibration"] == pol
                           and r["gate_status"] != "excluded"]
                    if sel:
                        per.append(m(sel, "worst_group_cov"))
                cells.append(f"{max(per) - min(per):.3f}" if len(per) > 1 else "---")
        sp_rows.append(" & ".join([NAME[bb]] + cells) + r" \\")

summary_block = "\n".join(
    [r"\begin{table}[H]", r"\centering",
     r"\caption{The dissociation across all three conformity scores "
     r"($\rhocal=\rhotest=0.95$): the spread in worst-group coverage across the kept training "
     r"methods, under one shared threshold and under per-group thresholds. Per-group spreads run "
     r"$0.001$--$0.029$ against $0.002$--$0.478$ for the shared rule, and are the smaller of the "
     r"two in $22$ of the $24$ pairs; the exceptions are ViT-B/16 on CelebA under APS and RAPS, "
     r"where the shared threshold already leaves almost nothing to recover.}",
     r"\label{tab:scorespread}", r"\footnotesize",
     r"\setlength{\tabcolsep}{4pt}",
     r"\begin{tabular}{@{}l" + "cc" * 3 + r"@{}}", r"\toprule",
     r" & \multicolumn{2}{c}{APS} & \multicolumn{2}{c}{RAPS} & \multicolumn{2}{c}{THR} \\",
     r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
     r"Setting & shared & per-gr. & shared & per-gr. & shared & per-gr. \\", r"\midrule"]
    + sp_rows + [r"\botrule", r"\end{tabular}", r"\end{table}", ""])
new_blocks.insert(0, summary_block)

# ---- splice the new blocks in place of the old three
old = io.open(OUT, encoding="utf-8", newline="").read()
tail_start = old.index(r"\begin{table}[H]", old.index("tab:gridTHR"))
tail = old[tail_start:]
io.open(OUT, "w", encoding="utf-8", newline="").write("\n".join(new_blocks) + "\n" + tail)

s2 = io.open(OUT, encoding="utf-8").read()
print(f"\nbaris berkas: {len(old.splitlines())} -> {len(s2.splitlines())}")
for lab in re.findall(r"\\label\{(tab:[^}]+)\}", s2):
    print(f"  {lab}")
for env, spec, body in re.findall(
        r"\\begin\{(longtable|tabular)\}(?:\[[^\]]*\])?\{([^}]*)\}(.*?)\\end\{\1\}", s2, re.S):
    n = len(re.findall(r"[lcr]", spec))
    rows = [l for l in body.splitlines()
            if "&" in l and "multicolumn" not in l and "cmidrule" not in l]
    widths = {l.count("&") + 1 for l in rows}
    print(f"  {env:9s} spec {n} kolom | baris {sorted(widths)} | {len(rows)} baris data "
          f"{'ok' if widths <= {n} else 'BEDA'}")
print(f"brace {s2.count('{')}/{s2.count('}')} | $ {'genap' if s2.count('$') % 2 == 0 else 'GANJIL'}")
