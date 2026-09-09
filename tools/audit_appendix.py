"""Re-derive every appendix table cell from the records, independently of the generator.

The generator's output agrees with the generator by construction, so re-running it proves nothing.
This parses the rendered LaTeX back into numbers and recomputes each cell from the CSV along a
separate code path, so a wrong column, a shifted row, or a heading that does not describe its
column shows up as a mismatch rather than as agreement.

Reads the .tex; does not write anything.
"""
import csv
import io
import re
from collections import defaultdict

import numpy as np

D = r"c:\jagr\vgscp\ACML_Journal___Robust_CP_Train_Study (1)"
R = r"c:\jagr\vgscp\results"
NAME = {"ResNet-50": "resnet50_erm", "CLIP": "clip_vitb32",
        "DINOv2": "dinov2_vitb14", "ViT-B/16": "vit_b16_in1k"}
MNAME = {"ERM": "erm", "DFR": "dfr", "AFR": "afr",
         "Bal. sub.": "balanced_subsample", "GroupDRO-LL": "groupdro_ll"}
abl = [r for r in csv.DictReader(open(f"{R}/calibration_ablation_4bb.csv"))
       if r["gate_status"] != "excluded"]
tex = io.open(f"{D}/appendix-tables.tex", encoding="utf-8").read()
bad = []


def table(label):
    i = tex.index(label)
    st = max(tex.rfind(r"\begin{table}", 0, i), tex.rfind(r"\begin{longtable}", 0, i))
    en = tex.find(r"\end{table}", i)
    en2 = tex.find(r"\end{longtable}", i)
    en = min(x for x in (en, en2) if x > 0)
    return tex[st:en]


def datarows(block):
    out, ds = [], None
    for ln in block.split("\\\\"):
        # a row may end "\\*" to forbid a page break after it, and may be followed by
        # \nopagebreak; neither is part of the next row's first cell
        ln = ln.lstrip("*").replace(r"\nopagebreak", " ").strip()
        if r"\emph{Waterbirds}" in ln:
            ds = "waterbirds"
            continue
        if r"\emph{CelebA}" in ln:
            ds = "celeba"
            continue
        if "&" not in ln or "rule" in ln or "endhead" in ln or "endfoot" in ln \
                or "caption" in ln or "multicolumn" in ln:
            continue
        cells = [c.strip() for c in ln.split("&")]
        if not any(re.search(r"\d\.\d", c) for c in cells):
            continue
        out.append((ds, cells))
    return out


def num(c):
    m = re.search(r"(\d+\.\d+)", c.replace("\\,", " "))
    return float(m.group(1)) if m else None


def mean(sel, col):
    v = [float(r[col]) for r in sel]
    return np.mean(v) if v else None


def check(label, got, want, where):
    if want is None or got is None:
        bad.append(f"  {label}: {where} tidak dapat dihitung (got={got} want={want})")
    elif abs(got - want) > 0.0006:
        bad.append(f"  {label}: {where} tertulis {got:.3f}, dihitung {want:.3f}")


# ---------------------------------------------------------------- grids
# read which grids exist rather than assuming three: RAPS is summarised, not tabulated in full
GRIDS = [sc for sc in ("APS", "RAPS", "THR") if f"tab:grid{sc}" in tex]
print(f"=== grid: {', '.join(GRIDS)} ===")
assert GRIDS, "tidak ada tabel grid sama sekali"
for sc in GRIDS:
    blk = table(f"tab:grid{sc}")
    rows, cur_bb = datarows(blk), None
    n = 0
    for ds, c in rows:
        if c[0]:
            cur_bb = c[0]
        bb, meth = NAME[cur_bb], MNAME[c[1].replace("\\ ", " ")]
        for k, pol in enumerate(("marginal_split", "mondrian", "shift_robust")):
            sel = [r for r in abl if r["backbone"] == bb and r["dataset"] == ds
                   and r["method"] == meth and r["score"] == sc
                   and r["rho_test"] == "0.95" and r["calibration"] == pol]
            check(f"grid{sc}", num(c[2 + 2 * k]), mean(sel, "worst_group_cov"),
                  f"{cur_bb}/{ds}/{c[1]} cov {pol}")
            check(f"grid{sc}", num(c[3 + 2 * k]), mean(sel, "worst_group_set_size"),
                  f"{cur_bb}/{ds}/{c[1]} size {pol}")
            n += 2
    print(f"  {sc}: {len(rows)} baris, {n} sel diperiksa")

# ---------------------------------------------------------------- divergences
print("=== tab:div ===", end=" ")
_rows = datarows(table("tab:div"))
print(f"{len(_rows)} baris")
assert _rows, "TIDAK ADA BARIS TERBACA: tab:div"
for ds, c in datarows(table("tab:div")):
    bb = NAME[c[0]]
    for k, sc in enumerate(("APS", "RAPS", "THR")):
        sel = [r for r in abl if r["backbone"] == bb and r["dataset"] == ds
               and r["score"] == sc and r["rho_test"] == "0.95"
               and r["calibration"] == "marginal_split"]
        check("div", num(c[1 + 2 * k]), mean(sel, "div_ks_stat"), f"{c[0]}/{ds} KS {sc}")
        check("div", num(c[2 + 2 * k]), mean(sel, "div_wasserstein1"), f"{c[0]}/{ds} W1 {sc}")

# ---------------------------------------------------------------- sweep
print("=== tab:sweep ===", end=" ")
_rows = datarows(table("tab:sweep"))
print(f"{len(_rows)} baris")
assert _rows, "TIDAK ADA BARIS TERBACA: tab:sweep"
RHOS = ["0.95", "0.9", "0.8", "0.7", "0.6", "0.5"]
cur = None
for ds, c in datarows(table("tab:sweep")):
    if c[0]:
        cur = c[0]
    pol = "marginal_split" if "shared" in c[1] else "mondrian"
    for k, rt in enumerate(RHOS):
        sel = [r for r in abl if r["backbone"] == NAME[cur] and r["dataset"] == ds
               and r["score"] == "APS" and r["rho_test"] == rt and r["calibration"] == pol]
        check("sweep", num(c[2 + k]), mean(sel, "worst_group_cov"), f"{cur}/{ds} {pol} rho={rt}")

# ---------------------------------------------------------------- disparity
print("=== tab:disparity ===", end=" ")
_rows = datarows(table("tab:disparity"))
print(f"{len(_rows)} baris")
assert _rows, "TIDAK ADA BARIS TERBACA: tab:disparity"
for ds, c in datarows(table("tab:disparity")):
    bb = NAME[c[0]]
    for k, (col, pol) in enumerate((("mean_group_cov", "marginal_split"),
                                    ("mean_group_cov", "mondrian"),
                                    ("set_size_disparity", "marginal_split"),
                                    ("set_size_disparity", "mondrian"))):
        sel = [r for r in abl if r["backbone"] == bb and r["dataset"] == ds
               and r["score"] == "APS" and r["rho_test"] == "0.95"
               and r["calibration"] == pol]
        check("disparity", num(c[1 + k]), mean(sel, col), f"{c[0]}/{ds} {col} {pol}")

# ---------------------------------------------------------------- variance
print("=== tab:variance ===", end=" ")
_rows = datarows(table("tab:variance"))
print(f"{len(_rows)} baris")
assert _rows, "TIDAK ADA BARIS TERBACA: tab:variance"
for ds, c in datarows(table("tab:variance")):
    bb = NAME[c[0]]
    seed_sd, split_sd, marg_sd = [], [], []
    for m in ("dfr", "balanced_subsample", "groupdro_ll"):
        for pol, sink in (("mondrian", split_sd), ("marginal_split", marg_sd)):
            sel = [r for r in abl if r["backbone"] == bb and r["dataset"] == ds
                   and r["method"] == m and r["score"] == "APS"
                   and r["rho_test"] == "0.95" and r["calibration"] == pol]
            if not sel:
                continue
            by = defaultdict(list)
            for r_ in sel:
                by[r_["train_seed"]].append(float(r_["worst_group_cov"]))
            if pol == "mondrian":
                mm = [np.mean(v) for v in by.values()]
                if len(mm) > 1:
                    seed_sd.append(np.std(mm))
            sink.append(np.mean([np.std(v) for v in by.values()]))
    for got, want, w in ((num(c[1]), np.mean(seed_sd), "SD seeds"),
                         (num(c[2]), np.mean(split_sd), "SD splits per-group"),
                         (num(c[3]), np.mean(marg_sd), "SD splits shared")):
        if got is None or abs(got - want) > 0.00006:
            bad.append(f"  variance: {c[0]}/{ds} {w} tertulis {got}, dihitung {want:.4f}")

# ---------------------------------------------------------------- cross-score spread
# This table now carries the claim the RAPS grid used to, so its 48 cells are verified too.
print("=== tab:scorespread ===", end=" ")
_rows = datarows(table("tab:scorespread"))
print(f"{len(_rows)} baris")
assert _rows, "TIDAK ADA BARIS TERBACA: tab:scorespread"
for ds, c in _rows:
    bb = NAME[c[0]]
    k = 1
    for sc in ("APS", "RAPS", "THR"):
        for pol in ("marginal_split", "mondrian"):
            per = []
            for meth in ("erm", "dfr", "afr", "balanced_subsample", "groupdro_ll"):
                sel = [r for r in abl if r["backbone"] == bb and r["dataset"] == ds
                       and r["method"] == meth and r["score"] == sc
                       and r["rho_test"] == "0.95" and r["calibration"] == pol]
                if sel:
                    per.append(mean(sel, "worst_group_cov"))
            want = (max(per) - min(per)) if len(per) > 1 else None
            check("scorespread", num(c[k]), want, f"{c[0]}/{ds} {sc} {pol}")
            k += 1

print()
if bad:
    print(f"{len(bad)} KETIDAKCOCOKAN:")
    for b in bad[:40]:
        print(b)
else:
    print("semua sel tabel appendix cocok dengan rekaman")
