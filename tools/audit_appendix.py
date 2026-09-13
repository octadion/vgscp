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
# Read which grids exist rather than assuming any. The per-method grids are no longer printed --
# the cross-score spread table below carries their claim and is checked cell by cell -- so an
# empty list is the expected state, not a failure.
GRIDS = [sc for sc in ("APS", "RAPS", "THR") if f"tab:grid{sc}" in tex]
print(f"=== grid: {', '.join(GRIDS) if GRIDS else 'tidak dicetak'} ===")
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
        check("div", num(c[1 + k]), mean(sel, "div_ks_stat"), f"{c[0]}/{ds} KS {sc}")
        if sc == "APS":
            # W1 is now given for the headline score alone, in the last column
            check("div", num(c[4]), mean(sel, "div_wasserstein1"), f"{c[0]}/{ds} W1 APS")

# ---------------------------------------------------------------- the sweep, now a prose bound
# The sweep table is gone; Appendix B states instead that no setting departs from its
# rho_test=0.95 worst-group coverage by more than a stated amount. That bound is a claim about the
# records like any table cell, so it is re-derived here.
print("=== sweep bound ===", end=" ")
RHOS = ["0.95", "0.9", "0.8", "0.7", "0.6", "0.5"]
dev = 0.0
for ds in ("waterbirds", "celeba"):
    for bb in NAME.values():
        for pol in ("marginal_split", "mondrian"):
            vals = []
            for rt in RHOS:
                sel = [r for r in abl if r["backbone"] == bb and r["dataset"] == ds
                       and r["score"] == "APS" and r["rho_test"] == rt
                       and r["calibration"] == pol]
                m_ = mean(sel, "worst_group_cov")
                vals.append(round(m_, 3) if m_ is not None else None)
            if vals[0] is None:
                continue
            dev = max([dev] + [abs(v - vals[0]) for v in vals[1:] if v is not None])
prose = io.open(f"{D}/sn-article.tex", encoding="utf-8").read()
# Anchored on the sentence itself: an unanchored "by more than $x$" matched the first such phrase in
# the paper, which after a later edit was a different claim in Section 5.5.
stated = float(re.search(r"departs from its \$\\rhotest=0\.95\$ value by more than\s+\$(\d\.\d+)\$",
                         re.sub(r"\s+", " ", prose)).group(1))
print(f"tertulis {stated:.3f}, dihitung {dev:.3f}")
if abs(stated - dev) > 0.0006:
    bad.append(f"  sweep: batas tertulis {stated:.3f}, dihitung {dev:.3f}")

# ---------------------------------------------------------------- how tight the KS bound is
# Section 5.1 quotes the shortfall as a fraction of its bound, per dataset. An audit called these
# ranges wrong after rebuilding them from three different tables -- the per-group table's re-run,
# the divergence table's method average, and a constant pi taken from the ERM row -- which is not
# the population the figure uses. Recompute them the way the figure does: keep the paired runs,
# average coverage, D_KS and pi per setting, then take the ratio.
print("=== ketatnya batas KS ===", end=" ")
paired = defaultdict(dict)
for r in csv.DictReader(open(f"{R}/calibration_ablation_4bb.csv")):
    if r["score"] == "APS" and r["rho_test"] == "0.95" and r["gate_status"] == "kept":
        paired[(r["backbone"], r["dataset"], r["method"],
                r["train_seed"], r["split_seed"])][r["calibration"]] = r
cells = defaultdict(list)
for d in paired.values():
    if "marginal_split" in d and "mondrian" in d:
        m = d["marginal_split"]
        cells[(m["dataset"], m["backbone"])].append(
            (float(m["worst_group_cov"]), float(m["div_ks_stat"]),
             int(m["n_cal_worst_group"]) / int(m["n_eval"])))
ratio, nviol = defaultdict(list), 0
for (ds, bb), v in cells.items():
    cov, kstat, pi_ = np.array(v).mean(axis=0)
    short_, bound_ = 0.90 - cov, (1 - pi_) * kstat
    nviol += short_ > bound_ + 1e-9
    ratio[ds].append(short_ / bound_)
got = re.search(r"shortfall reaches \$(\d\.\d+)\$--\$(\d\.\d+)\$ of the bound on Waterbirds "
                r"but only \$(\d\.\d+)\$--\$(\d\.\d+)\$ on CelebA", re.sub(r"\s+", " ", prose))
want = [min(ratio["waterbirds"]), max(ratio["waterbirds"]),
        min(ratio["celeba"]), max(ratio["celeba"])]
print(f"tertulis {got and got.groups()}, dihitung {[f'{x:.3f}' for x in want]}, "
      f"{nviol} pelanggaran")
if got is None:
    bad.append("  ketat: kalimat rentang di 5.1 tidak ditemukan")
else:
    for g, w, lab in zip(got.groups(), want, ("WB min", "WB max", "CelebA min", "CelebA max")):
        if abs(float(g) - round(w, 2)) > 0.0051:
            bad.append(f"  ketat: {lab} tertulis {g}, dihitung {w:.3f}")
if nviol:
    bad.append(f"  ketat: {nviol} setting melanggar batas; 5.1 mengklaim tidak ada")

# ---------------------------------------------------------------- per-group coverage + disparity
# tab:pergroupcov lives in the manuscript rather than the generated file, and it now carries the
# set-size disparity column that had a table of its own. Both are checked against the records.
print("=== tab:pergroupcov ===", end=" ")
_pgc_blk = prose[prose.index(r"\label{tab:pergroupcov}") - 2000:
                 prose.index(r"\label{tab:pergroupcov}") + 3000]
_pgc_blk = _pgc_blk[_pgc_blk.index(r"\begin{tabular}"):_pgc_blk.index(r"\end{tabular}")]
_pgc_rows, _cur = [], None
for ln in _pgc_blk.split("\\\\"):
    ln = ln.strip()
    if r"\emph{Waterbirds}" in ln:
        _ds = "waterbirds"
        continue
    if r"\emph{CelebA}" in ln:
        _ds = "celeba"
        continue
    cells = [x.strip() for x in ln.split("&")]
    if len(cells) != 9 or not any(re.search(r"\d\.\d", x) for x in cells):
        continue
    if cells[0]:
        _cur = cells[0]
    _pgc_rows.append((_ds, _cur, cells))
print(f"{len(_pgc_rows)} baris")
assert len(_pgc_rows) == 16, f"tab:pergroupcov: {len(_pgc_rows)} baris, harus 16"
for _ds, _cur, c in _pgc_rows:
    pol = "marginal_split" if "shared" in c[1] else "mondrian"
    sel = [r for r in abl if r["backbone"] == NAME[_cur] and r["dataset"] == _ds
           and r["score"] == "APS" and r["rho_test"] == "0.95"
           and r["calibration"] == pol]
    check("pergroupcov", num(c[6]), mean(sel, "mean_group_cov"),
          f"{_cur}/{_ds} mean {pol}")
    check("pergroupcov", num(c[8]), mean(sel, "set_size_disparity"),
          f"{_cur}/{_ds} disparity {pol}")

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

# ---------------------------------------------------------------- fine-tuning study
# Generated, but never re-derived here until now: the generator agreeing with itself proved
# nothing, and dropping the RAPS rows shifted every row of this table.
print("=== tab:reprfull ===", end=" ")
rep = [r for r in csv.DictReader(open(f"{R}/representation_records.csv"))]
RPNAME = {"ERM": "erm", "GroupDRO": "groupdro", "Reweighting": "reweight"}
_rows, _cur = datarows(table("tab:reprfull")), None
print(f"{len(_rows)} baris")
assert _rows, "TIDAK ADA BARIS TERBACA: tab:reprfull"
for ds, c in _rows:
    if c[0]:
        _cur = c[0]
    sc = c[1]
    base = [r for r in rep if r["dataset"] == ds and r["representation"] == RPNAME[_cur]
            and r["score"] == sc and r["rho_test"] == "0.95" and r["head"] == "erm"]
    check("reprfull", num(c[2]), mean(base, "worst_group_acc"), f"{_cur}/{ds}/{sc} wg acc")
    for k, pol in enumerate(("marginal_split", "mondrian")):
        p = [r for r in base if r["calibration"] == pol]
        by = defaultdict(list)
        for r_ in p:
            by[r_["ft_seed"]].append(float(r_["worst_group_cov"]))
        per_seed = [np.mean(v) for v in by.values()]
        check("reprfull", num(c[3 + k]), np.mean(per_seed) if per_seed else None,
              f"{_cur}/{ds}/{sc} cov {pol}")
        # the bracketed figure is the across-seed SD, so it needs its own parse
        got_sd = re.search(r"\[(\d\.\d+)\]", c[3 + k])
        want_sd = np.std(per_seed) if per_seed else None
        if got_sd is None or want_sd is None or abs(float(got_sd.group(1)) - want_sd) > 0.0006:
            bad.append(f"  reprfull: {_cur}/{ds}/{sc} SD {pol} tertulis {got_sd and got_sd.group(1)}"
                       f", dihitung {want_sd}")

# ---------------------------------------------------------------- predicted groups
print("=== tab:pgfull ===", end=" ")
pg = [r for r in csv.DictReader(open(f"{R}/predicted_group_mondrian_4bb.csv"))
      if r["gate_status"] != "excluded"]
COND = sorted({r["condition"] for r in pg})
_rows, _cur = datarows(table("tab:pgfull")), None
print(f"{len(_rows)} baris")
assert _rows, "TIDAK ADA BARIS TERBACA: tab:pgfull"
for ds, c in _rows:
    if c[0]:
        _cur = c[0]
    sc = c[1]
    for k, cond in enumerate(COND):
        sel = [r for r in pg if r["backbone"] == NAME[_cur] and r["dataset"] == ds
               and r["score"] == sc and r["condition"] == cond]
        check("pgfull", num(c[2 + k]), mean(sel, "worst_group_cov"),
              f"{_cur}/{ds}/{sc} {cond}")
    au = [r for r in pg if r["backbone"] == NAME[_cur] and r["dataset"] == ds]
    check("pgfull", num(c[2 + len(COND)]), mean(au, "probe_auroc"), f"{_cur}/{ds} AUROC")

# ---------------------------------------------------------------- body Tables 5 and 6
# These two come from the five-seed main grid, not the three-seed calibration ablation behind the
# appendix, and nothing checked them. Both ERM columns of Table 5 reproduce from either file --
# ERM and AFR are fitted with L-BFGS, whose solver ignores the seed, so their means are identical
# over three seeds or five -- which is exactly why reading them from the ablation looks right until
# the robust ranges are checked. Two sessions have now spent time on that; this ends it.
grid = [r for r in csv.DictReader(open(f"{R}/grid_records.csv"))
        if r["score"] == "APS" and r["rho_test"] == "0.95"
        and r["calibration"] == "marginal_split" and r["gate_status"] != "excluded"]
assert {r["train_seed"] for r in grid} == {"0", "1", "2", "3", "4"}, "grid bukan lima seed"
ROB = ["dfr", "afr", "balanced_subsample", "groupdro_ll"]


def _blk(label, src):
    # anchor on the \label, not on the first \ref of the same name earlier in the prose
    i = src.index("\\label{" + label + "}")
    return src[src.rindex(r"\begin{table}", 0, i):src.index(r"\end{table}", i)]


print("=== Tabel 5 (tab:h1) ===", end=" ")
rows5 = datarows(_blk("tab:h1", prose))
print(f"{len(rows5)} baris")
for ds, c in rows5:
    bb = NAME[c[0].replace("$^{\\dagger}$", "").strip()]
    base = [r for r in grid if r["dataset"] == ds and r["backbone"] == bb]
    if num(c[1]) is not None:
        e = [r for r in base if r["method"] == "erm"]
        check("T5", num(c[1]), mean(e, "div_wasserstein1"), f"{c[0]}/{ds} D_ERM")
        check("T5", num(c[2]), mean(e, "base_top1"), f"{c[0]}/{ds} acc_ERM")
    dv = [mean([r for r in base if r["method"] == m_], "div_wasserstein1") for m_ in ROB]
    av = [mean([r for r in base if r["method"] == m_], "base_top1") for m_ in ROB]
    dv = [x for x in dv if x is not None]
    av = [x for x in av if x is not None]
    for col, lo, hi in ((3, min(dv), max(dv)), (4, min(av), max(av))):
        got = re.findall(r"(\d\.\d+)", c[col])
        if len(got) != 2:
            bad.append(f"  T5 {c[0]}/{ds} kolom {col}: tidak terbaca sebagai rentang")
            continue
        check("T5", float(got[0]), lo, f"{c[0]}/{ds} kolom {col} min")
        check("T5", float(got[1]), hi, f"{c[0]}/{ds} kolom {col} max")

print("=== Tabel 6 (tab:h2) ===", end=" ")
MINV = {v: k for k, v in MNAME.items()}
rows6 = datarows(_blk("tab:h2", prose))
print(f"{len(rows6)} baris")
for ds, c in rows6:
    bb = NAME[c[0]]
    base = [r for r in grid if r["dataset"] == ds and r["backbone"] == bb]
    accs = {m_: mean([r for r in base if r["method"] == m_], "worst_group_acc")
            for m_ in MNAME.values()}
    accs = {k: v for k, v in accs.items() if v is not None}
    gaps = {k: 0.90 - mean([r for r in base if r["method"] == k], "worst_group_cov")
            for k in accs}
    ab, sb = max(accs, key=accs.get), min(gaps, key=gaps.get)
    for got, want, lab in ((c[1], MINV[ab], "acc-best"), (c[4], MINV[sb], "shortfall-best")):
        if got.replace("\\ ", " ") != want.replace("\\ ", " "):
            bad.append(f"  T6 {c[0]}/{ds} {lab}: tertulis {got}, dihitung {want}")
    check("T6", num(c[2]), accs[ab], f"{c[0]}/{ds} wg acc")
    check("T6", num(c[3]), gaps[ab], f"{c[0]}/{ds} covgap acc-best")
    check("T6", num(c[5]), gaps[sb], f"{c[0]}/{ds} covgap shortfall-best")

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
