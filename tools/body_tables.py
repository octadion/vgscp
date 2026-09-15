"""Re-derive the body tables the appendix auditor does not touch.

`audit_appendix.py` covers the generated appendix tables. The tables inside `sn-article.tex` were
guarded by nothing, and an audit found three stale cells there in one pass: Table 2's CelebA
ResNet-50 per-group mean, and two cells of Table A1. Every one of them had survived a full run of
the eleven checkers, because no checker read those tables.

Four tables, each recomputed from the records along a path independent of whatever produced the
number in the manuscript:

  tab:c1        Table 2, the calibration comparison   -> interval_tables.table2()
  tab:c2        Table 4, worst-group set size         -> the ablation records + reanalysis
  tab:argmin    Table A1, where the worst group lands -> the ablation records
  tab:predgroup Table 7, predicted groups             -> the predicted-group records

Reads the .tex; writes nothing.
"""
import collections
import csv
import io
import re
import sys

import numpy as np

sys.path.insert(0, r"c:\jagr\vgscp")
from study_robust_train.interval_tables import table2
from study_robust_train.reanalysis import arm_means, correlation_with_ci

# An optional directory argument lets the checker run against a copy, which is how its own checks
# are mutation-tested without touching the manuscript.
D = sys.argv[1] if len(sys.argv) > 1 else r"c:\jagr\vgscp\ACML_Journal___Robust_CP_Train_Study (1)"
R = r"c:\jagr\vgscp\results"
NAME = {"ResNet-50": "resnet50_erm", "CLIP": "clip_vitb32",
        "DINOv2": "dinov2_vitb14", "ViT-B/16": "vit_b16_in1k"}
tex = io.open(f"{D}/sn-article.tex", encoding="utf-8").read()
bad = []


def table(label):
    """The tabular that follows \\label{...}: these tables put caption and label before it.

    Anchored on the label definition, not on the bare name: every \\ref in the prose contains the
    name too, and the first of those sits pages earlier, in front of a different table.
    """
    st = tex.index(r"\begin{tabular}", tex.index("\\label{" + label + "}"))
    return tex[st:tex.index(r"\end{tabular}", st)]


def datarows(block):
    """(dataset, cells) per data row, dataset tracked from the \\emph{...} separator rows."""
    out, ds = [], None
    for ln in block.split("\\\\"):
        ln = ln.lstrip("*").strip()
        if r"\emph{Waterbirds}" in ln:
            ds = "waterbirds"
            continue
        if r"\emph{CelebA}" in ln:
            ds = "celeba"
            continue
        if "&" not in ln or "rule" in ln or "multicolumn" in ln:
            continue
        cells = [c.strip() for c in ln.split("&")]
        name = re.sub(r"\$\^\{?\\dagger\}?\$|\\emph\{|\}", "", cells[0]).strip()
        if name not in NAME:
            continue
        out.append((ds, name, cells))
    return out


def num(c):
    m = re.search(r"([+-]?\d+\.\d+)", c.replace("\\,", " ").replace("$", ""))
    return float(m.group(1)) if m else None


def check(label, where, got, want, tol=5e-4):
    if got is None or want is None:
        bad.append(f"  {label}: {where} tidak terbaca (got={got} want={want})")
    elif abs(got - want) > tol:
        bad.append(f"  {label}: {where} tertulis {got}, dihitung {want:.4f}")


# ─────────────────────────────────────────────────────────────────── Table 2 (tab:c1)
rows2 = {(r["dataset"], r["backbone"]): r for r in table2()}
n = 0
for ds, name, c in datarows(table("tab:c1")):
    w = rows2[(ds, NAME[name])]
    n += 1
    check("tab:c1", f"{name}/{ds} shared", num(c[1]), w["shared"])
    check("tab:c1", f"{name}/{ds} per-group", num(c[2]), w["per_group"])
    check("tab:c1", f"{name}/{ds} lift", num(c[3]), w["lift"])
    check("tab:c1", f"{name}/{ds} spread shared", num(c[4]), w["spread_shared"])
    check("tab:c1", f"{name}/{ds} spread per-group", num(c[5]), w["spread_per_group"])
print(f"=== tab:c1 === {n} baris")

# ─────────────────────────────────────────────────────────────────── Table 4 (tab:c2)
abl = [r for r in csv.DictReader(open(f"{R}/calibration_ablation_4bb.csv"))
       if r["gate_status"] != "excluded"]
for r in abl:                                   # the module's filters compare typed values
    r["rho_test"] = float(r["rho_test"])
    r["worst_group_set_size"] = float(r["worst_group_set_size"])
    r["worst_group_acc"] = float(r["worst_group_acc"])
EQ = dict(score="APS", rho_test=0.95, calibration="mondrian")
size = {k: float(np.mean(v[0])) for k, v in arm_means(abl, "worst_group_set_size", **EQ).items()}
corr = correlation_with_ci(abl, B=2000, alpha=0.05, seed=0, **EQ)
n = 0
for ds, name, c in datarows(table("tab:c2")):
    bb = NAME[name]
    per = {m: v for (b, d, m), v in size.items() if (b, d) == (bb, ds)}
    n += 1
    if "dagger" not in c[1]:
        check("tab:c2", f"{name}/{ds} ERM", num(c[1]), per.get("erm"))
    check("tab:c2", f"{name}/{ds} best robust", num(c[2]),
          min(v for m, v in per.items() if m != "erm"))
    check("tab:c2", f"{name}/{ds} spread", num(c[3]), max(per.values()) - min(per.values()))
    check("tab:c2", f"{name}/{ds} corr", num(c[4]), corr[(bb, ds)]["r"], tol=5e-3)
    ci = [float(x) for x in re.findall(r"[+-]?\d+\.\d+", c[5])]
    check("tab:c2", f"{name}/{ds} CI lo", ci[0] if ci else None, corr[(bb, ds)]["lo"], tol=5e-3)
    check("tab:c2", f"{name}/{ds} CI hi", ci[1] if len(ci) > 1 else None,
          corr[(bb, ds)]["hi"], tol=5e-3)
print(f"=== tab:c2 === {n} baris")

# ─────────────────────────────────────────────────────── Table A1 (tab:argmin)
raw = [r for r in csv.DictReader(open(f"{R}/calibration_ablation_4bb.csv"))
       if r["gate_status"] != "excluded"]
n = 0
for ds, name, c in datarows(table("tab:argmin")):
    bb, n = NAME[name], n + 1
    mo = [r for r in raw if r["dataset"] == ds and r["backbone"] == bb
          and r["calibration"] == "mondrian"]
    top = collections.Counter(r["worst_group"] for r in mo).most_common(1)[0][1]
    check("tab:argmin", f"{name}/{ds} most frequent %", int(c[4].strip()),
          100 * top / len(mo), tol=0.5)
    if "---" not in c[1]:                       # the ERM columns, present only where ERM is kept
        erm = [r for r in raw if r["dataset"] == ds and r["backbone"] == bb
               and r["calibration"] == "marginal_split" and r["method"] == "erm"]
        g = int(re.search(r"g_\{?(\d)", c[1]).group(1))
        share = sum(r["worst_group"] == str(g) for r in erm) / len(erm)
        check("tab:argmin", f"{name}/{ds} ERM % of evaluations",
              float(c[3].strip()), 100 * share, tol=0.5)
        # the named group must be ERM's own most frequent one, and its share of the calibration
        # set follows from rho=0.95: the two aligned cells hold 0.475 each, the other two 0.025
        top = collections.Counter(r["worst_group"] for r in erm).most_common(1)[0][0]
        if str(g) != top:
            bad.append(f"  tab:argmin: {name}/{ds} worst group tertulis g{g}, dihitung g{top}")
        check("tab:argmin", f"{name}/{ds} share of cal. set", num(c[2]),
              0.475 if g in (0, 3) else 0.025, tol=1e-6)
print(f"=== tab:argmin === {n} baris")

# ─────────────────────────────────────────────────── Table 7 (tab:predgroup)
pg = [r for r in csv.DictReader(open(f"{R}/predicted_group_mondrian_4bb.csv"))
      if r["gate_status"] != "excluded" and r["score"] == "APS"]
cov = collections.defaultdict(lambda: collections.defaultdict(list))
auroc = collections.defaultdict(list)
for r in pg:
    cov[(r["dataset"], r["backbone"], r["method"])][r["condition"]].append(float(r["worst_group_cov"]))
    auroc[(r["dataset"], r["backbone"])].append(float(r["probe_auroc"]))
gap = {k: abs(np.mean(v["a_true"]) - np.mean(v["c_pred_both"])) for k, v in cov.items()}
n = 0
for ds, name, c in datarows(table("tab:predgroup")):
    bb, n = NAME[name], n + 1
    g = {k[2]: v for k, v in gap.items() if (k[0], k[1]) == (ds, bb)}
    check("tab:predgroup", f"{name}/{ds} AUROC", num(c[1]), np.mean(auroc[(ds, bb)]))
    check("tab:predgroup", f"{name}/{ds} median gap", num(c[2]), float(np.median(list(g.values()))))
    check("tab:predgroup", f"{name}/{ds} max gap", num(c[3]), max(g.values()))
    want = f"{sum(v <= 0.02 for v in g.values())}/{len(g)}"
    if c[4].strip() != want:
        bad.append(f"  tab:predgroup: {name}/{ds} deploy. tertulis {c[4].strip()}, dihitung {want}")
    # the last column names the worst method only where one exceeds the bar
    MW = {"ERM": "erm", "DFR": "dfr", "AFR": "afr", "Bal.sub.": "balanced_subsample",
          "GroupDRO-LL": "groupdro_ll", "---": None}
    got = MW.get(c[5].strip().replace("\\", "").replace(" ", ""), "?")
    want_m = max(g, key=lambda m: g[m]) if max(g.values()) > 0.02 else None
    if got != want_m:
        bad.append(f"  tab:predgroup: {name}/{ds} worst method tertulis {c[5].strip()!r}, "
                   f"dihitung {want_m}")
print(f"=== tab:predgroup === {n} baris")

if bad:
    print(f"\n{len(bad)} KETIDAKCOCOKAN:")
    print("\n".join(bad))
else:
    print("\nsemua sel tabel badan cocok dengan rekaman")
