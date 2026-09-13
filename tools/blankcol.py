"""Catch the defect the column-count check cannot see: a column that is blank in every data row.

tab:reprfull declared seven columns and supplied seven, so every structural check passed -- but two
of them were empty in every row and the objective printed under the wrong heading. A column that is
never filled is either a mistake or dead weight, so flag it.
"""
import io, re
D = r"c:\jagr\vgscp\ACML_Journal___Robust_CP_Train_Study (1)"
bad = 0
for fn in ("sn-article.tex", "appendix-tables.tex"):
    s = io.open(f"{D}/{fn}", encoding="utf-8").read()
    for m in re.finditer(r"\begin\{(tabular|longtable)\}(?:\[[^\]]*\])?\{", s):
        env = m.group(1); i, d = m.end(), 1
        while d:
            d += {"{": 1, "}": -1}.get(s[i], 0); i += 1
        body = s[i:s.index(rf"\end{{{env}}}", i)]
        lab = re.findall(r"\label\{(tab:[^}]+)\}", s[max(0, m.start()-1200):m.start()] + body[:900])
        lab = lab[-1] if lab else "?"
        rows = [r.strip() for r in body.split("\\\\")
                if "&" in r and "multicolumn" not in r and "rule" not in r and "endhead" not in r]
        if len(rows) < 3:
            continue
        ncol = max(r.count("&") + 1 for r in rows)
        data = [r.split("&") for r in rows if r.count("&") + 1 == ncol]
        if len(data) < 3:
            continue
        for c in range(ncol):
            vals = [row[c].strip() for row in data[1:]]   # skip the header row
            if vals and all(v == "" for v in vals):
                print(f"  KOSONG  {lab} ({fn}): kolom {c+1} dari {ncol} kosong di semua {len(vals)} baris")
                bad += 1
print("tidak ada kolom kosong" if not bad else f"{bad} kolom kosong")
