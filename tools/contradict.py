r"""Assert the invariants that five self-inflicted bugs violated.

Every mechanical check I had caught numbers: cells against records, values across documents, column
counts, table widths. None of them could see two paragraphs disagreeing, which is the class that
produced every defect the fourth and fifth audits found in my own edits.

So this encodes the claims that must not reappear, and the pairs that must stay in step. Each entry
records what went wrong, so a future edit that reintroduces it fails with the reason attached.
"""
import io
import re

D = r"c:\jagr\vgscp\ACML_Journal___Robust_CP_Train_Study (1)"
DOCS = {f: io.open(f"{D}/{f}", encoding="utf-8").read()
        for f in ("sn-article.tex", "appendix-tables.tex",
                  "response-to-reviewers.tex", "before-after.tex")}
PAPER = DOCS["sn-article.tex"]
bad = []


def forbid(pattern, why, where=("sn-article.tex",)):
    for f in where:
        for m in re.finditer(pattern, re.sub(r"\s+", " ", DOCS[f])):
            bad.append(f"  [{f}] {why}\n      ...{m.group(0)[:120]}...")


def require(pattern, why, where="sn-article.tex"):
    if not re.search(pattern, re.sub(r"\s+", " ", DOCS[where])):
        bad.append(f"  [{where}] HILANG: {why}")


print("=== grup mana yang terburuk ===")
# The per-group minimum lands on g3 -- an aligned 47.5% group -- in all four Waterbirds settings,
# and no group holds it in more than 41% of runs. Calling it a minority group was wrong twice.
forbid(r"under per-group thresholds the minimum falls on a minority",
       "minimum per-grup TIDAK jatuh di grup minoritas (g3 di 4/4 Waterbirds)")
# On CelebA the shared-threshold minimum is g3 only on ResNet-50 and CLIP; g0 on the two ViTs.
forbid(r"On CelebA the lowest group is \$g_3\$",
       "di CelebA minimum jatuh di g3 hanya 2 dari 4 setting")
forbid(r"the minority group the spurious correlation disadvantages---waterbird-on-land on "
       r"Waterbirds, blond male on CelebA",
       "g3 CelebA memegang 47.5% himpunan kalibrasi; bukan minoritas grup")

print("=== reproduksi ulang: 2.400 run vs 36 AFR ===")
require(r"all but \$36\$ of the \$2\{,\}400\$ runs",
        "klaim reproduksi harus mengecualikan 36 run AFR")
forbid(r"reproduces the published worst-group coverage exactly on all \$2\{,\}400\$ runs",
       "bertentangan dengan catatan AFR di Appendix F")

print("=== rentang vs rerata ===")
# 0.856-0.884 is the spread of the per-setting minima; their mean is 0.8728.
forbid(r"mean of the minima of \$0\.856\$--\$0\.884\$",
       "itu rentang minimum per-setting, bukan reratanya (0.8728)")

print("=== keterangan Tabel 2 tidak memuat pembahasan margin ===")
i = PAPER.index(r"\label{tab:c1}")
cap = re.sub(r"\s+", " ", PAPER[PAPER.rindex(r"\caption{", 0, i):i])
for phrase in ("not pre-specified", "close to binding", "discussed in the text"):
    if phrase in cap:
        bad.append(f"  [caption tab:c1] memuat {phrase!r}; pembahasan margin milik badan 5.1")
if len(cap.split()) > 190:
    bad.append(f"  [caption tab:c1] {len(cap.split())} kata; batas 190")
require(r"That margin was not pre-specified", "pembahasan margin harus ada di badan 5.1")

print("=== pita cakupan ===")
# 0.838 is the smallest per-group value across all scores, so a score-agnostic band must be 0.83.
forbid(r"empirical Mondrian level of \$0\.84\$--\$0\.89\$",
       "pernyataan lintas-skor harus 0.83--0.89 (RAPS mencapai 0.838)")

print("=== hitungan tabel konsisten lintas dokumen ===")
n_tab = len(re.findall(r"\\label\{tab:", PAPER)) + len(re.findall(r"\\label\{tab:",
                                                                 DOCS["appendix-tables.tex"]))
app_i = PAPER.index(r"\begin{appendices}")
n_body = len(re.findall(r"\\label\{tab:", PAPER[:app_i]))
print(f"  manuskrip: {n_tab} tabel, {n_body} di badan, {n_tab - n_body} di appendiks")
for f, pat, got in (("response-to-reviewers.tex", r"twenty-two tables", n_tab == 22),
                    ("before-after.tex", r"& 6 & 22 &", n_tab == 22),
                    ("before-after.tex",
                     rf"{n_body} in the body, {n_tab - n_body} in the appendix", True)):
    if not re.search(pat, re.sub(r"\s+", " ", DOCS[f])):
        bad.append(f"  [{f}] hitungan tabel tidak cocok dengan {n_tab} ({n_body} badan)")

print("=== nomor tabel appendiks yang dirujuk ada ===")
labels = re.findall(r"\\label\{tab:([^}]+)\}", PAPER) + \
         re.findall(r"\\label\{tab:([^}]+)\}", DOCS["appendix-tables.tex"])
order = {}
inp = PAPER.index(r"\input{appendix-tables}")
full = PAPER[:inp] + DOCS["appendix-tables.tex"] + PAPER[inp:]
for k, m in enumerate(re.finditer(r"\\label\{tab:([^}]+)\}", full), 1):
    order[m.group(1)] = k
valid = {f"{chr(64 + 1)}1"} | {f"B{i}" for i in range(2, 12)} | {"C12", "D13", "E14", "F15"}
for f in ("response-to-reviewers.tex", "before-after.tex"):
    for m in re.finditer(r"Table~([A-F]\d+)", DOCS[f]):
        if m.group(1) not in valid:
            bad.append(f"  [{f}] Table~{m.group(1)} bukan label yang ada "
                       f"(sah: A1, B2--B11, C12, D13, E14, F15)")

print()
if bad:
    print(f"{len(bad)} KONTRADIKSI:")
    print("\n".join(bad))
else:
    print("tidak ada kontradiksi yang diketahui")
