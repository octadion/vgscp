r"""Find where the pages actually are, before proposing any cut.

Three things consume space and they are not interchangeable. Prose is cheap to trim and cheap to
damage. Captions are prose that grew unnoticed -- they are outside the section word counts and
nobody reads them as text, so they bloat quietly. Tables and figures consume vertical space
regardless of their word count, so a table that could merge with another saves far more page than
its words suggest.

This measures all three and flags the ones worth discussing, without cutting anything.
"""
import io
import re

D = r"c:\jagr\vgscp\ACML_Journal___Robust_CP_Train_Study (1)"
PAPER = io.open(f"{D}/sn-article.tex", encoding="utf-8").read()
APX = io.open(f"{D}/appendix-tables.tex", encoding="utf-8").read()
app_i = PAPER.index(r"\begin{appendices}")


def words(t):
    t = re.sub(r"\\begin\{(tabular|longtable)\}.*?\\end\{\1\}", " ", t, flags=re.S)
    return len(re.sub(r"[{}$&\\~^_%]", " ", re.sub(r"\\[a-zA-Z]+\*?", " ", t)).split())


print("=== keterangan tabel/gambar terpanjang ===")
caps = []
for src, tag in ((PAPER, "paper"), (APX, "apx")):
    for m in re.finditer(r"\\caption\{", src):
        i, d = m.end(), 1
        while d:
            d += {"{": 1, "}": -1}.get(src[i], 0)
            i += 1
        body = src[m.end():i - 1]
        lab = re.search(r"\\label\{([^}]+)\}", src[i:i + 300])
        nxt = re.search(r"\\label\{([^}]+)\}", body)
        name = (nxt or lab).group(1) if (nxt or lab) else "?"
        caps.append((words(body), name, tag))
tot = sum(c[0] for c in caps)
print(f"  {len(caps)} keterangan, {tot} kata total")
for w, name, tag in sorted(caps, reverse=True)[:10]:
    print(f"    {w:4d}  {name}")
print(f"  sepuluh terpanjang = {sum(c[0] for c in sorted(caps, reverse=True)[:10])} kata "
      f"({100*sum(c[0] for c in sorted(caps, reverse=True)[:10])/tot:.0f}% dari semua keterangan)")

print("\n=== tabel appendiks: baris data (proksi tinggi halaman) ===")
rows = []
for src, tag in ((PAPER[app_i:], "paper"), (APX, "apx")):
    for m in re.finditer(r"\\begin\{(tabular|longtable)\}", src):
        env = m.group(1)
        end = src.index(f"\\end{{{env}}}", m.end())
        body = src[m.end():end]
        lab = re.findall(r"\\label\{(tab:[^}]+)\}", src[max(0, m.start() - 1500):m.start()] + body[:900])
        n = len([l for l in body.split("\\\\")
                 if "&" in l and "multicolumn" not in l and "rule" not in l])
        rows.append((n, lab[-1] if lab else "?"))
for n, lab in sorted(rows, reverse=True):
    print(f"    {n:4d} baris  {lab}")
print(f"  total baris tabel appendiks: {sum(r[0] for r in rows)}")

print("\n=== kandidat pengulangan: frasa angka yang muncul di banyak tempat ===")
flat = re.sub(r"\s+", " ", PAPER)
for pat, label in (
    (r"0\.358", "spread 0.358"),
    (r"0\.024", "spread 0.024"),
    (r"0\.9024", "mean over groups"),
    (r"0\.018\$--\$0\.045", "gap 0.018-0.045"),
    (r"0\.036\$--\$0\.364", "efficiency spread"),
    (r"six of eight", "six of eight"),
    (r"\$140\$", "CelebA g3 distinct"),
    (r"with replacement", "with-replacement disclosure"),
    (r"\$0\.05\$ margin", "0.05 margin"),
    (r"min-over-groups|minimum over four", "min-over-groups idea"),
):
    k = len(re.findall(pat, flat))
    if k > 1:
        print(f"    {k}x  {label}")
