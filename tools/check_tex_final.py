"""Structural check on the manuscript after the figure swap.

There is no TeX installation on this machine, so this stands in for a compile: it catches the
classes of defect that have actually bitten this revision -- a table declaring more columns than it
supplies (a defect present in the submitted paper), a reference with no label, an included figure
with no file on disk, and the stray control characters a mangled shell heredoc once wrote into the
source.
"""
import io
import os
import re

DIR = r"c:\jagr\vgscp\ACML_Journal___Robust_CP_Train_Study (1)"
P = os.path.join(DIR, "sn-article.tex")
s = io.open(P, encoding="utf-8", newline="").read()

# Follow \input so labels defined in an included file are not reported as dangling.
for inc in re.findall(r"\\input\{([^}]+)\}", s):
    q = os.path.join(DIR, inc if inc.endswith(".tex") else inc + ".tex")
    if os.path.exists(q):
        s += "\n" + io.open(q, encoding="utf-8").read()
        print(f"  (menyertakan {inc})")

bad = 0


def check(ok, msg):
    global bad
    print(("  ok   " if ok else "  GAGAL ") + msg)
    bad += 0 if ok else 1


# --- delimiters
check(s.count("{") == s.count("}"), f"brace seimbang {s.count('{')}/{s.count('}')}")
check(s.count("$") % 2 == 0, f"pembatas matematika genap ({s.count('$')})")
ctrl = sorted({ord(c) for c in s if ord(c) < 32 and c != "\n"})
check(not ctrl, f"tanpa karakter kendali liar {ctrl or ''}")

# --- labels and references
labels = set(re.findall(r"\\label\{([^}]+)\}", s))
refs = set(re.findall(r"\\(?:ref|eqref|autoref)\{([^}]+)\}", s))
check(not (refs - labels), f"acuan menggantung: {sorted(refs - labels) or 'tidak ada'}")
unref = sorted(l for l in labels - refs if l.split(":")[0] in ("fig", "tab"))
check(not unref, f"gambar/tabel tak dirujuk: {unref or 'tidak ada'}")

# --- citations resolve against the bibliography.
# This check was missing, and a \citep whose entry had never been written passed silently; it
# would have compiled to "?" in the PDF.
bibf = os.path.join(DIR, "sn-bibliography.bib")
if os.path.exists(bibf):
    bib = io.open(bibf, encoding="utf-8").read()
    cites = set()
    for grp in re.findall(r"\\cite[a-zA-Z]*\*?(?:\[[^\]]*\])*\{([^}]+)\}", s):
        cites |= {c.strip() for c in grp.split(",")}
    keys = set(re.findall(r"@\w+\{\s*([^,\s]+)\s*,", bib))
    check(not (cites - keys), f"sitasi tanpa entri bib: {sorted(cites - keys) or 'tidak ada'}")
    check(not (keys - cites), f"entri bib tak disitasi: {sorted(keys - cites) or 'tidak ada'}")

# --- figure files present (paths are extensionless so pdflatex prefers the PDF)
inc = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", s)
for f in inc:
    check(os.path.exists(os.path.join(DIR, f + ".pdf")), f"berkas ada: {f}.pdf")

# --- table column counts: the alignment spec must supply what each row uses
for m in re.finditer(r"\\begin\{(tabular|longtable)\}(?:\[[^\]]*\])?\{", s):
    env = m.group(1)
    # The spec can hold @{} and p{..}, so a [^}]* group stops at the first inner brace and reports
    # zero columns -- a false alarm that would hide a real mismatch. Brace-match instead.
    i, depth = m.end(), 1
    while depth:
        depth += {"{": 1, "}": -1}.get(s[i], 0)
        i += 1
    spec = s[m.end():i - 1]
    body = s[i:s.index(f"\\end{{{env}}}", i)]
    # strip, in order: @{..} inserts, >{..}/<{..} cell modifiers (whose contents are full of
    # letters like \raggedright), then p{..} widths -- otherwise the modifiers are counted as
    # columns and a correct table is reported as broken.
    bare = re.sub(r"@\{[^}]*\}", "", spec)
    bare = re.sub(r"[><]\{(?:[^{}]|\{[^{}]*\})*\}", "", bare)
    ncol = len(re.findall(r"[lcrp]", re.sub(r"p\{[^}]*\}", "p", bare)))
    rows = [r for r in body.split(r"\\")
            if "&" in r and "multicolumn" not in r and "cmidrule" not in r]
    widths = {r.count("&") + 1 for r in rows}
    # a tabular is labelled before it, a longtable inside its own caption
    label = (re.search(r"\\label\{(tab:[^}]+)\}", s[max(0, m.start() - 1200):m.start()] + body[:900])
             or [None, "?"])[1]
    check(widths <= {ncol}, f"{label}: spesifikasi {ncol} kolom, baris memakai {sorted(widths)}")

print(f"\n{len(s.splitlines())} baris, {len(inc)} gambar, "
      f"{len(re.findall(r'begin.tabular', s))} tabel")
print("SEMUA LULUS" if not bad else f"{bad} GAGAL")
