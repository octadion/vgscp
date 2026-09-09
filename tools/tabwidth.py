"""Measure every tabular against the text block, since there is no LaTeX here to catch overflow.

sn-jnl sets the single-column text block to 31pc = 372pt. Springer's body face is Times, and
matplotlib ships the Adobe AFM metrics for Times-Roman and Times-Bold, so column widths can be
computed exactly rather than guessed from character counts -- which is what let the earlier
overflows through, digits and letters being very different widths in a proportional face.

Width of a tabular = sum of column maxima + 2 * tabcolsep per column. Reported as a percentage of
the text block; anything over 100 runs into the margin.
"""
import io
import os
import re

from matplotlib import _afm, rcParams

TEXTWIDTH = 31 * 12.0  # 31pc, from the geometry call in sn-jnl.cls
SIZE = {"footnotesize": 8.0, "small": 9.0, "scriptsize": 7.0, "normalsize": 10.0, "tiny": 5.0}

d = os.path.join(rcParams["datapath"] if "datapath" in rcParams else
                 os.path.join(os.path.dirname(_afm.__file__), "mpl-data"), "fonts", "afm")
FONTS = {}
for key, fn in (("rm", "ptmr8a.afm"), ("bf", "ptmb8a.afm"), ("it", "ptmri8a.afm")):
    with open(os.path.join(d, fn), "rb") as fh:
        FONTS[key] = _afm.AFM(fh)


def strip(cell):
    """Turn a LaTeX cell into the glyphs it actually sets, and say which face."""
    c = cell.strip()
    face = "bf" if "\\textbf" in c or "\\bfseries" in c else "rm"
    c = re.sub(r"\\multicolumn\{\d+\}\{[^}]*\}", "", c)
    c = re.sub(r"\\(?:textbf|emph|textit|mathrm|text)\{([^{}]*)\}", r"\1", c)
    c = re.sub(r"\\(?:rhocal|rhotest)", "rho", c)
    c = c.replace(r"\,", " ").replace(r"\ ", " ").replace(r"\%", "%")
    c = c.replace(r"$+$", "+").replace(r"$-$", "-").replace("$", "")
    c = re.sub(r"\^\{?\\dagger\}?", "+", c)          # a dagger is about a digit wide
    c = re.sub(r"\\[a-zA-Z]+", "", c)
    c = c.replace("{", "").replace("}", "").replace("~", " ")
    return c, face


def width(text, face, pt):
    f = FONTS[face]
    w = 0.0
    for ch in text:
        try:
            w += f.get_width_char(ch)
        except KeyError:
            w += 500.0
    return w / 1000.0 * pt


def measure(src, label):
    out = []
    for m in re.finditer(
            r"\\begin\{(longtable|tabular)\}(?:\[[^\]]*\])?\{", src):
        env = m.group(1)
        # brace-match the column spec: it can hold @{} and p{..}, so [^}]* stops far too early
        i, depth = m.end(), 1
        while depth:
            depth += {"{": 1, "}": -1}.get(src[i], 0)
            i += 1
        spec = src[m.end():i - 1]
        body = src[i:src.index(f"\\end{{{env}}}", i)]
        # a tabular's label sits before it, a longtable's inside its own caption
        head = src[max(0, m.start() - 1400):m.start()]
        lab = re.findall(r"\\label\{([^}]+)\}", head + body[:900])
        lab = lab[-1] if lab else "?"
        sz = 10.0
        for k, v in SIZE.items():
            if f"\\{k}" in head:
                sz = v
        tcs = re.findall(r"\\setlength\{\\tabcolsep\}\{([\d.]+)pt\}", head)
        tabcolsep = float(tcs[-1]) if tcs else 6.0
        ncol = len(re.findall(r"[lcrp]", re.sub(r"@\{[^}]*\}", "", spec)))

        cols = [0.0] * ncol
        for line in body.split("\\\\"):
            line = line.strip()
            if not line or "\\multicolumn" in line or "rule" in line or "endhead" in line:
                continue
            cells = line.split("&")
            if len(cells) != ncol:
                continue
            for i, cell in enumerate(cells):
                t, face = strip(cell)
                cols[i] = max(cols[i], width(t, face, sz))
        total = sum(cols) + 2 * tabcolsep * ncol
        pct = 100 * total / TEXTWIDTH
        out.append((lab, env, ncol, sz, tabcolsep, total, pct))
    for lab, env, ncol, sz, tcs, total, pct in sorted(out, key=lambda r: -r[6]):
        flag = "LEBIH" if pct > 100 else ("mepet" if pct > 94 else "ok")
        print(f"  {flag:5s} {pct:5.1f}%  {total:6.1f}pt  {ncol}kol {sz:.0f}pt "
              f"sep{tcs:g}  {lab}  [{src is A and 'app' or label}]")
    return out


base = r"c:\jagr\vgscp\ACML_Journal___Robust_CP_Train_Study (1)"
M = io.open(f"{base}/sn-article.tex", encoding="utf-8").read()
A = io.open(f"{base}/appendix-tables.tex", encoding="utf-8").read()
print(f"lebar blok teks: {TEXTWIDTH:.0f}pt\n")
print("badan utama:")
r1 = measure(M, "body")
print("\nappendiks:")
r2 = measure(A, "app")
bad = [r for r in r1 + r2 if r[6] > 100]
print(f"\n{len(bad)} tabel melewati batas" if bad else "\nsemua tabel muat")
