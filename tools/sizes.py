"""Word count per section, so the cutting is aimed at the fat rather than spread evenly."""
import io
import re

P = r"c:\jagr\vgscp\ACML_Journal___Robust_CP_Train_Study (1)\sn-article.tex"
s = io.open(P, encoding="utf-8").read()


def w(t):
    t = re.sub(r"\\begin\{tabular\}.*?\\end\{tabular\}", " ", t, flags=re.S)
    return len(re.sub(r"[{}$&\\~^_%]", " ", re.sub(r"\\[a-zA-Z]+\*?", " ", t)).split())


parts = re.split(r"(\\(?:sub)?section\{)", s)
# rebuild: find each heading and the text until the next heading
heads = [(m.start(), m.group(0), m.group(1))
         for m in re.finditer(r"\\(sub)?section\{([^}]*(?:\{[^}]*\}[^}]*)*)\}", s)]
tot_body = tot_app = 0
app_at = s.index(r"\begin{appendices}")
print(f"{'kata':>6}  bagian")
for i, (pos, full, sub) in enumerate(heads):
    end = heads[i + 1][0] if i + 1 < len(heads) else len(s)
    name = re.search(r"\\(?:sub)?section\{(.*)\}", full).group(1)
    nw = w(s[pos:end])
    if pos < app_at:
        tot_body += nw
    else:
        tot_app += nw
    print(f"{nw:6d}  {'  ' if sub else ''}{name[:64]}")
print(f"\nbadan utama {tot_body} kata | appendiks {tot_app} kata")
print(f"caption: {sum(w(c) for c in re.findall(r'\\caption\{(.*?)\}\s*\n\s*\\label', s, re.S))} kata total")
