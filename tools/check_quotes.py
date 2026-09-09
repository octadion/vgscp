"""Do the response letter's verbatim quotations still match the manuscript?

The letter quotes the revised paper inside `revised` environments, which is the strongest thing it
does: a reviewer can read the reply and the paper's own words side by side. That only works while
the words agree. This round changed several of the quoted passages -- most importantly the
Proposition 1 remark, whose "holds here by construction" clause was withdrawn as false -- so any
block still quoting the old text is now telling the reviewer something the paper does not say.

Matching is done on a normalised character run (LaTeX markup, math and whitespace stripped) and
compared by longest common substring, because the letter legitimately elides with \\ldots and
quotes fragments. A block scoring low is not automatically wrong -- it may be a fair paraphrase or
an elided quote -- but it is a block to read by hand.
"""
import io
import re
from difflib import SequenceMatcher

D = r"c:\jagr\vgscp\ACML_Journal___Robust_CP_Train_Study (1)"
paper = io.open(f"{D}/sn-article.tex", encoding="utf-8").read()
letter = io.open(f"{D}/response-to-reviewers.tex", encoding="utf-8").read()


def norm(t):
    # a comment starts at an UNescaped %; treating "98\%" as one truncates each file at whatever
    # column it happens to wrap at, which makes identical text look different
    t = re.sub(r"(?<!\\)%.*", " ", t)
    t = re.sub(r"\\(?:textbf|emph|textit|mathrm|text|citep|citet)\{([^{}]*)\}", r"\1", t)
    t = re.sub(r"\$[^$]*\$", " ", t)
    t = re.sub(r"\\[a-zA-Z]+\*?", " ", t)
    t = re.sub(r"[{}\\~^_&%`'\"()\[\]]", " ", t)
    t = re.sub(r"[^a-z0-9. ]", " ", t.lower())
    return re.sub(r"\s+", " ", t).strip()


P = norm(paper)
blocks = re.findall(r"\\begin\{revised\}(.*?)\\end\{revised\}", letter, re.S)
print(f"{len(blocks)} blok kutipan\n")

worst = []
for i, b in enumerate(blocks, 1):
    # the letter elides with \ldots; score each fragment separately and take the weakest
    frags = [f for f in re.split(r"\\ldots", b) if len(norm(f)) > 40]
    scores = []
    for f in frags:
        q = norm(f)
        m = SequenceMatcher(None, q, P, autojunk=False).find_longest_match(0, len(q), 0, len(P))
        scores.append((m.size / len(q), q))
    if not scores:
        continue
    score, q = min(scores)
    line = letter[:letter.index(b)].count("\n") + 1
    flag = "COCOK" if score > 0.85 else ("PERIKSA" if score > 0.5 else "TIDAK COCOK")
    print(f"  {flag:11s} {score:5.2f}  blok {i} (baris ~{line})")
    if score <= 0.85:
        worst.append((score, i, line, q))

print()
for score, i, line, q in sorted(worst):
    print(f"--- blok {i}, baris ~{line}, skor {score:.2f} ---")
    print(f"    kutipan: {q[:230]}...")
    # show the closest run actually present in the paper, to see what it drifted from
    m = SequenceMatcher(None, q, P, autojunk=False).find_longest_match(0, len(q), 0, len(P))
    print(f"    cocok  : ...{P[m.b:m.b + 160]}...\n")
