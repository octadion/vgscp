"""Guard against losing numbers or citations while rewriting prose.

Usage:
    numguard.py snapshot        -- record the current state
    numguard.py check           -- compare the current state against the snapshot

A rewrite of the prose is allowed to move a number out of a sentence and leave it in a table; it is
not allowed to make the number vanish from the paper. So the comparison is over the whole file, not
per section, and anything that disappears entirely is reported. Citations and cross-reference
labels are tracked the same way, since dropping a reference is the other quiet way a revision
loses something a reviewer asked for.
"""
import io
import json
import os
import re
import sys
from collections import Counter

P = r"c:\jagr\vgscp\ACML_Journal___Robust_CP_Train_Study (1)\sn-article.tex"
# The baseline lived in a temp scratchpad belonging to a session that had already ended, so it
# could be swept away at any point and the guard would silently have nothing to compare against.
# It belongs beside the tool, in the repository, where git records when it was last refreshed.
SNAP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "numguard.json")


def state():
    s = io.open(P, encoding="utf-8").read()
    cites = set()
    for m in re.findall(r"\\cite[a-zA-Z]*\*?(?:\[[^\]]*\])*\{([^}]+)\}", s):
        cites |= {c.strip() for c in m.split(",")}
    return {
        # every decimal figure, however written
        "numbers": dict(Counter(re.findall(r"\d+\.\d+", s))),
        "percents": dict(Counter(re.findall(r"\d+\\%", s))),
        "cites": sorted(cites),
        "labels": sorted(set(re.findall(r"\\label\{([^}]+)\}", s))),
        "tables": sorted(set(re.findall(r"\\label\{(tab:[^}]+)\}", s))),
        "figures": sorted(set(re.findall(r"\\label\{(fig:[^}]+)\}", s))),
        "lines": len(s.splitlines()),
    }


if sys.argv[1] == "snapshot":
    json.dump(state(), io.open(SNAP, "w", encoding="utf-8"), indent=1)
    st = state()
    print(f"tersimpan: {len(st['numbers'])} nilai desimal unik "
          f"({sum(st['numbers'].values())} kemunculan), {len(st['cites'])} sitasi, "
          f"{len(st['tables'])} tabel, {len(st['figures'])} gambar, {st['lines']} baris")
else:
    old = json.load(io.open(SNAP, encoding="utf-8"))
    new = state()
    bad = 0

    # Values checked by hand and accepted: each was a rounded restatement in prose of numbers
    # that remain exact in a table. "0.86" summarised the Waterbirds Mondrian column of
    # Table 5, which still reads 0.865 / 0.861 / 0.864.
    # Verified by hand: each is printed inside the opening figure itself, so a reader sees it
    # there; "0.86" was a rounded restatement of Table 5's 0.865 / 0.861 / 0.864.
    # "0.35" was a rounded restatement of the ERM lift range in the old contribution list;
    # the exact endpoints 0.063 and 0.345 both remain in Table 3 and Section 6.1.
    ALLOWED = {"0.86", "0.391", "0.892", "0.984", "1.001", "1.080", "0.35"}
    gone = sorted(set(old["numbers"]) - set(new["numbers"]) - ALLOWED)
    print(f"{'ok  ' if not gone else 'HILANG'} nilai desimal yang lenyap sama sekali: "
          f"{gone or 'tidak ada'}")
    bad += len(gone)

    fewer = {k: (old["numbers"][k], new["numbers"].get(k, 0))
             for k in old["numbers"] if new["numbers"].get(k, 0) < old["numbers"][k]}
    if fewer:
        print(f"  (berkurang kemunculannya, boleh saja bila pindah ke tabel: "
              f"{len(fewer)} nilai)")

    gone_p = sorted(set(old["percents"]) - set(new["percents"]))
    print(f"{'ok  ' if not gone_p else 'HILANG'} persen yang lenyap: {gone_p or 'tidak ada'}")
    bad += len(gone_p)

    for key in ("cites", "labels", "tables", "figures"):
        lost = sorted(set(old[key]) - set(new[key]))
        print(f"{'ok  ' if not lost else 'HILANG'} {key}: {lost or 'tidak ada'}")
        bad += len(lost)

    add = sorted(set(new["numbers"]) - set(old["numbers"]))
    if add:
        print(f"  catatan: nilai baru muncul -> {add}")
    print(f"\nbaris {old['lines']} -> {new['lines']}")
    print("AMAN" if not bad else f"{bad} MASALAH -- periksa sebelum lanjut")
