"""Every interval printed in Table 2 and Section 5.4 must be the output of the released module.

Table 2's per-group spread column once printed seed-only intervals while its caption promised a
two-stage bootstrap; nothing caught it because every other check compared numbers with records,
not intervals with the procedure that is supposed to produce them. This runs that procedure
(study_robust_train.interval_tables) and requires its exact output in the manuscript and the letter.
"""
import io
import os
import sys
import warnings

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
warnings.simplefilter("ignore", np.exceptions.RankWarning)
from study_robust_train.interval_tables import model_level_delta, table2  # noqa: E402

D = os.path.join(ROOT, "ACML_Journal___Robust_CP_Train_Study (1)")
paper = io.open(os.path.join(D, "sn-article.tex"), encoding="utf-8").read()
letter = io.open(os.path.join(D, "response-to-reviewers.tex"), encoding="utf-8").read()
bad = []

print("=== Tabel 2 ===")
for r in table2():
    lift = f"$+{r['lift']:.3f}$"
    if r["erm_retained"]:
        lo, hi = r["lift_ci"]
        lift += f"\\,[${lo:+.3f},{hi:+.3f}$]"
    slo, shi = r["spread_per_group_ci"]
    spread = f"\\textbf{{{r['spread_per_group']:.3f}}}\\,[{slo:.3f},{shi:.3f}]"
    for s in (lift, spread):
        if s not in paper:
            bad.append(f"  {r['dataset']}/{r['backbone']}: '{s}' tidak ada di naskah")

print("=== Bagian 5.4 ===")
for sc in ("APS", "RAPS", "THR"):
    d = model_level_delta(score=sc)
    lo, hi = d["ci"]
    in_paper = f"\\Delta_{{\\mathrm{{{sc}}}}}=+{d['delta']:.3f}\\,[{lo:+.3f},{hi:+.3f}]"
    in_letter = f"\\Delta_{{\\mathrm{{{sc}}}}} = +{d['delta']:.3f}\\,[{lo:+.3f}, {hi:+.3f}]"
    if in_paper not in paper:
        bad.append(f"  {sc}: '{in_paper}' tidak ada di naskah")
    if in_letter not in letter:
        bad.append(f"  {sc}: '{in_letter}' tidak ada di surat")

print("\n".join(bad) if bad else "")
print("SEMUA INTERVAL DARI MODUL RILIS" if not bad else f"{len(bad)} INTERVAL TIDAK COCOK")
