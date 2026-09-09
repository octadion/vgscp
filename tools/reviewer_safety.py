"""Check that every reviewer point is still answered, and answered in the right place.

Some things are fine in an appendix -- a proof, a hyperparameter list, a full grid. Others have to
be in the main body or a reviewer will read the revision as having quietly dropped their point: a
requested experiment, the statistic that replaced a criticised one, an ethics discussion that was
asked for explicitly.

So each point is checked twice: is the evidence present anywhere, and for the placement-sensitive
ones, is it present in the *body*. A point that has moved entirely to an appendix when it should
not have is reported.
"""
import io
import re

DIR = r"c:\jagr\vgscp\ACML_Journal___Robust_CP_Train_Study (1)"
s = io.open(f"{DIR}\\sn-article.tex", encoding="utf-8").read()
tabs = io.open(f"{DIR}\\appendix-tables.tex", encoding="utf-8").read()
# Collapse whitespace before matching: the source is hard-wrapped, so a phrase quoted from one
# place breaks across lines in another and a literal pattern misses it. Both flags this check
# raised the first time were that, not missing content.
def squash(t):
    return re.sub(r"\s+", " ", t)


body = squash(s[:s.index(r"\begin{appendices}")])
appx = squash(s[s.index(r"\begin{appendices}"):] + tabs)


def has(hay, *needles):
    return all(re.search(n, hay, re.I | re.S) for n in needles)


# (point, must be in BODY, description, patterns)
CHECKS = [
    ("R1.1", False, "sub-target level explained",
     [r"mean over groups", r"0\.9024|min-over-groups|minimum over four"]),
    ("R1.2", True, "end-to-end fine-tuning experiment + table",
     [r"fine-tun", r"tab:repr", r"0\.222"]),
    ("R1.2", True, "title/claim narrowed to comparative",
     [r"Beats Robust Training|More Than Robust Training"]),
    ("R1.3", True, "four backbones named + literature comparison table",
     [r"DINOv2", r"ViT-B/16", r"tab:lit"]),
    ("R1.4", True, "seed count + the deterministic-solver disclosure",
     [r"five training seeds|5.*training seeds", r"L-BFGS", r"0\.000"]),
    ("R1.5", True, "requested unfreezing citation, used substantively",
     [r"elmehdi2024unfreezing", r"unfreeze"]),
    ("R2.1", True, "scope narrowed in abstract + conclusions",
     [r"Fine-tuning the representation end-to-end does not change this"]),
    ("R2.2", True, "Prop 1(ii) as exact identity + KS bound + remark",
     [r"eq:shortfall", r"eq:ksbound", r"admits no such bound"]),
    ("R2.3", True, "model-level matching, cluster bootstrap, interval, table+figure",
     [r"model level", r"cluster bootstrap", r"\+0\.084", r"tab:h1", r"fig:h1"]),
    ("R2.4", True, "pre-specified vs preregistered stated in body",
     [r"pre-specified rather than preregistered"]),
    ("R2.4", False, "sensitivity analysis over excluded runs",
     [r"0\.028", r"excluded methods included|put back"]),
    ("R2.5", True, "rho-sweep scoped to group-prior shift",
     [r"group mixture", r"not a general covariate shift|group-prior shift"]),
    ("R2.5", True, "predicted-group mechanics + supervision + probe error",
     [r"hat g=2y\+\\hat a|2y\+\\hat a", r"training split", r"asymmetric"]),
    ("R2.5", True, "ethics of inferring a protected attribute",
     [r"self-identified gender", r"2\$--\$5|2--5"]),
    ("R2.6", True, "claims moderated (efficiency range, direction not estimate)",
     [r"not narrow|direction", r"two-one-sided"]),
    ("R2.6", False, "reproducibility detail",
     [r"IMAGENET1K|checkpoint", r"calibration split", r"tabcolsep|per-group counts|Per-group"]),
    ("R3.1", True, "confidence intervals in main tables",
     [r"cluster-bootstrap CI|bootstrap CI"]),
    ("R3.1", True, "equivalence test present",
     [r"equivalence"]),
    ("R3.2", True, "conditional on Mondrian + interval on correlations",
     [r"only once calibration is group-conditional|conditional on", r"\[-1,\+1\]|uninformative"]),
]

fail = 0
print(f"{'point':6s} {'where':10s} status  what")
for pt, body_required, desc, pats in CHECKS:
    in_body = has(body, *pats)
    in_appx = has(appx, *pats)
    if in_body:
        where, ok = "body", True
    elif in_appx:
        where, ok = "appendix", not body_required
    else:
        where, ok = "MISSING", False
    fail += 0 if ok else 1
    flag = "ok    " if ok else ("MOVED " if in_appx else "GONE  ")
    print(f"{pt:6s} {where:10s} {flag} {desc}")

print(f"\n{'SEMUA AMAN' if not fail else f'{fail} PERLU DIPERIKSA'}")
