"""LOCAL logic-validation of recoverability.py on SYNTHETIC features (no torch / no data).

Exercises Part A (recoverability AUROC + Mondrian worst-group coverage + verdict) and Part B (the
project-out-top-k sweep + figure + RECOVERABILITY.md), asserting the wiring. CLAIMS NO REAL NUMBERS.

    python -m study_robust_train.validate_recoverability     # exit 0 on PASS
"""
from __future__ import annotations

import math
import os

from .recoverability import (classify_tension, make_part_b_figure, needs_part_b, run_part_a,
                             run_part_b, spurious_from_group, write_recoverability_md)
from .synthetic import make_synthetic_griddata


def _check(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        raise AssertionError(f"recoverability logic-validation failed: {name}")


def main() -> int:
    print("=" * 78)
    print("study_robust_train — RECOVERABILITY logic validation (no real numbers claimed)")
    print("=" * 78)

    data = {
        ("resnet50_erm", "waterbirds"): make_synthetic_griddata(
            backbone="resnet50_erm", dataset="waterbirds", n_pool=4000, seed=0),
        ("clip_vitb32", "celeba"): make_synthetic_griddata(
            backbone="clip_vitb32", dataset="celeba", n_pool=4000, seed=1),
    }

    print("\n[1] spurious target derives from group (2*y+spurious -> spurious=group%2)")
    g = data[("resnet50_erm", "waterbirds")].train[2]
    _check("spurious_from_group in {0,1}", set(int(x) for x in spurious_from_group(g)) <= {0, 1})

    print("\n[2] Part A runs per cell (recoverability AUROC + Mondrian worst-group coverage + verdict)")
    A = run_part_a(data, seeds=(0, 1, 2), n_splits=4)
    _check("Part A has both cells", set(A["cells"]) == set(data))
    for key, c in A["cells"].items():
        r, cov = c["recoverability"], c["coverage"]
        _check(f"{key}: AUROC in [0,1]", 0.0 <= r["auroc_mean"] <= 1.0)
        _check(f"{key}: AUROC 95% CI ordered & in [0,1]",
               r["ci"][0] <= r["ci"][1] and 0.0 <= r["ci"][0] and r["ci"][1] <= 1.0)
        _check(f"{key}: >=3 seeds reported", len(r["auroc_per_seed"]) >= 3)
        _check(f"{key}: worst-group cov in [0,1] with CI", 0.0 <= cov["worst_group_cov_mean"] <= 1.0)
        _check(f"{key}: verdict valid", c["verdict"] in ("tension_dead", "tension_alive", "ambiguous"))

    print("\n[3] verdict classifier honors the spec bars")
    _check("good cov + high AUROC -> tension_dead", classify_tension(0.90, 0.80) == "tension_dead")
    _check("good cov + chance AUROC -> tension_alive", classify_tension(0.90, 0.52) == "tension_alive")
    _check("good cov + mid AUROC -> ambiguous", classify_tension(0.90, 0.60) == "ambiguous")
    _check("poor cov -> ambiguous", classify_tension(0.70, 0.90) == "ambiguous")
    _check("needs_part_b returns bool", isinstance(needs_part_b(A), bool))

    print("\n[4] Part B sweep (project out top-k) + tradeoff flag")
    B = run_part_b(data, ks=(0, 1, 5, 20), seeds=(0, 1, 2), n_splits=4)
    for key, c in B["cells"].items():
        ks = [p["k"] for p in c["curve"]]
        _check(f"{key}: curve covers k in {{0,1,5,20}}", ks == [0, 1, 5, 20])
        _check(f"{key}: removed<=k and AUROC/cov in [0,1]",
               all(p["removed"] <= p["k"] and 0 <= p["auroc"] <= 1 and 0 <= p["worst_group_cov"] <= 1
                   for p in c["curve"]))
        _check(f"{key}: negative_tradeoff is bool", isinstance(c["negative_tradeoff"], bool))
    # projecting out spurious directions should REDUCE recoverability (sanity on the toy)
    k0 = B["cells"][("resnet50_erm", "waterbirds")]["curve"][0]["auroc"]
    k20 = B["cells"][("resnet50_erm", "waterbirds")]["curve"][-1]["auroc"]
    _check("toy: recoverability drops after projecting out 20 spurious dirs", k20 <= k0 + 1e-9)

    print("\n[5] figure + RECOVERABILITY.md emitters")
    os.makedirs("results/study_synthetic", exist_ok=True)
    fig = make_part_b_figure(B, "results/study_synthetic/recoverability_partB_SYNTHETIC.png")
    md = "results/study_synthetic/RECOVERABILITY_SYNTHETIC.md"
    write_recoverability_md(A, md, part_b=B, fig_path=os.path.basename(fig), synthetic=True)
    _check("Part B figure written", os.path.exists(fig) and os.path.getsize(fig) > 0)
    _check("RECOVERABILITY.md written", os.path.exists(md) and os.path.getsize(md) > 0)
    # also exercise the Part-A-only report path (the STOP-after-A deliverable)
    write_recoverability_md(A, "results/study_synthetic/RECOVERABILITY_partA_only_SYNTHETIC.md", synthetic=True)
    _check("Part-A-only report written", os.path.getsize("results/study_synthetic/RECOVERABILITY_partA_only_SYNTHETIC.md") > 0)

    print("\n" + "=" * 78)
    print("RECOVERABILITY LOGIC OK — Part A + Part B + emitters validated on synthetic.")
    print("NO REAL NUMBERS CLAIMED. Real run: notebooks/recoverability.ipynb (Colab).")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
