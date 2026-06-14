"""LOCAL logic-validation of predicted_group_mondrian.py on SYNTHETIC features (no torch/data).

Exercises the three Mondrian conditions (true / predicted-test / predicted-both), the gap (a)-(c),
the >=3-seed × splits aggregation with CIs, and the MD/CSV emitters. CLAIMS NO REAL NUMBERS.

    python -m study_robust_train.validate_predicted_group_mondrian      # exit 0 on PASS
"""
from __future__ import annotations

import os

from .methods import METHODS
from .predicted_group_mondrian import CONDITIONS, run_predicted_group, write_csv, write_md
from .synthetic import make_synthetic_griddata
from .verdicts import SCORES


def _check(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        raise AssertionError(f"predicted-group-mondrian logic-validation failed: {name}")


def main() -> int:
    print("=" * 78)
    print("study_robust_train -- PREDICTED-GROUP MONDRIAN logic validation (no real numbers claimed)")
    print("=" * 78)

    data = {("clip_vitb32", "waterbirds"): make_synthetic_griddata(
                backbone="clip_vitb32", dataset="waterbirds", n_pool=4500, seed=0),
            ("resnet50_erm", "celeba"): make_synthetic_griddata(
                backbone="resnet50_erm", dataset="celeba", n_pool=4500, seed=1)}

    print("\n[1] runs across condition × method × score × seed × split")
    out = run_predicted_group(data, methods=METHODS, scores=SCORES, seeds=(0, 1, 2), n_splits=4)
    recs = out["records"]
    _check("records produced", len(recs) > 0)
    _check("all 3 conditions present", set(r["condition"] for r in recs) == set(CONDITIONS))
    _check("coverage + set size in valid ranges",
           all(0.0 <= r["worst_group_cov"] <= 1.0 and r["worst_group_set_size"] >= 0.0 for r in recs))
    _check("probe AUROC recorded in [0,1] (or nan)",
           all((r["probe_auroc"] != r["probe_auroc"]) or 0.0 <= r["probe_auroc"] <= 1.0 for r in recs))

    print("\n[2] verdict: per-method a/b/c aggregates + gap(a-c) + deployable")
    for key, v in out["verdicts"].items():
        _check(f"{key}: every method has a/b/c cov+size aggregates",
               all(all(c in v["methods"][m] for c in CONDITIONS) for m in v["methods"]))
        _check(f"{key}: gap_cov_a_minus_c + deployable present",
               all("gap_cov_a_minus_c" in v["methods"][m] and isinstance(v["methods"][m]["deployable"], bool)
                   for m in v["methods"]))
        _check(f"{key}: all_deployable is bool", isinstance(v["all_deployable"], bool))
        # CI present (>=3 seeds × splits aggregation)
        any_m = next(iter(v["methods"]))
        _check(f"{key}: 95% CI on cov(a)", len(v["methods"][any_m]["a_true"]["cov"]["ci"]) == 2)
        for m in v["methods"]:
            ga = v["methods"][m]
            print(f"      {key} {m}: AUROC={ga['probe_auroc']:.3f} cov(a)={ga['a_true']['cov']['mean']:.3f} "
                  f"cov(c)={ga['c_pred_both']['cov']['mean']:.3f} gap={ga['gap_cov_a_minus_c']:+.3f} "
                  f"deployable={ga['deployable']}")

    print("\n[3] seed vs split variance separated in aggregates")
    agg = next(iter(out["verdicts"].values()))["methods"]
    a0 = agg[next(iter(agg))]["a_true"]["cov"]
    _check("aggregate has seed_std + split_std", "seed_std" in a0 and "split_std" in a0)

    print("\n[4] CSV + MD emitters")
    os.makedirs("results/study_synthetic", exist_ok=True)
    csv_p = "results/study_synthetic/predicted_group_mondrian_SYNTHETIC.csv"
    md_p = "results/study_synthetic/PREDICTED_GROUP_MONDRIAN_SYNTHETIC.md"
    write_csv(recs, csv_p)
    write_md(out, md_p, synthetic=True)
    import csv as _csv
    with open(csv_p, encoding="utf-8") as f:
        header = next(_csv.reader(f))
    _check("CSV has condition column", "condition" in header)
    _check("MD has true vs predicted-both columns + appendix",
           os.path.getsize(md_p) > 0 and "gap(a" in open(md_p, encoding="utf-8").read()
           and "appendix" in open(md_p, encoding="utf-8").read())

    print("\n" + "=" * 78)
    print("PREDICTED-GROUP MONDRIAN LOGIC OK -- (a)/(b)/(c) + gap + emitters validated on synthetic.")
    print("NO REAL NUMBERS CLAIMED. Real run: notebooks/predicted_group_mondrian.ipynb (Colab).")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
