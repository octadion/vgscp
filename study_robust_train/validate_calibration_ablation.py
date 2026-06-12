"""LOCAL logic-validation of calibration_ablation.py on SYNTHETIC features (no torch / no data).

Exercises the calibration axis (marginal_split / mondrian / shift_robust), C1/C2/C3 verdicts, and
the CSV / MD / figure emitters. CLAIMS NO REAL NUMBERS.

    python -m study_robust_train.validate_calibration_ablation     # exit 0 on PASS
"""
from __future__ import annotations

import os

from .calibration_ablation import (make_c1_figure, run_ablation, write_calibration_ablation_md,
                                    write_csv)
from .conformal_eval import CALIBRATIONS
from .methods import METHODS
from .synthetic import make_synthetic_griddata
from .verdicts import SCORES


def _check(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        raise AssertionError(f"calibration-ablation logic-validation failed: {name}")


def main() -> int:
    print("=" * 78)
    print("study_robust_train -- CALIBRATION-ABLATION logic validation (no real numbers claimed)")
    print("=" * 78)

    data = {("clip_vitb32", "waterbirds"): make_synthetic_griddata(
                backbone="clip_vitb32", dataset="waterbirds", n_pool=4500, seed=0)}
    rho_sweep = (0.95, 0.80, 0.50)

    print("\n[1] ablation runs across calibration x method x score x rho x seed x split")
    out = run_ablation(data, methods=METHODS, scores=SCORES, rho_sweep=rho_sweep,
                       seeds=(0, 1, 2), n_splits=4)
    recs = out["records"]
    _check("records produced", len(recs) > 0)
    _check("all 3 calibration policies present",
           set(r["calibration"] for r in recs) == set(CALIBRATIONS))
    _check("every record has a calibration column + valid coverage",
           all("calibration" in r and 0.0 <= r["worst_group_cov"] <= 1.0 for r in recs))

    key = ("clip_vitb32", "waterbirds")
    v = out["verdicts"][key]

    print("\n[2] C1 -- coverage-is-calibration verdict structure")
    c1 = v["C1"]
    _check("C1 has all methods with all 3 calibration aggregates",
           all(set(("marginal_split", "mondrian", "shift_robust")).issubset(c1["methods"][m]) for m in c1["methods"]))
    _check("C1_holds is bool", isinstance(c1["C1_holds"], bool))
    _check("each method has mondrian_near_target + split_below_target bools",
           all(isinstance(c1["methods"][m]["mondrian_near_target"], bool)
               and isinstance(c1["methods"][m]["split_below_target"], bool) for m in c1["methods"]))
    # near-definitional sanity: Mondrian worst-group cov >= marginal_split's (per method, tolerance)
    ok = all(c1["methods"][m]["mondrian"]["mean"] >= c1["methods"][m]["marginal_split"]["mean"] - 0.03
             for m in c1["methods"])
    _check("Mondrian worst-group cov >= marginal_split (within tol) for all methods", ok)
    for m in c1["methods"]:
        mo = c1["methods"][m]["mondrian"]["mean"]; ms = c1["methods"][m]["marginal_split"]["mean"]
        print(f"      {m}: mondrian={mo:.3f}  marginal_split={ms:.3f}  shortfall={c1['methods'][m]['split_shortfall']:+.3f}")

    print("\n[3] C2 -- efficiency vs accuracy (under Mondrian)")
    c2 = v["C2"]
    _check("C2 has per-method worst-group set size + accuracy",
           all("worst_group_set_size" in c2["methods"][m] and "worst_group_acc" in c2["methods"][m]
               for m in c2["methods"]))
    _check("C2 efficiency_tracks_accuracy is bool", isinstance(c2["efficiency_tracks_accuracy"], bool))

    print("\n[4] C3 -- Mondrian shift validity per (method x score)")
    c3 = v["C3"]
    _check("C3 has a rho curve + survives flag per method/score",
           all(all(("curve" in c3["methods"][m][sc]) and isinstance(c3["methods"][m][sc]["survives"], bool)
                   for sc in SCORES) for m in c3["methods"]))

    print("\n[5] seed vs split variance separated in aggregates")
    agg = c1["methods"][next(iter(c1["methods"]))]["mondrian"]
    _check("aggregate reports seed_std and split_std", "seed_std" in agg and "split_std" in agg)

    print("\n[6] CSV (with calibration column) + MD + figure emitters")
    os.makedirs("results/study_synthetic", exist_ok=True)
    csv_p = "results/study_synthetic/calibration_ablation_SYNTHETIC.csv"
    md_p = "results/study_synthetic/CALIBRATION_ABLATION_SYNTHETIC.md"
    write_csv(recs, csv_p)
    figs = make_c1_figure(out, "results/study_synthetic/figures")
    write_calibration_ablation_md(out, md_p, fig_paths=figs, synthetic=True)
    import csv as _csv
    with open(csv_p, encoding="utf-8") as f:
        header = next(_csv.reader(f))
    _check("CSV has the calibration column", "calibration" in header)
    _check("MD + C1 figure written", os.path.getsize(md_p) > 0 and figs and os.path.getsize(figs[0]) > 0)

    print("\n[7] extend_ablation_to appends a NEW (CelebA) cell WITHOUT overwriting Waterbirds")
    from .calibration_ablation import extend_ablation_to, records_from_csv
    n_wb_rows = len(records_from_csv(csv_p))
    celeba = {("clip_vitb32", "celeba"): make_synthetic_griddata(
                  backbone="clip_vitb32", dataset="celeba", n_pool=4500, seed=2)}
    fig_dir = "results/study_synthetic/figures"
    ext = extend_ablation_to(celeba, csv_path=csv_p, md_path=md_p, figdir=fig_dir,
                             methods=METHODS, scores=SCORES, rho_sweep=rho_sweep, seeds=(0, 1, 2), n_splits=4)
    merged = records_from_csv(csv_p)
    keys = set((r["backbone"], r["dataset"]) for r in merged)
    _check("merged CSV keeps Waterbirds AND adds CelebA",
           ("clip_vitb32", "waterbirds") in keys and ("clip_vitb32", "celeba") in keys)
    wb_after = sum(1 for r in merged if (r["backbone"], r["dataset"]) == ("clip_vitb32", "waterbirds"))
    _check("Waterbirds rows preserved (not overwritten)", wb_after == n_wb_rows)
    _check("regenerated MD has both Waterbirds and CelebA sections",
           "clip_vitb32 / waterbirds" in open(md_p, encoding="utf-8").read()
           and "clip_vitb32 / celeba" in open(md_p, encoding="utf-8").read())
    _check("CelebA C1 figure emitted", os.path.exists(f"{fig_dir}/calib_C1_clip_vitb32_celeba.png"))
    # idempotent: re-extending the same cell must NOT duplicate rows
    extend_ablation_to(celeba, csv_path=csv_p, md_path=md_p, figdir=fig_dir,
                       methods=METHODS, scores=SCORES, rho_sweep=rho_sweep, seeds=(0, 1, 2), n_splits=4)
    _check("re-extend is idempotent (no duplicate rows)", len(records_from_csv(csv_p)) == len(merged))

    print("\n" + "=" * 78)
    print("CALIBRATION-ABLATION LOGIC OK -- C1/C2/C3 + emitters validated on synthetic.")
    print("NO REAL NUMBERS CLAIMED. Real run: notebooks/calibration_ablation.ipynb (Colab).")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
