"""LOCAL logic-validation of the FULL H1/H2/H3 grid on SYNTHETIC features (no torch / no data).

Exercises the whole chain — all 5 last-layer methods, APS/RAPS/THR, the rho sweep, the
accuracy-matched H1 readout, H2 ranking inversion, H3 shift survival, and the CSV / RESULTS_study.md
emitters — and CLAIMS NO REAL NUMBERS. Run:

    python -m study_robust_train.validate_grid

Exits 0 on PASS. Real H1/H2/H3 numbers come from the Colab grid notebook.
"""
from __future__ import annotations

import math
import os

from .grid import run_grid, write_csv, write_results_md
from .methods import METHODS
from .synthetic import make_synthetic_griddata
from .verdicts import SCORES


def _check(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        raise AssertionError(f"grid logic-validation failed: {name}")


def main() -> int:
    print("=" * 78)
    print("study_robust_train — SYNTHETIC GRID LOGIC VALIDATION (no real numbers claimed)")
    print("=" * 78)

    # two keys to exercise multi-(backbone,dataset) aggregation; small splits/seeds for speed
    data = {
        ("resnet50_erm", "waterbirds"): make_synthetic_griddata(
            backbone="resnet50_erm", dataset="waterbirds", n_pool=4500, seed=0),
        ("clip_vitb32", "waterbirds"): make_synthetic_griddata(
            backbone="clip_vitb32", dataset="waterbirds", n_pool=4500, seed=1),
    }
    rho_sweep = (0.95, 0.80, 0.50)     # reduced sweep for the local logic check
    seeds = (0, 1, 2)
    n_splits = 4

    print("\n[1] grid runs end-to-end (5 methods x 3 seeds x 3 scores x 3 rho x 4 splits x 2 keys)")
    out = run_grid(data, methods=METHODS, scores=SCORES, rho_sweep=rho_sweep,
                   seeds=seeds, n_splits=n_splits)
    recs = out["records"]
    _check("records produced", len(recs) > 0)
    expected_max = len(data) * len(METHODS) * len(seeds) * len(SCORES) * len(rho_sweep) * n_splits
    _check(f"record count <= theoretical max ({expected_max}) and >0",
           0 < len(recs) <= expected_max)

    print("\n[2] every record well-formed")
    ok = True
    for r in recs:
        ok &= (0.0 <= r["worst_group_cov"] <= 1.0) and (0.0 <= r["marginal_cov"] <= 1.0)
        ok &= r["mean_set_size"] >= 0.0 and r["set_size_disparity"] >= 0.0
        ok &= math.isfinite(r["div_wasserstein1"]) and 0.0 <= r["div_ks_stat"] <= 1.0
        ok &= 0.0 <= r["worst_group_acc"] <= 1.0 and 0.0 <= r["base_top1"] <= 1.0
    _check("coverages/sizes/divergence/accuracy all in valid ranges", ok)

    print("\n[3] verdicts produced per (backbone, dataset)")
    _check("verdicts for both keys", set(out["verdicts"]) == set(data))
    for key, v in out["verdicts"].items():
        _check(f"{key}: H1 has all robust methods", set(v["h1"]["methods"]) == set(m for m in METHODS if m != "erm"))
        _check(f"{key}: H1 GO flag is boolean per method",
               all(isinstance(mr["GO"], bool) for mr in v["h1"]["methods"].values()))
        _check(f"{key}: H2 reports two rankings + inversion bool",
               "ranking_by_accuracy" in v["h2"] and isinstance(v["h2"]["inversion"], bool))
        _check(f"{key}: H3 has a rho curve per robust method/score",
               all(all("curve" in v["h3"]["methods"][m][sc] for sc in SCORES)
                   for m in v["h3"]["methods"]))

    print("\n[4] accuracy-matching branch engaged (matched True and/or infeasible handled)")
    sample_key = next(iter(out["verdicts"]))
    h1 = out["verdicts"][sample_key]["h1"]
    matched_flags = [r.get("matched") for mr in h1["methods"].values() for r in mr["per_score"].values()]
    _check("matched flag present for every (method, score) H1 cell",
           all(isinstance(x, bool) for x in matched_flags))
    print(f"      matched=True cells: {sum(1 for x in matched_flags if x)}/{len(matched_flags)} "
          f"(infeasible cells report matched=False with a reason — both are valid)")

    print("\n[5] CSV + RESULTS_study.md emitters run")
    os.makedirs("results/study_synthetic", exist_ok=True)
    csv_path = "results/study_synthetic/grid_records_SYNTHETIC.csv"
    md_path = "results/study_synthetic/RESULTS_study_SYNTHETIC.md"
    write_csv(recs, csv_path)
    write_results_md(out, md_path, synthetic=True)
    _check("CSV written", os.path.exists(csv_path) and os.path.getsize(csv_path) > 0)
    _check("RESULTS_study (synthetic) written", os.path.exists(md_path) and os.path.getsize(md_path) > 0)

    print("\n" + "=" * 78)
    print("GRID LOGIC OK — full H1/H2/H3 chain + emitters validated on synthetic.")
    print("NO REAL NUMBERS CLAIMED. Real grid: notebooks/grid_robust_train.ipynb (Colab).")
    print(f"Synthetic artifacts: {csv_path} , {md_path}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
