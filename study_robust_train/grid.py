"""H1/H2/H3 grid orchestrator (spec §3, §6). Cheap-first last-layer arms.

Iterates: backbone x dataset x method x training-seed x score x rho_test x calibration-split,
producing tidy long records (one per conformal evaluation), then runs the pre-registered H1/H2/H3
verdicts per (backbone, dataset). Backbone-agnostic: consumes pre-extracted frozen features via
``GridData`` (no CUB coupling, no torch here). The heavy full-GroupDRO fine-tune and any 3rd/4th
dataset are NOT here — that is the post-checkpoint scope.

Per spec §2, every arm carries the Phase-0 gates: in-domain train+test, a per-method worst-group
accuracy gate (an arm below its floor is excluded with a logged reason, never shipped as valid),
the standardized probe, and predicted-only concepts (the verifiability section is separate).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import metrics
from .conformal_eval import RHO_SWEEP, evaluate
from .heads import head_probs
from .methods import METHODS, fit_method
from .verdicts import SCORES, h1_verdict, h2_verdict, h3_verdict

# Per-(dataset, method) worst-group accuracy floor (§2 gate). A method below its floor is EXCLUDED
# with a logged reason, never shipped as a valid arm. Floors are HARD; WG_ACC_SOFT_FLAG adds a soft
# warning band that is reported (not excluded). CelebA refs (spec): DFR worst-group >=0.85, ERM
# ~0.4-0.5 -> robust hard-floor 0.80 (tolerate seed variance), soft-flag DFR <0.85.
_DEFAULT_FLOOR = {"erm": 0.30, "dfr": 0.45, "afr": 0.40, "groupdro_ll": 0.45,
                  "balanced_subsample": 0.40}
_CELEBA_FLOOR = {"erm": 0.30, "dfr": 0.80, "afr": 0.75, "groupdro_ll": 0.75,
                 "balanced_subsample": 0.75}
WG_ACC_FLOOR = {"waterbirds": _DEFAULT_FLOOR, "celeba": _CELEBA_FLOOR}
WG_ACC_SOFT_FLAG = {("celeba", "dfr"): 0.85}   # (dataset, method) -> warn-if-below (no exclusion)


def worst_group_floor(dataset: str, method: str) -> float:
    return WG_ACC_FLOOR.get(dataset, _DEFAULT_FLOOR).get(method, 0.0)


@dataclass
class GridData:
    """Pre-extracted frozen features for one (backbone, dataset). y = binary task label;
    group = 2*y + spurious. All splits are the SAME composited distribution (in-domain)."""
    backbone: str
    dataset: str
    train: tuple      # (X, y, group) — ERM / GroupDRO / balanced subsample fit here
    reweight: tuple   # (X, y, group) — DFR / AFR fit here (held-out reweighting split)
    eval_domain: tuple  # (X, y, group) — conformal cal/test pools resampled from here
    n_classes: int


def run_grid(data_by_key: dict, *, methods=METHODS, scores=SCORES, rho_sweep=RHO_SWEEP,
             seeds=(0, 1, 2), n_splits=10, alpha=0.1, method_hp=None) -> dict:
    """Run the full last-layer grid over the provided (backbone, dataset) GridData objects.

    Returns {"records": [...], "excluded": [...], "verdicts": {key: {h1,h2,h3}}}. ``method_hp``
    optionally maps method name -> dict of hyperparameters.
    """
    method_hp = method_hp or {}
    records, excluded, flagged = [], [], []

    for key, gd in data_by_key.items():
        Xev, yev, gev = gd.eval_domain
        for method in methods:
            for seed in seeds:
                hp = method_hp.get(method, {})
                head = fit_method(method, gd.train, gd.reweight, seed=seed, **hp)
                probs = head_probs(head, Xev, gd.n_classes)
                y_pred = np.argmax(probs, axis=1)
                _, wg_acc = metrics.worst_group_accuracy(y_pred, yev, gev)

                # §2 per-method worst-group accuracy gate (HARD floor -> exclude)
                floor = worst_group_floor(gd.dataset, method)
                if wg_acc < floor:
                    excluded.append({"backbone": gd.backbone, "dataset": gd.dataset,
                                     "method": method, "seed": seed, "worst_group_acc": wg_acc,
                                     "floor": floor, "reason": "below per-method worst-group acc floor"})
                    continue
                # soft warning band (reported, NOT excluded)
                soft = WG_ACC_SOFT_FLAG.get((gd.dataset, method))
                if soft is not None and wg_acc < soft:
                    flagged.append({"backbone": gd.backbone, "dataset": gd.dataset,
                                    "method": method, "seed": seed, "worst_group_acc": wg_acc,
                                    "expected_min": soft,
                                    "reason": "below expected worst-group acc (kept; flag for review)"})

                for score in scores:
                    for rho in rho_sweep:
                        for sp in range(n_splits):
                            rec = evaluate(probs, yev, gev, score=score, alpha=alpha,
                                           rho_test=rho, split_seed=sp)
                            rec.update({"backbone": gd.backbone, "dataset": gd.dataset,
                                        "method": method, "train_seed": seed,
                                        "worst_group_acc": wg_acc})
                            records.append(rec)

    verdicts = build_verdicts(records, scores=scores, rho_sweep=rho_sweep)
    return {"records": records, "excluded": excluded, "flagged": flagged, "verdicts": verdicts}


def build_verdicts(records, *, scores=SCORES, rho_sweep=RHO_SWEEP) -> dict:
    """Compute H1/H2/H3 per (backbone, dataset) from tidy records. Used by run_grid AND reanalyze
    (so Tasks A/B re-run on an existing CSV with no retraining)."""
    keys = sorted({(r["backbone"], r["dataset"]) for r in records})
    verdicts = {}
    for key in keys:
        recs = [r for r in records if (r["backbone"], r["dataset"]) == key]
        present = sorted({r["method"] for r in recs})
        robust = [m for m in present if m != "erm"]
        if "erm" not in present or not robust:
            verdicts[key] = {"note": "ERM and/or robust arms missing/excluded; verdicts skipped",
                             "present_methods": present}
            continue
        verdicts[key] = {
            "h1": h1_verdict(recs, robust, scores=scores),
            "h2": h2_verdict(recs, present),
            "h3": h3_verdict(recs, robust, scores=scores, rho_sweep=rho_sweep),
        }
    return verdicts


# --------------------------------------------------------------------------------------
# tidy CSV + RESULTS_study.md emission
# --------------------------------------------------------------------------------------
_CSV_COLS = ["backbone", "dataset", "method", "train_seed", "score", "alpha", "rho_cal",
             "rho_test", "split_seed", "worst_group_acc", "base_top1", "marginal_cov",
             "worst_group", "worst_group_cov", "cov_gap", "mean_set_size",
             "worst_group_set_size", "set_size_disparity", "div_wasserstein1", "div_ks_stat",
             "div_ks_pvalue", "rho_test_realized"]


def write_csv(records, path):
    import csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_COLS, extrasaction="ignore")
        w.writeheader()
        for r in records:
            w.writerow(r)


def write_results_md(out: dict, path: str, *, synthetic: bool = False):
    """Write RESULTS_study.md: accuracy-matched comparison UP FRONT, uncontrolled labeled,
    H2 rankings, H3 survival. Negatives stated plainly."""
    L = []
    if synthetic:
        L += ["> **SYNTHETIC LOGIC-VALIDATION ONLY — NOT a scientific result.** Numbers below come",
              "> from a toy generator and validate the reporting/analysis machinery, not the",
              "> phenomenon. Real numbers come from the Colab grid run.\n"]
    L.append("# RESULTS_study.md — Conformal Burden v2 (H1/H2/H3, last-layer arms)\n")
    if out["excluded"]:
        L.append("## Excluded arms (§2 worst-group accuracy gate)\n")
        for e in out["excluded"]:
            L.append(f"- {e['backbone']}/{e['dataset']} {e['method']} seed{e['seed']}: "
                     f"worst-group acc {e['worst_group_acc']:.3f} < floor {e['floor']} — {e['reason']}")
        L.append("")

    for key, v in out["verdicts"].items():
        bb, ds = key
        L.append(f"## {bb} / {ds}\n")
        if "note" in v:
            L.append(f"_{v['note']}_ (present: {v.get('present_methods')})\n")
            continue

        # H1 — accuracy-matched FIRST, uncontrolled labeled
        h1 = v["h1"]
        L.append(f"### H1 (Transfer) — bar: {h1['bar']} @ rho={h1['rho']}\n")
        L.append("**Accuracy-matched divergence (PRIMARY — confound-controlled):**\n")
        L.append("| method | GO | scores reducing | per-score Delta(a*) [CI] |")
        L.append("|---|---|---|---|")
        for m, mr in h1["methods"].items():
            cells = []
            for sc, r in mr["per_score"].items():
                if r.get("matched"):
                    cells.append(f"{sc}: {r['delta_matched']:+.4f} [{r['ci'][0]:+.4f},{r['ci'][1]:+.4f}]"
                                 f"{'*' if r['reduces'] else ''}")
                else:
                    cells.append(f"{sc}: unmatched ({r.get('reason','')})")
            L.append(f"| {m} | {'GO' if mr['GO'] else 'no'} | {mr['n_scores_reduce']}/{mr['n_scores']} | "
                     + "<br>".join(cells) + " |")
        L.append("\n_* = reduction CI excludes 0. Delta>0 = robust method lowers divergence at matched accuracy._\n")
        L.append("**Uncontrolled (raw) divergence — reported with base accuracy, NOT the verdict:**\n")
        L.append("| method | score | raw divergence | base top-1 |")
        L.append("|---|---|---|---|")
        for m, mr in h1["methods"].items():
            for sc, r in mr["per_score"].items():
                rr = r.get("raw_robust", {})
                L.append(f"| {m} | {sc} | {rr.get('divergence_mean', float('nan')):.4f} | "
                         f"{rr.get('base_top1_mean', float('nan')):.3f} |")
        # ERM reference row
        any_m = next(iter(h1["methods"].values()))
        for sc, r in any_m["per_score"].items():
            re = r.get("raw_ref", {})
            L.append(f"| erm (ref) | {sc} | {re.get('divergence_mean', float('nan')):.4f} | "
                     f"{re.get('base_top1_mean', float('nan')):.3f} |")
        L.append("")

        # H2 — ranking inversion WITH CIs (Task B)
        h2 = v["h2"]
        L.append("### H2 (Ranking inversion — headline)\n")
        L.append(f"- by worst-group **accuracy** (higher=better): {' > '.join(h2['ranking_by_accuracy'])} "
                 f"(top: **{h2['top_by_accuracy']}**)")
        L.append(f"- by worst-group **burden** ({h2['burden_key']}, lower=better): "
                 f"{' > '.join(h2['ranking_by_burden'])} (top: **{h2['top_by_burden']}**)")
        L.append("")
        L.append(f"| method | worst-group acc | {h2['burden_key']} [95% CI] |")
        L.append("|---|---|---|")
        for m in h2["ranking_by_accuracy"]:
            ci = h2["burden_ci"][m]
            L.append(f"| {m} | {h2['worst_group_acc'][m]:.3f} | "
                     f"{h2['burden'][m]:.4f} [{ci[0]:.4f},{ci[1]:.4f}] |")
        diff_ci = h2["inversion_diff_ci"]
        L.append("")
        L.append(f"- point-estimate inversion: **{h2['inversion_point']}**")
        L.append(f"- inversion REAL (burden of acc-top minus burden-top separated, CI excludes 0): "
                 f"**{h2['inversion_real']}** "
                 f"(Δ{h2['burden_key']}={h2['inversion_diff']:+.4f} [{diff_ci[0]:+.4f},{diff_ci[1]:+.4f}])")
        if h2['inversion_point'] and not h2['inversion_real']:
            L.append("  - point inversion but CIs OVERLAP → treat as noise, not a real inversion.")
        L.append("")

        # H3 — burden survival (Task A): divergence channel + set-size relocation, coverage separate
        h3 = v["h3"]
        L.append(f"### H3 (Shift survival) — calibrate ρ={h3['rho_cal']}, sweep {h3['rho_sweep']}\n")
        L.append(f"_What H3 measures (NOT coverage): {h3['criterion']}_\n")
        L.append("**(1) Divergence survival — does the H1 accuracy-matched reduction hold across ρ?**\n")
        L.append("| method | score | label |")
        L.append("|---|---|---|")
        for m, mm in h3["methods"].items():
            for sc, r in mm["per_score"].items():
                L.append(f"| {m} | {sc} | {r['failure_type']} |")
        L.append("\n_Labels: survived / never_held / held_then_broke@ρ / undefined@ρ "
                 "(matched comparison undefined where accuracy supports don't overlap)._\n")
        L.append("**(2) Set-size disparity vs ρ — does the burden RELOCATE to set-size inflation? "
                 "(burden2026: relocate, not remove)**\n")
        L.append("| method | ρ=0.95 | ρ=0.50 | inflates under shift | (robust − ERM) @ρ=0.50 |")
        L.append("|---|---|---|---|---|")
        for m, mm in h3["methods"].items():
            sd = {d["rho_test"]: d for d in mm["setsize_disparity_curve"]}
            lo_rho = min(h3["rho_sweep"]); hi_rho = max(h3["rho_sweep"])
            L.append(f"| {m} | {sd[hi_rho]['robust']:.3f} | {sd[lo_rho]['robust']:.3f} | "
                     f"{mm['setsize_inflates_under_shift']} | {sd[lo_rho]['robust_minus_erm']:+.3f} |")
        L.append("")
        L.append("**Coverage stability (reported, NOT the H3 criterion):**\n")
        L.append("| method | worst-group cov range over ρ | flat (<0.05) |")
        L.append("|---|---|---|")
        for m, cs in h3["coverage_stability"].items():
            L.append(f"| {m} | {cs['range']:.3f} | {cs['flat']} |")
        L.append("\n_Flat coverage with growing set-size disparity = the burden relocated to set "
                 "size under shift, not removed — the expected 'relocate, not remove' story._\n")

    if out.get("flagged"):
        L.append("## Flagged arms (soft worst-group warning — kept, not excluded)\n")
        for fl in out["flagged"]:
            L.append(f"- {fl['backbone']}/{fl['dataset']} {fl['method']} seed{fl['seed']}: "
                     f"worst-group acc {fl['worst_group_acc']:.3f} < expected {fl['expected_min']} "
                     f"— {fl['reason']}")
        L.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))


# --------------------------------------------------------------------------------------
# reanalysis from a saved CSV (Tasks A/B with NO retraining)
# --------------------------------------------------------------------------------------
_FLOAT_COLS = {"alpha", "rho_cal", "rho_test", "rho_test_realized", "worst_group_acc", "base_top1",
               "marginal_cov", "worst_group_cov", "cov_gap", "mean_set_size",
               "worst_group_set_size", "set_size_disparity", "div_wasserstein1", "div_ks_stat",
               "div_ks_pvalue"}
_INT_COLS = {"train_seed", "split_seed", "worst_group"}


def records_from_csv(path: str) -> list:
    """Load tidy grid records from a CSV written by write_csv (coercing numeric columns)."""
    import csv
    recs = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            r = {}
            for k, val in row.items():
                if k in _FLOAT_COLS:
                    r[k] = float(val)
                elif k in _INT_COLS:
                    r[k] = int(float(val))
                else:
                    r[k] = val
            recs.append(r)
    return recs


def reanalyze(csv_path: str, *, results_md="RESULTS_study.md", figdir="results/study/figures",
              scores=SCORES, rho_sweep=None, make_figs=True) -> dict:
    """Recompute H1/H2/H3 (with Task-A H3 + Task-B H2 CIs) from a saved CSV and rewrite
    RESULTS_study.md + figures. NO retraining — pure re-analysis of existing records. The rho
    sweep is INFERRED from the CSV (so it matches whatever sweep was actually run)."""
    records = records_from_csv(csv_path)
    if rho_sweep is None:
        rho_sweep = tuple(sorted({r["rho_test"] for r in records}, reverse=True))
    out = {"records": records, "excluded": [], "flagged": [],
           "verdicts": build_verdicts(records, scores=scores, rho_sweep=rho_sweep)}
    write_results_md(out, results_md, synthetic=False)
    if make_figs:
        from .figures import make_figures
        make_figures(out, figdir)
    return out
