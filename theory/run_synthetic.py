"""Synthetic Gaussian theory testbed runner — verifies P1 (validity) and P2 (separation).

Runs the shortcut-strength sweep across seeds, computes ALL reliability signals from cached
model outputs, evaluates error-detection AUROC / contamination / minority-correctness, runs
verifiability-gated selective conformal to check retained coverage per group, and emits:
  - results/runs/<run_id>/logs/*.parquet   per-sample logs (one row per test sample)
  - results/runs/<run_id>/synthetic_results.json
  - results/figures/theory_p2.pdf          the P2 / regime-map theory figure
  - results/runs/<run_id>/manifest.json

Pure numpy/scipy/sklearn — no torch, no GPU. This is the piece intended to run anywhere.

Usage:
    python -m theory.run_synthetic --config configs/synthetic.yaml [--quick]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# allow running as a script from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config_util import load_config
from conformal import scores as cscores
from conformal import selective, split_conformal, validity
from eval import metrics
from eval.phase1_eval import evaluate_signals, kill_switch_verdict
from signals import ncv as ncv_sig
from signals import registry
from theory import gaussian_model as gm
from theory import synthetic_models as sm
from vgscp_logging.manifest import RunManifest, make_run_id
from vgscp_logging.parquet_logger import ParquetLogger

PRIMARY = "V_full"
EVAL_SIGNALS = ["conf_msp", "trust", "ensemble_disagree", "V_comp", "V_sound", "V_full"]


def _compute_split_signals(split, f, members, ncv, beta):
    x, y, a = split["x"], split["y"], split["a"]
    probs = f.predict_proba(x)
    y_pred = probs.argmax(axis=1)
    correct = (y_pred == y).astype(int)

    member_probs = sm.ensemble_member_probs(members, x)
    pA_SM = ncv.pA_given_SM(x)
    pA_SA = ncv.pA_given_SA(x, y_pred)

    sigs = registry.build_signals(
        probs,
        y_pred,
        train_feats=None,  # trust handled separately below to reuse train refs
        member_probs=member_probs,
        pA_given_SM=pA_SM,
        pA_given_SA=pA_SA,
        n_classes=2,
        beta=beta,
    )
    return {
        "x": x,
        "y": y,
        "a": a,
        "probs": probs,
        "y_pred": y_pred,
        "correct": correct,
        "is_minority": split["is_minority"],
        "group_id": split["group_id"],
        "pA_SM": pA_SM,
        "pA_SA": pA_SA,
        "signals": sigs,
    }


def _add_trust(split_data, train_x, train_y):
    from signals.trust import trust_score

    split_data["signals"]["trust"] = trust_score(
        train_x, train_y, split_data["x"], split_data["y_pred"], n_classes=2, prefer_torch=False
    )


def run_one(cfg_ds, common, strength, seed):
    cfg = gm.GaussianConfig(
        n=cfg_ds["n_per_split"],
        d_core=cfg_ds["d_core"],
        d_spurious=cfg_ds["d_spurious"],
        d_noise=cfg_ds["d_noise"],
        core_sep=cfg_ds["core_sep"],
        spurious_sep=cfg_ds["spurious_sep"],
        minority_frac=cfg_ds["minority_frac"],
        shortcut_strength=strength,
    )
    splits = gm.generate_splits(cfg, seed)
    train = splits["train"]

    f = sm.fit_base_classifier(train["x"], train["y"], seed=seed)
    members = sm.fit_ensemble(train["x"], train["y"], common["ensemble"]["n_members"], seed=seed)
    ncv = sm.SyntheticNCV(splits["train"]["core_slice"], splits["train"]["spurious_slice"]).fit(
        train["x"], train["y"], seed=seed
    )

    # compute V_comp/V_sound on D_learn to tune beta (minority error-detection AUROC)
    learn = _compute_split_signals(splits["d_learn"], f, members, ncv, beta=0.5)
    _add_trust(learn, train["x"], train["y"])
    vc_learn = learn["signals"]["V_comp"]
    vs_learn = learn["signals"]["V_sound"]
    beta, _ = ncv_sig.tune_beta(
        vc_learn,
        vs_learn,
        learn["correct"],
        common["ncv"]["beta_grid"],
        objective_fn=metrics.error_detection_auroc,
        minority_mask=learn["is_minority"],
    )

    cal = _compute_split_signals(splits["d_cal"], f, members, ncv, beta=beta)
    test = _compute_split_signals(splits["d_test"], f, members, ncv, beta=beta)
    _add_trust(cal, train["x"], train["y"])
    _add_trust(test, train["x"], train["y"])
    # recompute V_full on learn with tuned beta for completeness
    learn["signals"]["V_full"] = ncv_sig.v_full(vc_learn, vs_learn, beta)

    # ---- selective conformal validity (P1) using V_full gate, THR score ----
    alpha = common["conformal"]["alpha"]
    thr_cal = cscores.thr_scores_all(cal["probs"])
    thr_test = cscores.thr_scores_all(test["probs"])
    g_learn = learn["signals"]["V_full"]
    p1_rows = []
    for b in common["budgets"]:
        res = selective.selective_conformal(
            g_learn, cal["signals"]["V_full"], test["signals"]["V_full"],
            thr_cal, cal["y"], thr_test, test["y"], alpha, b,
        )
        # per-group coverage among retained test
        kept = res.retained_test_mask
        cov_overall = res.coverage
        groups = test["group_id"][kept]
        ytk = test["y"][kept]
        memb = res.membership
        cov_by_group = split_conformal.empirical_coverage_by_group(memb, ytk, groups)
        passed, lower = validity.check_marginal_coverage(cov_overall, alpha, res.n_retained_test)
        p1_rows.append({
            "shortcut_strength": strength, "seed": seed, "budget": b,
            "tau": res.tau, "qhat": res.qhat,
            "coverage": cov_overall, "coverage_lower_ci": lower,
            "coverage_ok": passed, "avg_set_size": res.avg_set_size,
            "abstention_rate": res.abstention_rate,
            "n_retained_test": res.n_retained_test,
            "cov_by_group": json_safe(cov_by_group),
        })

    # ---- build tidy test dataframe for metrics ----
    df = pd.DataFrame({
        "correct": test["correct"],
        "is_minority": test["is_minority"].astype(int),
        "spurious_attr": test["a"],
        "y_true": test["y"],
        "y_pred": test["y_pred"],
        "group_id": test["group_id"],
        "p_true": test["probs"][np.arange(len(test["y"])), test["y"]],
        "conf_msp": test["signals"]["conf_msp"],
        "trust": test["signals"]["trust"],
        "ensemble_disagree": test["signals"]["ensemble_disagree"],
        "V_comp": test["signals"]["V_comp"],
        "V_sound": test["signals"]["V_sound"],
        "V_full": test["signals"]["V_full"],
        "R_adv": test["signals"]["_R_adv"],
        "shortcut_strength": strength,
        "seed": seed,
    })

    # Per-run bootstrap is cheap here because the synthetic aggregate reports POINT estimates
    # across the (>=10) seeds with across-seed std as the uncertainty. A small per-run resample
    # count keeps point estimates stable while keeping the 70-run sweep fast.
    reports = evaluate_signals(df, EVAL_SIGNALS, budgets=tuple(common["budgets"][1:6]),
                               n_resamples=120, seed=seed)
    verdict = kill_switch_verdict(df, primary=PRIMARY, n_resamples=120, seed=seed)

    concept_w = ncv.concept_weights()
    return {
        "df": df,
        "reports": reports,
        "verdict": verdict,
        "p1_rows": p1_rows,
        "beta": beta,
        "concept_weights": concept_w,
        "minority_acc": float(test["correct"][test["is_minority"]].mean()),
        "majority_acc": float(test["correct"][~test["is_minority"].astype(bool)].mean()),
    }


def json_safe(d):
    return {str(k): (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in d.items()}


def ci_to_dict(ci):
    if ci is None:
        return None
    return {"est": ci.estimate, "lo": ci.lo, "hi": ci.hi, "se": ci.se}


def aggregate(results_by_strength):
    """Aggregate per-signal metrics across seeds for each shortcut strength."""
    agg = {}
    for strength, runs in results_by_strength.items():
        per_signal = {}
        for sig in EVAL_SIGNALS:
            aurocs_min = [r["reports"][sig].auroc_minority.estimate
                          for r in runs if sig in r["reports"] and r["reports"][sig].auroc_minority]
            aurocs_overall = [r["reports"][sig].auroc_overall.estimate
                              for r in runs if sig in r["reports"]]
            contam = [r["reports"][sig].contamination_auroc for r in runs if sig in r["reports"]]
            mi = [r["reports"][sig].mutual_info for r in runs if sig in r["reports"]]
            aurc_min = [r["reports"][sig].aurc_minority.estimate
                        for r in runs if sig in r["reports"] and r["reports"][sig].aurc_minority]
            per_signal[sig] = {
                "auroc_minority_mean": float(np.nanmean(aurocs_min)) if aurocs_min else float("nan"),
                "auroc_minority_std": float(np.nanstd(aurocs_min)) if aurocs_min else float("nan"),
                "auroc_overall_mean": float(np.nanmean(aurocs_overall)) if aurocs_overall else float("nan"),
                "contamination_auroc_mean": float(np.nanmean(contam)) if contam else float("nan"),
                "mutual_info_mean": float(np.nanmean(mi)) if mi else float("nan"),
                "aurc_minority_mean": float(np.nanmean(aurc_min)) if aurc_min else float("nan"),
            }
        go = sum(1 for r in runs if r["verdict"].decision == "GO")
        agg[strength] = {
            "per_signal": per_signal,
            "n_seeds": len(runs),
            "n_GO": go,
            "minority_acc_mean": float(np.mean([r["minority_acc"] for r in runs])),
            "majority_acc_mean": float(np.mean([r["majority_acc"] for r in runs])),
            "beta_mean": float(np.mean([r["beta"] for r in runs])),
            "arthur_core_l2_mean": float(np.mean([r["concept_weights"]["core_l2"] for r in runs])),
            "arthur_spurious_l2_mean": float(np.mean([r["concept_weights"]["spurious_l2"] for r in runs])),
        }
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/synthetic.yaml")
    ap.add_argument("--quick", action="store_true", help="fewer seeds/strengths for a smoke test")
    ap.add_argument("--timestamp", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    ds = cfg["dataset"]
    common = cfg["common"]
    exp = cfg["experiment"]

    strengths = ds["shortcut_strength_sweep"]
    seeds = exp["seeds"]
    if args.quick:
        strengths = [0.5, 0.9, 0.99]
        seeds = seeds[:3]

    ts = args.timestamp or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_id = make_run_id(exp["name"], ts)
    run_dir = os.path.join(exp["out_dir"], "runs", run_id)
    logger = ParquetLogger(os.path.join(run_dir, "logs"))
    manifest = RunManifest(run_dir, cfg)
    manifest.write_config_yaml(cfg)

    results_by_strength = {}
    p1_all = []
    all_test_frames = []
    for s in strengths:
        runs = []
        for seed in seeds:
            print(f"[synthetic] shortcut_strength={s} seed={seed}")
            r = run_one(ds, common, s, seed)
            runs.append(r)
            p1_all.extend(r["p1_rows"])
            all_test_frames.append(r["df"])
        results_by_strength[s] = runs

    # write per-sample logs (concatenated test rows)
    big = pd.concat(all_test_frames, ignore_index=True)
    big.insert(0, "split", "d_test")
    big.insert(0, "dataset", "synthetic_gaussian")
    big.insert(0, "sample_id", np.arange(len(big)))
    logger.write_frame(big, "synthetic_test")
    logger.write_frame(pd.DataFrame(p1_all), "synthetic_p1_coverage")

    agg = aggregate(results_by_strength)

    # P1 summary: worst retained coverage vs target across all budgets/strengths
    p1_df = pd.DataFrame(p1_all)
    target = 1 - common["conformal"]["alpha"]
    p1_summary = {
        "target_coverage": target,
        "min_retained_coverage": float(p1_df["coverage"].min()),
        "mean_retained_coverage": float(p1_df["coverage"].mean()),
        "frac_budgets_coverage_ok": float(p1_df["coverage_ok"].mean()),
    }

    # P2 verdict at the default strength
    default_s = ds["shortcut_strength_default"]
    p2 = agg.get(default_s) or agg[strengths[-1]]
    out = {
        "run_id": run_id,
        "config_path": os.path.abspath(args.config),
        "strengths": strengths,
        "seeds": seeds,
        "aggregate_by_strength": agg,
        "p1_validity": p1_summary,
        "p2_default_strength": default_s,
        "p2_summary": p2["per_signal"],
        "primary": PRIMARY,
    }
    with open(os.path.join(run_dir, "synthetic_results.json"), "w") as fh:
        json.dump(out, fh, indent=2, default=str)

    manifest.set("synthetic_results", out)
    manifest.save()

    # figure
    try:
        from viz.theory_figure import make_theory_figure

        fig_path = os.path.join(exp["out_dir"], "figures", "theory_p2.pdf")
        make_theory_figure(agg, p1_df, EVAL_SIGNALS, fig_path)
        print(f"[synthetic] wrote figure -> {fig_path}")
    except Exception as e:  # don't fail the run if matplotlib missing
        print(f"[synthetic] figure skipped: {e}")

    # console summary
    print("\n==== SYNTHETIC P1/P2 SUMMARY ====")
    print(f"run_id: {run_id}")
    print(f"P1 (validity): min retained coverage={p1_summary['min_retained_coverage']:.3f} "
          f"(target {target:.2f}); frac budgets OK={p1_summary['frac_budgets_coverage_ok']:.2f}")
    print(f"\nP2 (separation) at shortcut_strength={default_s} "
          f"[minority acc={p2['minority_acc_mean']:.3f}, majority acc={p2['majority_acc_mean']:.3f}]:")
    print(f"{'signal':<18}{'minAUROC':>10}{'contamAUROC':>13}{'MI':>8}{'minAURC':>10}")
    for sig in EVAL_SIGNALS:
        ps = p2["per_signal"][sig]
        print(f"{sig:<18}{ps['auroc_minority_mean']:>10.3f}{ps['contamination_auroc_mean']:>13.3f}"
              f"{ps['mutual_info_mean']:>8.3f}{ps['aurc_minority_mean']:>10.3f}")
    print(f"\nGO seeds at default strength: {p2['n_GO']}/{p2['n_seeds']}")
    print(f"Arthur concept weight L2: core={p2['arthur_core_l2_mean']:.2f} "
          f"spurious={p2['arthur_spurious_l2_mean']:.2f} (low spurious => earned robustness)")
    print(f"results -> {run_dir}/synthetic_results.json")


if __name__ == "__main__":
    main()
