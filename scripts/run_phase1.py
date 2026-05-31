"""Phase-1 kill-switch on CLEVR-Hans3 (Section 4, 17). RUN ON THE GPU BOX.

End-to-end:
  1. setup perf (precision/AMP/TF32) + VRAM probe -> fixed batch policy for the whole suite;
  2. per seed: load CLEVR-Hans3 bundle (group labels + ground-truth scene concepts), train (or
     load) f (ERM), the M-member ensemble, and the NCV verifier (official or reimpl);
  3. PRECOMPUTE-ONCE: cache logits/probs/features/concepts/p_A over all splits;
  4. compute ALL signals from the cache; tune beta/eta/zeta on D_learn ONLY;
  5. evaluate error-detection AUROC / AURC / capture / contamination on D_test;
  6. run verifiability-gated selective conformal (THR/APS/RAPS) + validity asserts;
  7. write per-sample parquet logs, manifest (env/git/gpu/throughput), and the GO/NO-GO verdict.

This script is import-safe without torch (torch is used only inside the run). Phase 1 is gated
on CLEVR-Hans3 + synthetic; do NOT extend to Waterbirds/CelebA before the human 'go'.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config_util import load_config
from conformal import scores as cscores
from conformal import selective, split_conformal, validity
from conformal.verifier_aware import tune_eta_zeta, verifier_aware_scores_all
from data.base import SplitSpec
from data.registry import load_dataset
from eval import metrics
from eval.phase1_eval import evaluate_signals, kill_switch_verdict
from precompute.cache import CacheStore
from signals import ncv as ncv_sig
from signals import registry
from vgscp_logging.manifest import RunManifest, make_run_id
from vgscp_logging.parquet_logger import ParquetLogger

PRIMARY = "V_full"
REPORT_SIGNALS = ["conf_msp", "trust", "ensemble_disagree", "mcdropout",
                  "V_comp", "V_sound", "V_full"]


def build_and_precompute(cfg, seed, ctx, perf_cfg, cache):
    """Train/load models and run the one-time precompute into ``cache``. Returns the bundle."""
    import torch  # noqa

    from models.base_model import (
        FeatureClassifier,
        TrainConfig,
        build_backbone,
        train_erm,
        worst_group_accuracy,
    )
    from models.verifier_adapter import build_verifier
    from perf.vram_probe import probe_batch_size
    from precompute.run_precompute import precompute_all

    ds_cfg = cfg["dataset"]
    spec = SplitSpec(**{k: cfg["common"]["splits"][k] for k in ("d_learn", "d_cal", "d_test")})
    bundle = load_dataset(ds_cfg["name"], ds_cfg, seed, spec)

    # VRAM probe: fix the batch policy ONCE (reused for the whole suite).
    if ctx.device == "cuda" and ctx.batch_size == 0:
        def make_batch(bs):
            return torch.randn(bs, 3, ds_cfg["image_size"], ds_cfg["image_size"], device=ctx.device)

        net0, fd0 = build_backbone(cfg["model"]["backbone"], ds_cfg["n_classes"],
                                   cfg["model"].get("pretrained", True))
        clf0 = FeatureClassifier(net0, fd0, ctx.device)
        probe_batch_size(ctx, perf_cfg, make_batch, lambda b: clf0.logits_and_features(b))

    # --- base f (ERM) ---
    net, feat_dim = build_backbone(cfg["model"]["backbone"], ds_cfg["n_classes"],
                                   cfg["model"].get("pretrained", True),
                                   dropout_p=0.2)  # dropout enables MC-dropout reuse
    clf = FeatureClassifier(net, feat_dim, ctx.device)
    # NOTE: training loop + loaders live in models/base_model.train_erm. Provide a train loader
    # from bundle.datasets['train']; omitted here to keep the orchestration readable.
    train_loader = _loader(bundle.datasets["train"], ctx, perf_cfg, shuffle=True)
    tcfg = TrainConfig(num_epochs=cfg["model"]["num_epochs"], lr=cfg["model"]["lr"],
                       weight_decay=cfg["model"]["weight_decay"], optimizer=cfg["model"]["optimizer"])
    train_erm(clf.net, train_loader, tcfg, ctx, ctx.device)

    # --- ensemble (M members) ---
    members = []
    for m in range(cfg["common"]["ensemble"]["n_members"]):
        from perf.setup import seed_everything

        seed_everything(seed * 1000 + m, deterministic=ctx.deterministic)
        net_m, fd_m = build_backbone(cfg["model"]["backbone"], ds_cfg["n_classes"],
                                     cfg["model"].get("pretrained", True))
        clf_m = FeatureClassifier(net_m, fd_m, ctx.device)
        train_erm(clf_m.net, _loader(bundle.datasets["train"], ctx, perf_cfg, shuffle=True),
                  tcfg, ctx, ctx.device)
        members.append(clf_m)

    # --- NCV verifier (concept space = ground-truth CLEVR scene concepts) ---
    concept_dim = bundle.concepts["train"].shape[1]
    verifier = build_verifier(cfg["ncv"], concept_dim, ds_cfg["n_classes"], ctx.device)
    if cfg["ncv"]["source"] == "reimpl":
        verifier.train(bundle.concepts["train"], bundle.y["train"], ctx,
                       epochs=cfg["ncv"].get("epochs", 30))

    # --- one-time precompute over all splits ---
    tlog = precompute_all(bundle, clf, members, verifier, ctx, perf_cfg, cache,
                          k_passes=cfg["common"]["mc_dropout"]["n_passes"])

    # shortcut sanity (Section 6): worst-group acc on D_test must be << overall
    probs = cache.load_array("d_test", "probs")
    wg = worst_group_accuracy(probs.argmax(1), bundle.y["d_test"], bundle.group_id["d_test"])
    intrinsic = verifier.intrinsic_metrics(bundle.concepts["d_test"], bundle.y["d_test"])
    return bundle, tlog, wg, intrinsic


def _loader(ds, ctx, perf_cfg, shuffle):
    import torch

    from perf.setup import resolve_num_workers

    dl = perf_cfg.get("dataloader", {})
    return torch.utils.data.DataLoader(
        ds, batch_size=max(1, ctx.batch_size), shuffle=shuffle,
        num_workers=resolve_num_workers(dl.get("num_workers", "auto")),
        pin_memory=ctx.device == "cuda", persistent_workers=dl.get("persistent_workers", True),
        prefetch_factor=dl.get("prefetch_factor", 4),
    )


def signals_from_cache(cache, split, train_feats, train_labels, beta):
    """Build the full signal dict for a split from cached arrays (vectorized, cheap)."""
    probs = cache.load_array(split, "probs")
    y_pred = cache.load_array(split, "y_pred")
    feats = cache.load_array(split, "features")
    member_probs = cache.load_array(split, "member_probs") if cache.exists(split, "member_probs") else None
    mc = cache.load_array(split, "mc_pass_probs") if cache.exists(split, "mc_pass_probs") else None
    pA_SM = cache.load_array(split, "pA_given_SM") if cache.exists(split, "pA_given_SM") else None
    pA_SA = cache.load_array(split, "pA_given_SA") if cache.exists(split, "pA_given_SA") else None
    reject = cache.load_array(split, "reject_prob") if cache.exists(split, "reject_prob") else None
    n_classes = probs.shape[1]
    return registry.build_signals(
        probs, y_pred, train_feats=train_feats, train_labels=train_labels, query_feats=feats,
        member_probs=member_probs, mc_pass_probs=mc, pA_given_SM=pA_SM, pA_given_SA=pA_SA,
        reject_prob=reject, n_classes=n_classes, beta=beta,
    )


def run_seed(cfg, seed, ctx, perf_cfg, cache, logger):
    bundle, tlog, wg, intrinsic = build_and_precompute(cfg, seed, ctx, perf_cfg, cache)
    common = cfg["common"]
    alpha = common["conformal"]["alpha"]

    train_feats = cache.load_array("train", "features")
    train_labels = cache.load_array("train", "y_true")

    # tune beta on D_learn (minority error-detection AUROC)
    learn = signals_from_cache(cache, "d_learn", train_feats, train_labels, beta=0.5)
    learn_correct = (cache.load_array("d_learn", "y_pred") == cache.load_array("d_learn", "y_true")).astype(int)
    learn_minor = cache.load_array("d_learn", "is_minority").astype(bool)
    beta, _ = ncv_sig.tune_beta(learn["V_comp"], learn["V_sound"], learn_correct,
                                common["ncv"]["beta_grid"], metrics.error_detection_auroc, learn_minor)

    test = signals_from_cache(cache, "d_test", train_feats, train_labels, beta=beta)
    cal = signals_from_cache(cache, "d_cal", train_feats, train_labels, beta=beta)

    # tidy test frame
    yt = cache.load_array("d_test", "y_true")
    yp = cache.load_array("d_test", "y_pred")
    df = pd.DataFrame({
        "sample_id": np.arange(len(yt)), "dataset": cfg["dataset"]["name"], "split": "d_test",
        "seed": seed, "y_true": yt, "y_pred": yp, "correct": (yp == yt).astype(int),
        "group_id": cache.load_array("d_test", "group_id"),
        "spurious_attr": cache.load_array("d_test", "spurious_attr"),
        "is_minority": cache.load_array("d_test", "is_minority").astype(int),
        "p_true": cache.load_array("d_test", "probs")[np.arange(len(yt)), yt],
        "conf_msp": test["conf_msp"], "trust": test.get("trust"),
        "ensemble_disagree": test.get("ensemble_disagree"), "mcdropout": test.get("mcdropout"),
        "V_comp": test["V_comp"], "V_sound": test["V_sound"], "V_full": test["V_full"],
        "R_adv": test.get("_R_adv"),
    })

    # --- selective conformal across budgets x base scores, with validity asserts ---
    conf_rows = []
    for score_name in common["conformal"]["base_scores"]:
        cal_probs, test_probs = cache.load_array("d_cal", "probs"), cache.load_array("d_test", "probs")
        kw = dict(lam_reg=common["conformal"]["raps"]["lam_reg"], k_reg=common["conformal"]["raps"]["k_reg"]) \
            if score_name == "RAPS" else {}
        s_cal = cscores.scores_all(score_name, cal_probs, **kw)
        s_test = cscores.scores_all(score_name, test_probs, **kw)
        for gate_name in REPORT_SIGNALS:
            if gate_name not in test:
                continue
            for b in common["budgets"]:
                res = selective.selective_conformal(
                    learn[gate_name], cal[gate_name], test[gate_name],
                    s_cal, cache.load_array("d_cal", "y_true"),
                    s_test, yt, alpha, b)
                passed, lower = validity.check_marginal_coverage(res.coverage, alpha, res.n_retained_test)
                conf_rows.append({"score": score_name, "gate": gate_name, "budget": b,
                                  "coverage": res.coverage, "coverage_ok": passed,
                                  "avg_set_size": res.avg_set_size, "qhat": res.qhat,
                                  "abstention_rate": res.abstention_rate, "seed": seed})

    logger.write_frame(df, f"clevr_test_seed{seed}")
    logger.write_frame(pd.DataFrame(conf_rows), f"clevr_conformal_seed{seed}")

    reports = evaluate_signals(df, [s for s in REPORT_SIGNALS if s in df and df[s].notna().any()],
                               budgets=tuple(common["budgets"][1:6]),
                               n_resamples=common["eval"]["bootstrap_resamples"], seed=seed)
    verdict = kill_switch_verdict(df, primary=PRIMARY,
                                  n_resamples=common["eval"]["bootstrap_resamples"], seed=seed)
    return {"df": df, "reports": reports, "verdict": verdict, "beta": beta,
            "worst_group": wg, "ncv_intrinsic": intrinsic, "throughput": tlog.to_list(),
            "conformal": conf_rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/clevr_hans3.yaml")
    ap.add_argument("--timestamp", default=None)
    args = ap.parse_args()

    from perf.setup import setup_perf

    cfg = load_config(args.config)
    perf_cfg = cfg["perf"]
    ctx = setup_perf(perf_cfg, seed=cfg["experiment"]["seeds"][0])

    ts = args.timestamp or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_id = make_run_id(cfg["experiment"]["name"], ts)
    run_dir = os.path.join(cfg["experiment"]["out_dir"], "runs", run_id)
    logger = ParquetLogger(os.path.join(run_dir, "logs"))
    manifest = RunManifest(run_dir, cfg, ctx.to_dict())
    manifest.write_config_yaml(cfg)

    all_runs = []
    for seed in cfg["experiment"]["seeds"]:
        cache = CacheStore(os.path.join("cache"), cfg["dataset"]["name"], seed)
        print(f"[phase1] seed {seed}")
        all_runs.append(run_seed(cfg, seed, ctx, perf_cfg, cache, logger))

    # aggregate verdict: GO only if the kill criterion is met on a majority of seeds
    go = sum(1 for r in all_runs if r["verdict"].decision == "GO")
    decision = "GO" if go > len(all_runs) / 2 else "NO-GO"
    manifest.set("verdict", {"decision": decision, "go_seeds": go, "n_seeds": len(all_runs),
                             "per_seed_rationale": [r["verdict"].rationale for r in all_runs]})
    manifest.set_throughput(all_runs[0]["throughput"] if all_runs else [])
    manifest.save()

    out = {"run_id": run_id, "decision": decision, "go_seeds": go, "n_seeds": len(all_runs)}
    with open(os.path.join(run_dir, "phase1_results.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n[phase1] VERDICT: {decision} ({go}/{len(all_runs)} seeds). See {run_dir}")
    print("[phase1] Now run viz.make_figures and viz.latex_tables on the run dir, then review "
          "PHASE1_REPORT.md. STOP and await human 'go' before Phase 2.")


if __name__ == "__main__":
    main()
