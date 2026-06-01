"""PREMISE-2 GATE — does adversarial verifiability (Morgana) add value over a plain concept probe?

Concept-space CONTRAST on the SAME Waterbirds f (task Sections 1-5):
  A. attributes_only : 312 CUB per-image attributes (clean/causal)        — cleanliness control
  B. mixed           : [312 CUB attributes | K frozen-CLIP scene concepts] — decisive test

For each space, on the Waterbirds MINORITY group, we compare (all "higher = more reliable"):
  f-baselines        : conf_msp, trust (on f), ensemble_disagree, mcdropout
  concept controls   : probe_concept (plain concept->y probe confidence), trust_concept
  ours               : V_comp (Morgana off), V_sound, V_full (Morgana on)
and apply the pre-committed premise-2 criterion (eval.phase1_eval.premise2_verdict). A downstream
verifiability-gated selective-conformal pass (THR/APS/RAPS) confirms validity on the winner.

GPU once (ERM ResNet-50 + CLIP scene concepts), then everything CPU/vectorized. The REAL run needs
Waterbirds + CUB + torch + open_clip (handed to Colab, like the probes). A pure-numpy/CPU
``--smoke`` path fabricates both concept spaces + a shortcut f and exercises the ENTIRE
verifier+signals+verdict+conformal pipeline locally with no dataset.

    python -m scripts.run_premise2 --config configs/premise2_waterbirds.yaml   # real (Colab/GPU)
    python -m scripts.run_premise2 --smoke                                     # local self-test
"""
from __future__ import annotations

# Import torch FIRST (before numpy/pandas/sklearn): on Windows, pandas/sklearn load OpenMP/VC
# runtime DLLs that break torch's later DLL initialization. Harmless ordering no-op on Linux/Colab.
try:  # pragma: no cover - environment dependent
    import torch  # noqa: F401
except Exception:
    pass

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
from conformal import selective, validity
from eval import metrics
from eval.phase1_eval import evaluate_signals, premise2_verdict
from models.concept_extractor_clip import ConceptStandardizer
from signals import ncv as ncv_sig
from signals import registry
from vgscp_logging.manifest import RunManifest, make_run_id

# Signals reported per concept space (premise-2). f-baselines + concept controls + ours.
REPORT_SIGNALS = [
    "conf_msp", "trust", "ensemble_disagree", "mcdropout",   # f-feature baselines
    "probe_concept", "trust_concept",                        # concept-space controls (key)
    "V_comp", "V_sound", "V_full", "V_comp_moff",            # ours (V_comp_moff = Morgana-off)
]
SPLITS = ("train", "d_learn", "d_cal", "d_test")
PRIMARY = "V_full"


# ======================================================================================
# Core (shared by real + smoke): train the verifiers and compute every signal for a space
# ======================================================================================
def _train_verifier(ncv_cfg, concept_dim, n_classes, device, train_concepts, train_y, seed):
    from models.verifier_adapter import build_verifier

    v = build_verifier(ncv_cfg, concept_dim, n_classes, device=device)
    v.train(train_concepts, train_y, seed=seed)
    return v


def evaluate_concept_space(space, concepts_raw, fdata, cfg, device="cpu", seed=0,
                           n_resamples=1000):
    """Run the full premise-2 comparison for ONE concept space.

    ``concepts_raw[split]`` is an (N_split, D) raw concept matrix aligned to ``fdata[split]``.
    ``fdata[split]`` holds cached f outputs: probs, y_pred, y_true, features, member_probs,
    mc_pass_probs, group_id, spurious_attr, is_minority. Returns a result dict (signals, tuned
    beta, verdict, test DataFrame, conformal rows, intrinsic metrics, contamination)."""
    common = cfg["common"]
    n_classes = int(fdata["train"]["probs"].shape[1])

    # --- TRAIN-only standardization of the concept space (no leakage), reused everywhere ---
    if cfg.get("concepts", {}).get("standardize", True):
        std = ConceptStandardizer.fit(concepts_raw["train"])
        concepts = {s: std.transform(concepts_raw[s]).astype(np.float32) for s in concepts_raw}
    else:
        concepts = {s: concepts_raw[s].astype(np.float32) for s in concepts_raw}

    concept_dim = concepts["train"].shape[1]
    train_y = fdata["train"]["y_true"]

    # --- two verifiers: Morgana ON (primary) and Morgana OFF (required ablation) ---
    ncv_on = dict(cfg["ncv"]); ncv_on["morgana"] = "on"; ncv_on["standardize"] = False
    ncv_off = dict(cfg["ncv"]); ncv_off["morgana"] = "off"; ncv_off["standardize"] = False
    v_on = _train_verifier(ncv_on, concept_dim, n_classes, device, concepts["train"], train_y, seed)
    v_off = _train_verifier(ncv_off, concept_dim, n_classes, device, concepts["train"], train_y, seed)

    intrinsic = {
        "morgana_on": v_on.intrinsic_metrics(concepts["d_test"], fdata["d_test"]["y_true"]),
        "morgana_off": v_off.intrinsic_metrics(concepts["d_test"], fdata["d_test"]["y_true"]),
    }

    # --- verifier outputs per eval split (train concepts only feed verifier training) ---
    eval_splits = ("d_learn", "d_cal", "d_test")
    vo_on = {s: v_on.predict(concepts[s], fdata[s]["y_pred"]) for s in eval_splits}
    vo_off = {s: v_off.predict(concepts[s], fdata[s]["y_pred"]) for s in eval_splits}

    train_feats = fdata["train"]["features"]
    train_labels = fdata["train"]["y_true"]
    probe_kind = cfg.get("probe", {}).get("concept_probe_kind", "logistic")

    def signals_for(split, beta):
        f = fdata[split]
        out = registry.build_signals(
            f["probs"], f["y_pred"],
            train_feats=train_feats, train_labels=train_labels, query_feats=f["features"],
            member_probs=f.get("member_probs"), mc_pass_probs=f.get("mc_pass_probs"),
            pA_given_SM=vo_on[split].pA_given_SM, pA_given_SA=vo_on[split].pA_given_SA,
            reject_prob=vo_on[split].reject_prob, n_classes=n_classes, beta=beta,
            train_concepts=concepts["train"], query_concepts=concepts[split],
            concept_probe_kind=probe_kind,
        )
        # Morgana-OFF completeness signal (its V_full == V_comp by construction)
        out["V_comp_moff"] = ncv_sig.v_comp(vo_off[split].pA_given_SM, f["y_pred"])
        return out

    # --- tune beta on D_learn (minority error-detection AUROC), V_comp/V_sound are beta-free ---
    learn = signals_for("d_learn", beta=0.5)
    lf = fdata["d_learn"]
    learn_correct = (lf["y_pred"] == lf["y_true"]).astype(int)
    learn_minor = lf["is_minority"].astype(bool)
    beta, _ = ncv_sig.tune_beta(learn["V_comp"], learn["V_sound"], learn_correct,
                                common["ncv"]["beta_grid"], metrics.error_detection_auroc,
                                learn_minor)
    # recompute V_full at tuned beta (vectorized) for all splits
    sig = {s: signals_for(s, beta) for s in ("d_learn", "d_cal", "d_test")}

    # --- tidy D_test frame ---
    ft = fdata["d_test"]
    yt, yp = ft["y_true"], ft["y_pred"]
    df = pd.DataFrame({
        "sample_id": np.arange(len(yt)), "concept_space": space, "seed": seed,
        "y_true": yt, "y_pred": yp, "correct": (yp == yt).astype(int),
        "group_id": ft["group_id"], "spurious_attr": ft["spurious_attr"],
        "is_minority": ft["is_minority"].astype(int),
        "p_true": ft["probs"][np.arange(len(yt)), yt],
    })
    for name in REPORT_SIGNALS:
        if name in sig["d_test"]:
            df[name] = sig["d_test"][name]

    # --- premise-2 verdict (minority group) ---
    verdict = premise2_verdict(df, space=space, n_resamples=n_resamples, seed=seed,
                               alpha_sig=common["eval"].get("significance_alpha", 0.05))

    # --- per-signal metric suite (AUROC / AURC / contamination / capture) ---
    present = [s for s in REPORT_SIGNALS if s in df and df[s].notna().any()]
    reports = evaluate_signals(df, present, budgets=tuple(common["budgets"][1:6]),
                               n_resamples=n_resamples, seed=seed)

    # --- verifiability-gated selective conformal on the winner (V_full), validity asserts ---
    conf_rows = _selective_conformal_confirmation(common, sig, fdata, gate=PRIMARY)

    return {"space": space, "beta": float(beta), "verdict": verdict, "df": df,
            "reports": reports, "conformal": conf_rows, "intrinsic": intrinsic,
            "present_signals": present}


def _selective_conformal_confirmation(common, sig, fdata, gate="V_full"):
    """Run the gated selective-conformal pipeline (THR/APS/RAPS x budgets) on ``gate``; assert
    retained marginal coverage ~ 1-alpha and record minority retained selective risk."""
    alpha = common["conformal"]["alpha"]
    cal_probs, test_probs = fdata["d_cal"]["probs"], fdata["d_test"]["probs"]
    cal_y, test_y = fdata["d_cal"]["y_true"], fdata["d_test"]["y_true"]
    test_minor = fdata["d_test"]["is_minority"].astype(bool)
    test_correct = (fdata["d_test"]["y_pred"] == test_y).astype(int)
    rows = []
    for score_name in common["conformal"]["base_scores"]:
        kw = (dict(lam_reg=common["conformal"]["raps"]["lam_reg"],
                   k_reg=common["conformal"]["raps"]["k_reg"]) if score_name == "RAPS" else {})
        s_cal = cscores.scores_all(score_name, cal_probs, **kw)
        s_test = cscores.scores_all(score_name, test_probs, **kw)
        for b in common["budgets"]:
            res = selective.selective_conformal(
                sig["d_learn"][gate], sig["d_cal"][gate], sig["d_test"][gate],
                s_cal, cal_y, s_test, test_y, alpha, b)
            passed, lower = validity.check_marginal_coverage(res.coverage, alpha, res.n_retained_test)
            kept = res.retained_test_mask
            kept_minor = kept & test_minor
            minor_risk = (float(1.0 - test_correct[kept_minor].mean())
                          if kept_minor.any() else float("nan"))
            rows.append({"gate": gate, "score": score_name, "budget": float(b),
                         "coverage": res.coverage, "coverage_ok": bool(passed),
                         "coverage_lower_ci": lower, "avg_set_size": res.avg_set_size,
                         "abstention_rate": res.abstention_rate,
                         "n_retained_test": res.n_retained_test,
                         "minority_selective_risk": minor_risk})
    return rows


# ======================================================================================
# Smoke: fabricate both concept spaces + a shortcut f, run the full pipeline on CPU
# ======================================================================================
def _make_smoke_fdata_and_concepts(seed=0, n=1200):
    """Synthetic Waterbirds-like setup. f is a SHORTCUT classifier that reads the background
    (place), so it is confident-but-wrong on the minority (y != place). attributes_only concepts
    carry clean y-signal; the mixed space appends spurious place-tracking concepts the adversary
    can exploit. This is a pipeline/logic self-test, NOT a premise-2 claim."""
    rng = np.random.default_rng(seed)
    n_classes = 2

    def gen(m, s):
        r = np.random.default_rng(1000 + s)
        y = r.integers(0, 2, m)
        minority = r.random(m) < 0.25
        place = np.where(minority, 1 - y, y)
        # f leans on `place` (the shortcut) but not perfectly: it predicts `place` on ~60% of
        # samples and the true `y` on the rest. On the majority (place==y) it is ~always right;
        # on the minority (place!=y) it is right ~40% of the time -> confident-but-wrong mix,
        # so minority error-detection AUROC is well defined (both correct and wrong present).
        use_shortcut = r.random(m) < 0.6
        pred = np.where(use_shortcut, place, y)
        conf = 0.78 + r.random(m) * 0.18                      # confident (0.78-0.96)
        p1 = np.where(pred == 1, conf, 1 - conf)
        probs = np.stack([1 - p1, p1], axis=1)
        probs = probs / probs.sum(1, keepdims=True)
        y_pred = probs.argmax(1)
        # f features: dominated by place (so trust-on-f is contaminated, blind to minority errors)
        feats = np.concatenate([
            (2 * place - 1)[:, None] * 1.6 + r.normal(0, 0.5, (m, 4)),
            r.normal(0, 1, (m, 4)),
        ], axis=1).astype(np.float32)
        # ensemble: 3 members all sharing the shortcut -> agree on minority errors
        member = np.stack([probs + r.normal(0, 0.02, probs.shape) for _ in range(3)], 0)
        member = np.clip(member, 1e-3, 1 - 1e-3); member /= member.sum(2, keepdims=True)
        mc = np.stack([probs + r.normal(0, 0.02, probs.shape) for _ in range(10)], 0)
        mc = np.clip(mc, 1e-3, 1 - 1e-3); mc /= mc.sum(2, keepdims=True)
        # concepts: clean core (y) attributes; spurious (place) scene concepts
        sy, sp = (2 * y - 1).astype(np.float32), (2 * place - 1).astype(np.float32)
        core = np.stack([1.6 * sy + r.normal(0, 0.7, m) for _ in range(8)], 1).astype(np.float32)
        scene = np.stack([2.0 * sp + r.normal(0, 0.5, m) for _ in range(6)], 1).astype(np.float32)
        f = {"probs": probs.astype(np.float32), "y_pred": y_pred.astype(np.int64),
             "y_true": y.astype(np.int64), "features": feats,
             "member_probs": member.astype(np.float32), "mc_pass_probs": mc.astype(np.float32),
             "group_id": (2 * y + place).astype(np.int64), "spurious_attr": place.astype(np.int64),
             "is_minority": (y != place)}
        return f, core, scene

    fdata, attrs, mixed = {}, {}, {}
    sizes = {"train": n, "d_learn": n // 2, "d_cal": n // 2, "d_test": n // 2}
    for i, sp in enumerate(SPLITS):
        f, core, scene = gen(sizes[sp], i)
        fdata[sp] = f
        attrs[sp] = core
        mixed[sp] = np.concatenate([core, scene], axis=1)
    return fdata, {"attributes_only": attrs, "mixed": mixed}


def run_smoke(cfg, n_resamples=300):
    fdata, concept_sets = _make_smoke_fdata_and_concepts(seed=0)
    cfg = dict(cfg)
    cfg.setdefault("ncv", {})
    cfg["ncv"].update({"source": "reimpl", "epochs": 20, "hidden": 32, "merlin_sparsity": 3,
                       "morgana_sparsity": 3, "n_train_max": None, "batch_size": 256})
    results = {}
    for space in cfg.get("concept_spaces", ["attributes_only", "mixed"]):
        results[space] = evaluate_concept_space(space, concept_sets[space], fdata, cfg,
                                                 device="cpu", seed=0, n_resamples=n_resamples)
    return results


# ======================================================================================
# Real run (Colab/GPU): train f + ensemble + mc-dropout, build both concept spaces, evaluate
# ======================================================================================
def _build_fdata_real(cfg, seed):
    """Train ERM ResNet-50 f + ensemble + MC-dropout on Waterbirds and precompute f outputs over
    all splits into an in-memory fdata dict (reusing precompute/ stage helpers). GPU once."""
    import torch

    from data.base import SplitSpec
    from data.waterbirds import load_waterbirds
    from models.base_model import (FeatureClassifier, TrainConfig, build_backbone, train_erm,
                                    worst_group_accuracy)
    from perf.setup import seed_everything, setup_perf
    from precompute.run_precompute import (precompute_classifier, precompute_mc_dropout,
                                           precompute_member_probs)

    ctx = setup_perf(cfg["perf"], seed=seed)
    if ctx.batch_size == 0:
        ctx.batch_size = 64 if ctx.device == "cuda" else 16
    spec = SplitSpec(**{k: cfg["common"]["splits"][k] for k in ("d_learn", "d_cal", "d_test")})
    mps = cfg.get("probe", {}).get("max_per_split")
    bundle = load_waterbirds(cfg["dataset"], seed, split_spec=spec, max_per_split=mps,
                             build_datasets=True)

    mcfg = cfg["model"]
    net, fd = build_backbone(mcfg["backbone"], cfg["dataset"]["n_classes"],
                             mcfg.get("pretrained", True), dropout_p=mcfg.get("dropout_p", 0.2))
    clf = FeatureClassifier(net, fd, ctx.device)
    tcfg = TrainConfig(num_epochs=mcfg["num_epochs"], lr=mcfg["lr"],
                       weight_decay=mcfg["weight_decay"], optimizer=mcfg["optimizer"])
    train_loader = torch.utils.data.DataLoader(bundle.datasets["train"],
                                               batch_size=max(8, ctx.batch_size), shuffle=True,
                                               num_workers=2)
    train_erm(clf.net, train_loader, tcfg, ctx, ctx.device)

    members = []
    for m in range(cfg["common"]["ensemble"]["n_members"]):
        seed_everything(seed * 1000 + m, deterministic=ctx.deterministic)
        nm, fdm = build_backbone(mcfg["backbone"], cfg["dataset"]["n_classes"],
                                 mcfg.get("pretrained", True), dropout_p=mcfg.get("dropout_p", 0.2))
        cm = FeatureClassifier(nm, fdm, ctx.device)
        dl = torch.utils.data.DataLoader(bundle.datasets["train"], batch_size=max(8, ctx.batch_size),
                                         shuffle=True, num_workers=2)
        train_erm(cm.net, dl, tcfg, ctx, ctx.device)
        members.append(cm)

    from perf.throughput import ThroughputLog
    tlog = ThroughputLog()
    fdata = {}
    k_passes = cfg["common"]["mc_dropout"]["n_passes"]
    for sp in SPLITS:
        ds = bundle.datasets[sp]
        logits, probs, feats = precompute_classifier(clf, ds, ctx, cfg["perf"], tlog, f"f::{sp}")
        member_probs = precompute_member_probs(members, ds, ctx, cfg["perf"], tlog)
        mc = precompute_mc_dropout(clf, ds, k_passes, ctx, cfg["perf"], tlog)
        fdata[sp] = {"probs": probs, "y_pred": probs.argmax(1).astype(np.int64),
                     "y_true": bundle.y[sp].astype(np.int64), "features": feats,
                     "member_probs": member_probs, "mc_pass_probs": mc,
                     "group_id": bundle.group_id[sp], "spurious_attr": bundle.spurious_attr[sp],
                     "is_minority": bundle.is_minority[sp]}
    wg = worst_group_accuracy(fdata["d_test"]["y_pred"], fdata["d_test"]["y_true"],
                              fdata["d_test"]["group_id"])
    return bundle, fdata, ctx, wg, tlog


def _build_concept_spaces_real(cfg, bundle, ctx):
    """Build attributes_only (CUB per-image attributes) and mixed ([CUB | CLIP scene]) concept
    matrices aligned to every split. CUB attrs are CPU; CLIP scene concepts need GPU + open_clip."""
    from data.cub_attributes import load_cub_attribute_concepts

    ccfg = cfg["cub"]
    attrs, attr_names, join_info = load_cub_attribute_concepts(
        bundle, ccfg["root"], splits=SPLITS, download=ccfg.get("download", False),
        url=ccfg.get("url"), use_certainty=ccfg.get("use_certainty", False),
        min_coverage=ccfg.get("min_coverage", 0.99))

    spaces = {"attributes_only": {s: attrs[s] for s in SPLITS}}
    info = {"cub_join": join_info, "attr_dim": int(attrs["train"].shape[1])}

    if "mixed" in cfg.get("concept_spaces", []):
        from models.concept_extractor_clip import CLIPConceptExtractor

        clipcfg = cfg["clip"]
        extractor = CLIPConceptExtractor(clipcfg["model_name"], clipcfg["pretrained"],
                                         clipcfg["scene_concept_bank"], device=ctx.device,
                                         temperature_softmax=clipcfg.get("temperature_softmax", False),
                                         temperature=clipcfg.get("temperature", 0.01))
        scene = {s: extractor.encode_paths(bundle.meta["paths"][s]) for s in SPLITS}
        spaces["mixed"] = {s: np.concatenate([attrs[s], scene[s]], axis=1).astype(np.float32)
                           for s in SPLITS}
        info["clip_scene_dim"] = int(scene["train"].shape[1])
    return spaces, info


def run_real(cfg, seed):
    bundle, fdata, ctx, wg, tlog = _build_fdata_real(cfg, seed)
    spaces, concept_info = _build_concept_spaces_real(cfg, bundle, ctx)
    n_resamples = cfg["common"]["eval"]["bootstrap_resamples"]
    results = {}
    for space in cfg.get("concept_spaces", list(spaces)):
        results[space] = evaluate_concept_space(space, spaces[space], fdata, cfg,
                                                 device=ctx.device, seed=seed,
                                                 n_resamples=n_resamples)
    return {"results": results, "worst_group": wg, "concept_info": concept_info,
            "throughput": tlog.to_list(), "perf": ctx.to_dict()}


# ======================================================================================
# Reporting
# ======================================================================================
def _ci_str(ci):
    return "--" if ci is None or not np.isfinite(ci.estimate) else \
        f"{ci.estimate:.3f} [{ci.lo:.3f}, {ci.hi:.3f}]"


def _space_table(res):
    reps = res["reports"]
    lines = ["| Signal | minority AUROC ↑ | minority AURC ↓ | contamination AUROC ↓ |",
             "|---|---|---|---|"]
    for s in REPORT_SIGNALS:
        if s not in reps:
            continue
        r = reps[s]
        lines.append(f"| `{s}` | {_ci_str(r.auroc_minority)} | {_ci_str(r.aurc_minority)} | "
                     f"{r.contamination_auroc:.3f} |")
    return "\n".join(lines)


def _conformal_summary(rows):
    if not rows:
        return "_(no conformal rows)_"
    ok = sum(1 for r in rows if r["coverage_ok"])
    covs = [r["coverage"] for r in rows if np.isfinite(r["coverage"])]
    worst = min(covs) if covs else float("nan")
    return (f"{ok}/{len(rows)} (score × budget) points pass retained marginal coverage ≥ 1−α; "
            f"worst realized coverage = {worst:.3f}.")


def write_report(path, cfg, payload, mode):
    res = payload["results"]
    decisive = "mixed" if "mixed" in res else list(res)[0]
    dv = res[decisive]["verdict"]
    parts = [
        "# PREMISE-2 REPORT — does adversarial verifiability (Morgana) beat a plain concept probe?",
        "",
        f"**Date:** {datetime.now(timezone.utc).date()} · **Run mode:** {mode} · "
        f"**Decisive space:** `{decisive}` · **VERDICT: {dv.label}**",
        "",
        "The premise-2 question: does the Prover–Verifier adversarial mechanism add value **over a "
        "plain probe on the same concept space**, and over trust + ensemble, at flagging "
        "**minority** errors? Contribution = selective prediction / set trustworthiness, NOT a new "
        "coverage guarantee. **A NULL/PARTIAL result is valid and reported honestly.**",
        "",
        "## Pre-committed criterion (committed before numbers; eval.phase1_eval.premise2_verdict)",
        "- **NOVELTY-VALIDATED** iff, in the **mixed** space, `V_full` beats `probe_concept` AND "
        "`trust_concept` AND `trust` AND `ensemble_disagree` on minority AUROC (CI/paired p<0.05) "
        "**AND** `V_full` > `V_comp` (Morgana adds value).",
        "- **PARTIAL** if V beats the f-baselines but not the concept-space controls (⇒ \"clean "
        "concepts help\"), or beats controls but Morgana on/off shows no gap.",
        "- **NULL** if V beats no concept-space control anywhere.",
        "",
    ]
    for space in res:
        r = res[space]
        intr = r["intrinsic"]
        parts += [
            f"## Concept space: `{space}`  (β={r['beta']:.2f})",
            f"**Verdict: {r['verdict'].label}**",
            "",
            r["verdict"].rationale,
            "",
            "Verifier sanity (D_test): "
            f"Morgana-on completeness={intr['morgana_on']['merlin_acc']:.3f} / "
            f"morgana_acc={intr['morgana_on']['morgana_acc']:.3f}; "
            f"Morgana-off completeness={intr['morgana_off']['merlin_acc']:.3f}.",
            "",
            _space_table(r),
            "",
            f"**Selective-conformal confirmation (gate=V_full):** {_conformal_summary(r['conformal'])}",
            "",
        ]
    if "worst_group" in payload:
        wg = payload["worst_group"]
        parts += [f"## Regime check (f)\nWorst-group acc = {wg['worst_group']:.3f} vs overall "
                  f"{wg['overall']:.3f} (per-group {wg['per_group']}).", ""]
    parts += [
        "## Regime map (honesty)",
        "- **attributes_only** (clean/causal concepts): expected V_comp wins via concept-bottleneck "
        "cleanliness while Morgana is ~idle — the cleanliness control.",
        "- **mixed** ([CUB | CLIP scene]): the decisive test — the CLIP scene concepts are the "
        "spurious concepts the adversary can exploit.",
        "- **clip_only**: known RED from the earlier probe (background-dominated, minority "
        "AUROC(concepts→y) ≈ 0.49) — the third regime point, not rerun here.",
        "",
        "> Contribution is selective prediction / set trustworthiness. No new coverage guarantee is "
        "claimed; the conformal pass only confirms retained validity. **Do NOT proceed to a "
        "multi-seed paper sweep before a human reviews this report.**",
    ]
    if mode == "smoke":
        parts.insert(3, "\n> ⚠️ **SMOKE (synthetic) run** — fabricated concept spaces + shortcut f. "
                        "This validates the verifier+signals+verdict+conformal PIPELINE end-to-end "
                        "on CPU. The real GREEN/AMBER/RED premise-2 numbers come from the Colab run.\n")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts) + "\n")


def _to_jsonable(o):
    if isinstance(o, dict):
        return {k: _to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_to_jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def _payload_json(payload):
    out = {"results": {}}
    for space, r in payload["results"].items():
        out["results"][space] = {
            "space": space, "beta": r["beta"], "verdict": r["verdict"].label,
            "verdict_rationale": r["verdict"].rationale, "intrinsic": r["intrinsic"],
            "minority_auroc": {s: (rep.auroc_minority.estimate if rep.auroc_minority else None)
                               for s, rep in r["reports"].items()},
            "minority_aurc": {s: (rep.aurc_minority.estimate if rep.aurc_minority else None)
                              for s, rep in r["reports"].items()},
            "contamination_auroc": {s: rep.contamination_auroc for s, rep in r["reports"].items()},
            "conformal": r["conformal"],
        }
    for k in ("worst_group", "concept_info", "perf"):
        if k in payload:
            out[k] = payload[k]
    return _to_jsonable(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/premise2_waterbirds.yaml")
    ap.add_argument("--smoke", action="store_true", help="synthetic CPU self-test (no dataset)")
    ap.add_argument("--timestamp", default=None)
    args = ap.parse_args()
    ts = args.timestamp or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    if args.smoke:
        cfg = load_config(args.config) if os.path.exists(args.config) else {
            "common": load_config("configs/common.yaml")["common"]}
        # ensure the nested 'common' is present (smoke can run before perf/torch are configured)
        cfg.setdefault("common", load_config("configs/common.yaml").get("common", {}))
        cfg.setdefault("concept_spaces", ["attributes_only", "mixed"])
        results = run_smoke(cfg)
        payload = {"results": results}
        for space, r in results.items():
            print(f"[{space}] VERDICT: {r['verdict'].label} (beta={r['beta']:.2f})")
            print("   " + r["verdict"].rationale)
        write_report("PREMISE2_REPORT.md", cfg, payload, mode="smoke")
        print("\n[premise2] SMOKE OK - full pipeline ran on synthetic data. Real numbers: Colab.")
        print("[premise2] wrote PREMISE2_REPORT.md (smoke). STOP - await the real run + review.")
        return

    cfg = load_config(args.config)
    seed = cfg["experiment"]["seeds"][0]
    run_id = make_run_id(cfg["experiment"]["name"], ts)
    run_dir = os.path.join(cfg["experiment"]["out_dir"], "runs", run_id)
    os.makedirs(run_dir, exist_ok=True)
    manifest = RunManifest(run_dir, cfg)
    manifest.write_config_yaml(cfg)

    from vgscp_logging.parquet_logger import ParquetLogger
    logger = ParquetLogger(os.path.join(run_dir, "logs"))

    payload = run_real(cfg, seed)
    for space, r in payload["results"].items():
        logger.write_frame(r["df"], f"premise2_{space}_test_seed{seed}")
        logger.write_frame(pd.DataFrame(r["conformal"]), f"premise2_{space}_conformal_seed{seed}")
        print(f"[{space}] VERDICT: {r['verdict'].label} (beta={r['beta']:.2f})")
        print("   " + r["verdict"].rationale)

    pj = _payload_json(payload)
    manifest.set("premise2", pj)
    manifest.save()
    with open(os.path.join(run_dir, "premise2_results.json"), "w") as f:
        json.dump(pj, f, indent=2)
    write_report(os.path.join(run_dir, "PREMISE2_REPORT.md"), cfg, payload, mode="real")
    write_report("PREMISE2_REPORT.md", cfg, payload, mode="real")
    # paper-ready tables/figures (reuse viz/) from the logged per-sample frames
    try:
        from viz import latex_tables, make_figures  # noqa: F401
        print("[premise2] tables/figures: run `python -m viz.latex_tables --run "
              f"{run_dir}` and `python -m viz.make_figures --run {run_dir}`")
    except Exception:
        pass
    print(f"\n[premise2] results -> {run_dir}. STOP. Await human review of PREMISE2_REPORT.md "
          "before any multi-seed paper sweep.")


if __name__ == "__main__":
    main()
