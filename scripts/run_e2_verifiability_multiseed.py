"""E2 — Verifiability collapse, CLEAN MULTI-SEED regeneration of the existing kill-check.

For minority-error detection, compare the error-detection AUROC of:
  * feature-space trust            (`trust`)
  * concept-distance trust         (`trust_concept`)            <- THE BAR
  * verifiability signals          (`V_comp` Morgana-on / `V_comp_moff` Morgana-off / `V_sound` /
                                     `V_full`)
  * counterfactual support gap     (`V_gap`)
plus each signal's SPURIOUS-ATTRIBUTE (contamination) AUROC (near 0.5 = clean), in BOTH a clean
concept space (`attributes_only`) and a MIXED clean+spurious concept space (`mixed`), reported as
mean ± std across >=10 seeds.

This REUSES the single-seed machinery verbatim: premise-2's synthetic fabricator / real builders
(scripts.run_premise2), the reimpl verifier (models.verifier_adapter), the canonical signal set
(signals.registry.build_signals -- which produces trust, trust_concept, V_comp/V_sound/V_full AND
the support-gap V_gap in one call), the spuriousness-derived clean_mask (signals.spurious_gap), and
the AUROC / contamination metrics (eval.metrics). It only adds the multi-seed loop + aggregation.

Pre-committed expectation (report whatever appears): verifiability TIES/LOSES to concept-distance
trust; COLLAPSES below chance in the mixed space; support-gap uninformative (~0.5).

    python -m scripts.run_e2_verifiability_multiseed --smoke --seeds 10        # CPU self-test
    python -m scripts.run_e2_verifiability_multiseed --config configs/premise2_waterbirds.yaml --seeds 10
"""
from __future__ import annotations

try:  # pragma: no cover
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
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

from config_util import load_config
from eval import metrics
from models.concept_extractor_clip import ConceptStandardizer
from models.verifier_adapter import build_verifier
from signals import ncv as ncv_sig
from signals import registry
from signals.spurious_gap import clean_mask_from_rho, concept_spuriousness
from scripts.run_premise2 import _make_smoke_fdata_and_concepts

# E2 signal set: display label -> registry key. Order = report order.
E2_SIGNALS = [
    ("feature_trust", "trust"),
    ("concept_trust", "trust_concept"),
    ("probe_concept", "probe_concept"),
    ("V_comp_moff", "V_comp_moff"),
    ("V_comp", "V_comp"),
    ("V_sound", "V_sound"),
    ("V_full", "V_full"),
    ("support_gap", "V_gap"),
    ("support_clean", "support_clean"),
]
SPACES = ("attributes_only", "mixed")
SPLITS = ("train", "d_learn", "d_cal", "d_test")
BETA_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
# Spurious background/scene concepts mixed into the "mixed" space (the channel the adversary can
# exploit). Overridable via cfg.clip.scene_concept_bank.
DEFAULT_SCENE = ["a photo taken near water", "an ocean in the background", "a lake in the background",
                 "a forest in the background", "trees in the background", "dry land"]


def _smoke_ncv(cfg):
    cfg = dict(cfg)
    cfg.setdefault("ncv", {})
    cfg["ncv"] = dict(cfg["ncv"])
    cfg["ncv"].update({"source": "reimpl", "epochs": 20, "hidden": 32, "merlin_sparsity": 3,
                       "morgana_sparsity": 3, "n_train_max": None, "batch_size": 256})
    return cfg


def _resolve_device(cfg):
    """Use CUDA when available (so the verifier's Arthur MLP + greedy prover search run on GPU);
    fall back to CPU. Overridable via cfg.e2.device."""
    dev = cfg.get("e2", {}).get("device")
    if dev:
        return dev
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _subsample_eval(fdata, concepts_raw, splits, max_per_group, seed):
    """Stratified (by is_minority) per-seed subsample of the named eval splits to <= max_per_group
    samples PER group, so the verifier's greedy search runs on a few thousand points instead of the
    full ~11k (AUROC is stable there). Returns reduced (fdata, concepts) COPIES (originals untouched)
    and a dict of the realized n per split. No-op when max_per_group is None."""
    if not max_per_group:
        return fdata, concepts_raw, {s: int(len(fdata[s]["y_true"])) for s in splits}
    rng = np.random.default_rng(seed + 99)
    fd, cc, ns = dict(fdata), {k: v for k, v in concepts_raw.items()}, {}
    for s in splits:
        minor = np.asarray(fdata[s]["is_minority"]).astype(bool)
        keep = []
        for grp_mask in (minor, ~minor):
            idx = np.where(grp_mask)[0]
            if len(idx) > max_per_group:
                idx = rng.choice(idx, size=max_per_group, replace=False)
            keep.append(idx)
        keep = np.sort(np.concatenate(keep))
        fd[s] = {k: (v[keep] if isinstance(v, np.ndarray) and v.shape[:1] == minor.shape else v)
                 for k, v in fdata[s].items()}
        cc[s] = concepts_raw[s][keep]
        ns[s] = int(len(keep))
    return fd, cc, ns


def evaluate_space_seed(space, concepts_raw, fdata, cfg, seed, standardize=True, device="cpu"):
    """Per-(space, seed): train verifiers, build the full signal set, return per-signal minority
    error-detection AUROC + spurious-attribute (contamination) AUROC on d_test."""
    max_per_group = cfg.get("e2", {}).get("eval_max_per_group")
    fdata, concepts_raw, eval_ns = _subsample_eval(fdata, concepts_raw, ("d_learn", "d_test"),
                                                   max_per_group, seed)
    n_classes = int(fdata["train"]["probs"].shape[1])
    if standardize:
        std = ConceptStandardizer.fit(concepts_raw["train"])
        concepts = {s: std.transform(concepts_raw[s]).astype(np.float32) for s in concepts_raw}
    else:
        concepts = {s: concepts_raw[s].astype(np.float32) for s in concepts_raw}
    concept_dim = concepts["train"].shape[1]
    train_y = fdata["train"]["y_true"]

    ncv_on = dict(cfg["ncv"]); ncv_on["morgana"] = "on"; ncv_on["standardize"] = False
    ncv_off = dict(cfg["ncv"]); ncv_off["morgana"] = "off"; ncv_off["standardize"] = False
    v_on = build_verifier(ncv_on, concept_dim, n_classes, device=device); v_on.train(
        concepts["train"], train_y, seed=seed)
    v_off = build_verifier(ncv_off, concept_dim, n_classes, device=device); v_off.train(
        concepts["train"], train_y, seed=seed)

    # clean_mask from TRAIN-only concept spuriousness vs the spurious attr (place)
    rho = concept_spuriousness(concepts["train"], np.asarray(fdata["train"]["spurious_attr"]))
    clean_mask, _ = clean_mask_from_rho(rho)

    train_feats = fdata["train"]["features"]
    vo_on = {s: v_on.predict(concepts[s], fdata[s]["y_pred"]) for s in ("d_learn", "d_test")}
    vo_off = {s: v_off.predict(concepts[s], fdata[s]["y_pred"]) for s in ("d_learn", "d_test")}

    def sigs(split, beta):
        f = fdata[split]
        out = registry.build_signals(
            f["probs"], f["y_pred"], train_feats=train_feats, train_labels=train_y,
            query_feats=f["features"], member_probs=f.get("member_probs"),
            mc_pass_probs=f.get("mc_pass_probs"), pA_given_SM=vo_on[split].pA_given_SM,
            pA_given_SA=vo_on[split].pA_given_SA, reject_prob=vo_on[split].reject_prob,
            n_classes=n_classes, beta=beta, train_concepts=concepts["train"],
            query_concepts=concepts[split], verifier=v_on, clean_mask=clean_mask, gap_lam=0.0)
        out["V_comp_moff"] = ncv_sig.v_comp(vo_off[split].pA_given_SM, f["y_pred"])
        return out

    # tune beta on d_learn minority error-detection AUROC (faithful to premise-2)
    learn = sigs("d_learn", 0.5)
    lf = fdata["d_learn"]
    lc = (lf["y_pred"] == lf["y_true"]).astype(int)
    lm = lf["is_minority"].astype(bool)
    beta, _ = ncv_sig.tune_beta(learn["V_comp"], learn["V_sound"], lc, BETA_GRID,
                                metrics.error_detection_auroc, lm)
    test = sigs("d_test", beta)

    ft = fdata["d_test"]
    correct = (ft["y_pred"] == ft["y_true"]).astype(int)
    minor = ft["is_minority"].astype(bool)
    spur = np.asarray(ft["spurious_attr"])
    yt = np.asarray(ft["y_true"])
    rows = []
    for label, key in E2_SIGNALS:
        if key not in test:
            continue
        s = np.asarray(test[key])
        auroc_min = (metrics.error_detection_auroc(s[minor], correct[minor])
                     if minor.sum() > 0 and len(np.unique(correct[minor])) > 1 else float("nan"))
        contam = metrics.contamination_auroc(s, spur, yt)
        rows.append({"space": space, "seed": int(seed), "signal": label,
                     "minority_auroc": auroc_min, "contamination_auroc": contam, "beta": float(beta),
                     "n_eval": int(eval_ns["d_test"]), "device": device})
    return rows


def run(cfg, mode, n_seeds):
    rows = []
    # SMOKE stays on CPU (tiny synthetic data) so its numbers are byte-identical to before; REAL
    # uses CUDA when available (the verifier's MLP + greedy prover search are the hot path).
    device = "cpu" if mode == "smoke" else _resolve_device(cfg)
    if mode == "smoke":
        cfg = _smoke_ncv(cfg)
        for s in range(n_seeds):
            fdata, concept_sets = _make_smoke_fdata_and_concepts(seed=s)
            for space in SPACES:
                rows.extend(evaluate_space_seed(space, concept_sets[space], fdata, cfg, seed=s,
                                                device=device))
    else:
        # REAL (Colab/GPU): build the binary f + concept spaces ONCE on cached CLIP features (no
        # large-model training); the "mixed" space = CUB attributes + CLIP scene-concept cosines
        # computed from the SAME cached features (no image re-encode). Seeds then vary only the
        # verifier init + signal splits.
        from experiments.real_data import (build_binary_fdata, clip_text_concepts,
                                            clip_zeroshot_attribute_features, fit_attribute_probe,
                                            load_real_bundle, predict_attribute_features)
        pop_seed = int(cfg.get("pop_seed", 0))
        bundle = load_real_bundle(cfg, seed=pop_seed)
        fdata = build_binary_fdata(bundle, cfg, seed=pop_seed, with_ensemble_mc=True)
        clipcfg = cfg.get("clip", {})
        cargs = (clipcfg.get("model_name", "ViT-B-32"), clipcfg.get("pretrained", "openai"),
                 clipcfg.get("device", "cuda"))
        # §3 (v3): the concept channel is IMAGE-DERIVED (predicted), NOT the leaky ground-truth MTurk
        # attributes. Same `cbm` source as the unified 2x2 -> the verifiability falsification rests on
        # honest concepts. (`zeroshot`/`gt_attrs_leaky` selectable; gt_attrs_leaky is the prior leak.)
        concept_source = cfg.get("concept_source", "cbm")
        if concept_source == "cbm":
            probes = fit_attribute_probe(bundle.features["train"], bundle.attrs["train"], seed=pop_seed)
            attr = {sp: predict_attribute_features(probes, bundle.features[sp]) for sp in SPLITS}
        elif concept_source == "zeroshot":
            attr = {sp: clip_zeroshot_attribute_features(bundle.features[sp], bundle.attr_names, *cargs)
                    for sp in SPLITS}
        else:  # gt_attrs_leaky — the prior, invalid path (kept only as a flagged demo)
            print("[e2 §3] WARNING: concept_source='gt_attrs_leaky' uses GROUND-TRUTH attributes "
                  "(the prior leak); not the honest headline.")
            attr = {sp: bundle.attrs[sp] for sp in SPLITS}
        prompts = clipcfg.get("scene_concept_bank", DEFAULT_SCENE)
        scene = {sp: clip_text_concepts(bundle.features[sp], *cargs, prompts) for sp in SPLITS}
        concept_sets = {
            "attributes_only": {sp: attr[sp] for sp in SPLITS},
            "mixed": {sp: np.concatenate([attr[sp], scene[sp]], axis=1) for sp in SPLITS}}
        for s in range(n_seeds):
            for space in SPACES:
                rows.extend(evaluate_space_seed(space, concept_sets[space], fdata, cfg, seed=s,
                                                device=device))
    return {"rows": rows, "mode": mode, "n_seeds": n_seeds, "device": device}


# -------------------------------------------------------------------- aggregation / report
def _agg(df):
    return df.groupby(["space", "signal"]).agg(
        min_auroc_m=("minority_auroc", "mean"), min_auroc_s=("minority_auroc", "std"),
        contam_m=("contamination_auroc", "mean"), contam_s=("contamination_auroc", "std"),
        n=("seed", "count")).reset_index()


def write_report(path, df, payload):
    g = _agg(df)
    lines = [
        "# E2 — Verifiability collapse (clean multi-seed regeneration)",
        "",
        f"**Date:** {datetime.now(timezone.utc).date()} · **Run mode:** {payload['mode']} · "
        f"**Seeds:** {payload['n_seeds']} · minority error-detection AUROC (↑ better) + "
        f"spurious-attribute contamination AUROC (≈0.5 = clean).",
        "",
    ]
    if payload["mode"] == "smoke":
        lines += ["> ⚠️ **SMOKE (synthetic) run** — fabricated clean + mixed concept spaces and a "
                  "shortcut f. Validates the multi-seed verifier→signals→AUROC pipeline on CPU. Real "
                  "numbers come from the Colab/GPU run.", ""]
    lines += [
        "## Pre-committed expectation (report regardless)",
        "Verifiability ties/loses to concept-distance trust; **collapses below chance** in the mixed "
        "space; support-gap uninformative (~0.5).",
        "",
    ]
    order = [lbl for lbl, _ in E2_SIGNALS]
    for space in SPACES:
        sub = g[g.space == space]
        if sub.empty:
            continue
        lines += [f"## Concept space: `{space}`", "",
                  "| signal | minority AUROC ↑ | contamination AUROC (≈0.5 clean) |",
                  "|---|---|---|"]
        for lbl in order:
            r = sub[sub.signal == lbl]
            if r.empty:
                continue
            r = r.iloc[0]
            lines.append(f"| `{lbl}` | {r.min_auroc_m:.3f}±{r.min_auroc_s:.3f} | "
                         f"{r.contam_m:.3f}±{r.contam_s:.3f} |")
        lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/premise2_waterbirds.yaml")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--concept-source", default="cbm",
                    choices=("cbm", "zeroshot", "gt_attrs_leaky"),
                    help="§3: image-derived predicted concept channel (cbm) vs the prior leak")
    ap.add_argument("--out", default="results/e2")
    args = ap.parse_args()

    if os.path.exists(args.config):
        cfg = load_config(args.config)
    else:
        cfg = {"common": load_config("configs/common.yaml").get("common", {})}
    cfg.setdefault("common", load_config("configs/common.yaml").get("common", {}))
    cfg["concept_source"] = args.concept_source     # §3: real branch uses predicted concepts

    mode = "smoke" if args.smoke else "real"
    payload = run(cfg, mode=mode, n_seeds=args.seeds)

    os.makedirs(args.out, exist_ok=True)
    df = pd.DataFrame(payload["rows"])
    df.sort_values(["space", "signal", "seed"]).to_csv(
        os.path.join(args.out, "e2_verifiability_metrics.csv"), index=False)
    with open(os.path.join(args.out, "e2_results.json"), "w") as f:
        json.dump({"mode": mode, "n_seeds": payload["n_seeds"],
                   "agg": _agg(df).to_dict(orient="records")}, f, indent=2)
    write_report(os.path.join(args.out, "E2_REPORT.md"), df, payload)
    write_report("E2_REPORT.md", df, payload)

    g = _agg(df)
    print(f"[e2] mode={mode} seeds={payload['n_seeds']} -> {args.out}")
    for space in SPACES:
        sub = g[g.space == space]
        if sub.empty:
            continue
        def grab(sig):
            r = sub[sub.signal == sig]
            return r.min_auroc_m.iloc[0] if len(r) else float("nan")
        print(f"[e2] {space:15s}: concept_trust={grab('concept_trust'):.3f} "
              f"V_full={grab('V_full'):.3f} support_gap={grab('support_gap'):.3f}")
    if mode == "smoke":
        print("[e2] SMOKE OK — multi-seed verifiability pipeline validated. Real numbers: Colab.")


if __name__ == "__main__":
    main()
