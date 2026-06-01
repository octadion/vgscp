"""KILL-SWITCH for Arah 2 — counterfactual support-gap verifiability (go/no-go gate).

ONE hypothesis: does the DROP in verifier support when spurious concepts are removed (the
counterfactual support gap) carry minority-error information that beats BOTH `trust_concept`
(THE BAR) AND a clean-support-only control (`support_clean`)? If a "spuriousness-aware verifier"
merely deletes spurious concepts it collapses to the attributes_only space and only TIES
`trust_concept` — so the gap must add something neither distance-in-concept-space nor clean-support
alone has. See the pre-committed GREEN/RED logic in eval/killswitch_verdict.py.

Dose axis: number of CLIP spurious scene concepts mixed into the bank (all 312 CUB attributes
always present). f + ensemble + MC-dropout + CLIP concepts are IDENTICAL across doses (they
classify images, not concepts) — built/cached ONCE, reused per dose. Per dose only: rebuild the
concept subset -> recompute rho & clean_mask -> train one small Arthur -> compute signals ->
evaluate.

    python -m scripts.run_killswitch_gap --config configs/killswitch_arah2.yaml   # real (Colab/GPU)
    python -m scripts.run_killswitch_gap --smoke                                  # local CPU self-test

HARD STOP after writing KILLSWITCH_ARAH2_REPORT.md. No multi-seed / CelebA / figures / ablations.
"""
from __future__ import annotations

# Import torch FIRST (Windows DLL ordering; harmless no-op on Linux/Colab).
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
from eval import metrics
from eval.killswitch_verdict import killswitch_verdict
from eval.phase1_eval import evaluate_signals
from models.concept_extractor_clip import ConceptStandardizer
from models.verifier_adapter import build_verifier
from signals import registry
from signals.spurious_gap import (clean_mask_from_rho, concept_spuriousness,
                                  rho_recovery_summary)

SPLITS = ("train", "d_learn", "d_cal", "d_test")
# Reported per dose. Convention: higher = more reliable.
REPORT_SIGNALS = ["V_gap", "V_gap_pure", "support_clean", "V_comp", "trust_concept",
                  "trust", "ensemble_disagree", "conf_msp"]


# ======================================================================================
# Per-dose evaluation (the only thing that varies across the sweep)
# ======================================================================================
def _worst_group(y_pred, y_true, group_id):
    accs = {}
    for g in np.unique(group_id):
        m = group_id == g
        accs[int(g)] = float((y_pred[m] == y_true[m]).mean())
    return {"worst_group": float(min(accs.values())), "overall": float((y_pred == y_true).mean()),
            "per_group": accs}


def _build_bank(dose, attrs, scene_top):
    """Concept bank for a dose = [312 CUB attrs | top-`dose` CLIP scene concepts]."""
    if dose == 0:
        return {s: attrs[s].astype(np.float32) for s in SPLITS}
    return {s: np.concatenate([attrs[s], scene_top[s]], axis=1).astype(np.float32) for s in SPLITS}


def evaluate_dose(dose, attrs, scene_top, fdata, cfg, device="cpu", seed=0, n_resamples=1000):
    """Full kill-switch evaluation for ONE dose. Returns a result dict (rho, clean_mask, tuned lam,
    recovery summary, per-signal reports, d_test DataFrame, raw arrays)."""
    common = cfg["common"]
    lam_grid = cfg.get("lam_grid", [0, 0.5, 1, 2, 4])
    n_classes = int(fdata["train"]["probs"].shape[1])
    n_clean_true = int(attrs["train"].shape[1])  # CUB attributes are the genuinely-clean dims

    bank_raw = _build_bank(dose, attrs, scene_top)

    # --- TRAIN-only standardization of the concept bank (no leakage), reused everywhere ---
    if cfg.get("concepts", {}).get("standardize", True):
        std = ConceptStandardizer.fit(bank_raw["train"])
        bank = {s: std.transform(bank_raw[s]).astype(np.float32) for s in SPLITS}
    else:
        bank = {s: bank_raw[s].astype(np.float32) for s in SPLITS}

    # --- rho_j + clean_mask on TRAIN only (spurious attr a = place, train-only) ---
    a_train = np.asarray(fdata["train"]["spurious_attr"])
    rho = concept_spuriousness(bank["train"], a_train)
    clean_mask, tau = clean_mask_from_rho(rho)
    recovery = rho_recovery_summary(rho, clean_mask, n_clean_true)

    # --- ONE Arthur (Morgana on, so V_comp is available too) on the full bank for this dose ---
    ncv_on = dict(cfg["ncv"]); ncv_on["morgana"] = "on"; ncv_on["standardize"] = False
    v = build_verifier(ncv_on, bank["train"].shape[1], n_classes, device=device)
    v.train(bank["train"], fdata["train"]["y_true"], seed=seed)

    probe_kind = cfg.get("probe", {}).get("concept_probe_kind", "logistic")
    train_feats = fdata["train"]["features"]
    train_labels = fdata["train"]["y_true"]

    def signals_for(split, lam):
        f = fdata[split]
        vo = v.predict(bank[split], f["y_pred"])
        return registry.build_signals(
            f["probs"], f["y_pred"],
            train_feats=train_feats, train_labels=train_labels, query_feats=f["features"],
            member_probs=f.get("member_probs"), mc_pass_probs=f.get("mc_pass_probs"),
            pA_given_SM=vo.pA_given_SM, pA_given_SA=vo.pA_given_SA, reject_prob=vo.reject_prob,
            n_classes=n_classes, beta=0.5,
            train_concepts=bank["train"], query_concepts=bank[split], concept_probe_kind=probe_kind,
            verifier=v, clean_mask=clean_mask, gap_lam=lam,
        )

    # --- tune lam on D_learn minority error-detection AUROC (support_clean & _gap are lam-free) ---
    learn = signals_for("d_learn", 0.0)
    lf = fdata["d_learn"]
    lc = (lf["y_pred"] == lf["y_true"]).astype(int)
    lm = lf["is_minority"].astype(bool)
    sc, gp = learn["support_clean"], learn["_gap"]
    best_lam, best_score = lam_grid[0], -np.inf
    lam_scores = {}
    for lam in lam_grid:
        vg = sc - float(lam) * gp
        score = metrics.error_detection_auroc(vg[lm], lc[lm]) if lm.any() else float("nan")
        lam_scores[float(lam)] = float(score) if np.isfinite(score) else None
        if np.isfinite(score) and score > best_score:
            best_score, best_lam = score, float(lam)

    # --- D_test signals at the tuned lam ---
    test = signals_for("d_test", best_lam)
    ft = fdata["d_test"]
    yt, yp = ft["y_true"], ft["y_pred"]
    df = pd.DataFrame({
        "sample_id": np.arange(len(yt)), "dose": dose, "lam": best_lam, "seed": seed,
        "y_true": yt, "y_pred": yp, "correct": (yp == yt).astype(int),
        "group_id": ft["group_id"], "spurious_attr": ft["spurious_attr"],
        "is_minority": ft["is_minority"].astype(int),
    })
    for name in REPORT_SIGNALS:
        if name in test:
            df[name] = test[name]
    # keep the raw gap internals for reproduction/inspection
    df["_support_full"] = test["_support_full"]
    df["_gap"] = test["_gap"]

    present = [s for s in REPORT_SIGNALS if s in df and df[s].notna().any()]
    reports = evaluate_signals(df, present, budgets=tuple(common["budgets"][1:6]),
                               n_resamples=n_resamples, seed=seed)

    return {"dose": dose, "lam": best_lam, "lam_scores": lam_scores, "tau": tau, "rho": rho,
            "clean_mask": clean_mask, "recovery": recovery, "reports": reports, "df": df,
            "concept_dim": int(bank["train"].shape[1]), "n_clean_true": n_clean_true}


# ======================================================================================
# Concept ranking: pick the highest-rho CLIP scene concepts for the dose subsets
# ======================================================================================
def rank_scene_by_rho(scene_raw, a_train, standardize=True):
    """Rank the CLIP scene concepts by TRAIN spuriousness rho (descending) so dose d takes the d
    most-spurious. Standardize on TRAIN before scoring (same as the bank). Returns (order, rho)."""
    if standardize:
        std = ConceptStandardizer.fit(scene_raw["train"])
        train_scene = std.transform(scene_raw["train"])
    else:
        train_scene = scene_raw["train"]
    rho = concept_spuriousness(train_scene, np.asarray(a_train))
    order = np.argsort(-rho)  # most spurious first
    return order, rho


def _take_top(scene_raw, order, d):
    if d == 0:
        return {s: np.zeros((scene_raw[s].shape[0], 0), dtype=np.float32) for s in SPLITS}
    idx = order[:d]
    return {s: scene_raw[s][:, idx].astype(np.float32) for s in SPLITS}


# ======================================================================================
# Real run (Colab/GPU): build/cache f + ensemble + MC-dropout + CUB attrs + CLIP scene ONCE
# ======================================================================================
def _build_raw_concepts_real(cfg, bundle, ctx):
    """Build raw (un-standardized) CUB attributes (312) and the full CLIP scene bank (K), each as
    {split: (N, .)} — kept SEPARATE so doses can slice the scene concepts. Standardization is
    per-dose (train-only) inside evaluate_dose."""
    from data.cub_attributes import load_cub_attribute_concepts

    ccfg = cfg["cub"]
    attrs, attr_names, join_info = load_cub_attribute_concepts(
        bundle, ccfg["root"], splits=SPLITS, download=ccfg.get("download", False),
        url=ccfg.get("url"), use_certainty=ccfg.get("use_certainty", False),
        min_coverage=ccfg.get("min_coverage", 0.99))

    from models.concept_extractor_clip import CLIPConceptExtractor

    clipcfg = cfg["clip"]
    extractor = CLIPConceptExtractor(clipcfg["model_name"], clipcfg["pretrained"],
                                     clipcfg["scene_concept_bank"], device=ctx.device,
                                     temperature_softmax=clipcfg.get("temperature_softmax", False),
                                     temperature=clipcfg.get("temperature", 0.01))
    scene = {s: extractor.encode_paths(bundle.meta["paths"][s]) for s in SPLITS}
    info = {"cub_join": join_info, "attr_dim": int(attrs["train"].shape[1]),
            "clip_scene_dim": int(scene["train"].shape[1]),
            "scene_bank": list(clipcfg["scene_concept_bank"])}
    return {s: attrs[s].astype(np.float32) for s in SPLITS}, \
           {s: scene[s].astype(np.float32) for s in SPLITS}, info


_FDATA_ARTIFACTS = ["probs", "y_pred", "y_true", "features", "member_probs", "mc_pass_probs",
                    "group_id", "spurious_attr", "is_minority"]


def _cache_dir(cfg, seed):
    return os.path.join(cfg["experiment"]["out_dir"], "cache_killswitch", f"seed{seed}")


def _try_load_cache(cfg, seed):
    """Reuse a previously-built f+ensemble+concepts cache if present (so doses never retrain the
    ResNets). Returns (fdata, attrs, scene, info) or None."""
    base = _cache_dir(cfg, seed)
    if not os.path.isfile(os.path.join(base, "info.json")):
        return None
    with open(os.path.join(base, "info.json")) as f:
        info = json.load(f)
    fdata, attrs, scene = {}, {}, {}
    try:
        for sp in SPLITS:
            d = os.path.join(base, sp)
            fdata[sp] = {a: np.load(os.path.join(d, f"{a}.npy")) for a in _FDATA_ARTIFACTS}
            fdata[sp]["is_minority"] = fdata[sp]["is_minority"].astype(bool)
            attrs[sp] = np.load(os.path.join(d, "attrs.npy"))
            scene[sp] = np.load(os.path.join(d, "scene.npy"))
    except FileNotFoundError:
        return None
    print(f"[killswitch] reusing cached f+ensemble+concepts from {base}")
    return fdata, attrs, scene, info


def _save_cache(cfg, seed, fdata, attrs, scene, info):
    base = _cache_dir(cfg, seed)
    for sp in SPLITS:
        d = os.path.join(base, sp)
        os.makedirs(d, exist_ok=True)
        for a in _FDATA_ARTIFACTS:
            np.save(os.path.join(d, f"{a}.npy"), np.asarray(fdata[sp][a]))
        np.save(os.path.join(d, "attrs.npy"), attrs[sp])
        np.save(os.path.join(d, "scene.npy"), scene[sp])
    with open(os.path.join(base, "info.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(f"[killswitch] cached f+ensemble+concepts -> {base}")


def build_or_load(cfg, seed, reuse=True):
    """Get (fdata, attrs, scene, info, worst_group). Reuse the cache if present; else build the
    ResNets + CLIP concepts ONCE (GPU) and cache them."""
    if reuse:
        cached = _try_load_cache(cfg, seed)
        if cached is not None:
            fdata, attrs, scene, info = cached
            wg = _worst_group(fdata["d_test"]["y_pred"], fdata["d_test"]["y_true"],
                              fdata["d_test"]["group_id"])
            return fdata, attrs, scene, info, wg

    # Build once (reuse the premise-2 real f/ensemble/mc-dropout builder verbatim).
    from scripts.run_premise2 import _build_fdata_real

    bundle, fdata, ctx, wg, _tlog = _build_fdata_real(cfg, seed)
    attrs, scene, info = _build_raw_concepts_real(cfg, bundle, ctx)
    _save_cache(cfg, seed, fdata, attrs, scene, info)
    return fdata, attrs, scene, info, wg


# ======================================================================================
# Smoke: fabricate a shortcut f + clean CUB-like attrs + a spurious CLIP-like scene pool
# ======================================================================================
def _make_smoke(seed=0, n=1400, n_attrs=12, n_scene=16):
    """Synthetic Waterbirds-like setup for a CPU PIPELINE self-test (NOT a claim). f is a shortcut
    classifier that reads the background (place); the CUB-like core concepts track the TRUE label y
    (clean), while the CLIP-like scene concepts track `place` (spurious) with VARYING strength so
    rho ranks them. On minority (y!=place) f is confidently wrong and the full verifier can lean on
    the scene concepts to 'support' the wrong label -> a large counterfactual support gap."""
    n_classes = 2

    def gen(m, s):
        r = np.random.default_rng(2000 + s)
        y = r.integers(0, 2, m)
        minority = r.random(m) < 0.25
        place = np.where(minority, 1 - y, y)
        use_shortcut = r.random(m) < 0.65
        pred = np.where(use_shortcut, place, y)
        conf = 0.80 + r.random(m) * 0.16
        p1 = np.where(pred == 1, conf, 1 - conf)
        probs = np.stack([1 - p1, p1], axis=1)
        probs = probs / probs.sum(1, keepdims=True)
        y_pred = probs.argmax(1)
        feats = np.concatenate([
            (2 * place - 1)[:, None] * 1.6 + r.normal(0, 0.5, (m, 4)),
            r.normal(0, 1, (m, 4)),
        ], axis=1).astype(np.float32)
        member = np.stack([probs + r.normal(0, 0.02, probs.shape) for _ in range(3)], 0)
        member = np.clip(member, 1e-3, 1 - 1e-3); member /= member.sum(2, keepdims=True)
        mc = np.stack([probs + r.normal(0, 0.02, probs.shape) for _ in range(20)], 0)
        mc = np.clip(mc, 1e-3, 1 - 1e-3); mc /= mc.sum(2, keepdims=True)
        sy = (2 * y - 1).astype(np.float32)
        sp = (2 * place - 1).astype(np.float32)
        # clean core attrs (track y); a couple weakly y-aligned to make the clean set non-degenerate
        attrs = np.stack([1.6 * sy + r.normal(0, 0.7, m) for _ in range(n_attrs)], 1).astype(np.float32)
        # scene concepts: track place with strength fading across the bank (rho high -> low)
        scene = np.stack([
            (2.2 * (1.0 - 0.9 * k / max(1, n_scene - 1))) * sp + r.normal(0, 0.5, m)
            for k in range(n_scene)
        ], 1).astype(np.float32)
        f = {"probs": probs.astype(np.float32), "y_pred": y_pred.astype(np.int64),
             "y_true": y.astype(np.int64), "features": feats,
             "member_probs": member.astype(np.float32), "mc_pass_probs": mc.astype(np.float32),
             "group_id": (2 * y + place).astype(np.int64), "spurious_attr": place.astype(np.int64),
             "is_minority": (y != place)}
        return f, attrs, scene

    fdata, attrs, scene = {}, {}, {}
    sizes = {"train": n, "d_learn": n // 2, "d_cal": n // 2, "d_test": n // 2}
    for i, sp in enumerate(SPLITS):
        f, a, sc = gen(sizes[sp], i)
        fdata[sp], attrs[sp], scene[sp] = f, a, sc
    info = {"attr_dim": n_attrs, "clip_scene_dim": n_scene, "scene_bank": [f"scene_{k}" for k in range(n_scene)],
            "smoke": True}
    return fdata, attrs, scene, info


# ======================================================================================
# Orchestration
# ======================================================================================
def run(cfg, seed, mode, reuse_cache=True, n_resamples=None):
    if mode == "smoke":
        fdata, attrs, scene, info = _make_smoke(seed)
        wg = _worst_group(fdata["d_test"]["y_pred"], fdata["d_test"]["y_true"],
                          fdata["d_test"]["group_id"])
        device = "cpu"
        n_resamples = n_resamples or 300
        # smaller / faster verifier for the CPU self-test
        cfg = dict(cfg); cfg.setdefault("ncv", {})
        cfg["ncv"] = dict(cfg["ncv"]); cfg["ncv"].update(
            {"source": "reimpl", "epochs": 20, "hidden": 32, "merlin_sparsity": 3,
             "morgana_sparsity": 3, "n_train_max": None, "batch_size": 256})
    else:
        fdata, attrs, scene, info, wg = build_or_load(cfg, seed, reuse=reuse_cache)
        device = _device_from_cfg(cfg)
        n_resamples = n_resamples or cfg["common"]["eval"]["bootstrap_resamples"]

    # rank scene concepts by TRAIN rho so dose d takes the d most spurious
    standardize = cfg.get("concepts", {}).get("standardize", True)
    order, scene_rho = rank_scene_by_rho(scene, fdata["train"]["spurious_attr"], standardize)

    dose_grid = [d for d in cfg.get("dose_grid", [0, 4, 8, 12, 16])
                 if d <= scene["train"].shape[1]]
    dose_results = []
    for d in dose_grid:
        scene_top = _take_top(scene, order, d)
        print(f"[killswitch] dose={d} (concept_dim={attrs['train'].shape[1] + d}) ...")
        res = evaluate_dose(d, attrs, scene_top, fdata, cfg, device=device, seed=seed,
                            n_resamples=n_resamples)
        dose_results.append(res)
        dv_signals = ", ".join(
            f"{s}={res['reports'][s].auroc_minority.estimate:.3f}"
            for s in ("V_gap", "support_clean", "trust_concept")
            if s in res["reports"] and res["reports"][s].auroc_minority)
        print(f"            lam*={res['lam']:g} tau={res['tau']:.3f} | minAUROC: {dv_signals}")

    verdict = killswitch_verdict([{"dose": r["dose"], "lam": r["lam"], "df": r["df"]}
                                  for r in dose_results], n_resamples=n_resamples, seed=seed)
    return {"dose_results": dose_results, "verdict": verdict, "worst_group": wg, "info": info,
            "scene_rho": scene_rho, "scene_order": order, "dose_grid": dose_grid, "seed": seed,
            "mode": mode}


def _device_from_cfg(cfg):
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


# ======================================================================================
# Reporting
# ======================================================================================
def _ci_str(ci):
    return "--" if ci is None or not np.isfinite(ci.estimate) else \
        f"{ci.estimate:.3f} [{ci.lo:.3f}, {ci.hi:.3f}]"


def _dose_table(res):
    reps = res["reports"]
    lines = ["| Signal | minority AUROC ↑ | contamination AUROC ↓ |", "|---|---|---|"]
    for s in REPORT_SIGNALS:
        if s not in reps:
            continue
        r = reps[s]
        lines.append(f"| `{s}` | {_ci_str(r.auroc_minority)} | {r.contamination_auroc:.3f} |")
    return "\n".join(lines)


def _rho_summary_table(dose_results):
    lines = ["| dose | concept_dim | τ | clean recall (CUB→clean) | spurious recall (CLIP→spurious) "
             "| ρ̄ clean | ρ̄ spurious |", "|---|---|---|---|---|---|---|"]
    for r in dose_results:
        rec = r["recovery"]
        def _f(x):
            return "--" if x is None or not np.isfinite(x) else f"{x:.3f}"
        lines.append(f"| {r['dose']} | {r['concept_dim']} | {r['tau']:.3f} | "
                     f"{_f(rec['clean_recall'])} ({rec['n_flagged_clean']} clean) | "
                     f"{_f(rec['spurious_recall'])} ({rec['n_flagged_spurious']} spur) | "
                     f"{_f(rec['rho_clean_mean'])} | {_f(rec['rho_spurious_mean'])} |")
    return "\n".join(lines)


def write_report(path, cfg, payload):
    v = payload["verdict"]
    wg = payload["worst_group"]
    mode = payload["mode"]
    parts = [
        "# KILL-SWITCH ARAH 2 REPORT — counterfactual support-gap verifiability",
        "",
        f"**Date:** {datetime.now(timezone.utc).date()} · **Run mode:** {mode} · "
        f"**Seed:** {payload['seed']} (single-seed gate) · **OVERALL VERDICT: {v.label}**",
        "",
        "Go/no-go gate for ONE hypothesis: does the counterfactual support gap "
        "(`gap = support_full − support_clean`) carry minority-error information that beats BOTH "
        "`trust_concept` (THE BAR) AND a clean-support-only control (`support_clean`)? A "
        "spuriousness-aware verifier that merely deletes spurious concepts collapses to the "
        "attributes_only space and only TIES `trust_concept`; the gap must add something neither "
        "distance-in-concept-space nor clean-support alone has.",
        "",
        "## Pre-committed criterion (committed in code before any numbers — eval/killswitch_verdict.py)",
        "- **GREEN (mechanism alive):** at ≥1 dose, `V_gap` beats `trust_concept` (paired-delta 95% "
        "CI excludes 0) **AND** `V_gap` beats `support_clean` (paired-delta 95% CI excludes 0). The "
        "second clause is mandatory — it rules out the `V_gap ≈ support_clean` degeneracy.",
        "- **RED (dead):** `V_gap` ties/loses `trust_concept` at every dose, OR `V_gap` never beats "
        "`support_clean`. Any non-GREEN outcome is RED (not softened).",
        "",
        f"**OVERALL: {v.label}.** {v.rationale}",
        "",
        "## Regime check (f) — is the shortcut real?",
        f"Worst-group acc = {wg['worst_group']:.3f} vs overall {wg['overall']:.3f} "
        f"(per-group {wg['per_group']}).",
        "",
        "## ρ separation summary (validates the spuriousness scorer; the method uses the ρ-derived "
        "mask, NOT the CUB-vs-CLIP identity)",
        _rho_summary_table(payload["dose_results"]),
        "",
    ]
    for r in payload["dose_results"]:
        dv = next((d for d in v.per_dose if d.dose == r["dose"]), None)
        parts += [
            f"## Dose d={r['dose']}  (concept_dim={r['concept_dim']}, tuned λ*={r['lam']:g}, "
            f"τ={r['tau']:.3f}) — **{dv.label if dv else '??'}**",
            (dv.rationale if dv else ""),
            "",
            _dose_table(r),
            "",
        ]
    parts += [
        "## Per-dose verdict summary",
        "| dose | λ* | beats trust_concept | beats support_clean | dose verdict |",
        "|---|---|---|---|---|",
    ]
    for dv in v.per_dose:
        parts.append(f"| {dv.dose} | {dv.lam:g} | {'yes' if dv.beats_trust_concept else 'no'} | "
                     f"{'yes' if dv.beats_support_clean else 'no'} | **{dv.label}** |")
    parts += [
        "",
        f"**OVERALL VERDICT: {v.label}** (GREEN doses: {v.green_doses or 'none'}).",
        "",
        "> Single-seed kill-switch. Significance is the bootstrap CIs only; no claim beyond them. "
        "**STOP — await human review before any strength-controlled sweep, CelebA, multi-seed, "
        "figures, or paper machinery.**",
    ]
    if mode == "smoke":
        parts.insert(3, "\n> ⚠️ **SMOKE (synthetic) run** — fabricated shortcut f + CUB-like clean "
                        "attrs + CLIP-like spurious scene pool. Validates the rho/mask/verifier/gap/"
                        "verdict PIPELINE end-to-end on CPU. The real GREEN/RED numbers come from the "
                        "Colab run.\n")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts) + "\n")


def _to_jsonable(o):
    if isinstance(o, dict):
        return {k: _to_jsonable(x) for k, x in o.items()}
    if isinstance(o, (list, tuple)):
        return [_to_jsonable(x) for x in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def _payload_json(payload):
    v = payload["verdict"]
    doses = []
    for r in payload["dose_results"]:
        reps = r["reports"]
        doses.append({
            "dose": r["dose"], "lam": r["lam"], "lam_scores": r["lam_scores"], "tau": r["tau"],
            "concept_dim": r["concept_dim"], "n_clean_true": r["n_clean_true"],
            "rho": r["rho"], "clean_mask": r["clean_mask"], "recovery": r["recovery"],
            "minority_auroc": {s: (rep.auroc_minority.estimate if rep.auroc_minority else None)
                               for s, rep in reps.items()},
            "minority_auroc_ci": {s: ([rep.auroc_minority.lo, rep.auroc_minority.hi]
                                      if rep.auroc_minority else None)
                                  for s, rep in reps.items()},
            "contamination_auroc": {s: rep.contamination_auroc for s, rep in reps.items()},
        })
    per_dose = [{"dose": d.dose, "lam": d.lam, "label": d.label,
                 "beats_trust_concept": d.beats_trust_concept,
                 "beats_support_clean": d.beats_support_clean,
                 "vs_trust_concept": (None if d.vs_trust_concept is None else
                                      {"delta": d.vs_trust_concept.delta,
                                       "delta_lo": d.vs_trust_concept.delta_lo,
                                       "delta_hi": d.vs_trust_concept.delta_hi,
                                       "p_value": d.vs_trust_concept.p_value}),
                 "vs_support_clean": (None if d.vs_support_clean is None else
                                      {"delta": d.vs_support_clean.delta,
                                       "delta_lo": d.vs_support_clean.delta_lo,
                                       "delta_hi": d.vs_support_clean.delta_hi,
                                       "p_value": d.vs_support_clean.p_value}),
                 "rationale": d.rationale} for d in v.per_dose]
    out = {"verdict": v.label, "verdict_rationale": v.rationale, "green_doses": v.green_doses,
           "worst_group": payload["worst_group"], "dose_grid": payload["dose_grid"],
           "scene_rho": payload["scene_rho"], "scene_order": payload["scene_order"],
           "info": payload["info"], "doses": doses, "per_dose_verdict": per_dose,
           "seed": payload["seed"], "mode": payload["mode"]}
    return _to_jsonable(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/killswitch_arah2.yaml")
    ap.add_argument("--smoke", action="store_true", help="synthetic CPU self-test (no dataset)")
    ap.add_argument("--rebuild", action="store_true", help="rebuild f+ensemble+concepts (ignore cache)")
    ap.add_argument("--timestamp", default=None)
    args = ap.parse_args()
    ts = args.timestamp or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    if args.smoke:
        cfg = load_config(args.config) if os.path.exists(args.config) else {}
        cfg.setdefault("common", load_config("configs/common.yaml").get("common", {}))
        cfg.setdefault("dose_grid", [0, 4, 8, 12, 16])
        cfg.setdefault("lam_grid", [0, 0.5, 1, 2, 4])
        cfg.setdefault("concepts", {"standardize": True})
        cfg["common"].setdefault("ensemble", {})["n_members"] = 3
        payload = run(cfg, seed=0, mode="smoke")
        write_report("KILLSWITCH_ARAH2_REPORT.md", cfg, payload)
        print(f"\n[killswitch] SMOKE OK — full pipeline ran on synthetic data.")
        print(f"[killswitch] OVERALL VERDICT: {payload['verdict'].label}")
        print("[killswitch] wrote KILLSWITCH_ARAH2_REPORT.md (smoke). STOP — real numbers: Colab.")
        return

    cfg = load_config(args.config)
    seed = cfg["experiment"]["seeds"][0]
    # The prior premise-2 run wrongly used the inherited default n_members=5; here speed matters and
    # the ensemble is only context, so force 3 (the code reads cfg['common']['ensemble']).
    cfg["common"].setdefault("ensemble", {})["n_members"] = 3

    run_id = f"killswitch_arah2_{ts}"
    run_dir = os.path.join(cfg["experiment"]["out_dir"], "runs", run_id)
    os.makedirs(run_dir, exist_ok=True)

    payload = run(cfg, seed, mode="real", reuse_cache=not args.rebuild)

    # raw per-sample d_test frames (sufficient to reproduce the table) + JSON summary
    for r in payload["dose_results"]:
        r["df"].to_csv(os.path.join(run_dir, f"dose{r['dose']}_dtest.csv"), index=False)
    pj = _payload_json(payload)
    with open(os.path.join(run_dir, "killswitch_results.json"), "w") as f:
        json.dump(pj, f, indent=2)
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        import yaml
        yaml.safe_dump(cfg, f, sort_keys=False)
    write_report(os.path.join(run_dir, "KILLSWITCH_ARAH2_REPORT.md"), cfg, payload)
    write_report("KILLSWITCH_ARAH2_REPORT.md", cfg, payload)

    v = payload["verdict"]
    for dv in v.per_dose:
        print(f"[dose {dv.dose}] {dv.label}: {dv.rationale}")
    print(f"\n[killswitch] results -> {run_dir}")
    print(f"[killswitch] OVERALL VERDICT: {v.label}")
    print("[killswitch] STOP. Await human review of KILLSWITCH_ARAH2_REPORT.md before any "
          "strength-controlled sweep, CelebA, multi-seed, figures, or paper machinery.")


if __name__ == "__main__":
    main()
