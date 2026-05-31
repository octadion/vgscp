"""Phase A0 PROBE — are frozen CLIP concepts on Waterbirds usable for our claim? (Section 8)

A fast, cheap script (a few hundred images, 1 seed) that answers THREE questions, then STOPS:
  1. Concept usefulness  : does the CLIP concept vector predict bird type y? (overall / majority
                           / minority subsets) — non-degenerate & informative?
  2. Concept contamination: does the concept vector (and which individual concepts) encode the
                           background `place` shortcut? Some contamination is fine; TOTAL
                           contamination would predict NO-GO.
  3. Separation preview   : rough first look — do confidence / trust-on-f miss minority errors
                           that a concept-based signal catches? (tiny ERM f, few epochs.)

Outputs: results/runs/<run_id>/probe_results.json, the exact concept bank, cached CLIP scores,
and an env/git/GPU manifest. Then STOP and write PROBE_REPORT.md — do NOT build the kill-switch.

    python -m scripts.probe_waterbirds_clip --config configs/waterbirds.yaml
    python -m scripts.probe_waterbirds_clip --smoke   # synthetic self-test, no torch/dataset
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config_util import load_config
from eval import metrics
from vgscp_logging.manifest import RunManifest, make_run_id


# ----------------------------------------------------------------------------------------
# Analysis (pure numpy/sklearn — testable in --smoke without torch or the dataset)
# ----------------------------------------------------------------------------------------
def _fit_linear_probe(train_X, train_y, seed=0):
    """Logistic-regression probe on concepts (TRAIN only). Returns the fitted model."""
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
    clf.fit(train_X, train_y)
    return clf


def probe_concept_usefulness(train_X, train_y, test_X, test_y, test_minority, seed=0):
    """AUROC of the concept-probe for bird type y on overall / majority / minority subsets."""
    clf = _fit_linear_probe(train_X, train_y, seed)
    p1 = clf.predict_proba(test_X)[:, 1]
    minority = test_minority.astype(bool)
    out = {
        "probe_test_acc": float((clf.predict(test_X) == test_y).mean()),
        "auroc_y_overall": metrics.auroc(p1, test_y),
        "auroc_y_majority": metrics.auroc(p1[~minority], test_y[~minority]) if (~minority).any() else float("nan"),
        "auroc_y_minority": metrics.auroc(p1[minority], test_y[minority]) if minority.any() else float("nan"),
    }
    return out, p1


def probe_contamination(train_X, train_place, test_X, test_place, concept_bank, seed=0):
    """How much does the concept space encode the background `place` shortcut?

    - whole-vector contamination: AUROC of a place-probe (train->test).
    - per-concept contamination/usefulness: |AUROC-0.5| of each single concept for place and y.
    """
    clf = _fit_linear_probe(train_X, train_place, seed)
    pp = clf.predict_proba(test_X)[:, 1]
    whole = metrics.auroc(pp, test_place)
    whole = max(whole, 1 - whole) if not np.isnan(whole) else whole

    per_concept = []
    for k, name in enumerate(concept_bank):
        a_place = metrics.auroc(test_X[:, k], test_place)
        a_place = max(a_place, 1 - a_place) if not np.isnan(a_place) else a_place
        per_concept.append((name, float(a_place)))
    per_concept.sort(key=lambda kv: kv[1], reverse=True)
    return {
        "whole_vector_place_auroc": float(whole),
        "top_contaminating_concepts": per_concept[:10],
        "least_contaminating_concepts": per_concept[-5:],
    }


def separation_preview(signals: dict, correct, minority):
    """Minority error-detection AUROC for each available reliability signal (rough preview)."""
    minority = minority.astype(bool)
    out = {}
    for name, sig in signals.items():
        if sig is None:
            continue
        out[name] = {
            "auroc_minority": metrics.auroc(sig[minority], correct[minority]) if minority.any() else float("nan"),
            "auroc_overall": metrics.auroc(sig, correct),
        }
    return out


# ----------------------------------------------------------------------------------------
# Real run (needs torch + open_clip + the dataset)
# ----------------------------------------------------------------------------------------
def run_real(cfg, args):
    from data.waterbirds import load_waterbirds
    from models.concept_extractor_clip import CLIPConceptExtractor
    from perf.setup import setup_perf

    ctx = setup_perf(cfg["perf"], seed=cfg["experiment"]["seeds"][0])
    seed = cfg["experiment"]["seeds"][0]
    pcfg = cfg["probe"]

    bundle = load_waterbirds(cfg["dataset"], seed, max_per_split=pcfg["max_per_split"],
                             build_datasets=pcfg.get("train_tiny_f", False))

    # --- frozen CLIP concepts over train + d_test (cache once) ---
    extractor = CLIPConceptExtractor(
        cfg["clip"]["model_name"], cfg["clip"]["pretrained"], cfg["clip"]["concept_bank"],
        device=ctx.device, temperature_softmax=cfg["clip"]["temperature_softmax"],
        temperature=cfg["clip"]["temperature"],
    )
    train_raw = extractor.encode_paths(bundle.meta["paths"]["train"])
    test_raw = extractor.encode_paths(bundle.meta["paths"]["d_test"])
    extractor.fit_standardizer(train_raw)  # TRAIN-only
    standardize = cfg["clip"]["standardize"]
    train_X = extractor.apply_standardizer(train_raw, standardize)
    test_X = extractor.apply_standardizer(test_raw, standardize)

    train_y, test_y = bundle.y["train"], bundle.y["d_test"]
    train_place, test_place = bundle.spurious_attr["train"], bundle.spurious_attr["d_test"]
    test_minority = bundle.is_minority["d_test"]

    usefulness, clip_p1 = probe_concept_usefulness(train_X, train_y, test_X, test_y, test_minority, seed)
    contamination = probe_contamination(train_X, train_place, test_X, test_place,
                                        cfg["clip"]["concept_bank"], seed)

    # --- separation preview: tiny ERM f vs concept signal (rough) ---
    sep = {"note": "tiny_f disabled (probe.train_tiny_f=false)"}
    if pcfg.get("train_tiny_f", False):
        sep = _separation_with_tiny_f(cfg, bundle, ctx, train_X, train_y, test_X, test_y,
                                      test_minority, clip_p1)

    return {
        "n_train": int(len(train_y)), "n_test": int(len(test_y)),
        "n_concepts": extractor.n_concepts, "standardize": standardize,
        "minority_frac_test": float(test_minority.mean()),
        "usefulness": usefulness, "contamination": contamination,
        "separation_preview": sep,
    }, {"train_X": train_X, "test_X": test_X, "train_y": train_y, "test_y": test_y,
        "train_place": train_place, "test_place": test_place}


def _separation_with_tiny_f(cfg, bundle, ctx, train_X, train_y, test_X, test_y, test_minority, clip_p1):
    """Train a tiny ERM f (few epochs), compute conf/trust on f vs the concept signal."""
    import torch

    from models.base_model import (FeatureClassifier, TrainConfig, build_backbone,
                                    train_erm, worst_group_accuracy)
    from signals.confidence import msp
    from signals.trust import trust_score

    net, fd = build_backbone(cfg["model"]["backbone"], cfg["dataset"]["n_classes"],
                             cfg["model"].get("pretrained", True), cfg["model"].get("dropout_p", 0.2))
    clf = FeatureClassifier(net, fd, ctx.device)
    loader = torch.utils.data.DataLoader(bundle.datasets["train"],
                                         batch_size=max(8, ctx.batch_size or 32), shuffle=True,
                                         num_workers=2)
    train_erm(clf.net, loader, TrainConfig(num_epochs=cfg["probe"]["tiny_f_epochs"],
              lr=cfg["model"]["lr"], weight_decay=cfg["model"]["weight_decay"],
              optimizer=cfg["model"]["optimizer"]), ctx, ctx.device)

    # forward d_test (and train, for trust refs)
    def forward(ds):
        clf.net.eval()
        probs_l, feats_l = [], []
        dl = torch.utils.data.DataLoader(ds, batch_size=max(8, ctx.batch_size or 32))
        with torch.inference_mode():
            for x, _ in dl:
                x = x.to(ctx.device)
                logits, feats = clf.logits_and_features(x)
                probs_l.append(torch.softmax(logits.float(), 1).cpu().numpy())
                feats_l.append(feats.float().cpu().numpy())
        return np.concatenate(probs_l), np.concatenate(feats_l)

    test_probs, test_feats = forward(bundle.datasets["d_test"])
    _, train_feats = forward(bundle.datasets["train"])
    y_pred = test_probs.argmax(1)
    correct = (y_pred == test_y).astype(int)
    conf = msp(test_probs)
    trust_f = trust_score(train_feats, train_y, test_feats, y_pred, n_classes=cfg["dataset"]["n_classes"])
    # concept signal: confidence of the clip probe (max(p, 1-p))
    concept_sig = np.maximum(clip_p1, 1 - clip_p1)
    wg = worst_group_accuracy(y_pred, test_y, bundle.group_id["d_test"])
    sep = separation_preview({"conf_msp": conf, "trust_f": trust_f, "clip_probe": concept_sig},
                             correct, test_minority)
    sep["tiny_f_worst_group"] = wg
    sep["tiny_f_minority_acc"] = float(correct[test_minority.astype(bool)].mean())
    return sep


# ----------------------------------------------------------------------------------------
# Smoke (synthetic) self-test of the analysis logic
# ----------------------------------------------------------------------------------------
def run_smoke():
    """Fabricate concept scores where some concepts encode y, some encode place, some are noise.
    Validates the probe analysis end-to-end without torch / open_clip / the dataset."""
    rng = np.random.default_rng(0)
    K, n = 20, 800
    bank = [f"y_concept_{i}" for i in range(6)] + [f"place_concept_{i}" for i in range(6)] + \
           [f"noise_concept_{i}" for i in range(8)]

    def gen(m):
        y = rng.integers(0, 2, m)
        minority = rng.random(m) < 0.2
        place = np.where(minority, 1 - y, y)
        X = rng.normal(0, 1, (m, K))
        X[:, :6] += (2 * y - 1)[:, None] * 1.2          # y-encoding concepts
        X[:, 6:12] += (2 * place - 1)[:, None] * 1.5    # place-encoding (shortcut) concepts
        return X, y, place, minority

    train_X, train_y, train_place, _ = gen(n)
    test_X, test_y, test_place, test_minority = gen(n)

    usefulness, clip_p1 = probe_concept_usefulness(train_X, train_y, test_X, test_y, test_minority)
    contamination = probe_contamination(train_X, train_place, test_X, test_place, bank)
    # fake separation: confidence tracks place (contaminated), concept signal tracks y
    correct = (clip_p1 > 0.5).astype(int) == test_y
    correct = correct.astype(int)
    conf = np.maximum(test_X[:, 6], 0)  # tracks a place concept -> contaminated
    sep = separation_preview({"conf_msp": conf, "clip_probe": np.maximum(clip_p1, 1 - clip_p1)},
                             correct, test_minority)
    return {"n_train": n, "n_test": n, "n_concepts": K, "minority_frac_test": float(test_minority.mean()),
            "usefulness": usefulness, "contamination": contamination, "separation_preview": sep}


def _to_jsonable(o):
    if isinstance(o, dict):
        return {k: _to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_to_jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/waterbirds.yaml")
    ap.add_argument("--smoke", action="store_true", help="synthetic self-test (no torch/dataset)")
    ap.add_argument("--timestamp", default=None)
    args = ap.parse_args()

    ts = args.timestamp or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    if args.smoke:
        res = run_smoke()
        print(json.dumps(_to_jsonable(res), indent=2))
        print("\n[probe] SMOKE OK — analysis logic runs. Use the real run for the actual verdict.")
        return

    cfg = load_config(args.config)
    run_id = make_run_id(cfg["experiment"]["name"] + "_probe", ts)
    run_dir = os.path.join(cfg["experiment"]["out_dir"], "runs", run_id)
    os.makedirs(run_dir, exist_ok=True)
    manifest = RunManifest(run_dir, cfg)
    manifest.write_config_yaml(cfg)

    res, cache_arrays = run_real(cfg, args)

    # cache the CLIP concept scores + the exact bank
    np.savez(os.path.join(run_dir, "clip_concepts.npz"), **cache_arrays)
    with open(os.path.join(run_dir, "concept_bank.json"), "w") as f:
        json.dump(cfg["clip"]["concept_bank"], f, indent=2)
    with open(os.path.join(run_dir, "probe_results.json"), "w") as f:
        json.dump(_to_jsonable(res), f, indent=2)
    manifest.set("probe_results", _to_jsonable(res))
    manifest.save()

    u, c = res["usefulness"], res["contamination"]
    print("\n==== WATERBIRDS CLIP PROBE ====")
    print(f"run_id: {run_id}  (n_train={res['n_train']}, n_test={res['n_test']}, K={res['n_concepts']})")
    print(f"[usefulness] AUROC(concepts->y): overall={u['auroc_y_overall']:.3f} "
          f"majority={u['auroc_y_majority']:.3f} minority={u['auroc_y_minority']:.3f}")
    print(f"[contamination] AUROC(concepts->place)={c['whole_vector_place_auroc']:.3f}")
    print(f"  top contaminating concepts: {[n for n, _ in c['top_contaminating_concepts'][:5]]}")
    if "auroc_minority" in str(res["separation_preview"]):
        print(f"[separation preview] {json.dumps(_to_jsonable(res['separation_preview']), indent=2)}")
    print(f"\nresults -> {run_dir}/probe_results.json")
    print("[probe] STOP. Review PROBE_REPORT.md, then say 'go' for the A1 kill-switch.")


if __name__ == "__main__":
    main()
