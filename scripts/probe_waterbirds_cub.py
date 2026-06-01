"""PROBE — does the CUB per-image attribute concept space have minority bird-type signal AND
low background contamination on Waterbirds? (premise 1 only — then STOP).

Mirrors ``scripts/probe_waterbirds_clip.py`` but swaps the concept source: instead of frozen-CLIP
cosine scores, the concept vector for each image is its own 312 CUB per-image MTurk attribute
labels (Section 3 of the task). The CLIP probe found the global CLIP embedding is dominated by the
background (place AUROC ~0.97, minority y-signal ~0.49). This probe tests the hypothesis that the
annotation-based concept space fixes that: bird-describing attributes should carry minority signal
WITHOUT tracking the pasted background.

Reuses the pure-numpy/sklearn analysis fns from the CLIP probe (concept-source-agnostic):
  - probe_concept_usefulness : AUROC(attributes -> y), overall / majority / minority
  - probe_contamination      : AUROC(attributes -> place) whole-vector + per-attribute ranking

CPU-only: logistic regression over a 312-dim attribute matrix. No GPU, no torch, no CLIP.

    python -m scripts.probe_waterbirds_cub --config configs/waterbirds_cub.yaml
    python -m scripts.probe_waterbirds_cub --smoke   # synthetic self-test, no dataset
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
from models.concept_extractor_clip import ConceptStandardizer
from scripts.probe_waterbirds_clip import probe_concept_usefulness, probe_contamination
from vgscp_logging.manifest import RunManifest, make_run_id

# Pre-committed reading thresholds (committed before the numbers exist — see PROBE_REPORT_CUB.md).
GREEN_MIN_MINORITY_AUROC = 0.65   # minority AUROC(attrs->y) >= this
GREEN_MAX_PLACE_AUROC = 0.80      # AUROC(attrs->place) clearly below this (CLIP failed at ~0.97)
RED_MINORITY_AUROC = 0.55         # at/below ~chance => no recoverable bird signal
RED_PLACE_AUROC = 0.95            # ~CLIP-level contamination => likely a join bug


def _read_verdict(minority_auroc: float, place_auroc: float) -> tuple[str, str]:
    """Apply the pre-committed GREEN/AMBER/RED rule from the task to the probe numbers."""
    if np.isnan(minority_auroc) or np.isnan(place_auroc):
        return "INCONCLUSIVE", "Degenerate AUROC (NaN) — check subsample / split composition."
    if minority_auroc <= RED_MINORITY_AUROC or place_auroc >= RED_PLACE_AUROC:
        return "RED", (
            "This concept space also fails: minority bird signal is ~chance OR contamination is "
            "~CLIP-level. If contamination is high, double-check the CUB<->Waterbirds join."
        )
    if minority_auroc >= GREEN_MIN_MINORITY_AUROC and place_auroc < GREEN_MAX_PLACE_AUROC:
        return "GREEN", (
            "Premise 1 solved on real data: per-image attributes carry minority bird-type signal "
            "AND are far less background-contaminated than CLIP. Proceed to build the real "
            "pipeline + premise-2 (Morgana) test."
        )
    return "AMBER", (
        "Minority signal present but modest, and/or moderate contamination. Proceed, but the "
        "premise-2 control baselines (plain attribute-probe) become decisive."
    )


# ----------------------------------------------------------------------------------------
# Real run (needs the Waterbirds metadata + the CUB attribute txt files; CPU-only)
# ----------------------------------------------------------------------------------------
def run_real(cfg):
    from data.cub_attributes import load_cub_attribute_concepts
    from data.waterbirds import load_waterbirds
    from perf.setup import seed_everything

    seed = cfg["experiment"]["seeds"][0]
    seed_everything(seed, deterministic=True)
    pcfg = cfg["probe"]
    ccfg = cfg["cub"]
    splits = ("train", "d_test")

    # paths-only Waterbirds bundle (no torch Datasets needed for the attribute probe)
    bundle = load_waterbirds(cfg["dataset"], seed, max_per_split=pcfg["max_per_split"],
                             build_datasets=False)

    concepts, attr_names, join_info = load_cub_attribute_concepts(
        bundle, ccfg["root"], splits=splits,
        download=ccfg.get("download", False), url=ccfg.get("url"),
        use_certainty=ccfg.get("use_certainty", False),
        min_coverage=ccfg.get("min_coverage", 0.99),
    )

    train_raw, test_raw = concepts["train"], concepts["d_test"]
    standardize = bool(ccfg.get("standardize", False))  # binary attrs: standardization optional
    if standardize:
        std = ConceptStandardizer.fit(train_raw)  # TRAIN-only stats
        train_X, test_X = std.transform(train_raw), std.transform(test_raw)
    else:
        train_X, test_X = train_raw, test_raw

    train_y, test_y = bundle.y["train"], bundle.y["d_test"]
    train_place, test_place = bundle.spurious_attr["train"], bundle.spurious_attr["d_test"]
    test_minority = bundle.is_minority["d_test"]

    usefulness, _ = probe_concept_usefulness(train_X, train_y, test_X, test_y, test_minority, seed)
    contamination = probe_contamination(train_X, train_place, test_X, test_place, attr_names, seed)

    verdict, rationale = _read_verdict(usefulness["auroc_y_minority"],
                                       contamination["whole_vector_place_auroc"])
    res = {
        "concept_source": "cub_per_image_attributes",
        "n_train": int(len(train_y)), "n_test": int(len(test_y)),
        "n_concepts": int(train_X.shape[1]), "standardize": standardize,
        "use_certainty": bool(ccfg.get("use_certainty", False)),
        "minority_frac_test": float(test_minority.mean()),
        "join_info": join_info,
        "usefulness": usefulness, "contamination": contamination,
        "verdict": verdict, "verdict_rationale": rationale,
        "thresholds": {
            "green_min_minority_auroc": GREEN_MIN_MINORITY_AUROC,
            "green_max_place_auroc": GREEN_MAX_PLACE_AUROC,
            "red_minority_auroc": RED_MINORITY_AUROC,
            "red_place_auroc": RED_PLACE_AUROC,
        },
    }
    cache = {"train_X": train_X, "test_X": test_X, "train_y": train_y, "test_y": test_y,
             "train_place": train_place, "test_place": test_place}
    return res, cache, attr_names


# ----------------------------------------------------------------------------------------
# Smoke (synthetic) self-test — fabricated per-image binary attributes, no dataset
# ----------------------------------------------------------------------------------------
def run_smoke():
    """Per-image binary attrs where some encode the bird type y, some the background place, rest
    noise. Validates the join->probe analysis path without CUB / Waterbirds."""
    rng = np.random.default_rng(0)
    K, n = 312, 800

    def gen(m):
        y = rng.integers(0, 2, m)
        minority = rng.random(m) < 0.2
        place = np.where(minority, 1 - y, y)
        X = (rng.random((m, K)) < 0.1).astype(np.float32)          # sparse base noise
        # ~30 bird-attr concepts correlate with y; ~6 with place (weak background leak)
        for k in range(30):
            X[:, k] = (rng.random(m) < np.where(y == 1, 0.7, 0.2)).astype(np.float32)
        for k in range(30, 36):
            X[:, k] = (rng.random(m) < np.where(place == 1, 0.6, 0.3)).astype(np.float32)
        return X, y, place, minority

    train_X, train_y, train_place, _ = gen(n)
    test_X, test_y, test_place, test_minority = gen(n)
    attr_names = [f"attr_{i}" for i in range(1, K + 1)]

    usefulness, _ = probe_concept_usefulness(train_X, train_y, test_X, test_y, test_minority)
    contamination = probe_contamination(train_X, train_place, test_X, test_place, attr_names)
    verdict, rationale = _read_verdict(usefulness["auroc_y_minority"],
                                       contamination["whole_vector_place_auroc"])
    return {"concept_source": "smoke_synthetic", "n_train": n, "n_test": n, "n_concepts": K,
            "minority_frac_test": float(test_minority.mean()),
            "usefulness": usefulness, "contamination": contamination,
            "verdict": verdict, "verdict_rationale": rationale}


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


def _print_summary(res, run_id=None):
    u, c = res["usefulness"], res["contamination"]
    print("\n==== WATERBIRDS CUB PER-IMAGE ATTRIBUTE PROBE ====")
    if run_id:
        print(f"run_id: {run_id}", end="  ")
    print(f"(n_train={res['n_train']}, n_test={res['n_test']}, K={res['n_concepts']}, "
          f"standardize={res.get('standardize')})")
    if "join_info" in res:
        ji = res["join_info"]
        print(f"[join] coverage={ji['coverage']:.4f} ({ji['n_matched']}/{ji['n_total']}), "
              f"malformed_attr_rows={ji['n_malformed_attr_rows']}")
    print(f"[usefulness] AUROC(attrs->y): overall={u['auroc_y_overall']:.3f} "
          f"majority={u['auroc_y_majority']:.3f} minority={u['auroc_y_minority']:.3f}")
    print(f"[contamination] AUROC(attrs->place)={c['whole_vector_place_auroc']:.3f}")
    print(f"  top contaminating attrs: {[n for n, _ in c['top_contaminating_concepts'][:5]]}")
    print(f"\n>>> READING: {res['verdict']} - {res['verdict_rationale']}")


def _write_report(path, res, run_id):
    u, c = res["usefulness"], res["contamination"]
    ji = res.get("join_info", {})
    top = "\n".join(f"  - `{n}` (place AUROC {a:.3f})" for n, a in c["top_contaminating_concepts"][:10])
    lines = f"""# PROBE REPORT — CUB per-image attributes as the concept space on Waterbirds

**Concept source:** CUB-200-2011 per-image MTurk attributes (312-dim, frozen annotations,
per-image NOT class-level — no oracle). **Probe gates premise 1 only, then STOPS.**
**run_id:** `{run_id}`

## Why this probe
The frozen-CLIP concept-bottleneck was RED: the global CLIP image embedding is dominated by the
background, so it had ~0.97 AUROC for `place` and ~0.49 minority AUROC(concepts->y). This probe
swaps the concept source to the dataset's own per-image bird attributes, which describe the bird
and are independent of the pasted background — hypothesis: minority bird-type signal WITHOUT
background contamination.

## Join (CUB <-> Waterbirds)
- coverage: **{ji.get('coverage', float('nan')):.4f}** ({ji.get('n_matched')}/{ji.get('n_total')} images matched a CUB attribute vector)
- malformed attribute rows skipped (known file issue): {ji.get('n_malformed_attr_rows')}
- use_certainty: {res.get('use_certainty')} · standardize: {res.get('standardize')}

## Results
| Metric | Value |
|---|---|
| usefulness AUROC(attrs->y) overall / majority / **minority** | {u['auroc_y_overall']:.3f} / {u['auroc_y_majority']:.3f} / **{u['auroc_y_minority']:.3f}** |
| contamination AUROC(attrs->place), whole-vector | {c['whole_vector_place_auroc']:.3f} |
| n_train / n_test / K | {res['n_train']} / {res['n_test']} / {res['n_concepts']} |
| minority frac (test) | {res['minority_frac_test']:.3f} |

Top contaminating attributes (single-attr AUROC for `place`):
{top}

## Pre-committed reading rule
- **GREEN** (proceed to real pipeline + premise-2/Morgana): minority AUROC(attrs->y) >= ~{GREEN_MIN_MINORITY_AUROC}
  AND AUROC(attrs->place) clearly < ~{GREEN_MAX_PLACE_AUROC} (ideally ~0.5-0.65). This is the exact
  contrast showing annotations fix the CLIP background-domination.
- **AMBER** (~0.6-0.65 minority, or moderate contamination): proceed; premise-2 baselines decisive.
- **RED** (minority ~0.5, or contamination ~CLIP's 0.97 => suspect a join bug).

## READING: {res['verdict']}
{res['verdict_rationale']}

> The probe gates premise 1, not the paper. The real GO/NO-GO is premise 2 (does Morgana/V beat a
> plain attribute probe + trust + ensemble on the minority), tested only after GREEN. **STOP here
> and await human review.**
"""
    with open(path, "w") as f:
        f.write(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/waterbirds_cub.yaml")
    ap.add_argument("--smoke", action="store_true", help="synthetic self-test (no dataset)")
    ap.add_argument("--timestamp", default=None)
    args = ap.parse_args()

    ts = args.timestamp or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    if args.smoke:
        res = run_smoke()
        print(json.dumps(_to_jsonable(res), indent=2))
        _print_summary(res)
        print("\n[probe] SMOKE OK — join->analysis logic runs. Use the real run for the verdict.")
        return

    cfg = load_config(args.config)
    run_id = make_run_id(cfg["experiment"]["name"] + "_probe", ts)
    run_dir = os.path.join(cfg["experiment"]["out_dir"], "runs", run_id)
    os.makedirs(run_dir, exist_ok=True)
    manifest = RunManifest(run_dir, cfg)
    manifest.write_config_yaml(cfg)

    res, cache_arrays, attr_names = run_real(cfg)

    # cache the joined attribute matrices + the attribute-name list + results
    np.savez(os.path.join(run_dir, "cub_concepts.npz"), **cache_arrays)
    with open(os.path.join(run_dir, "attribute_names.json"), "w") as f:
        json.dump(attr_names, f, indent=2)
    with open(os.path.join(run_dir, "probe_results.json"), "w") as f:
        json.dump(_to_jsonable(res), f, indent=2)
    manifest.set("probe_results", _to_jsonable(res))
    manifest.save()

    _print_summary(res, run_id)
    # write the report both into the run dir and at repo root for review
    _write_report(os.path.join(run_dir, "PROBE_REPORT_CUB.md"), res, run_id)
    _write_report("PROBE_REPORT_CUB.md", res, run_id)
    print(f"\nresults -> {run_dir}/probe_results.json")
    print("[probe] STOP. Review PROBE_REPORT_CUB.md, then await 'go' for the premise-2 build.")


if __name__ == "__main__":
    main()
