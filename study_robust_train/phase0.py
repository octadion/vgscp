"""PHASE 0 orchestrator (spec §1) — ERM + DFR last-layer on Waterbirds, then STOP.

Spec §1: run a minimal pilot (ERM and DFR only, Waterbirds only), report worst-group
accuracy of each + a sample cross-group conformity-score divergence (APS), over 3 seeds,
then HALT for human review. This module DELIBERATELY does not contain the full H1/H2/H3
grid — Phase-0-then-STOP is a hard discipline (see README.md).

It is distribution-agnostic: it consumes a ``Phase0Data`` (per-split features / y / place /
group, all in-domain composited). The Colab notebook builds that from the real Waterbirds
CLIP features; the local synthetic validator builds it from generated features. No real
numbers are produced anywhere except the Colab run.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import metrics
from .divergence import cross_group_divergence, true_label_conformity_scores
from .heads import fit_dfr, fit_erm, head_probs

# Spec §1 sanity reference (the HUMAN STOP gate, checked when real numbers exist on Colab).
DFR_WG_REF = (0.86, 0.92)    # DFR worst-group accuracy should land here on Waterbirds
ERM_WG_REF = (0.60, 0.75)    # ERM worst-group much lower
PHASE0_STOP = "PHASE-0 COMPLETE -- STOP for human review. Do NOT proceed to the H1/H2/H3 grid."


@dataclass
class Phase0Data:
    """In-domain composited splits. Features L2-normalized; y in 0..n_classes-1; place in {0,1}."""
    feats_train: np.ndarray
    y_train: np.ndarray
    place_train: np.ndarray
    feats_test: np.ndarray
    y_test: np.ndarray
    place_test: np.ndarray
    n_classes: int
    synthetic: bool = False     # True => numbers are NOT real results, logic-validation only

    def group_train(self) -> np.ndarray:
        return metrics.group_ids(self.y_train, self.place_train)

    def group_test(self) -> np.ndarray:
        return metrics.group_ids(self.y_test, self.place_test)


@dataclass
class ModelResult:
    method: str
    seed: int
    per_group_acc: dict
    worst_group: int
    worst_group_acc: float
    overall_acc: float
    # APS cross-group burden (worst group vs rest) — the §1 "sample divergence"
    aps_wasserstein1: float
    aps_ks_stat: float
    aps_ks_pvalue: float


def _eval_model(method: str, head, data: Phase0Data, seed: int) -> ModelResult:
    probs = head_probs(head, data.feats_test, data.n_classes)
    y_pred = np.argmax(probs, axis=1)
    g_test = data.group_test()
    pg = metrics.per_group_accuracy(y_pred, data.y_test, g_test)
    worst_g, worst_acc = metrics.worst_group_accuracy(y_pred, data.y_test, g_test)
    overall = float((y_pred == data.y_test).mean())

    aps = true_label_conformity_scores(probs, data.y_test, score="APS", seed=seed)
    div = cross_group_divergence(aps, g_test, worst_group=worst_g, score_name="APS")
    return ModelResult(
        method=method, seed=seed, per_group_acc=pg, worst_group=worst_g,
        worst_group_acc=worst_acc, overall_acc=overall,
        aps_wasserstein1=div.wasserstein1, aps_ks_stat=div.ks_stat,
        aps_ks_pvalue=div.ks_pvalue,
    )


def run_phase0(data: Phase0Data, seeds=(0, 1, 2), *, dfr_subsets: int = 10,
               C: float = 1.0, max_iter: int = 5000,
               dfr_fit: tuple | None = None) -> dict:
    """Train ERM + DFR per seed, evaluate worst-group acc + APS divergence, aggregate, STOP.

    ERM fits on ``data.feats_train``. DFR fits on ``dfr_fit`` if given as
    ``(feats, y, group)`` -- the textbook held-out reweighting split (e.g. the bundle's
    ``d_learn``); otherwise it falls back to the same in-domain ``train`` split. All fit splits
    must be the SAME composited distribution as ``d_test`` (in-domain, §2).

    Returns a dict with per-(method, seed) ModelResults and mean+/-std aggregates. Does NOT
    proceed to any grid. The DFR/ERM worst-group sanity check vs the published reference is a
    HUMAN step performed when this runs on real Waterbirds features in Colab.
    """
    if dfr_fit is None:
        dfr_feats, dfr_y, dfr_group = data.feats_train, data.y_train, data.group_train()
    else:
        dfr_feats, dfr_y, dfr_group = dfr_fit

    results: list[ModelResult] = []
    for seed in seeds:
        erm = fit_erm(data.feats_train, data.y_train, C=C, max_iter=max_iter, seed=seed)
        dfr = fit_dfr(dfr_feats, dfr_y, dfr_group,
                      n_subsets=dfr_subsets, C=C, max_iter=max_iter, seed=seed)
        results.append(_eval_model("ERM", erm, data, seed))
        results.append(_eval_model("DFR", dfr, data, seed))

    agg = {}
    for method in ("ERM", "DFR"):
        rows = [r for r in results if r.method == method]
        wg = np.array([r.worst_group_acc for r in rows])
        w1 = np.array([r.aps_wasserstein1 for r in rows])
        ks = np.array([r.aps_ks_stat for r in rows])
        ov = np.array([r.overall_acc for r in rows])
        agg[method] = {
            "worst_group_acc_mean": float(wg.mean()), "worst_group_acc_std": float(wg.std()),
            "overall_acc_mean": float(ov.mean()), "overall_acc_std": float(ov.std()),
            "aps_w1_mean": float(w1.mean()), "aps_w1_std": float(w1.std()),
            "aps_ks_mean": float(ks.mean()), "aps_ks_std": float(ks.std()),
        }

    return {
        "synthetic": bool(data.synthetic),
        "seeds": list(seeds),
        "n_classes": int(data.n_classes),
        "per_run": results,
        "aggregate": agg,
        "dfr_wg_reference": DFR_WG_REF,
        "erm_wg_reference": ERM_WG_REF,
        "stop": PHASE0_STOP,
    }


def format_report(out: dict) -> str:
    """Plain-text Phase-0 summary. Prepends a SYNTHETIC banner when not a real run."""
    lines = []
    if out["synthetic"]:
        lines += [
            "=" * 78,
            "SYNTHETIC LOGIC-VALIDATION ONLY -- these are NOT real results.",
            "Numbers below validate machinery (shapes, DFR reweighting, divergence, wiring),",
            "not the Waterbirds phenomenon. Real Phase-0 numbers come from the Colab notebook.",
            "=" * 78,
        ]
    lines.append(f"Phase-0 pilot (ERM + DFR, {len(out['seeds'])} seeds, n_classes={out['n_classes']})")
    for method in ("ERM", "DFR"):
        a = out["aggregate"][method]
        lines.append(
            f"  {method}: worst-group acc {a['worst_group_acc_mean']:.3f}+/-{a['worst_group_acc_std']:.3f}"
            f" | overall {a['overall_acc_mean']:.3f}"
            f" | APS W1(worst,rest) {a['aps_w1_mean']:.4f}+/-{a['aps_w1_std']:.4f}"
            f" | APS KS {a['aps_ks_mean']:.3f}"
        )
    lines.append(
        f"  [HUMAN STOP GATE] expect (real run): DFR worst-group in {out['dfr_wg_reference']}, "
        f"ERM worst-group in {out['erm_wg_reference']}."
    )
    lines.append(f"  {out['stop']}")
    return "\n".join(lines)
