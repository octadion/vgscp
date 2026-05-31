"""Regenerate all Phase-1 figures (PDF) from the per-sample parquet logs (Section 11).

    python -m viz.make_figures --run results/runs/<run_id>

Figures produced (from cached logs => no model re-runs):
  - risk_coverage_overall.pdf / risk_coverage_minority.pdf : per-signal risk-coverage + AURC
  - separation_contamination.pdf : AUROC(signal->spurious) vs minority error-detection AUROC
  - confident_but_wrong_capture.pdf : capture-rate bars per signal at a fixed budget
The synthetic theory figure is produced by theory/run_synthetic via viz/theory_figure.py.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from eval import metrics
from .style import SIGNAL_COLORS, SIGNAL_LABELS, apply_style

SIGNALS = ["conf_msp", "trust", "ensemble_disagree", "mcdropout", "V_comp", "V_sound", "V_full"]


def _present_signals(df):
    return [s for s in SIGNALS if s in df.columns]


def fig_risk_coverage(df, out_path, minority_only=False, title=""):
    plt = apply_style()
    sub = df[df["is_minority"] == 1] if minority_only else df
    correct = sub["correct"].to_numpy().astype(int)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for s in _present_signals(sub):
        sig = sub[s].to_numpy().astype(float)
        cov, risk = metrics.risk_coverage_curve(sig, correct)
        a = metrics.aurc(sig, correct)
        ax.plot(cov, risk, color=SIGNAL_COLORS.get(s),
                label=f"{SIGNAL_LABELS.get(s, s)} (AURC={a:.3f})")
    ax.set_xlabel("coverage (retained fraction)")
    ax.set_ylabel("selective risk (error rate)")
    ax.set_title(title or ("Minority risk-coverage" if minority_only else "Overall risk-coverage"))
    ax.legend(fontsize=7)
    _save(fig, out_path)


def fig_separation_contamination(df, out_path):
    plt = apply_style()
    correct = df["correct"].to_numpy().astype(int)
    minority = df["is_minority"].to_numpy().astype(bool)
    spur = df["spurious_attr"].to_numpy() if "spurious_attr" in df else None
    ytrue = df["y_true"].to_numpy() if "y_true" in df else None
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for s in _present_signals(df):
        sig = df[s].to_numpy().astype(float)
        y = metrics.error_detection_auroc(sig[minority], correct[minority]) if minority.any() else np.nan
        x = metrics.contamination_auroc(sig, spur, ytrue) if spur is not None else np.nan
        ax.scatter(x, y, s=80, color=SIGNAL_COLORS.get(s), zorder=3)
        ax.annotate(SIGNAL_LABELS.get(s, s), (x, y), textcoords="offset points",
                    xytext=(6, 4), fontsize=8)
    ax.axhline(0.5, ls=":", c="grey"); ax.axvline(0.5, ls=":", c="grey")
    ax.set_xlabel("contamination AUROC (signal $\\to$ spurious attr | label)")
    ax.set_ylabel("minority error-detection AUROC")
    ax.set_title("P2 separation: top-left = robust & useful")
    _save(fig, out_path)


def fig_capture_rate(df, out_path, budget=0.2):
    plt = apply_style()
    correct = df["correct"].to_numpy().astype(int)
    conf = df["conf_msp"].to_numpy()
    sigs = _present_signals(df)
    rates = [metrics.confident_but_wrong_capture_rate(df[s].to_numpy().astype(float),
                                                      correct, conf, budget) for s in sigs]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(range(len(sigs)), rates, color=[SIGNAL_COLORS.get(s) for s in sigs])
    ax.set_xticks(range(len(sigs)))
    ax.set_xticklabels([SIGNAL_LABELS.get(s, s) for s in sigs], rotation=30, ha="right")
    ax.set_ylabel("confident-but-wrong capture rate")
    ax.set_title(f"Confident-but-wrong capture @ budget b={budget}")
    _save(fig, out_path)


def _save(fig, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    import matplotlib.pyplot as plt

    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="results/runs/<run_id> directory")
    ap.add_argument("--out", default="results/figures")
    ap.add_argument("--budget", type=float, default=0.2)
    args = ap.parse_args()

    from vgscp_logging.parquet_logger import load_run_logs

    df = load_run_logs(args.run, split="test")
    fig_risk_coverage(df, os.path.join(args.out, "risk_coverage_overall.pdf"), False)
    fig_risk_coverage(df, os.path.join(args.out, "risk_coverage_minority.pdf"), True)
    fig_separation_contamination(df, os.path.join(args.out, "separation_contamination.pdf"))
    fig_capture_rate(df, os.path.join(args.out, "confident_but_wrong_capture.pdf"), args.budget)


if __name__ == "__main__":
    main()
