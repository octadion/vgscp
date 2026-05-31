"""Synthetic theory figure (Section 11): P2 separation + regime map + P1 validity.

Four panels:
  (a) minority error-detection AUROC vs shortcut strength (V stays high; conf/trust/ensemble fall)
  (b) contamination AUROC(signal -> spurious attr) vs shortcut strength (V ~0.5; others rise)
  (c) P2 separation scatter at the strongest regime: contamination (x) vs minority AUROC (y)
  (d) P1 validity: retained marginal coverage vs abstention budget (>= target line)
"""
from __future__ import annotations

import numpy as np

from .style import SIGNAL_COLORS, SIGNAL_LABELS, apply_style


def make_theory_figure(agg: dict, p1_df, signals, out_path: str):
    plt = apply_style()
    strengths = sorted(agg.keys())

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    def series(metric_key, sig):
        return [agg[s]["per_signal"][sig][metric_key] for s in strengths]

    # (a) minority AUROC vs strength
    for sig in signals:
        ax_a.plot(strengths, series("auroc_minority_mean", sig),
                  marker="o", color=SIGNAL_COLORS.get(sig), label=SIGNAL_LABELS.get(sig, sig))
    ax_a.axhline(0.5, ls=":", c="grey", lw=1)
    ax_a.set_xlabel("shortcut strength $s$")
    ax_a.set_ylabel("minority error-detection AUROC")
    ax_a.set_title("(a) P2: minority correctness detection")
    ax_a.legend(ncol=2)

    # (b) contamination AUROC vs strength
    for sig in signals:
        ax_b.plot(strengths, series("contamination_auroc_mean", sig),
                  marker="s", color=SIGNAL_COLORS.get(sig), label=SIGNAL_LABELS.get(sig, sig))
    ax_b.axhline(0.5, ls=":", c="grey", lw=1)
    ax_b.set_xlabel("shortcut strength $s$")
    ax_b.set_ylabel("contamination AUROC (signal $\\to$ spurious attr)")
    ax_b.set_title("(b) P2: contamination by the shortcut")

    # (c) separation scatter at strongest strength
    s_max = strengths[-1]
    for sig in signals:
        x = agg[s_max]["per_signal"][sig]["contamination_auroc_mean"]
        y = agg[s_max]["per_signal"][sig]["auroc_minority_mean"]
        ax_c.scatter(x, y, s=70, color=SIGNAL_COLORS.get(sig), zorder=3)
        ax_c.annotate(SIGNAL_LABELS.get(sig, sig), (x, y), textcoords="offset points",
                      xytext=(6, 4), fontsize=8)
    ax_c.axhline(0.5, ls=":", c="grey", lw=1)
    ax_c.axvline(0.5, ls=":", c="grey", lw=1)
    ax_c.set_xlabel("contamination AUROC (lower = robust)")
    ax_c.set_ylabel("minority AUROC (higher = useful)")
    ax_c.set_title(f"(c) P2 separation @ $s$={s_max}\n(top-left = robust & useful)")

    # (d) P1 validity: coverage vs budget
    target = 1.0 - 0.10
    if p1_df is not None and len(p1_df):
        for s in strengths:
            sub = p1_df[p1_df["shortcut_strength"] == s]
            if len(sub):
                g = sub.groupby("budget")["coverage"].mean()
                ax_d.plot(g.index, g.values, marker=".", alpha=0.7, label=f"s={s}")
        ax_d.axhline(target, ls="--", c="red", lw=1.2, label="target $1-\\alpha$")
        ax_d.set_xlabel("abstention budget $b$")
        ax_d.set_ylabel("retained marginal coverage")
        ax_d.set_title("(d) P1: selective-conformal validity")
        ax_d.legend(ncol=2, fontsize=7)

    fig.suptitle("Synthetic Gaussian theory testbed — P1 (validity) & P2 (separation)", y=1.0)
    fig.tight_layout()
    import os

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
