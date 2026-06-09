"""Minimal matplotlib helpers for the kill-switch deliverables (Agg backend, no display)."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_group_coverage(per_group_by_method: dict, alpha: float, path: str, title: str):
    """Grouped bar chart: true-attribute-conditional coverage per group, one cluster per method,
    with the nominal (1-alpha) line. ``per_group_by_method`` = {method: {group: (mean, std)}}."""
    methods = list(per_group_by_method)
    groups = sorted({g for m in per_group_by_method.values() for g in m})
    x = np.arange(len(groups))
    w = 0.8 / max(1, len(methods))
    fig, ax = plt.subplots(figsize=(1.6 * len(groups) + 3, 4))
    for i, m in enumerate(methods):
        means = [per_group_by_method[m].get(g, (np.nan, 0))[0] for g in groups]
        stds = [per_group_by_method[m].get(g, (np.nan, 0))[1] for g in groups]
        ax.bar(x + i * w, means, w, yerr=stds, capsize=3, label=m)
    ax.axhline(1 - alpha, color="k", ls="--", lw=1, label=f"nominal {1-alpha:.2f}")
    ax.set_xticks(x + w * (len(methods) - 1) / 2)
    ax.set_xticklabels([f"A_true={g}" for g in groups])
    ax.set_ylabel("true-conditional coverage")
    ax.set_ylim(0, 1.02)
    ax.set_title(title)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_setsize(size_by_method: dict, path: str, title: str, trivial: float | None = None):
    """Bar chart of mean overall set size per method (+ optional trivial |Y| reference line).
    ``size_by_method`` = {method: (mean, std)}."""
    methods = list(size_by_method)
    means = [size_by_method[m][0] for m in methods]
    stds = [size_by_method[m][1] for m in methods]
    fig, ax = plt.subplots(figsize=(1.3 * len(methods) + 2, 4))
    ax.bar(np.arange(len(methods)), means, yerr=stds, capsize=3, color="C2")
    if trivial is not None:
        ax.axhline(trivial, color="r", ls="--", lw=1, label=f"trivial |Y|={trivial:g}")
        ax.legend(fontsize=8)
    ax.set_xticks(np.arange(len(methods)))
    ax.set_xticklabels(methods, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("mean set size")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_lines(x, series: dict, xlabel: str, ylabel: str, path: str, title: str,
               hline: float | None = None, hlabel: str | None = None):
    """Generic multi-series line plot (series = {label: (y_means, y_stds_or_None)})."""
    fig, ax = plt.subplots(figsize=(6.2, 4))
    for lab, val in series.items():
        ys, es = (val if isinstance(val, tuple) else (val, None))
        if es is not None:
            ax.errorbar(x, ys, yerr=es, marker="o", capsize=3, label=lab)
        else:
            ax.plot(x, ys, "o-", label=lab)
    if hline is not None:
        ax.axhline(hline, color="k", ls="--", lw=1, label=hlabel or f"{hline:g}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_breakdown_vs_corr(corrs: list, gaps_by_group: dict, path: str, title: str):
    """KS-1b: under-coverage of the true minority group vs the score<->flip correlation."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for g, gaps in gaps_by_group.items():
        ax.plot(corrs, gaps, "o-", label=f"A_true={g}")
    ax.axhline(0, color="k", lw=1)
    ax.set_xlabel("score<->flip dependence (AUROC of score predicting probe error)")
    ax.set_ylabel("coverage gap to nominal (deconv − (1−α))")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
