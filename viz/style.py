"""Consistent matplotlib rcParams for paper-ready PDF figures (Section 11)."""
from __future__ import annotations


def apply_style():
    import matplotlib

    matplotlib.use("Agg")  # headless / no display
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,   # editable text in PDF
            "ps.fonttype": 42,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.linewidth": 1.8,
            "lines.markersize": 5,
        }
    )
    return plt


# stable color per signal so every figure is consistent
SIGNAL_COLORS = {
    "conf_msp": "#1f77b4",
    "trust": "#ff7f0e",
    "ensemble_disagree": "#2ca02c",
    "mcdropout": "#9467bd",
    "V_comp": "#d62728",
    "V_sound": "#e377c2",
    "V_full": "#000000",
}

SIGNAL_LABELS = {
    "conf_msp": "Confidence (MSP)",
    "trust": "Trust score",
    "ensemble_disagree": "Ensemble disagree",
    "mcdropout": "MC-dropout",
    "V_comp": "$V_{comp}$",
    "V_sound": "$V_{sound}$",
    "V_full": "$V_{full}$ (ours)",
}
