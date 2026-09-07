"""Publication figure style, implementing the figures4papers `scientific-figure-making` spec.

That skill documents conventions to implement rather than a package to import
(`references/api.md`: "Implement these in your own code or adapt from the figure_* demos"), so
this module is that implementation, kept in the repository so the archived code reproduces the
figures as well as the tables.

What it takes from the spec, and why each matters here:

  * PALETTE / DEFAULT_COLORS verbatim, and the semantic mapping -- blue for the key mechanism, red
    for the baseline it is contrasted against, neutral grey for background categories. In this
    paper the key mechanism is Mondrian calibration and the baseline is marginal split, so those
    take blue and red in every figure rather than being distinguished by line style.
  * Frameless legends and top/right spines removed.
  * Vector export. `finalize_figure` writes PDF alongside PNG and sets `pdf.fonttype = 42` and
    `svg.fonttype = 'none'`, so text stays selectable and the figure stays sharp at any zoom --
    the single largest visible difference in a compiled paper.
  * Print-safe bars: black edges, so groups remain distinguishable in greyscale.
  * Dynamic y-limits, so differences are visible rather than squashed against zero -- except on
    bar charts, where a truncated axis would exaggerate them.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

__all__ = ["PALETTE", "DEFAULT_COLORS", "FigureStyle", "apply_publication_style",
           "create_subplots", "finalize_figure", "annotate_bars", "make_grouped_bar",
           "make_trend", "dynamic_ylim"]

PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "green_1": "#DDF3DE", "green_2": "#AADCA9", "green_3": "#8BCF8B",
    "red_1": "#F6CFCB", "red_2": "#E9A6A1", "red_strong": "#B64342",
    "neutral": "#CFCECE", "highlight": "#FFD700",
    "teal": "#42949E", "violet": "#9A4D8E",
}

DEFAULT_COLORS = [PALETTE["blue_main"], PALETTE["green_3"], PALETTE["red_strong"],
                  PALETTE["teal"], PALETTE["violet"], PALETTE["neutral"]]


@dataclass(frozen=True)
class FigureStyle:
    font_size: int = 16
    axes_linewidth: float = 2.5
    use_tex: bool = False
    # Arial/Helvetica first, per design-theory.md: the house style is Helvetica, with Arial as the
    # portable stand-in. DejaVu Sans is matplotlib's own default and stays only as the last resort,
    # since it is the font that makes a figure read as an unstyled matplotlib plot.
    font_family: tuple = field(default=("Arial", "Helvetica", "DejaVu Sans", "sans-serif"))


def apply_publication_style(style: FigureStyle | None = None) -> None:
    """Set rcParams once, before any figure is created."""
    import matplotlib

    if matplotlib.get_backend().lower() not in ("agg", "pdf", "svg", "ps"):
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    st = style or FigureStyle()
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": list(st.font_family),
        "font.size": st.font_size,
        "axes.linewidth": st.axes_linewidth,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.labelsize": st.font_size,
        "axes.titlesize": st.font_size,
        "xtick.labelsize": st.font_size - 2,
        "ytick.labelsize": st.font_size - 2,
        "xtick.major.width": st.axes_linewidth,
        "ytick.major.width": st.axes_linewidth,
        "legend.fontsize": st.font_size - 3,
        "legend.frameon": False,
        "text.usetex": st.use_tex,
        # Editable, non-rasterised text in vector output.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.dpi": 300,
        "figure.dpi": 120,
    })


def create_subplots(nrows: int = 1, ncols: int = 1, figsize=None, **kwargs):
    """(fig, axes) with axes always a flat 1-D array, so callers need no shape branching."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)
    axes = np.atleast_1d(np.asarray(axes)).ravel()
    return fig, axes


def finalize_figure(fig, out_path: str, formats=None, dpi: int = 300, close: bool = True,
                    pad: float = 0.05, **kwargs) -> list:
    """Save to every requested format. Defaults to PDF plus PNG: PDF for the paper, PNG to look at.

    LaTeX including a 300-dpi PNG is the usual reason a figure looks soft in a compiled PDF; a
    vector PDF has no such ceiling.
    """
    ALLOWED = {"pdf", "svg", "eps", "png", "jpg", "jpeg", "tif", "tiff"}
    root, ext = os.path.splitext(out_path)
    if formats is None:
        formats = [ext.lstrip(".").lower()] if ext else ["pdf", "png"]
    bad = [f for f in formats if f.lower() not in ALLOWED]
    if bad:
        raise ValueError(f"format tidak didukung: {bad}; pilih dari {sorted(ALLOWED)}")
    os.makedirs(os.path.dirname(root) or ".", exist_ok=True)
    written = []
    for f in formats:
        p = f"{root}.{f.lower()}"
        fig.savefig(p, dpi=dpi, bbox_inches="tight", pad_inches=pad, **kwargs)
        written.append(p)
    if close:
        import matplotlib.pyplot as plt

        plt.close(fig)
    return written


def dynamic_ylim(ax, values, margin: float = 0.12, floor=None, ceil=None) -> None:
    """Tighten the y-axis to the data. Never use on bars: a truncated bar axis misleads."""
    v = np.asarray([x for x in np.ravel(values) if np.isfinite(x)], dtype=float)
    lo, hi = v.min(), v.max()
    pad = max((hi - lo) * margin, 1e-3)
    lo, hi = lo - pad, hi + pad
    if floor is not None:
        lo = max(lo, floor)
    if ceil is not None:
        hi = min(hi, ceil)
    ax.set_ylim(lo, hi)


def annotate_bars(ax, bars, fmt: str = "{:.2f}", fontsize: int = 10, padding: int = 3,
                  color: str | None = None) -> None:
    for b in bars:
        h = b.get_height()
        ax.annotate(fmt.format(h), xy=(b.get_x() + b.get_width() / 2, h),
                    xytext=(0, padding), textcoords="offset points",
                    ha="center", va="bottom", fontsize=fontsize,
                    color=color or "black", fontweight="bold")


def make_grouped_bar(ax, categories, series, labels, ylabel: str = "Value", colors=None,
                     annotate: bool = False, edgecolor: str = "black", edgewidth: float = 1.5):
    """Grouped bars with print-safe edges. Returns the last BarContainer."""
    series = [np.asarray(s, dtype=float) for s in series]
    for s in series:
        if s.shape[0] != len(categories):
            raise ValueError(f"panjang seri {s.shape[0]} != jumlah kategori {len(categories)}")
    colors = colors or DEFAULT_COLORS
    x = np.arange(len(categories), dtype=float)
    w = 0.8 / len(series)
    bars = None
    for i, (s, lab) in enumerate(zip(series, labels)):
        off = (i - (len(series) - 1) / 2) * w
        bars = ax.bar(x + off, s, w, label=lab, color=colors[i % len(colors)],
                      edgecolor=edgecolor, linewidth=edgewidth)
        if annotate:
            annotate_bars(ax, bars)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel(ylabel)
    return bars


def make_trend(ax, x, y_series, labels, colors=None, ylabel=None, xlabel=None,
               show_shadow: bool = True, bands=None, linewidth: float = 2.6, marker: str = "o",
               markersize: float = 5.0, linestyle: str = "-", alpha: float = 1.0):
    """Line plot with an optional uncertainty band per series.

    ``bands`` is a list of (lo, hi) arrays. When omitted and ``show_shadow`` is set, the band is
    the min-max envelope of the series themselves -- which is what shows an across-condition
    spread rather than a within-condition interval.
    """
    x = np.asarray(x, dtype=float)
    y_series = [np.asarray(y, dtype=float) for y in y_series]
    for y in y_series:
        if y.shape[0] != x.shape[0]:
            raise ValueError(f"panjang seri {y.shape[0]} != panjang x {x.shape[0]}")
    colors = colors or DEFAULT_COLORS
    for i, (y, lab) in enumerate(zip(y_series, labels)):
        c = colors[i % len(colors)]
        ax.plot(x, y, linestyle, marker=marker, ms=markersize, lw=linewidth, color=c,
                label=lab, alpha=alpha)
        if bands is not None and bands[i] is not None:
            lo, hi = bands[i]
            ax.fill_between(x, lo, hi, color=c, alpha=0.18, linewidth=0)
    if show_shadow and bands is None and len(y_series) > 1:
        stack = np.vstack(y_series)
        ax.fill_between(x, stack.min(axis=0), stack.max(axis=0),
                        color=colors[0], alpha=0.12, linewidth=0)
    if ylabel:
        ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
