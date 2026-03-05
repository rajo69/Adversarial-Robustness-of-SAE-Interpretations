"""
Shared figure styling configuration for the SAE Adversarial Robustness project.

All notebooks and analysis scripts should import from this module to ensure
consistent visual style across all figures.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import Optional


# ---------------------------------------------------------------------------
# Color palette (colorblind-friendly)
# ---------------------------------------------------------------------------
COLORS = {
    "adversarial": "#E63946",   # Red — adversarial perturbations
    "random":      "#457B9D",   # Blue — random baseline
    "clean":       "#2A9D8F",   # Teal — clean / unperturbed
    "eps_001":     "#264653",
    "eps_005":     "#2A9D8F",
    "eps_01":      "#E9C46A",
    "eps_05":      "#E63946",
    "highlight":   "#F4A261",   # Orange — highlighted points
    "neutral":     "#ADB5BD",   # Grey — neutral reference lines
}

EPSILON_COLORS = {
    0.01: COLORS["eps_001"],
    0.05: COLORS["eps_005"],
    0.1:  COLORS["eps_01"],
    0.5:  COLORS["eps_05"],
}

LAYER_CMAP = "viridis"


# ---------------------------------------------------------------------------
# Font and size settings
# ---------------------------------------------------------------------------
FONT_SIZE_TITLE  = 14
FONT_SIZE_LABEL  = 12
FONT_SIZE_TICK   = 10
FONT_SIZE_LEGEND = 10

FIGURE_DPI    = 300
FIGURE_WIDTH  = 7.0   # inches — single column
FIGURE_HEIGHT = 5.0

LINEWIDTH = 1.8
MARKERSIZE = 6


# ---------------------------------------------------------------------------
# Apply global rcParams
# ---------------------------------------------------------------------------
def apply_style() -> None:
    """Apply the project-wide matplotlib style.

    Call once at the top of each notebook or script before creating any figures.

    Examples
    --------
    >>> from src.plot_config import apply_style
    >>> apply_style()
    """
    mpl.rcParams.update({
        # Fonts
        "font.family":       "sans-serif",
        "font.sans-serif":   ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size":         FONT_SIZE_LABEL,
        "axes.titlesize":    FONT_SIZE_TITLE,
        "axes.labelsize":    FONT_SIZE_LABEL,
        "xtick.labelsize":   FONT_SIZE_TICK,
        "ytick.labelsize":   FONT_SIZE_TICK,
        "legend.fontsize":   FONT_SIZE_LEGEND,
        # Lines and markers
        "lines.linewidth":   LINEWIDTH,
        "lines.markersize":  MARKERSIZE,
        # Axes
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         True,
        "grid.alpha":        0.3,
        "grid.linestyle":    "--",
        # Figure
        "figure.dpi":        100,   # screen DPI; save at FIGURE_DPI
        "figure.facecolor":  "white",
        "savefig.dpi":       FIGURE_DPI,
        "savefig.bbox":      "tight",
        "savefig.facecolor": "white",
    })


# ---------------------------------------------------------------------------
# Helper: create a standard figure
# ---------------------------------------------------------------------------
def make_fig(
    width: float = FIGURE_WIDTH,
    height: float = FIGURE_HEIGHT,
    ncols: int = 1,
    nrows: int = 1,
    **kwargs,
) -> tuple:
    """Create a figure with the project style applied.

    Parameters
    ----------
    width : float
        Figure width in inches.
    height : float
        Figure height in inches.
    ncols : int
        Number of subplot columns.
    nrows : int
        Number of subplot rows.
    **kwargs
        Additional keyword arguments passed to ``plt.subplots``.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : matplotlib.axes.Axes or ndarray of Axes
    """
    apply_style()
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=(width * ncols, height * nrows), **kwargs)
    return fig, axes


# ---------------------------------------------------------------------------
# Helper: save figure
# ---------------------------------------------------------------------------
def save_fig(fig: mpl.figure.Figure, path: str) -> None:
    """Save a figure at 300 DPI with tight bounding box.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to save.
    path : str
        Destination file path (e.g., ``"figures/exp1_cosine_vs_epsilon.png"``).
    """
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Helper: epsilon label for legends
# ---------------------------------------------------------------------------
def epsilon_label(eps: float) -> str:
    """Return a formatted legend label for an epsilon value.

    Parameters
    ----------
    eps : float
        Perturbation magnitude.

    Returns
    -------
    str
        E.g. ``"ε = 0.10"``.
    """
    return f"ε = {eps:.2f}"
