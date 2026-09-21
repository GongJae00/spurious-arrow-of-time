"""Shared publication style for manuscript figures."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

TEXT: Final = "#1F2933"
MUTED_TEXT: Final = "#5B6470"
LIGHT_TEXT: Final = "#7A8591"
GRID: Final = "#E7EBEF"
SPINE: Final = "#B9C1CA"
PANEL_BORDER: Final = "#D9DEE5"
MISSING: Final = "#F1F3F5"

NUISANCE: Final = "#D06B45"
SEQUENCE: Final = "#273F4D"

AUDIT_CMAP: Final = LinearSegmentedColormap.from_list(
    "audit_accuracy",
    ["#F8FAFC", "#E8EEF3", "#CADAE4", "#8DB3C4", "#4F849B", "#1F526C"],
)


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 400,
            "font.family": "DejaVu Sans",
            "font.size": 7.8,
            "axes.titlesize": 8.2,
            "axes.labelsize": 7.6,
            "xtick.labelsize": 6.9,
            "ytick.labelsize": 6.9,
            "legend.fontsize": 6.9,
            "axes.linewidth": 0.65,
            "axes.edgecolor": SPINE,
            "axes.labelcolor": TEXT,
            "text.color": TEXT,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.8,
            "ytick.major.size": 2.8,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, out_dir: Path, name: str) -> None:
    figure_dir = out_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / f"{name}.pdf", bbox_inches="tight", pad_inches=0.025)
    fig.savefig(figure_dir / f"{name}.png", bbox_inches="tight", pad_inches=0.025, dpi=400)
    plt.close(fig)


def style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(SPINE)
    ax.spines["left"].set_linewidth(0.65)
    ax.spines["bottom"].set_color(SPINE)
    ax.spines["bottom"].set_linewidth(0.65)
    ax.tick_params(length=2.8, width=0.6, colors=TEXT)
    ax.grid(axis="y", color=GRID, linewidth=0.55)
    ax.set_axisbelow(True)
