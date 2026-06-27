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

CORE: Final = "#1F8A83"
CORE_LIGHT: Final = "#E8F4F2"
NUISANCE: Final = "#D06B45"
NUISANCE_LIGHT: Final = "#FAEEE8"
SEQUENCE: Final = "#273F4D"
SEQUENCE_LIGHT: Final = "#EAF0F3"
FINAL: Final = "#8E98A5"
FINAL_LIGHT: Final = "#F0F2F4"
COUNTERFACTUAL: Final = "#527DA3"
COUNTERFACTUAL_LIGHT: Final = "#EAF1F7"
POSITIVE: Final = "#2E7D72"
NEGATIVE: Final = "#B76855"

METHOD_COLORS: Final = {
    "core_only_oracle": CORE,
    "sequence_erm": SEQUENCE,
    "final_frame_mlp": FINAL,
    "nuisance_only_oracle": NUISANCE,
    "counterfactual_invariance": COUNTERFACTUAL,
}

METHOD_LIGHT: Final = {
    "core_only_oracle": CORE_LIGHT,
    "sequence_erm": SEQUENCE_LIGHT,
    "final_frame_mlp": FINAL_LIGHT,
    "nuisance_only_oracle": NUISANCE_LIGHT,
    "counterfactual_invariance": COUNTERFACTUAL_LIGHT,
}

METHOD_LABELS: Final = {
    "core_only_oracle": "Core-only\noracle",
    "sequence_erm": "Sequence\nERM",
    "final_frame_mlp": "Final-frame\nMLP",
    "nuisance_only_oracle": "Nuisance-only\noracle",
    "counterfactual_invariance": "Counterfactual\nreplacement",
}

AUDIT_CMAP: Final = LinearSegmentedColormap.from_list(
    "audit_accuracy",
    ["#F8FAFC", "#E8EEF3", "#CADAE4", "#8DB3C4", "#4F849B", "#1F526C"],
)


def apply_style() -> None:
    """Apply a shared, non-default style to all manuscript figures."""

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
    """Save a figure as vector PDF and high-resolution PNG."""

    figure_dir = out_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / f"{name}.pdf", bbox_inches="tight", pad_inches=0.025)
    fig.savefig(figure_dir / f"{name}.png", bbox_inches="tight", pad_inches=0.025, dpi=400)
    plt.close(fig)


def style_axis(
    ax: plt.Axes,
    *,
    grid: bool = True,
    hide_left: bool = False,
    hide_bottom: bool = False,
) -> None:
    """Apply shared axis styling."""

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(not hide_left)
    ax.spines["bottom"].set_visible(not hide_bottom)
    if not hide_left:
        ax.spines["left"].set_color(SPINE)
        ax.spines["left"].set_linewidth(0.65)
    if not hide_bottom:
        ax.spines["bottom"].set_color(SPINE)
        ax.spines["bottom"].set_linewidth(0.65)
    ax.tick_params(length=2.8, width=0.6, colors=TEXT)
    if grid:
        ax.grid(axis="y", color=GRID, linewidth=0.55)
        ax.set_axisbelow(True)


def panel_title(ax: plt.Axes, label: str, title: str, subtitle: str | None = None) -> None:
    """Draw a consistent panel label and title."""

    ax.text(
        0.0,
        1.085,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.6,
        fontweight="bold",
        color=TEXT,
    )
    ax.text(
        0.075,
        1.085,
        title,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.0,
        fontweight="bold",
        color=TEXT,
    )
    if subtitle:
        ax.text(
            0.075,
            1.01,
            subtitle,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=6.7,
            color=MUTED_TEXT,
        )
