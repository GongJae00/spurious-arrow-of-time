"""Regenerate manuscript figures 1 and 7.

Figure 2 is `audit_flow_figure.py`. Figure 3 is `paper_figures.py`.
Figures 4a/4b are `temporal_audit_figure.py`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects as pe
from matplotlib.cm import ScalarMappable
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.visualization.paper_data import MetricStore  # noqa: E402
from src.visualization.paper_style import (  # noqa: E402
    AUDIT_CMAP,
    LIGHT_TEXT,
    MISSING,
    MUTED_TEXT,
    PANEL_BORDER,
    SPINE,
    TEXT,
    apply_style,
    save_figure,
)


def figure_conceptual(out_dir: Path) -> None:
    C_CORE, C_CORE_L = "#2F6F8F", "#E3EEF4"
    C_NUI, C_NUI_L = "#B85C38", "#F8ECE5"
    C_BAD = "#C0392B"
    fig, ax = plt.subplots(figsize=(7.35, 2.7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()

    def flat_node(x, y, w, h, label, edge, face, fs=7.5):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.010,rounding_size=0.028",
            linewidth=1.0, edgecolor=edge, facecolor=face, zorder=3))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=fs, color=TEXT, zorder=4)

    def flat_arrow(p, q, color, lw=1.8, rad=0.0, ls="-"):
        ax.add_patch(FancyArrowPatch(
            p, q, arrowstyle="-|>", mutation_scale=9, linewidth=lw,
            color=color, connectionstyle=f"arc3,rad={rad}", linestyle=ls,
            shrinkA=4, shrinkB=4, zorder=4))

    NW, NH = 0.13, 0.115
    X0, X1, X2 = 0.10, 0.295, 0.49
    PAD = 0.012

    def panel(pb, header, names, nuis, valid):
        ax.plot([0.012, 0.70], [pb + 0.44, pb + 0.44], color=PANEL_BORDER,
                lw=0.8)
        ax.text(0.012, pb + 0.385, header, fontsize=8.8,
                fontweight="bold", color=TEXT)
        yc = pb + 0.185
        for x, name in zip((X0, X1, X2), names):
            flat_node(x, yc, NW, NH, name, C_CORE, C_CORE_L)
        ym = yc + NH / 2
        flat_arrow((X0 + NW + PAD, ym), (X1 - PAD, ym), C_CORE, lw=1.9)
        flat_arrow((X1 + NW + PAD, ym), (X2 - PAD, ym), C_CORE, lw=1.9)
        nx, nw2 = 0.10, 0.265
        ny = pb - 0.005
        ax.add_patch(FancyBboxPatch(
            (nx, ny), nw2, NH,
            boxstyle="round,pad=0.010,rounding_size=0.028",
            linewidth=1.0, edgecolor=C_NUI, facecolor=C_NUI_L, zorder=3))
        ymid = ny + NH / 2
        ax.text(nx + nw2 / 2, ymid + 0.024, nuis, ha="center",
                va="center", fontsize=7.2, color=TEXT, zorder=4)
        order = ("$s_0 \\rightarrow s_1 \\rightarrow \\cdots "
                 "\\rightarrow s_{L-1}$" if valid else
                 "$s_0 \\leftarrow s_1 \\leftarrow \\cdots "
                 "\\leftarrow s_{L-1}$")
        ax.text(nx + nw2 / 2, ymid - 0.028, order, ha="center",
                va="center", fontsize=7.0, color=C_NUI, zorder=4)
        xr = X2 + 0.065
        color = C_NUI if valid else C_BAD
        ax.plot([nx + nw2 + PAD, xr], [ymid, ymid], color=color,
                lw=1.5, ls=(0, (4, 2)), zorder=3)
        flat_arrow((xr, ymid), (xr, yc - PAD - 0.002), color, lw=1.5)
        if valid:
            ax.text((nx + nw2 + xr) / 2 + 0.012, ymid + 0.048,
                    "correlated with the label", fontsize=6.6,
                    color=C_NUI, ha="center")
        else:
            ax.scatter([(nx + nw2 + xr) / 2 - 0.014], [ymid], marker="x",
                       s=110, color=C_BAD, linewidths=2.8, zorder=5)
            ax.text((nx + nw2 + xr) / 2 + 0.005, ymid + 0.048,
                    "relation now invalid", fontsize=6.6, color=C_BAD,
                    ha="center")

    panel(0.545, "Train / IID",
          ("latent\nsource", "diffusive\ncore", "label"),
          "directional nuisance", True)
    panel(0.045, "OOD",
          ("same\nsource", "same\ncore", "same\nlabel"),
          "reversed nuisance", False)

    ax.plot([0.735, 0.735], [0.03, 0.97], color=PANEL_BORDER, lw=0.8)
    ax.text(0.868, 0.925, "model choice", fontsize=8.8, fontweight="bold",
            color=TEXT, ha="center")
    flat_node(0.748, 0.44, 0.105, 0.13, "mixed\nsequence", "#43505E",
              "#EDF0F3")
    flat_node(0.885, 0.645, 0.098, 0.12, "robust\nOOD", C_CORE, C_CORE_L)
    flat_node(0.885, 0.21, 0.098, 0.12, "OOD\ncollapse", C_NUI, C_NUI_L)
    flat_arrow((0.867, 0.55), (0.926, 0.628), C_CORE, lw=1.8, rad=0.22)
    flat_arrow((0.867, 0.46), (0.926, 0.352), C_NUI, lw=1.8, rad=-0.22)
    ax.text(0.815, 0.755, "core\npath", fontsize=6.9, color=C_CORE,
            ha="center", va="center", fontweight="bold")
    ax.text(0.815, 0.195, "shortcut\npath", fontsize=6.9, color=C_NUI,
            ha="center", va="center", fontweight="bold")

    fig.subplots_adjust(left=0.004, right=0.998, top=0.995, bottom=0.005)
    save_figure(fig, out_dir, "fig1_conceptual_problem")


def figure_scenario_audit(out_dir: Path, store: MetricStore) -> None:
    scenarios = [
        "main_spurious_arrow",
        "no_spurious_correlation",
        "residue_visible_control",
        "ood_randomized",
        "ood_partial_shift",
    ]
    columns = ["Main\nreversal", "No\nspurious", "Residue\nvisible", "OOD\nrandom", "Partial\nshift"]
    methods = ["sequence_erm", "final_frame_mlp", "nuisance_only_oracle", "counterfactual_invariance"]
    rows = ["Seq. ERM", "Final frame", "Nuis. reference", "Counterfactual"]
    matrix = np.full((len(methods), len(scenarios)), np.nan)
    for i, method in enumerate(methods):
        for j, scenario in enumerate(scenarios):
            try:
                matrix[i, j] = store.aggregate(method, "ood_test_accuracy", scenario).mean
            except (KeyError, ValueError):
                matrix[i, j] = np.nan

    fig, ax = plt.subplots(figsize=(7.25, 3.12))
    ax.set_xlim(-0.5, len(scenarios) - 0.5)
    ax.set_ylim(len(methods) - 0.5, -1.08)
    norm = plt.Normalize(0.0, 1.0)
    for i in range(len(methods)):
        for j in range(len(scenarios)):
            val = matrix[i, j]
            if np.isnan(val):
                rect = Rectangle(
                    (j - 0.5, i - 0.5),
                    1,
                    1,
                    facecolor=MISSING,
                    edgecolor="#D6DBE1",
                    linewidth=0.8,
                    hatch="////",
                    zorder=1,
                )
                ax.add_patch(rect)
                ax.text(j, i, "N/A", ha="center", va="center", fontsize=7.0, color=LIGHT_TEXT, zorder=3)
                continue
            color = AUDIT_CMAP(norm(val))
            rect = Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=color, edgecolor="white", linewidth=1.0, zorder=1)
            ax.add_patch(rect)
            txt_color = "white" if val >= 0.64 else TEXT
            if val <= 0.20:
                tag = "collapse"
            elif 0.40 <= val <= 0.60:
                tag = "near chance"
            elif val >= 0.80:
                tag = "robust"
            else:
                tag = "partial"
            txt = ax.text(j, i - 0.08, f"{val:.2f}", ha="center", va="center", fontsize=7.5, fontweight="bold", color=txt_color, zorder=3)
            tag_color = "white" if val >= 0.64 else MUTED_TEXT
            tag_txt = ax.text(j, i + 0.18, tag, ha="center", va="center", fontsize=5.55, color=tag_color, zorder=3)
            if txt_color == "white":
                txt.set_path_effects([pe.withStroke(linewidth=1.0, foreground="#173F56", alpha=0.35)])
                tag_txt.set_path_effects([pe.withStroke(linewidth=1.0, foreground="#173F56", alpha=0.3)])
    ax.set_xticks(np.arange(len(scenarios)))
    ax.set_xticklabels(columns)
    ax.set_yticks(np.arange(len(methods)))
    ax.set_yticklabels(rows)
    ax.tick_params(axis="both", length=0, pad=5)
    for spine in ax.spines.values():
        spine.set_visible(False)

    groups = [
        (0, 0, "Main stress test"),
        (1, 2, "Controls"),
        (3, 4, "Shift variants"),
    ]
    for start, end, label in groups:
        y = -0.78
        ax.plot([start - 0.42, end + 0.42], [y, y], color=SPINE, linewidth=0.75, clip_on=False)
        ax.text((start + end) / 2, y - 0.16, label, ha="center", va="top", fontsize=6.2, color=MUTED_TEXT, clip_on=False)

    sm = ScalarMappable(norm=norm, cmap=AUDIT_CMAP)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.032, pad=0.03)
    cbar.set_label("OOD accuracy", labelpad=7)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(length=2.5, width=0.45, color=SPINE, labelcolor=TEXT)

    fig.subplots_adjust(left=0.125, right=0.94, top=0.84, bottom=0.13)
    save_figure(fig, out_dir, "fig7_scenario_audit")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=Path("results/extended/scenario_audit/summary.json"))
    parser.add_argument("--metrics", type=Path, default=Path("results/extended/scenario_audit/metrics.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("."))
    args = parser.parse_args()

    apply_style()
    store = MetricStore(args.summary, args.metrics)
    figure_conceptual(args.out_dir)
    figure_scenario_audit(args.out_dir, store)


if __name__ == "__main__":
    main()
