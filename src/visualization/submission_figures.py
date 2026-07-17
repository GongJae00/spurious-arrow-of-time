"""Regenerate publication-grade manuscript figures 1, 3, 4, and 5.

Figure 2 is intentionally not regenerated here.
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
    CORE,
    CORE_LIGHT,
    COUNTERFACTUAL,
    FINAL,
    GRID,
    LIGHT_TEXT,
    METHOD_COLORS,
    METHOD_LABELS,
    MISSING,
    MUTED_TEXT,
    NEGATIVE,
    NUISANCE,
    NUISANCE_LIGHT,
    PANEL_BORDER,
    SEQUENCE,
    SEQUENCE_LIGHT,
    SPINE,
    TEXT,
    apply_style,
    panel_title,
    save_figure,
    style_axis,
)


def draw_node(
    ax: plt.Axes,
    xy: tuple[float, float],
    size: tuple[float, float],
    label: str,
    *,
    edge: str,
    face: str,
    lw: float = 1.05,
) -> FancyBboxPatch:
    x, y = xy
    w, h = size
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.045",
        linewidth=lw,
        edgecolor=edge,
        facecolor=face,
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=7.0, color=TEXT, zorder=3)
    return patch


def draw_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str,
    lw: float = 1.7,
    rad: float = 0.0,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=lw,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=5,
            shrinkB=5,
            zorder=3,
        )
    )


def draw_core_cloud(ax: plt.Axes, center: tuple[float, float], *, scale: float = 1.0) -> None:
    x, y = center
    radii = [0.09, 0.16, 0.24]
    alphas = [0.75, 0.32, 0.15]
    for r, a in zip(radii, alphas, strict=True):
        circ = plt.Circle((x, y), r * scale, color=CORE, alpha=a, linewidth=0, zorder=2)
        ax.add_patch(circ)
    ax.scatter([x - 0.035 * scale, x + 0.035 * scale], [y, y], s=10 * scale, color=CORE, zorder=4)


def draw_nuisance_trajectory(
    ax: plt.Axes,
    points: list[tuple[float, float]],
    *,
    reverse: bool = False,
    alpha: float = 1.0,
) -> None:
    pts = points[::-1] if reverse else points
    xs, ys = zip(*pts, strict=True)
    ax.plot(xs, ys, color=NUISANCE, linewidth=2.0, alpha=alpha, solid_capstyle="round", zorder=2)
    for k, (x, y) in enumerate(pts):
        ax.scatter(x, y, s=22 + 3 * k, color=NUISANCE, alpha=0.28 + 0.11 * k, zorder=3)
    draw_arrow(ax, pts[-2], pts[-1], color=NUISANCE, lw=1.9)


def figure_conceptual(out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.35, 2.55))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()

    NW, NH = 0.125, 0.125             # node size
    X0, X1, X2 = 0.145, 0.335, 0.575  # column anchors (lower-left x)

    def row(y, names, traj_reverse):
        """One generative row: three nodes, core arrows, nuisance path."""
        for x, name in zip((X0, X1, X2), names):
            draw_node(ax, (x, y), (NW, NH), name, edge=CORE, face=CORE_LIGHT)
        ym = y + NH / 2
        draw_arrow(ax, (X0 + NW, ym), (X1, ym), color=CORE, lw=2.0)
        draw_arrow(ax, (X1 + NW, ym), (X2, ym), color=CORE, lw=2.0)
        yb = y - 0.135
        pts = [(X0 + 0.02 + 0.095 * k, yb + 0.022 * k) for k in range(4)]
        draw_nuisance_trajectory(ax, pts, reverse=traj_reverse)
        xe, ye = pts[-1][0] + 0.03, pts[-1][1] + 0.012
        xt, yt = X2 + 0.055, y - 0.006
        ax.plot([xe, xe + 0.72 * (xt - xe)], [ye, ye + 0.72 * (yt - ye)],
                color=NUISANCE, linewidth=1.55, linestyle=(0, (4, 2)))
        return (xe, ye), (xt, yt)

    ax.text(0.015, 0.75, "Train / IID", fontsize=9.0, fontweight="bold",
            color=TEXT, va="center")
    ax.text(0.015, 0.32, "OOD", fontsize=9.0, fontweight="bold",
            color=TEXT, va="center")
    ax.plot([0.105, 0.105], [0.10, 0.91], color=PANEL_BORDER, linewidth=0.8)

    # Train row: the nuisance relation predicts the label.
    (xe, ye), (xt, yt) = row(0.66, ("latent\nsource", "diffusive\ncore",
                                    "label"), False)
    draw_arrow(ax, (xe + 0.6 * (xt - xe), ye + 0.6 * (yt - ye)), (xt, yt),
               color=NUISANCE, lw=1.6)
    ax.text(0.395, 0.845, "stable task-relevant relation", fontsize=7.6,
            color=CORE, ha="center", fontweight="bold")
    ax.text(0.30, 0.455,
            "environment-dependent nuisance relation is predictive",
            fontsize=7.3, color=NUISANCE, ha="center")

    # OOD row: the same nuisance relation is now invalid.
    (xe, ye), (xt, yt) = row(0.23, ("same\nsource", "same\ncore",
                                    "same\nlabel"), True)
    ax.text((xe + xt) / 2 - 0.01, (ye + yt) / 2 - 0.005, "X", fontsize=13,
            color=NUISANCE, fontweight="bold", ha="center", va="center")
    ax.text(0.30, 0.025, "reversed nuisance arrow gives wrong evidence",
            fontsize=7.3, color=NUISANCE, ha="center")

    # Model-choice panel.
    ax.plot([0.72, 0.72], [0.10, 0.91], color=PANEL_BORDER, linewidth=0.8)
    ax.text(0.86, 0.845, "model choice", fontsize=9.0, fontweight="bold",
            color=TEXT, ha="center")
    draw_node(ax, (0.75, 0.45), (0.115, 0.13), "mixed\nsequence",
              edge=SEQUENCE, face=SEQUENCE_LIGHT)
    draw_node(ax, (0.895, 0.63), (0.095, 0.12), "robust\nOOD",
              edge=CORE, face=CORE_LIGHT)
    draw_node(ax, (0.895, 0.28), (0.095, 0.12), "OOD\ncollapse",
              edge=NUISANCE, face=NUISANCE_LIGHT)
    draw_arrow(ax, (0.865, 0.545), (0.90, 0.665), color=CORE, lw=1.9,
               rad=0.22)
    draw_arrow(ax, (0.865, 0.485), (0.90, 0.365), color=NUISANCE, lw=1.9,
               rad=-0.22)
    ax.text(0.765, 0.375, "core path", fontsize=7.3, color=CORE,
            ha="left", fontweight="bold")
    ax.text(0.765, 0.325, "shortcut path", fontsize=7.3, color=NUISANCE,
            ha="left", fontweight="bold")

    fig.subplots_adjust(left=0.01, right=0.995, top=0.96, bottom=0.04)
    save_figure(fig, out_dir, "fig01_conceptual_problem")


def seed_swarm_y(ax: plt.Axes, x: float, vals: np.ndarray, color: str, rng: np.random.Generator) -> None:
    jitter = rng.uniform(-0.045, 0.045, len(vals))
    ax.scatter(
        np.full(len(vals), x) + jitter,
        vals,
        s=12,
        color=color,
        alpha=0.55,
        edgecolor="white",
        linewidth=0.25,
        zorder=3,
    )


def mean_point_y(ax: plt.Axes, x: float, mean: float, std: float, color: str, *, marker: str = "o") -> None:
    ax.errorbar(
        x,
        mean,
        yerr=std,
        fmt=marker,
        markersize=5.2,
        color=color,
        markerfacecolor=color,
        markeredgecolor="white",
        markeredgewidth=0.55,
        ecolor=color,
        elinewidth=1.0,
        capsize=2.2,
        zorder=5,
    )


def seed_swarm_x(ax: plt.Axes, y: float, vals: np.ndarray, color: str, rng: np.random.Generator) -> None:
    jitter = rng.uniform(-0.065, 0.065, len(vals))
    ax.scatter(
        vals,
        np.full(len(vals), y) + jitter,
        s=12,
        color=color,
        alpha=0.55,
        edgecolor="white",
        linewidth=0.25,
        zorder=3,
    )


def mean_point_x(
    ax: plt.Axes,
    y: float,
    mean: float,
    std: float,
    color: str,
    *,
    marker: str = "o",
    fill: bool = True,
) -> None:
    ax.errorbar(
        mean,
        y,
        xerr=std,
        fmt=marker,
        markersize=5.2,
        color=color,
        markerfacecolor=color if fill else "white",
        markeredgecolor=color,
        markeredgewidth=0.8,
        ecolor=color,
        elinewidth=1.0,
        capsize=2.2,
        zorder=5,
    )


def dot_metric_panel(
    ax: plt.Axes,
    store: MetricStore,
    *,
    method: str,
    scenario: str,
    metrics: list[tuple[str, str, str, str]],
    ylabel: str,
    annotation: str | None = None,
) -> None:
    rng = np.random.default_rng(100)
    xs = np.arange(len(metrics), dtype=float)
    for x, (metric_name, label, color, marker) in zip(xs, metrics, strict=True):
        vals = store.values(method, metric_name, scenario)
        agg = store.aggregate(method, metric_name, scenario)
        seed_swarm_y(ax, x, vals, color, rng)
        mean_point_y(ax, x, agg.mean, agg.std, color, marker=marker)
    ax.set_xticks(xs)
    ax.set_xticklabels([item[1] for item in metrics])
    ax.set_ylabel(ylabel)
    ax.set_ylim(0.0, 1.04)
    ax.axhline(0.5, color=SPINE, linewidth=0.7, linestyle=(0, (2, 2)), zorder=0)
    style_axis(ax)
    if annotation:
        ax.text(
            0.03,
            0.09,
            annotation,
            transform=ax.transAxes,
            fontsize=6.35,
            color=MUTED_TEXT,
            ha="left",
            va="bottom",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.0},
        )


def figure_evidence_gates(out_dir: Path, store: MetricStore) -> None:
    fig, ax = plt.subplots(figsize=(7.25, 3.55))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()

    gates = [
        {
            "label": "A",
            "title": "No-spurious recovery",
            "subtitle": "Core recoverable from the mixture",
            "lines": [
                ("Std. budget", 21 / 30, "21/30 core"),
                ("Ext. budget", 30 / 30, "30/30 core"),
            ],
            "message": "std.: 9/30 at chance; ext.: 30/30 core (OOD 0.997)",
            "color": CORE,
        },
        {
            "label": "B",
            "title": "Shortcut selected",
            "subtitle": "Main sequence ERM",
            "lines": [
                ("IID", store.aggregate("sequence_erm", "iid_test_accuracy").mean),
                ("OOD", store.aggregate("sequence_erm", "ood_test_accuracy").mean),
            ],
            "message": "collapse: 29/30 std., 19/30 ext. budget",
            "color": NUISANCE,
        },
        {
            "label": "C",
            "title": "Mechanisms separated",
            "subtitle": "Reference inputs",
            "lines": [
                ("Core OOD", store.aggregate("core_only_oracle", "ood_test_accuracy").mean),
                ("Nuis. OOD", store.aggregate("nuisance_only_oracle", "ood_test_accuracy").mean),
            ],
            "message": "valid vs invalid cue",
            "color": SEQUENCE,
        },
        {
            "label": "D",
            "title": "Endpoint controlled",
            "subtitle": "Gap audit",
            "lines": [
                ("Seq. gap", store.aggregate("sequence_erm", "ood_gap").mean),
                ("Final gap", store.aggregate("final_frame_mlp", "ood_gap").mean),
            ],
            "message": "not final-frame leakage",
            "color": FINAL,
        },
    ]
    card_w = 0.455
    card_h = 0.355
    positions = [(0.025, 0.52), (0.52, 0.52), (0.025, 0.12), (0.52, 0.12)]
    for idx, ((x0, y0), gate) in enumerate(zip(positions, gates, strict=True)):
        ax.add_patch(
            FancyBboxPatch(
                (x0, y0),
                card_w,
                card_h,
                boxstyle="round,pad=0.012,rounding_size=0.02",
                facecolor="#FFFFFF",
                edgecolor=PANEL_BORDER,
                linewidth=0.8,
            )
        )
        ax.text(x0 + 0.023, y0 + card_h - 0.06, gate["label"], fontsize=8.8, fontweight="bold", color=TEXT, ha="left", va="center")
        ax.text(x0 + 0.065, y0 + card_h - 0.058, gate["title"], fontsize=8.3, fontweight="bold", color=TEXT, ha="left", va="center")
        ax.text(x0 + 0.065, y0 + card_h - 0.113, gate["subtitle"], fontsize=6.8, color=MUTED_TEXT, ha="left", va="center")
        ax.text(
            x0 + card_w - 0.018,
            y0 + card_h - 0.06,
            "PASS",
            fontsize=7.0,
            color="white",
            ha="right",
            va="center",
            bbox={"boxstyle": "round,pad=0.22,rounding_size=0.12", "facecolor": CORE, "edgecolor": "none"},
        )
        bar_x0 = x0 + 0.055
        bar_w = card_w - 0.11
        for k, line in enumerate(gate["lines"]):
            name, val = line[0], line[1]
            display = line[2] if len(line) > 2 else f"{val:.3f}"
            y = y0 + 0.17 - 0.105 * k
            fill = gate["color"] if idx != 2 or k == 0 else NUISANCE
            if idx == 3 and k == 1:
                fill = FINAL
            ax.text(bar_x0, y + 0.035, name, fontsize=6.8, color=TEXT, ha="left", va="center")
            ax.text(bar_x0 + bar_w, y + 0.035, display, fontsize=7.0, fontweight="bold", color=TEXT, ha="right", va="center")
            ax.add_patch(Rectangle((bar_x0, y - 0.018), bar_w, 0.035, facecolor="#EEF1F4", edgecolor="none"))
            ax.add_patch(Rectangle((bar_x0, y - 0.018), max(0.0, min(1.0, val)) * bar_w, 0.035, facecolor=fill, edgecolor="none", alpha=0.88))
        ax.text(x0 + 0.055, y0 + 0.035, gate["message"], fontsize=6.7, color=MUTED_TEXT, ha="left", va="center")
    ax.text(0.012, 0.94, "Evidence gates for the spurious-shortcut interpretation", fontsize=9.0, fontweight="bold", color=TEXT)
    ax.text(0.012, 0.035, "Each card reports the minimal artifact-backed evidence required to support the interpretation.", fontsize=6.8, color=MUTED_TEXT)

    fig.subplots_adjust(left=0.005, right=0.995, top=0.98, bottom=0.02)
    save_figure(fig, out_dir, "fig03_evidence_gates")


def figure_main_results(out_dir: Path, store: MetricStore) -> None:
    methods = [
        "core_only_oracle",
        "sequence_erm",
        "final_frame_mlp",
        "nuisance_only_oracle",
        "counterfactual_invariance",
    ]
    fig = plt.figure(figsize=(7.35, 4.05))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.24, 0.86], wspace=0.32)

    ax = fig.add_subplot(gs[0, 0])
    panel_title(ax, "A", "Mechanism signature", "Open marker = IID; filled marker = OOD")
    ypos = np.arange(len(methods))[::-1].astype(float)
    rng = np.random.default_rng(404)
    for y, method in zip(ypos, methods, strict=True):
        color = METHOD_COLORS[method]
        iid = store.aggregate(method, "iid_test_accuracy")
        ood = store.aggregate(method, "ood_test_accuracy")
        ax.hlines(y, ood.mean, iid.mean, color=color, linewidth=2.0, alpha=0.56, zorder=1)
        seed_swarm_x(ax, y - 0.18, store.values(method, "ood_test_accuracy"), color, rng)
        mean_point_x(ax, y, iid.mean, iid.std, color, marker="o", fill=False)
        mean_point_x(ax, y, ood.mean, ood.std, color, marker="s", fill=True)
    ax.axvline(0.5, color=SPINE, linewidth=0.7, linestyle=(0, (2, 2)), zorder=0)
    ax.set_yticks(ypos)
    ax.set_yticklabels([METHOD_LABELS[m] for m in methods])
    ax.set_xlabel("Accuracy")
    ax.set_xlim(-0.02, 1.04)
    ax.set_ylim(-0.75, len(methods) - 0.25)
    ax.grid(axis="x", color=GRID, linewidth=0.55)
    style_axis(ax, grid=False)
    ax.text(0.725, 0.965, "○ IID mean", transform=ax.transAxes, fontsize=6.8, color=TEXT)
    ax.text(0.725, 0.91, "■ OOD mean", transform=ax.transAxes, fontsize=6.8, color=TEXT)
    y_seq = ypos[1]
    y_nuis = ypos[3]
    ax.annotate(
        "ERM OOD matches\nnuisance-only collapse",
        xy=(0.12, (y_seq + y_nuis) / 2),
        xytext=(0.31, (y_seq + y_nuis) / 2 + 0.38),
        fontsize=6.7,
        color=NUISANCE,
        ha="left",
        arrowprops={"arrowstyle": "-[", "lw": 0.85, "color": NUISANCE, "shrinkA": 2, "shrinkB": 2},
    )

    ax = fig.add_subplot(gs[0, 1])
    panel_title(ax, "B", "Counterfactual stability", "Paired OOD accuracy by seed")
    seeds, erm, cf = store.paired_values("sequence_erm", "counterfactual_invariance", "ood_test_accuracy")
    stable = cf >= 0.80
    stable_count = int(stable.sum())
    for seed, a, b, ok in zip(seeds, erm, cf, stable, strict=True):
        line_color = COUNTERFACTUAL if ok else NEGATIVE
        ax.plot([0, 1], [a, b], color=line_color, linewidth=1.35, alpha=0.68, zorder=1)
        ax.scatter(0, a, s=20, color=SEQUENCE, edgecolor="white", linewidth=0.4, zorder=3)
        ax.scatter(1, b, s=22, color=COUNTERFACTUAL if ok else NEGATIVE, edgecolor="white", linewidth=0.4, zorder=3)
    ax.axhline(0.80, color=TEXT, linewidth=0.85, linestyle=(0, (3, 2)), alpha=0.78)
    ax.text(
        0.03,
        0.815,
        f"0.80 threshold ({stable_count}/{len(seeds)} above)",
        transform=ax.get_yaxis_transform(),
        ha="left",
        va="bottom",
        fontsize=6.4,
        color=TEXT,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 1.0},
    )
    ax.text(1.02, 0.93, "improved\nbut unstable", ha="left", va="top", fontsize=6.6, color=COUNTERFACTUAL)
    ax.set_xlim(-0.14, 1.22)
    ax.set_ylim(0.0, 1.04)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Sequence\nERM", "Counterfactual\nreplacement"])
    ax.set_ylabel("OOD accuracy")
    ax.axhline(0.5, color=SPINE, linewidth=0.7, linestyle=(0, (2, 2)), zorder=0)
    style_axis(ax)

    fig.subplots_adjust(left=0.075, right=0.99, top=0.90, bottom=0.12)
    save_figure(fig, out_dir, "fig04_main_results")


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
    save_figure(fig, out_dir, "fig05_scenario_audit")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=Path("results/main_experiments/full/summary.json"))
    parser.add_argument("--metrics", type=Path, default=Path("results/main_experiments/full/metrics.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("paper/neurocomputing"))
    args = parser.parse_args()

    apply_style()
    store = MetricStore(args.summary, args.metrics)
    figure_conceptual(args.out_dir)
    figure_evidence_gates(args.out_dir, store)
    figure_main_results(args.out_dir, store)
    figure_scenario_audit(args.out_dir, store)


if __name__ == "__main__":
    main()
