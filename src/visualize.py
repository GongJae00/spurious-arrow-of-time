import argparse
import json
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib import gridspec
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from src.data import GeneratorConfig, generate_splits
from src.evaluate import MetricStore

# Figure 1 concept, Figure 2 Algorithm 1, Figure 3 construction, Figure 4 locality, Figure A3.


TEXT = "#1F2933"
MUTED_TEXT = "#5B6470"
GRID = "#E7EBEF"
SPINE = "#B9C1CA"
PANEL_BORDER = "#D9DEE5"
NUISANCE = "#D06B45"
SEQUENCE = "#273F4D"
CORE_COLOR = np.asarray([0.05, 0.55, 0.50])
NUISANCE_COLOR = np.asarray([0.81, 0.38, 0.28])
AUDIT_CMAP = LinearSegmentedColormap.from_list("audit_accuracy", ["#F8FAFC", "#E8EEF3", "#CADAE4", "#8DB3C4", "#4F849B", "#1F526C"])


def apply_style():
    plt.rcParams.update({
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
    })


def save_figure(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.025)
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", pad_inches=0.025, dpi=400)
    plt.close(fig)


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(SPINE)
    ax.spines["left"].set_linewidth(0.65)
    ax.spines["bottom"].set_color(SPINE)
    ax.spines["bottom"].set_linewidth(0.65)
    ax.tick_params(length=2.8, width=0.6, colors=TEXT)
    ax.grid(axis="y", color=GRID, linewidth=0.55)
    ax.set_axisbelow(True)


def fig1(out: Path):
    # Figure 1. Core path vs nuisance shortcut under OOD reversal.
    C_CORE, C_CORE_L = "#2F6F8F", "#E3EEF4"
    C_NUI, C_NUI_L = "#B85C38", "#F8ECE5"
    C_BAD = "#C0392B"
    fig, ax = plt.subplots(figsize=(7.35, 2.7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()

    def flat_node(x, y, w, h, label, edge, face, fs=7.5):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.010,rounding_size=0.028", linewidth=1.0, edgecolor=edge, facecolor=face, zorder=3))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fs, color=TEXT, zorder=4)

    def flat_arrow(p, q, color, lw=1.8, rad=0.0, ls="-"):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=9, linewidth=lw, color=color, connectionstyle=f"arc3,rad={rad}", linestyle=ls, shrinkA=4, shrinkB=4, zorder=4))

    NW, NH = 0.13, 0.115
    X0, X1, X2 = 0.10, 0.295, 0.49
    PAD = 0.012

    def panel(pb, header, names, nuis, valid):
        ax.plot([0.012, 0.70], [pb + 0.44, pb + 0.44], color=PANEL_BORDER, lw=0.8)
        ax.text(0.012, pb + 0.385, header, fontsize=8.8, fontweight="bold", color=TEXT)
        yc = pb + 0.185
        for x, name in zip((X0, X1, X2), names):
            flat_node(x, yc, NW, NH, name, C_CORE, C_CORE_L)
        ym = yc + NH / 2
        flat_arrow((X0 + NW + PAD, ym), (X1 - PAD, ym), C_CORE, lw=1.9)
        flat_arrow((X1 + NW + PAD, ym), (X2 - PAD, ym), C_CORE, lw=1.9)
        nx, nw2 = 0.10, 0.265
        ny = pb - 0.005
        ax.add_patch(FancyBboxPatch((nx, ny), nw2, NH, boxstyle="round,pad=0.010,rounding_size=0.028", linewidth=1.0, edgecolor=C_NUI, facecolor=C_NUI_L, zorder=3))
        ymid = ny + NH / 2
        ax.text(nx + nw2 / 2, ymid + 0.024, nuis, ha="center", va="center", fontsize=7.2, color=TEXT, zorder=4)
        order = ("$s_0 \\rightarrow s_1 \\rightarrow \\cdots \\rightarrow s_{L-1}$" if valid else "$s_0 \\leftarrow s_1 \\leftarrow \\cdots \\leftarrow s_{L-1}$")
        ax.text(nx + nw2 / 2, ymid - 0.028, order, ha="center", va="center", fontsize=7.0, color=C_NUI, zorder=4)
        xr = X2 + 0.065
        color = C_NUI if valid else C_BAD
        ax.plot([nx + nw2 + PAD, xr], [ymid, ymid], color=color, lw=1.5, ls=(0, (4, 2)), zorder=3)
        flat_arrow((xr, ymid), (xr, yc - PAD - 0.002), color, lw=1.5)
        if valid:
            ax.text((nx + nw2 + xr) / 2 + 0.012, ymid + 0.048, "correlated with the label", fontsize=6.6, color=C_NUI, ha="center")
        else:
            ax.scatter([(nx + nw2 + xr) / 2 - 0.014], [ymid], marker="x", s=110, color=C_BAD, linewidths=2.8, zorder=5)
            ax.text((nx + nw2 + xr) / 2 + 0.005, ymid + 0.048, "relation now invalid", fontsize=6.6, color=C_BAD, ha="center")

    panel(0.545, "Train / IID", ("latent\nsource", "diffusive\ncore", "label"), "directional nuisance", True)
    panel(0.045, "OOD", ("same\nsource", "same\ncore", "same\nlabel"), "reversed nuisance", False)
    ax.plot([0.735, 0.735], [0.03, 0.97], color=PANEL_BORDER, lw=0.8)
    ax.text(0.868, 0.925, "model choice", fontsize=8.8, fontweight="bold", color=TEXT, ha="center")
    flat_node(0.748, 0.44, 0.105, 0.13, "mixed\nsequence", "#43505E", "#EDF0F3")
    flat_node(0.885, 0.645, 0.098, 0.12, "core\nOOD", C_CORE, C_CORE_L)
    flat_node(0.885, 0.21, 0.098, 0.12, "OOD\ncollapse", C_NUI, C_NUI_L)
    flat_arrow((0.867, 0.55), (0.926, 0.628), C_CORE, lw=1.8, rad=0.22)
    flat_arrow((0.867, 0.46), (0.926, 0.352), C_NUI, lw=1.8, rad=-0.22)
    ax.text(0.815, 0.755, "core\npath", fontsize=6.9, color=C_CORE, ha="center", va="center", fontweight="bold")
    ax.text(0.815, 0.195, "shortcut\npath", fontsize=6.9, color=C_NUI, ha="center", va="center", fontweight="bold")
    fig.subplots_adjust(left=0.004, right=0.998, top=0.995, bottom=0.005)
    save_figure(fig, out)


def fig2(out: Path):
    # Figure 2. Algorithm 1 gate diagram.
    fig, ax = plt.subplots(figsize=(4.2, 6.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(1.5, 21)
    ax.axis("off")
    BOX = dict(boxstyle="round,pad=0.32", fc="#eef3fb", ec="#33518a", lw=1.1)
    FAIL = dict(boxstyle="round,pad=0.28", fc="#fbeeee", ec="#8a3333", lw=1.0)
    OUT = dict(boxstyle="round,pad=0.32", fc="#eefbef", ec="#2f7a3a", lw=1.1)
    gates = [
        (19.6, "G1: Core learnable\nalone?", "reject /\ntask too hard"),
        (16.9, "G2: Nuisance\npredictive alone?", "reject /\nweak shortcut cue"),
        (14.2, "G3: Endpoint leakage\ncontrolled?", "reject /\nendpoint leakage"),
        (11.5, "G4: Reference-learner\nrecovery (no-spurious)?", "reject /\ncore not recovered\nunder ref. budget"),
        (8.8, "G5: Reversal collapse\n(signature match)?", "inconclusive /\nsignature mismatch"),
    ]
    cx = 3.9
    for y, q, fail in gates:
        ax.text(cx, y, q, ha="center", va="center", fontsize=8.3, bbox=BOX)
        ax.text(7.0, y, fail, ha="left", va="center", fontsize=7.5, bbox=FAIL)
        ax.add_patch(FancyArrowPatch((6.1, y), (6.82, y), arrowstyle="-|>", mutation_scale=9, color="#8a3333", lw=0.9))
        ax.text(6.45, y + 0.32, "no", fontsize=7.5, color="#8a3333", ha="center")
    for (y1, _, _), (y2, _, _) in zip(gates, gates[1:]):
        ax.add_patch(FancyArrowPatch((cx, y1 - 0.62), (cx, y2 + 0.62), arrowstyle="-|>", mutation_scale=10, color="#33518a", lw=1.1))
        ax.text(cx + 0.28, (y1 + y2) / 2, "yes", fontsize=7.5, color="#33518a")
    y6 = 6.0
    ax.text(cx, y6, "G6: Cue-locality audit\n(probes + order interventions;\nRoute A construction-certified /\nRoute B intervention-certified)", ha="center", va="center", fontsize=8.0, bbox=BOX)
    ax.add_patch(FancyArrowPatch((cx, gates[-1][0] - 0.85), (cx, y6 + 0.95), arrowstyle="-|>", mutation_scale=10, color="#33518a", lw=1.1))
    ax.text(cx + 0.28, (gates[-1][0] + y6) / 2 - 0.1, "yes", fontsize=7.5, color="#33518a")
    outs = [(1.15, 3.45, "frame-local\n(single-frame\nprobe $\\geq$ 0.8)"), (4.6, 3.7, "order-invariant\nmulti-frame\n(set probe\n$\\geq$ 0.8)"), (8.3, 3.45, "order-encoded\n(only ordered\nreadout $\\geq$ 0.8)")]
    for x, ye, t in outs:
        ax.text(x, 2.6, t, ha="center", va="center", fontsize=7.5, bbox=OUT)
        ax.add_patch(FancyArrowPatch((cx, y6 - 1.05), (x, ye), arrowstyle="-|>", mutation_scale=9, color="#2f7a3a", lw=1.0))
    save_figure(fig, out)


def normalize_image(img, vmax=None):
    img = np.clip(np.asarray(img, dtype=float), 0.0, None)
    if vmax is None:
        vmax = float(np.quantile(img, 0.995))
    return np.clip(img / max(vmax, 1e-8), 0.0, 1.0)


def smooth_image(img, passes=1):
    out = np.asarray(img, dtype=float)
    kernel = np.asarray([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]]) / 16.0
    for _ in range(passes):
        padded = np.pad(out, 1, mode="edge")
        out = (
            kernel[0, 0] * padded[:-2, :-2] + kernel[0, 1] * padded[:-2, 1:-1] + kernel[0, 2] * padded[:-2, 2:]
            + kernel[1, 0] * padded[1:-1, :-2] + kernel[1, 1] * padded[1:-1, 1:-1] + kernel[1, 2] * padded[1:-1, 2:]
            + kernel[2, 0] * padded[2:, :-2] + kernel[2, 1] * padded[2:, 1:-1] + kernel[2, 2] * padded[2:, 2:]
        )
    return out


def colorize(img, color, vmax, gamma=0.62):
    norm = np.power(normalize_image(smooth_image(img, passes=1), vmax=vmax), gamma)
    return norm[..., None] * color[None, None, :]


def channel_frame(arr, t, channel):
    frame = arr[t]
    if frame.ndim == 3:
        return np.clip(frame[channel], 0.0, None)
    return np.clip(frame, 0.0, None)


def composite_frame(arr, t, core_vmax, nuisance_vmax):
    frame = arr[t]
    if frame.ndim == 3:
        return np.clip(colorize(frame[0], CORE_COLOR, core_vmax) + colorize(frame[1], NUISANCE_COLOR, nuisance_vmax), 0.0, 1.0)
    return np.clip(colorize(frame, CORE_COLOR, core_vmax), 0.0, 1.0)


def fig3(out: Path, config_path: Path):
    # Figure 3. Core, nuisance, mixed, counterfactual, OOD, and γ=0 nuisance.
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data = raw["data"]
    allowed = set(GeneratorConfig.__dataclass_fields__)
    config = GeneratorConfig(**{k: v for k, v in data.items() if k in allowed})
    config = replace(config, n_train=64, n_val_iid=16, n_iid_test=16, n_ood_test=64, seed=19)
    splits = generate_splits(config)
    train = splits["train"]
    ood = splits["ood_test"]
    train_oe = generate_splits(replace(config, nuisance_trail_decay=0.0))["train"]
    idx = int(np.where(train.y == 1)[0][0])
    idx_ood = int(np.where(ood.y == 1)[0][0])
    times = list(range(config.length))
    fig = plt.figure(figsize=(7.55, 4.25))
    grid = gridspec.GridSpec(6, 1 + len(times), figure=fig, width_ratios=[1.25] + [1] * len(times), hspace=0.075, wspace=0.045, left=0.015, right=0.998, top=0.945, bottom=0.02)
    rows = [
        ("A", "Core source", "task-relevant trace", train.core_only[idx], "core"),
        ("B", "Nuisance arrow", "shortcut", train.nuisance_only[idx], "nuisance"),
        ("C", "Mixed input", "train/IID", train.mixed[idx], "mixed"),
        ("D", "Counterfactual", "core fixed", train.counterfactual[idx], "mixed"),
        ("E", "OOD input", "arrow shifted", ood.mixed[idx_ood], "mixed"),
        ("F", "Order-encoded", "nuisance, no residue", train_oe.nuisance_only[idx], "nuisance"),
    ]
    core_stack = [train.core_only[idx], train.mixed[idx][:, 0], train.counterfactual[idx][:, 0], ood.mixed[idx_ood][:, 0]]
    nuisance_stack = [train.nuisance_only[idx], train.mixed[idx][:, 1], train.counterfactual[idx][:, 1], ood.mixed[idx_ood][:, 1]]
    core_vmax = max(float(np.quantile(arr, 0.995)) for arr in core_stack)
    nuisance_vmax = max(float(np.quantile(arr, 0.995)) for arr in nuisance_stack)
    for r, (label, row_name, sublabel, seq, mode) in enumerate(rows):
        ax_label = fig.add_subplot(grid[r, 0])
        ax_label.set_axis_off()
        ax_label.text(0.02, 0.64, label, fontsize=8.2, fontweight="bold", color=TEXT)
        ax_label.text(0.18, 0.64, row_name, fontsize=6.8, fontweight="bold", ha="left", color=TEXT)
        ax_label.text(0.18, 0.30, sublabel, fontsize=5.8, color=MUTED_TEXT, ha="left")
        for c, t in enumerate(times):
            ax = fig.add_subplot(grid[r, c + 1])
            if mode == "core":
                img = colorize(channel_frame(seq, t, 0), CORE_COLOR, core_vmax)
            elif mode == "nuisance":
                img = colorize(channel_frame(seq, t, 0), NUISANCE_COLOR, nuisance_vmax)
            else:
                img = composite_frame(seq, t, core_vmax, nuisance_vmax)
            ax.imshow(np.clip(img, 0.0, 1.0), interpolation="bicubic")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.35)
                spine.set_color("#D6DADE")
            if r == 0:
                ax.set_title(f"t={t}", pad=2.5, fontsize=5.9, color=MUTED_TEXT)
    save_figure(fig, out)


def fig4(trail_path: Path, oe_path: Path, probe_path: Path, out_a: Path, out_b: Path):
    # Figure 4. Per-frame direction probes and nuisance-only order interventions.
    trail = json.loads(trail_path.read_text(encoding="utf-8"))
    oe = json.loads(oe_path.read_text(encoding="utf-8"))
    probe = json.loads(probe_path.read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(3.6, 2.55))
    L = len(trail["per_frame"]["dir_iid"])
    ts = np.arange(L)
    for data, color, label in [(trail, NUISANCE, "trail variant"), (oe, SEQUENCE, "order-encoded variant")]:
        means = np.array([d["mean"] for d in data["per_frame"]["dir_iid"]])
        stds = np.array([d["std"] for d in data["per_frame"]["dir_iid"]])
        ax.errorbar(ts, means, yerr=stds, color=color, linewidth=1.7, marker="o", markersize=4.2, capsize=2.0, label=label, zorder=3)
    ax.axhline(0.5, color=MUTED_TEXT, linewidth=0.9, linestyle=(0, (4, 2)))
    ax.text(3.5, 0.458, "chance", fontsize=7.6, color=MUTED_TEXT, ha="center", va="top")
    style_axis(ax)
    ax.set_xlabel("frame index $t$", fontsize=9.0)
    ax.set_ylabel("single-frame direction acc.", fontsize=9.0)
    ax.tick_params(labelsize=8.2)
    ax.set_ylim(0.38, 1.05)
    ax.set_xticks(ts)
    ax.legend(fontsize=7.8, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, columnspacing=1.2, handletextpad=0.5)
    fig.subplots_adjust(left=0.16, right=0.985, top=0.90, bottom=0.19)
    save_figure(fig, out_a)
    fig, ax = plt.subplots(figsize=(3.6, 2.55))
    conditions = ["ood", "ood_shuffled", "ood_reversed_order"]
    cond_labels = ["ordered", "frame-\nshuffled", "order-\nreversed"]
    width = 0.36
    xs = np.arange(len(conditions))
    for k, (variant, color, label) in enumerate([("trail", NUISANCE, "trail variant"), ("simple_oe", SEQUENCE, "order-encoded variant")]):
        key = next(n for n in {"trail": ["trail", "trail_fl"], "simple_oe": ["simple_oe", "order_encoded"]}[variant] if n in probe)
        means = [probe[key][c]["mean"] for c in conditions]
        stds = [probe[key][c]["std"] for c in conditions]
        ax.bar(xs + (k - 0.5) * width, means, width, yerr=stds, capsize=2.0, color=color, edgecolor="white", linewidth=0.6, label=label, zorder=3)
        for xpos, m in zip(xs + (k - 0.5) * width, means, strict=True):
            ax.text(xpos, m + 0.05, f"{m:.2f}", fontsize=7.4, ha="center", color=TEXT)
    ax.axhline(0.5, color=MUTED_TEXT, linewidth=0.9, linestyle=(0, (4, 2)))
    style_axis(ax)
    ax.set_xticks(xs)
    ax.set_xticklabels(cond_labels, fontsize=8.6)
    ax.set_ylabel("OOD accuracy (nuisance-only)", fontsize=9.0)
    ax.tick_params(axis="y", labelsize=8.2)
    ax.set_ylim(0.0, 1.16)
    ax.legend(fontsize=7.8, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, columnspacing=1.2, handletextpad=0.5)
    fig.subplots_adjust(left=0.16, right=0.985, top=0.90, bottom=0.19)
    save_figure(fig, out_b)


def fig_a3(out: Path, store: MetricStore):
    # Figure A3. OOD accuracy by scenario and method.
    scenarios = ["main_spurious_arrow", "no_spurious_correlation", "residue_visible_control", "ood_randomized", "ood_partial_shift"]
    columns = ["Main\nreversal", "No\nspurious", "Residue\nvisible", "OOD\nrandom", "Partial\nshift"]
    methods = ["sequence_erm", "final_frame_mlp", "nuisance_only_oracle", "counterfactual_invariance"]
    rows = ["Seq. ERM", "Final frame", "Nuis. reference", "Counterfactual"]
    matrix = np.full((len(methods), len(scenarios)), np.nan)
    for i, method in enumerate(methods):
        for j, scenario in enumerate(scenarios):
            matrix[i, j] = store.aggregate(method, "ood_test_accuracy", scenario).mean
    fig, ax = plt.subplots(figsize=(7.25, 3.12))
    ax.set_xlim(-0.5, len(scenarios) - 0.5)
    ax.set_ylim(len(methods) - 0.5, -1.08)
    norm = plt.Normalize(0.0, 1.0)
    for i in range(len(methods)):
        for j in range(len(scenarios)):
            val = matrix[i, j]
            color = AUDIT_CMAP(norm(val))
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=color, edgecolor="white", linewidth=1.0, zorder=1))
            txt_color = "white" if val >= 0.64 else TEXT
            tag = "collapse" if val <= 0.20 else "near chance" if 0.40 <= val <= 0.60 else "core" if val >= 0.80 else "partial"
            ax.text(j, i - 0.08, f"{val:.2f}", ha="center", va="center", fontsize=7.5, fontweight="bold", color=txt_color, zorder=3)
            ax.text(j, i + 0.18, tag, ha="center", va="center", fontsize=5.55, color=("white" if val >= 0.64 else MUTED_TEXT), zorder=3)
    ax.set_xticks(np.arange(len(scenarios)))
    ax.set_xticklabels(columns)
    ax.set_yticks(np.arange(len(methods)))
    ax.set_yticklabels(rows)
    ax.tick_params(axis="both", length=0, pad=5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for start, end, label in [(0, 0, "Main stress test"), (1, 2, "Controls"), (3, 4, "Shift variants")]:
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
    save_figure(fig, out)


def render(name: str):
    apply_style()
    figures = {
        "fig1": lambda: fig1(Path("figures/main/fig1_conceptual_problem")),
        "fig2": lambda: fig2(Path("figures/main/fig2_audit_flow")),
        "fig3": lambda: fig3(Path("figures/main/fig3_benchmark_construction"), Path("configs/default.yaml")),
        "fig4": lambda: fig4(
            Path("results/main/trail_fl_audit.json"),
            Path("results/main/simple_oe_audit.json"),
            Path("results/main/nuisance_order.json"),
            Path("figures/main/fig4a_perframe_probes"),
            Path("figures/main/fig4b_order_interventions"),
        ),
        "fig_a3": lambda: fig_a3(
            Path("figures/appendix/fig_a3_scenario_audit"),
            MetricStore(Path("results/ablation/scenario/summary.json")),
        ),
    }
    figures[name]()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--figure", default="all")
    a = p.parse_args()
    if a.figure == "all":
        for n in ["fig1", "fig2", "fig3", "fig4", "fig_a3"]:
            render(n)
        return
    render(a.figure)


if __name__ == "__main__":
    main()
