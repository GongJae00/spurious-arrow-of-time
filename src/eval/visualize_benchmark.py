"""Visualize irreversible source inference benchmark components."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.data.irreversible_source_inference import generate_irreversible_source_splits, load_config


FIGURE_DPI = 300


def write_visualizations(config_path: str | Path, out_dir: str | Path) -> None:
    config = load_config(config_path)
    splits = generate_irreversible_source_splits(config)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    plot_problem_schematic(splits, out / "problem_schematic.png")
    plot_components(splits, out / "component_visualization.png")
    plot_recoverability(out / "diagnostics.json", out / "source_recoverability.png")


def plot_problem_schematic(splits: dict, path: Path) -> None:
    examples = [
        ("Train/IID shortcut", splits["train"], select_example(splits["train"], y_value=1)),
        ("OOD reversed shortcut", splits["ood_test"], select_example(splits["ood_test"], y_value=1)),
    ]
    columns = [
        "initial core\nhidden source",
        "diffused core\nfinal frame",
        "nuisance arrow\ntrajectory",
        "observed mixed\nfinal frame",
        "counterfactual\nfinal frame",
    ]
    panels: list[list[np.ndarray]] = []
    row_labels: list[str] = []
    for row_name, split, idx in examples:
        y = int(split.y[idx])
        source = source_label(int(split.source_orientation[idx]))
        nuisance = direction_label(int(split.nuisance_direction[idx]))
        cf = direction_label(int(split.counterfactual_direction[idx]))
        panels.append(
            [
                split.core_only[idx, 0],
                split.core_only[idx, -1],
                trajectory_projection(split.nuisance_only[idx]),
                split.mixed[idx, -1],
                split.counterfactual[idx, -1],
            ]
        )
        row_labels.append(
            f"{row_name}\ny={y}, {source}\nshortcut arrow={nuisance}, cf={cf}"
        )

    fig, axes = plt.subplots(
        len(panels),
        len(columns),
        figsize=(11.5, 4.8),
        constrained_layout=True,
        squeeze=False,
    )
    all_values = np.concatenate([panel.reshape(-1) for row in panels for panel in row])
    vmin = float(np.quantile(all_values, 0.01))
    vmax = float(np.quantile(all_values, 0.995))
    if vmax <= vmin:
        vmax = vmin + 1e-6
    for r, row in enumerate(panels):
        for c, image in enumerate(row):
            ax = axes[r, c]
            ax.imshow(image, cmap="magma", vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(columns[c], fontsize=10)
            if c == 0:
                ax.set_ylabel(row_labels[r], fontsize=9, rotation=0, ha="right", va="center")
    fig.suptitle(
        "Irreversible inverse inference with a spurious arrow shortcut",
        fontsize=13,
    )
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def trajectory_projection(sequence: np.ndarray) -> np.ndarray:
    weights = np.linspace(0.25, 1.0, sequence.shape[0], dtype=np.float32)
    projected = np.max(sequence * weights[:, None, None], axis=0)
    return projected.astype(np.float32)


def plot_components(splits: dict, path: Path) -> None:
    train = select_example(splits["train"], y_value=1)
    ood = select_example(splits["ood_test"], y_value=1)
    examples = [("Train", splits["train"], train), ("OOD", splits["ood_test"], ood)]
    length = splits["train"].mixed.shape[1]
    frame_ids = np.linspace(0, length - 1, num=min(4, length), dtype=int).tolist()
    rows = []
    for label, split, idx in examples:
        y = int(split.y[idx])
        source = source_label(int(split.source_orientation[idx]))
        nuisance = direction_label(int(split.nuisance_direction[idx]))
        cf = direction_label(int(split.counterfactual_direction[idx]))
        rows.extend(
            [
                (f"{label} core\ny={y}, {source}", split.core_only[idx]),
                (f"{label} nuisance\narrow={nuisance}", split.nuisance_only[idx]),
                (f"{label} mixed\nobserved", split.mixed[idx]),
                (f"{label} counterfactual\narrow={cf}", split.counterfactual[idx]),
            ]
        )

    fig, axes = plt.subplots(
        len(rows),
        len(frame_ids),
        figsize=(10.8, 12.6),
        constrained_layout=True,
        squeeze=False,
    )
    scales = component_scales(rows)
    for r, (row_name, values) in enumerate(rows):
        vmin, vmax = scales[row_group(row_name)]
        for c, frame_idx in enumerate(frame_ids):
            ax = axes[r, c]
            ax.imshow(values[frame_idx], cmap="magma", vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(row_name, fontsize=9, rotation=0, ha="right", va="center")
            if r == 0:
                ax.set_title(f"t={frame_idx}", fontsize=10, pad=6)
    fig.suptitle(
        "Irreversible source task: core diffusion, nuisance arrow, observation, counterfactual",
        fontsize=13,
    )
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def component_scales(rows: list[tuple[str, np.ndarray]]) -> dict[str, tuple[float, float]]:
    values_by_group: dict[str, list[np.ndarray]] = {"core": [], "nuisance": [], "observed": []}
    for name, values in rows:
        values_by_group[row_group(name)].append(values)
    scales = {}
    for group, values in values_by_group.items():
        stacked = np.concatenate([v.reshape(-1) for v in values])
        vmin = float(np.quantile(stacked, 0.01))
        vmax = float(np.quantile(stacked, 0.995))
        if vmax <= vmin:
            vmax = vmin + 1e-6
        scales[group] = (vmin, vmax)
    return scales


def row_group(row_name: str) -> str:
    if "core" in row_name:
        return "core"
    if "nuisance" in row_name:
        return "nuisance"
    return "observed"


def source_label(source_orientation: int) -> str:
    if source_orientation == 0:
        return "horizontal source"
    if source_orientation == 1:
        return "vertical source"
    return f"source={source_orientation}"


def direction_label(direction: int) -> str:
    if direction > 0:
        return "right"
    if direction < 0:
        return "left"
    return "none"


def plot_recoverability(diagnostics_path: Path, path: Path) -> None:
    if not diagnostics_path.exists():
        return
    payload = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    metrics = payload["metrics"]
    rows = [
        ("Final core oracle", metrics["final_frame_core_oracle_accuracy"], "#7f8c8d"),
        ("Full core oracle", metrics["full_sequence_core_oracle_accuracy"], "#0072b2"),
        ("Core-only OOD", metrics["core_only_ood_accuracy"], "#0072b2"),
        ("Nuisance-only IID", metrics["nuisance_only_iid_accuracy"], "#e69f00"),
        ("Nuisance-only OOD", metrics["nuisance_only_ood_accuracy"], "#e69f00"),
        ("Mixed feature probe IID", metrics["mixed_feature_probe_iid_accuracy"], "#cc79a7"),
        ("Mixed feature probe OOD", metrics["mixed_feature_probe_ood_accuracy"], "#cc79a7"),
    ]
    labels = [row[0] for row in rows]
    values = [row[1] for row in rows]
    colors = [row[2] for row in rows]
    y_pos = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(8.2, 5.0), constrained_layout=True)
    ax.barh(y_pos, values, color=colors)
    ax.axvline(0.5, color="black", lw=1, ls="--", alpha=0.65)
    ax.set_xlim(0.0, 1.04)
    ax.set_yticks(y_pos, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Accuracy")
    ax.set_title("Benchmark smoke gates: recoverability and shortcut failure", fontsize=12)
    ax.grid(axis="x", color="#dddddd", lw=0.7)
    ax.set_axisbelow(True)
    for i, value in enumerate(values):
        ax.text(min(value + 0.02, 1.0), i, f"{value:.2f}", va="center", fontsize=9)
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def select_example(split, y_value: int) -> int:
    candidates = np.flatnonzero(split.y == y_value)
    if len(candidates) == 0:
        return 0
    return int(candidates[0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_visualizations(args.config, args.out)


if __name__ == "__main__":
    main()
