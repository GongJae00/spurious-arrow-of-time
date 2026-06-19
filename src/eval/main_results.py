"""Aggregate and plot neural main-experiment results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


METHOD_COLORS = {
    "final_frame_mlp": "#7f8c8d",
    "sequence_erm": "#d55e00",
    "core_only_oracle": "#0072b2",
    "nuisance_only_oracle": "#e69f00",
    "time_reversed_sequence": "#009e73",
    "counterfactual_invariance": "#cc79a7",
    "group_invariance_light": "#56b4e9",
}

METHOD_ORDER = [
    "core_only_oracle",
    "sequence_erm",
    "nuisance_only_oracle",
    "final_frame_mlp",
    "time_reversed_sequence",
    "counterfactual_invariance",
    "group_invariance_light",
]

SCENARIO_ORDER = [
    "main_reversed",
    "ood_randomized",
    "ood_partial_shift",
    "nuisance_scale_low",
    "nuisance_scale_mid",
    "core_difficulty_easy",
    "core_difficulty_hard",
    "no_spurious_correlation",
    "random_labels",
    "core_only_no_nuisance",
]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def grouped_values(rows: list[dict[str, Any]], metric: str) -> dict[str, list[float]]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        grouped.setdefault(str(row["method"]), []).append(float(row[metric]))
    return grouped


def mean_sem(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    if len(arr) <= 1:
        return float(arr.mean()), 0.0
    return float(arr.mean()), float(arr.std(ddof=1) / np.sqrt(len(arr)))


def format_cell(value: float) -> str:
    if abs(value) < 0.005:
        value = 0.0
    return f"{value:.2f}"


def primary_rows(rows: list[dict[str, Any]], manifest: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not manifest:
        return rows
    primary = manifest.get("primary_scenario")
    if not primary:
        return rows
    filtered = [row for row in rows if row.get("scenario") == primary]
    return filtered or rows


def write_method_table(rows: list[dict[str, Any]], path: Path) -> None:
    methods = ordered_labels({str(row["method"]) for row in rows}, METHOD_ORDER)
    lines = [
        "# Main Neural Method Table",
        "",
        "| Method | Val IID | IID Test | OOD Test | OOD Gap | Seeds |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        vals = {
            key: mean_sem([float(row[key]) for row in method_rows])[0]
            for key in ["val_iid_accuracy", "iid_test_accuracy", "ood_test_accuracy", "ood_gap"]
        }
        lines.append(
            f"| {method} | {vals['val_iid_accuracy']:.3f} | "
            f"{vals['iid_test_accuracy']:.3f} | {vals['ood_test_accuracy']:.3f} | "
            f"{vals['ood_gap']:.3f} | {len(method_rows)} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_scenario_table(rows: list[dict[str, Any]], path: Path) -> None:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row.get("scenario", "main_reversed")), str(row["method"]))
        grouped.setdefault(key, []).append(row)
    lines = [
        "# Scenario Sweep Table",
        "",
        "| Scenario | Method | IID Test | OOD Test | OOD Gap | Seeds |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for (scenario, method), method_rows in sorted(grouped.items()):
        vals = {
            key: mean_sem([float(row[key]) for row in method_rows])[0]
            for key in ["iid_test_accuracy", "ood_test_accuracy", "ood_gap"]
        }
        lines.append(
            f"| {scenario} | {method} | {vals['iid_test_accuracy']:.3f} | "
            f"{vals['ood_test_accuracy']:.3f} | {vals['ood_gap']:.3f} | {len(method_rows)} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def ordered_labels(labels: set[str], preferred: list[str]) -> list[str]:
    preferred_present = [label for label in preferred if label in labels]
    remainder = sorted(labels - set(preferred_present))
    return preferred_present + remainder


def plot_iid_ood(rows: list[dict[str, Any]], path: Path) -> None:
    methods = ordered_labels({str(row["method"]) for row in rows}, METHOD_ORDER)
    x = np.arange(len(methods))
    width = 0.36
    iid_means = []
    iid_errs = []
    ood_means = []
    ood_errs = []
    colors = [METHOD_COLORS.get(method, "#666666") for method in methods]
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        iid_mean, iid_sem = mean_sem([float(row["iid_test_accuracy"]) for row in method_rows])
        ood_mean, ood_sem = mean_sem([float(row["ood_test_accuracy"]) for row in method_rows])
        iid_means.append(iid_mean)
        iid_errs.append(iid_sem)
        ood_means.append(ood_mean)
        ood_errs.append(ood_sem)

    fig, ax = plt.subplots(figsize=(10.5, 5.2), constrained_layout=True)
    ax.bar(x - width / 2, iid_means, width, yerr=iid_errs, label="IID test", color=colors, alpha=0.55)
    ax.bar(x + width / 2, ood_means, width, yerr=ood_errs, label="OOD test", color=colors, alpha=0.95)
    ax.axhline(0.5, color="black", lw=1, ls="--", alpha=0.65)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title("Raw neural model performance under spurious-arrow shift")
    ax.set_xticks(x, [label.replace("_", "\n") for label in methods], fontsize=9)
    ax.legend(frameon=False)
    ax.grid(axis="y", color="#dddddd", lw=0.7)
    ax.set_axisbelow(True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_ood_gap(rows: list[dict[str, Any]], path: Path) -> None:
    grouped = grouped_values(rows, "ood_gap")
    methods = ordered_labels(set(grouped), METHOD_ORDER)
    means = []
    errs = []
    for method in methods:
        mean, sem = mean_sem(grouped[method])
        means.append(mean)
        errs.append(sem)
    colors = [METHOD_COLORS.get(method, "#666666") for method in methods]
    fig, ax = plt.subplots(figsize=(9.5, 4.8), constrained_layout=True)
    ax.barh(np.arange(len(methods)), means, xerr=errs, color=colors)
    ax.axvline(0.0, color="black", lw=1)
    ax.set_yticks(np.arange(len(methods)), [method.replace("_", " ") for method in methods])
    ax.invert_yaxis()
    ax.set_xlabel("OOD gap = IID accuracy - OOD accuracy")
    ax.set_title("Shortcut reliance measured by OOD gap")
    ax.grid(axis="x", color="#dddddd", lw=0.7)
    ax.set_axisbelow(True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_scenario_heatmap(
    rows: list[dict[str, Any]],
    *,
    metric: str,
    path: Path,
    title: str,
    colorbar_label: str,
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> None:
    scenarios = ordered_labels(
        {str(row.get("scenario", "main_reversed")) for row in rows},
        SCENARIO_ORDER,
    )
    methods = ordered_labels({str(row["method"]) for row in rows}, METHOD_ORDER)
    if len(scenarios) <= 1:
        return

    values = np.full((len(scenarios), len(methods)), np.nan, dtype=float)
    for i, scenario in enumerate(scenarios):
        for j, method in enumerate(methods):
            selected = [
                float(row[metric])
                for row in rows
                if str(row.get("scenario", "main_reversed")) == scenario
                and str(row["method"]) == method
            ]
            if selected:
                values[i, j] = float(np.mean(selected))

    fig_height = max(4.5, 0.48 * len(scenarios) + 1.8)
    fig_width = max(8.5, 0.95 * len(methods) + 3.0)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)
    masked = np.ma.masked_invalid(values)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="#f0f0f0")
    image = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xticks(np.arange(len(methods)), [m.replace("_", "\n") for m in methods], fontsize=8)
    ax.set_yticks(np.arange(len(scenarios)), [s.replace("_", " ") for s in scenarios], fontsize=9)
    ax.set_xlabel("Method")
    ax.set_ylabel("Scenario")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if np.isfinite(values[i, j]):
                color = "white" if values[i, j] > (vmin + vmax) / 2 else "black"
                ax.text(
                    j,
                    i,
                    format_cell(values[i, j]),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=color,
                )
            else:
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=8, color="#555555")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label(colorbar_label)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_claim_audit(rows: list[dict[str, Any]], path: Path, manifest: dict[str, Any] | None) -> None:
    by_method = {method: vals for method, vals in grouped_values(rows, "ood_gap").items()}
    lines = ["# Result Claim Audit", ""]
    diagnostic_only = bool(
        manifest and (manifest.get("runtime_limited") or manifest.get("profile") in {"smoke", "pilot"})
    )
    if manifest and (manifest.get("runtime_limited") or manifest.get("profile") in {"smoke", "pilot"}):
        lines.extend(
            [
                "This is a runtime-limited diagnostic run. It cannot support a paper-level main claim.",
                "",
            ]
        )
    if "sequence_erm" not in by_method:
        lines.append("`sequence_erm` is missing; no main neural shortcut claim is allowed.")
    else:
        erm_gap = float(np.mean(by_method["sequence_erm"]))
        lines.append(f"`sequence_erm` mean OOD gap: `{erm_gap:.3f}`")
        if erm_gap >= 0.20:
            if diagnostic_only:
                lines.append("Diagnostic neural ERM shortcut target is met for this limited run.")
            else:
                lines.append("Neural ERM shortcut evidence target is met.")
        else:
            lines.append("Neural ERM shortcut evidence target is not met.")
    if "counterfactual_invariance" in by_method and "sequence_erm" in by_method:
        cf_gap = float(np.mean(by_method["counterfactual_invariance"]))
        erm_gap = float(np.mean(by_method["sequence_erm"]))
        lines.append(f"`counterfactual_invariance` mean OOD gap: `{cf_gap:.3f}`")
        cf_iid = float(
            np.mean(
                [
                    float(row["iid_test_accuracy"])
                    for row in rows
                    if row["method"] == "counterfactual_invariance"
                ]
            )
        )
        erm_iid = float(
            np.mean(
                [
                    float(row["iid_test_accuracy"])
                    for row in rows
                    if row["method"] == "sequence_erm"
                ]
            )
        )
        iid_preserved = cf_iid >= max(0.75, erm_iid - 0.15)
        if cf_gap < erm_gap:
            if iid_preserved:
                lines.append("Counterfactual invariance reduces the mean OOD gap without destroying IID accuracy.")
            else:
                lines.append("Counterfactual invariance reduces the mean OOD gap but destroys IID accuracy.")
        else:
            lines.append("Counterfactual invariance does not reduce the mean OOD gap.")
    lines.append("")
    lines.append("Feature-probe diagnostics are not neural model claims.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    rows = load_jsonl(Path(args.metrics))
    manifest_path = out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
    rows_for_main = primary_rows(rows, manifest)
    write_method_table(rows_for_main, out_dir / "method_table.md")
    write_scenario_table(rows, out_dir / "scenario_table.md")
    write_claim_audit(rows_for_main, out_dir / "claim_audit.md", manifest)
    plot_iid_ood(rows_for_main, out_dir / "main_results_bars.png")
    plot_ood_gap(rows_for_main, out_dir / "ood_gap_bars.png")
    plot_scenario_heatmap(
        rows,
        metric="ood_gap",
        path=out_dir / "scenario_ood_gap_heatmap.png",
        title="OOD gap across scenarios",
        colorbar_label="OOD gap",
    )
    plot_scenario_heatmap(
        rows,
        metric="ood_test_accuracy",
        path=out_dir / "scenario_ood_accuracy_heatmap.png",
        title="OOD accuracy across scenarios",
        colorbar_label="OOD accuracy",
    )


if __name__ == "__main__":
    main()
