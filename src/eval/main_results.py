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
    "main_spurious_arrow",
    "ood_randomized",
    "ood_partial_shift",
    "nuisance_scale_low",
    "nuisance_scale_mid",
    "residue_visible_control",
    "core_difficulty_easy",
    "core_difficulty_hard",
    "no_spurious_correlation",
    "random_labels",
    "core_label_randomized_spurious_nuisance",
    "core_only_no_nuisance",
]

SCENARIO_LABELS = {
    "random_labels": "core label randomized\nspurious nuisance",
    "core_label_randomized_spurious_nuisance": "core label randomized\nspurious nuisance",
}


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


def scenario_display_label(label: str) -> str:
    return SCENARIO_LABELS.get(label, label.replace("_", " "))


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
        key = (str(row.get("scenario", "main_spurious_arrow")), str(row["method"]))
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
        scenario_label = scenario_display_label(scenario).replace("\n", " ")
        lines.append(
            f"| {scenario_label} | {method} | {vals['iid_test_accuracy']:.3f} | "
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
        {str(row.get("scenario", "main_spurious_arrow")) for row in rows},
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
                if str(row.get("scenario", "main_spurious_arrow")) == scenario
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
    ax.set_yticks(np.arange(len(scenarios)), [scenario_display_label(s) for s in scenarios], fontsize=9)
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
        primary = manifest.get("primary_scenario", "main_spurious_arrow") if manifest else "main_spurious_arrow"
        cf_success = success_rate_metric(
            rows,
            str(primary),
            "counterfactual_invariance",
            "ood_test_accuracy",
            0.80,
            ">=",
        )
        if cf_success is not None:
            lines.append(f"`counterfactual_invariance` OOD>=0.80 seed success rate: `{cf_success:.3f}`")
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
                if cf_success is not None and cf_success < 0.80:
                    lines.append(
                        "Counterfactual invariance reduces the mean OOD gap, but is not seed-stable enough for a primary method-success claim."
                    )
                else:
                    lines.append("Counterfactual invariance reduces the mean OOD gap without destroying IID accuracy.")
            else:
                lines.append("Counterfactual invariance reduces the mean OOD gap but destroys IID accuracy.")
        else:
            lines.append("Counterfactual invariance does not reduce the mean OOD gap.")
    lines.append("")
    lines.append("Feature-probe diagnostics are not neural model claims.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def scenario_method_rows(
    rows: list[dict[str, Any]],
    scenario: str,
    method: str,
) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if str(row.get("scenario", "main_spurious_arrow")) == scenario
        and str(row["method"]) == method
    ]


def mean_metric(
    rows: list[dict[str, Any]],
    scenario: str,
    method: str,
    metric: str,
) -> float | None:
    selected = scenario_method_rows(rows, scenario, method)
    if not selected:
        return None
    return float(np.mean([float(row[metric]) for row in selected]))


def success_rate_metric(
    rows: list[dict[str, Any]],
    scenario: str,
    method: str,
    metric: str,
    threshold: float,
    op: str,
) -> float | None:
    selected = scenario_method_rows(rows, scenario, method)
    if not selected:
        return None
    values = np.asarray([float(row[metric]) for row in selected], dtype=float)
    if op == ">=":
        passed = values >= threshold
    elif op == "<=":
        passed = values <= threshold
    else:
        raise ValueError(op)
    return float(np.mean(passed))


def add_gate(
    checks: list[dict[str, Any]],
    name: str,
    value: float | None,
    op: str,
    threshold: float,
    note: str,
) -> None:
    if value is None:
        checks.append(
            {
                "name": name,
                "value": None,
                "op": op,
                "threshold": threshold,
                "passed": False,
                "note": note,
            }
        )
        return
    if op == ">=":
        passed = value >= threshold
    elif op == "<=":
        passed = value <= threshold
    else:
        raise ValueError(op)
    checks.append(
        {
            "name": name,
            "value": value,
            "op": op,
            "threshold": threshold,
            "passed": passed,
            "note": note,
        }
    )


def final_gate_checks(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    no_spur = "no_spurious_correlation"
    main = "main_spurious_arrow"
    endpoint: str | None = None
    main_rows = [row for row in rows if str(row.get("scenario", "")) == main]
    main_is_endpoint = bool(
        main_rows
        and all(
            (row.get("dataset_config") or {}).get("benchmark_variant") == "endpoint_matched"
            for row in main_rows
        )
    )
    if main_is_endpoint:
        endpoint = main

    no_iid = mean_metric(rows, no_spur, "sequence_erm", "iid_test_accuracy")
    no_ood = mean_metric(rows, no_spur, "sequence_erm", "ood_test_accuracy")
    no_gap = mean_metric(rows, no_spur, "sequence_erm", "ood_gap")
    no_success = success_rate_metric(
        rows,
        no_spur,
        "sequence_erm",
        "ood_test_accuracy",
        0.80,
        ">=",
    )
    add_gate(checks, "gate_a_no_spurious_iid", no_iid, ">=", 0.80, "Mixed ERM must learn the causal core when nuisance is label-independent.")
    add_gate(checks, "gate_a_no_spurious_ood", no_ood, ">=", 0.80, "No-spurious OOD should remain high.")
    add_gate(checks, "gate_a_no_spurious_gap", no_gap, "<=", 0.10, "No-spurious gap should be small.")
    add_gate(checks, "gate_a_no_spurious_seed_success_rate", no_success, ">=", 0.80, "Most seeds should learn the core when nuisance is label-independent.")

    main_iid = mean_metric(rows, main, "sequence_erm", "iid_test_accuracy")
    main_ood = mean_metric(rows, main, "sequence_erm", "ood_test_accuracy")
    main_gap = mean_metric(rows, main, "sequence_erm", "ood_gap")
    add_gate(checks, "gate_b_main_iid", main_iid, ">=", 0.80, "Main shortcut setting should be learnable IID.")
    add_gate(checks, "gate_b_main_gap", main_gap, ">=", 0.25, "Main setting should induce a substantial OOD gap.")
    add_gate(checks, "gate_b_main_ood", main_ood, "<=", 0.60, "Main OOD should fall when the spurious arrow shifts.")

    nuisance_iid = mean_metric(rows, main, "nuisance_only_oracle", "iid_test_accuracy")
    nuisance_ood = mean_metric(rows, main, "nuisance_only_oracle", "ood_test_accuracy")
    core_iid = mean_metric(rows, main, "core_only_oracle", "iid_test_accuracy")
    core_ood = mean_metric(rows, main, "core_only_oracle", "ood_test_accuracy")
    add_gate(checks, "gate_b_nuisance_iid", nuisance_iid, ">=", 0.80, "Nuisance-only predictor should expose the tempting shortcut.")
    add_gate(checks, "gate_b_nuisance_ood", nuisance_ood, "<=", 0.40, "Nuisance-only predictor should fail under reversed OOD.")
    add_gate(checks, "gate_b_core_iid", core_iid, ">=", 0.80, "Core-only oracle should remain learnable.")
    add_gate(checks, "gate_b_core_ood", core_ood, ">=", 0.80, "Core-only oracle should be robust.")

    final_gap = mean_metric(rows, main, "final_frame_mlp", "ood_gap")
    if final_gap is not None and main_gap is not None:
        sequence_minus_final = main_gap - final_gap
    else:
        sequence_minus_final = None
    add_gate(checks, "gate_d_final_frame_gap_recorded", final_gap, ">=", -1.00, "Final-frame audit must be present; high values limit temporal-only claims.")
    add_gate(checks, "gate_d_sequence_minus_final_gap", sequence_minus_final, ">=", -0.10, "If final-frame collapse matches sequence collapse, claim residue-aided shortcutting, not pure temporal reasoning.")

    endpoint_seq_gap = (
        mean_metric(rows, endpoint, "sequence_erm", "ood_gap") if endpoint else None
    )
    endpoint_final_gap = (
        mean_metric(rows, endpoint, "final_frame_mlp", "ood_gap") if endpoint else None
    )
    if endpoint is not None:
        add_gate(checks, "gate_e_endpoint_sequence_gap", endpoint_seq_gap, ">=", 0.20, "Endpoint-matched sequence model should still show temporal shortcutting.")
        add_gate(checks, "gate_e_endpoint_final_gap", endpoint_final_gap, "<=", 0.15, "Endpoint-matched final-frame leakage should be controlled.")

    cf_iid = mean_metric(rows, main, "counterfactual_invariance", "iid_test_accuracy")
    cf_ood = mean_metric(rows, main, "counterfactual_invariance", "ood_test_accuracy")
    cf_gap = mean_metric(rows, main, "counterfactual_invariance", "ood_gap")
    if cf_gap is not None:
        min_cf_iid = max(0.75, (main_iid or 0.0) - 0.15)
        add_gate(checks, "gate_f_counterfactual_iid", cf_iid, ">=", min_cf_iid, "Counterfactual method must not buy robustness by destroying IID accuracy.")
        if main_ood is not None:
            add_gate(checks, "gate_f_counterfactual_ood", cf_ood, ">=", main_ood + 0.05, "Counterfactual method should improve OOD over ERM.")
        if main_gap is not None:
            add_gate(checks, "gate_f_counterfactual_gap", cf_gap, "<=", main_gap - 0.05, "Counterfactual method should reduce the OOD gap.")
        cf_success = success_rate_metric(
            rows,
            main,
            "counterfactual_invariance",
            "ood_test_accuracy",
            0.80,
            ">=",
        )
        add_gate(checks, "gate_f_counterfactual_seed_success_rate", cf_success, ">=", 0.80, "Counterfactual improvement should not be driven by only a few seeds.")
    return checks


def write_final_gate_audit(
    rows: list[dict[str, Any]],
    path: Path,
    manifest: dict[str, Any] | None,
) -> None:
    checks = final_gate_checks(rows)
    phenomenon_checks = [check for check in checks if not str(check["name"]).startswith("gate_f_")]
    mitigation_checks = [check for check in checks if str(check["name"]).startswith("gate_f_")]
    phenomenon_passed = all(bool(check["passed"]) for check in phenomenon_checks)
    mitigation_passed = all(bool(check["passed"]) for check in mitigation_checks) if mitigation_checks else None
    diagnostic_only = bool(
        manifest and (manifest.get("runtime_limited") or manifest.get("profile") in {"smoke", "pilot", "calibration"})
    )
    lines = [
        "# Final Evidence Gate Audit",
        "",
        f"Phenomenon gates passed: `{phenomenon_passed}`",
        f"Counterfactual mitigation gates passed: `{mitigation_passed}`",
        f"Runtime-limited diagnostic: `{diagnostic_only}`",
        "",
        "| Gate | Value | Rule | Pass | Note |",
        "|---|---:|---|---|---|",
    ]
    for check in checks:
        value = "missing" if check["value"] is None else f"{float(check['value']):.3f}"
        lines.append(
            f"| `{check['name']}` | `{value}` | `{check['op']} {check['threshold']:.3f}` | `{check['passed']}` | {check['note']} |"
        )
    lines.extend(
        [
            "",
            "A failed gate does not invalidate the run. It limits which manuscript claim is allowed.",
            "Feature-probe diagnostics cannot substitute for these neural-result gates.",
        ]
    )
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
    write_final_gate_audit(rows, out_dir / "final_gate_audit.md", manifest)
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
