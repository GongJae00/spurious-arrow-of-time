"""Minimal plotting utilities for logged aggregate results."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

METHOD_LABELS = {
    "erm": "ERM",
    "ib": "IB",
    "ep_min": "EP-Min",
    "ep_max": "EP-Max",
    "ocp_style": "OCP-style",
    "lens_like_arrow_classifier": "LENS-like",
    "sib": "SIB",
    "sid": "SID",
    "itm": "ITM",
}

COLORS = {
    "iid": "#9CA3AF",
    "ood": "#4B5563",
    "core": "#2F6FBB",
    "spur": "#C44E52",
    "accent": "#1B9E77",
    "bar": "#6B7280",
    "grid": "#E5E7EB",
}


def _set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _save_figure_source(output: Path, payload: dict) -> None:
    source_path = output.with_suffix(".json")
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _save_current(output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, bbox_inches="tight", dpi=350)
    plt.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()


def plot_ood_bars(aggregate_json: str | Path, output: str | Path) -> None:
    aggregate_json = Path(aggregate_json)
    _set_style()
    data = json.loads(aggregate_json.read_text())
    methods = sorted(k for k, v in data.items() if k != "by_condition" and isinstance(v, dict))
    iid = []
    ood = []
    rows = []
    for method in methods:
        iid_value = data[method].get(
            "iid_test_accuracy_mean",
            data[method].get("fine_tuned_encoder_iid_test_accuracy_mean", 0.0),
        )
        ood_value = data[method].get(
            "ood_test_accuracy_mean",
            data[method].get("fine_tuned_encoder_ood_test_accuracy_mean", 0.0),
        )
        iid.append(iid_value)
        ood.append(ood_value)
        rows.append({"method": method, "iid_test_accuracy": iid_value, "ood_test_accuracy": ood_value})
    y = list(range(len(methods)))[::-1]
    plt.figure(figsize=(6.4, max(3.2, len(methods) * 0.34 + 0.8)))
    plt.errorbar(iid, [v + 0.12 for v in y], fmt="o", color=COLORS["iid"], label="IID test")
    plt.errorbar(ood, [v - 0.12 for v in y], fmt="o", color=COLORS["ood"], label="OOD test")
    plt.yticks(y, [METHOD_LABELS.get(method, method) for method in methods])
    plt.xlabel("Accuracy")
    plt.grid(axis="x", color=COLORS["grid"], linewidth=0.8)
    plt.legend(frameon=False, loc="lower right")
    plt.xlim(0.0, 1.03)
    output = Path(output)
    _save_current(output)
    _save_figure_source(
        output,
        {
            "kind": "ood",
            "input_json": str(aggregate_json),
            "rows": rows,
        },
    )


def _manifest_runs(manifest_json: str | Path) -> list[dict]:
    runs = json.loads(Path(manifest_json).read_text()).get("runs", [])
    return [run for run in runs if run.get("status", "success") == "success"]


def _aggregate_values(rows: list[dict], key_fields: tuple[str, ...], value_field: str) -> list[dict]:
    grouped: dict[tuple, list[float]] = defaultdict(list)
    exemplars: dict[tuple, dict] = {}
    for row in rows:
        key = tuple(row[field] for field in key_fields)
        grouped[key].append(float(row[value_field]))
        exemplars[key] = {field: row[field] for field in key_fields}
    out = []
    for key in sorted(grouped):
        values = grouped[key]
        std = stdev(values) if len(values) > 1 else 0.0
        aggregate = {
            **exemplars[key],
            "n": len(values),
            f"{value_field}_mean": mean(values),
            f"{value_field}_std": std,
            f"{value_field}_sem": std / (len(values) ** 0.5) if len(values) > 1 else 0.0,
            "raw_values": values,
        }
        out.append(aggregate)
    return out


def plot_setpoint_sweep(
    manifest_json: str | Path,
    output: str | Path,
    method: str = "sib",
    metric: str = "ood_test_accuracy",
) -> None:
    manifest_json = Path(manifest_json)
    _set_style()
    raw_rows = []
    for run in _manifest_runs(manifest_json):
        sweep = run.get("sweep") or {}
        if run.get("method") != method or sweep.get("type") != "setpoint":
            continue
        target = sweep.get("target")
        if target is None or metric not in run.get("metrics", {}):
            continue
        raw_rows.append(
            {
                "method": method,
                "sigma_target": float(target),
                "setpoint_mode": str(sweep.get("mode", "fixed_grid")),
                "setpoint_multiplier": sweep.get("multiplier"),
                metric: float(run["metrics"][metric]),
            }
        )
    raw_rows.sort(key=lambda item: item["sigma_target"])
    if not raw_rows:
        raise ValueError("no fixed-grid setpoint runs found for plotting")
    aggregate_rows = _aggregate_values(raw_rows, ("method", "sigma_target", "setpoint_mode"), metric)
    x = [row["sigma_target"] for row in aggregate_rows]
    y = [row[f"{metric}_mean"] for row in aggregate_rows]
    yerr = [row[f"{metric}_sem"] for row in aggregate_rows]
    plt.figure(figsize=(5.2, 3.4))
    plt.errorbar(x, y, yerr=yerr, marker="o", capsize=3, color=COLORS["accent"])
    plt.xlabel("Sigma target")
    plt.ylabel(metric)
    plt.title(f"{METHOD_LABELS.get(method, method)} setpoint sweep")
    plt.grid(axis="y", color=COLORS["grid"], linewidth=0.8)
    output = Path(output)
    _save_current(output)
    _save_figure_source(
        output,
        {
            "kind": "setpoint",
            "input_json": str(manifest_json),
            "method": method,
            "metric": metric,
            "rows": aggregate_rows,
            "aggregate_rows": aggregate_rows,
            "raw_rows": raw_rows,
        },
    )


def plot_ep_ratio_sweep(
    manifest_json: str | Path,
    output: str | Path,
    method: str = "sib",
    metric: str = "ood_test_accuracy",
) -> None:
    manifest_json = Path(manifest_json)
    _set_style()
    raw_rows = []
    for run in _manifest_runs(manifest_json):
        sweep = run.get("sweep") or {}
        if run.get("method") != method or sweep.get("type") != "ep_ratio":
            continue
        ratio = sweep.get("actual_ratio")
        if ratio is None or metric not in run.get("metrics", {}):
            continue
        raw_rows.append(
            {
                "method": method,
                "actual_sigma_s_over_sigma_c": float(ratio),
                metric: float(run["metrics"][metric]),
            }
        )
    raw_rows.sort(key=lambda item: item["actual_sigma_s_over_sigma_c"])
    if not raw_rows:
        raise ValueError("no EP-ratio runs found for plotting")
    aggregate_rows = _aggregate_values(
        raw_rows, ("method", "actual_sigma_s_over_sigma_c"), metric
    )
    x = [row["actual_sigma_s_over_sigma_c"] for row in aggregate_rows]
    y = [row[f"{metric}_mean"] for row in aggregate_rows]
    yerr = [row[f"{metric}_sem"] for row in aggregate_rows]
    plt.figure(figsize=(5.2, 3.4))
    plt.errorbar(x, y, yerr=yerr, marker="o", capsize=3, color=COLORS["accent"])
    plt.xlabel("Actual Sigma_s / Sigma_c")
    plt.ylabel(metric)
    plt.title(f"{METHOD_LABELS.get(method, method)} EP-ratio sweep")
    plt.grid(axis="y", color=COLORS["grid"], linewidth=0.8)
    output = Path(output)
    _save_current(output)
    _save_figure_source(
        output,
        {
            "kind": "ep_ratio",
            "input_json": str(manifest_json),
            "method": method,
            "metric": metric,
            "rows": aggregate_rows,
            "aggregate_rows": aggregate_rows,
            "raw_rows": raw_rows,
        },
    )


def plot_counterfactual_sensitivity(
    manifest_json: str | Path,
    output: str | Path,
    method: str = "sib",
    metric: str = "iid_test_cf_delta_arrow_total",
) -> None:
    manifest_json = Path(manifest_json)
    _set_style()
    raw_rows = []
    for run in _manifest_runs(manifest_json):
        sweep = run.get("counterfactual_sweep") or {}
        if run.get("method") != method or not sweep:
            continue
        metrics = run.get("metrics", {})
        if metric not in metrics:
            continue
        label = sweep["spurious_cf_mode"]
        if sweep.get("no_change"):
            label = "no_change"
        raw_rows.append(
            {
                "method": method,
                "counterfactual_mode": label,
                "spurious_cf_mode": sweep["spurious_cf_mode"],
                "no_change": bool(sweep.get("no_change")),
                metric: float(metrics[metric]),
            }
        )
    if not raw_rows:
        raise ValueError("no counterfactual sensitivity runs found for plotting")
    aggregate_rows = _aggregate_values(
        raw_rows,
        ("method", "counterfactual_mode", "spurious_cf_mode", "no_change"),
        metric,
    )
    labels = [row["counterfactual_mode"] for row in aggregate_rows]
    values = [row[f"{metric}_mean"] for row in aggregate_rows]
    yerr = [row[f"{metric}_sem"] for row in aggregate_rows]
    plt.figure(figsize=(max(5.4, len(aggregate_rows) * 1.0), 3.4))
    plt.bar(range(len(values)), values, color=COLORS["bar"])
    plt.errorbar(range(len(values)), values, yerr=yerr, fmt="none", ecolor="black", capsize=3)
    plt.xticks(range(len(labels)), labels, rotation=30, ha="right")
    plt.ylabel(metric)
    plt.title(f"{METHOD_LABELS.get(method, method)} counterfactual sensitivity")
    plt.grid(axis="y", color=COLORS["grid"], linewidth=0.8)
    output = Path(output)
    _save_current(output)
    _save_figure_source(
        output,
        {
            "kind": "counterfactual",
            "input_json": str(manifest_json),
            "method": method,
            "metric": metric,
            "rows": aggregate_rows,
            "aggregate_rows": aggregate_rows,
            "raw_rows": raw_rows,
        },
    )


def plot_final_decomposition(
    diagnostic_summary_json: str | Path,
    output: str | Path,
    *,
    split: str = "iid_test",
    sigma_prefix: str = "calibrated",
) -> None:
    diagnostic_summary_json = Path(diagnostic_summary_json)
    _set_style()
    payload = json.loads(diagnostic_summary_json.read_text())
    summary = payload.get("summary", payload)
    rows = []
    core_key = f"{split}_corr_{sigma_prefix}_sigma_core_dynamic_stat_mean"
    spur_key = f"{split}_corr_{sigma_prefix}_sigma_spurious_dynamic_stat_mean"
    for condition, values in sorted(summary.items()):
        if not isinstance(values, dict):
            continue
        if core_key not in values and spur_key not in values:
            continue
        rows.append(
            {
                "condition": condition,
                "core_corr": values.get(core_key),
                "spurious_corr": values.get(spur_key),
                "n_runs": values.get("n_runs", 0),
            }
        )
    if not rows:
        raise ValueError("no final decomposition correlation rows found for plotting")

    labels = [row["condition"] for row in rows]
    core_values = [float(row["core_corr"] or 0.0) for row in rows]
    spur_values = [float(row["spurious_corr"] or 0.0) for row in rows]
    x = range(len(rows))
    plt.figure(figsize=(max(5.8, len(rows) * 1.0), 3.4))
    plt.bar([i - 0.2 for i in x], core_values, width=0.4, label="sigma-core corr", color=COLORS["core"])
    plt.bar(
        [i + 0.2 for i in x],
        spur_values,
        width=0.4,
        label="sigma-spurious corr",
        color=COLORS["spur"],
    )
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.xticks(list(x), labels, rotation=30, ha="right")
    plt.ylabel("Correlation")
    plt.title(f"Final latent-arrow decomposition ({split}, {sigma_prefix})")
    plt.legend(frameon=False)
    plt.grid(axis="y", color=COLORS["grid"], linewidth=0.8)
    output = Path(output)
    _save_current(output)
    _save_figure_source(
        output,
        {
            "kind": "final_decomposition",
            "input_json": str(diagnostic_summary_json),
            "split": split,
            "sigma_prefix": sigma_prefix,
            "rows": rows,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot logged experiment results.")
    parser.add_argument("input_json")
    parser.add_argument("--output", default="figures/ood_bars.png")
    parser.add_argument(
        "--kind",
        choices=["ood", "setpoint", "ep_ratio", "counterfactual", "final_decomposition"],
        default="ood",
    )
    parser.add_argument("--method", default="sib")
    parser.add_argument("--metric", default=None)
    parser.add_argument("--split", default="iid_test")
    parser.add_argument("--sigma-prefix", default="calibrated")
    args = parser.parse_args()
    if args.kind == "ood":
        plot_ood_bars(args.input_json, args.output)
    elif args.kind == "setpoint":
        plot_setpoint_sweep(
            args.input_json,
            args.output,
            method=args.method,
            metric=args.metric or "ood_test_accuracy",
        )
    elif args.kind == "ep_ratio":
        plot_ep_ratio_sweep(
            args.input_json,
            args.output,
            method=args.method,
            metric=args.metric or "ood_test_accuracy",
        )
    elif args.kind == "counterfactual":
        plot_counterfactual_sensitivity(
            args.input_json,
            args.output,
            method=args.method,
            metric=args.metric or "iid_test_cf_delta_arrow_total",
        )
    elif args.kind == "final_decomposition":
        plot_final_decomposition(
            args.input_json,
            args.output,
            split=args.split,
            sigma_prefix=args.sigma_prefix,
        )


if __name__ == "__main__":
    main()
