"""Generate paper-oriented figures from audited final-study artifacts.

The figures produced here are explanatory manuscript assets, not training
artifacts. They intentionally separate mechanism, data visualization, and
claim-gated results so the paper does not rely on low-resolution diagnostic
contact sheets.
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec, patches
import numpy as np

from src.eval.audit_smoke import EXPECTED_BENCHMARKS, REQUIRED_METHODS
from src.eval.ink_advection_diffusion_diagnostics import generate_splits_from_config
from src.train.common import load_config, save_json


DEFAULT_OUTPUT_DIR = "paper/visuals"
DEFAULT_RESULT_ROOT = "results/full_run"
DEFAULT_INK_CONFIG = "configs/ink_advection_diffusion_full.yaml"

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

BENCHMARK_LABELS = {
    "sta": "STA-Bench",
    "ink_advection_diffusion": "Ink Advection-Diffusion",
}

PALETTE = {
    "core": "#2F6FBB",
    "spur": "#C44E52",
    "cf": "#8172B2",
    "neutral": "#4D4D4D",
    "light": "#F3F4F6",
    "itm": "#1B9E77",
    "sid": "#7570B3",
    "sib": "#D95F02",
    "baseline": "#6B7280",
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
            "figure.titlesize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": 350,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _save_figure(fig: plt.Figure, output: Path, source: Mapping[str, Any]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    pdf_output = output.with_suffix(".pdf")
    fig.savefig(pdf_output)
    plt.close(fig)
    save_json(
        output.with_suffix(".json"),
        {
            "png": str(output),
            "pdf": str(pdf_output),
            **dict(source),
        },
    )


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _metric_row(payload: Mapping[str, Any], method: str) -> dict[str, float | int | None]:
    row = payload.get(method, {})
    if not isinstance(row, Mapping):
        row = {}
    if method in {"ocp_style", "lens_like_arrow_classifier"}:
        return {
            "n_runs": row.get("n_runs"),
            "iid": row.get("fine_tuned_encoder_iid_test_accuracy_mean"),
            "iid_std": row.get("fine_tuned_encoder_iid_test_accuracy_std"),
            "ood": row.get("fine_tuned_encoder_ood_test_accuracy_mean"),
            "ood_std": row.get("fine_tuned_encoder_ood_test_accuracy_std"),
            "gap": row.get("fine_tuned_encoder_ood_gap_mean"),
            "gap_std": row.get("fine_tuned_encoder_ood_gap_std"),
        }
    return {
        "n_runs": row.get("n_runs"),
        "iid": row.get("iid_test_accuracy_mean"),
        "iid_std": row.get("iid_test_accuracy_std"),
        "ood": row.get("ood_test_accuracy_mean"),
        "ood_std": row.get("ood_test_accuracy_std"),
        "gap": row.get("ood_gap_mean"),
        "gap_std": row.get("ood_gap_std"),
    }


def _method_payloads(aggregate: Mapping[str, Any]) -> dict[str, Any]:
    top_level = {
        method: aggregate[method]
        for method in REQUIRED_METHODS
        if isinstance(aggregate.get(method), Mapping)
    }
    if top_level:
        return top_level
    by_condition = aggregate.get("by_condition", {})
    if isinstance(by_condition, Mapping):
        for _, payload in sorted(by_condition.items()):
            if not isinstance(payload, Mapping):
                continue
            condition_methods = {
                method: payload[method]
                for method in REQUIRED_METHODS
                if isinstance(payload.get(method), Mapping)
            }
            if condition_methods:
                return condition_methods
    return {}


def _load_result_rows(result_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for benchmark in EXPECTED_BENCHMARKS:
        aggregate_path = result_root / benchmark / "aggregate.json"
        aggregate = _method_payloads(_load_json(aggregate_path))
        for method in REQUIRED_METHODS:
            metrics = _metric_row(aggregate, method)
            rows.append(
                {
                    "benchmark": benchmark,
                    "method": method,
                    "label": METHOD_LABELS.get(method, method),
                    **metrics,
                }
            )
    interpretation_path = result_root / "result_interpretation.json"
    interpretation = _load_json(interpretation_path) if interpretation_path.exists() else {}
    return rows, interpretation


def _figure_results_summary(result_root: Path, output_dir: Path) -> None:
    rows, interpretation = _load_result_rows(result_root)
    _set_style()
    benchmarks = list(EXPECTED_BENCHMARKS)
    fig, axes = plt.subplots(1, len(benchmarks), figsize=(7.2, 4.1), sharey=True)
    if len(benchmarks) == 1:
        axes = np.asarray([axes])
    methods = list(REQUIRED_METHODS)
    y_positions = np.arange(len(methods))[::-1]
    label_by_method = [METHOD_LABELS.get(method, method) for method in methods]
    claim_gates = {
        gate.get("benchmark"): gate for gate in interpretation.get("benchmark_gates", [])
        if isinstance(gate, Mapping)
    }

    for ax, benchmark in zip(axes, benchmarks, strict=True):
        bench_rows = {row["method"]: row for row in rows if row["benchmark"] == benchmark}
        for idx, method in enumerate(methods):
            row = bench_rows[method]
            y = y_positions[idx]
            iid = row.get("iid")
            ood = row.get("ood")
            iid_std = row.get("iid_std") or 0.0
            ood_std = row.get("ood_std") or 0.0
            color = (
                PALETTE["itm"]
                if method == "itm"
                else PALETTE["sid"]
                if method == "sid"
                else PALETTE["sib"]
                if method == "sib"
                else PALETTE["baseline"]
            )
            if iid is not None:
                ax.errorbar(
                    float(iid),
                    y + 0.13,
                    xerr=float(iid_std),
                    marker="o",
                    markersize=4,
                    color="#9CA3AF",
                    ecolor="#D1D5DB",
                    capsize=2,
                    linewidth=1,
                    label="IID test" if idx == 0 and benchmark == benchmarks[0] else None,
                )
            if ood is not None:
                ax.errorbar(
                    float(ood),
                    y - 0.13,
                    xerr=float(ood_std),
                    marker="o",
                    markersize=5.5 if method in {"itm", "sid", "sib"} else 4.5,
                    color=color,
                    ecolor=color,
                    capsize=2,
                    linewidth=1.2,
                    label="OOD test" if idx == 0 and benchmark == benchmarks[0] else None,
                )
                if method == "itm":
                    ax.scatter(
                        [float(ood)],
                        [y - 0.13],
                        s=75,
                        facecolors="none",
                        edgecolors="#111827",
                        linewidths=1.1,
                        zorder=4,
                    )
        gate = claim_gates.get(benchmark, {})
        gate_text = "claim gate passed" if gate.get("passed") is True else "claim gate not passed"
        ax.text(
            0.03,
            0.04,
            gate_text,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8,
            color="#374151",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "#F9FAFB", "edgecolor": "#D1D5DB"},
        )
        ax.set_title(BENCHMARK_LABELS.get(benchmark, benchmark))
        ax.set_xlim(0.0, 1.03)
        ax.set_xlabel("Accuracy")
        ax.grid(axis="x", color="#E5E7EB", linewidth=0.8)
        ax.set_axisbelow(True)
    axes[0].set_yticks(y_positions)
    axes[0].set_yticklabels(label_by_method)
    axes[0].set_ylabel("Method")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    _save_figure(
        fig,
        output_dir / "results_summary.png",
        {
            "kind": "results_summary",
            "result_root": str(result_root),
            "rows": rows,
            "claim_mode": interpretation.get("claim_mode"),
            "primary_method": interpretation.get("primary_method"),
        },
    )


def _plot_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    *,
    color: str,
    text_color: str = "#111827",
) -> patches.FancyBboxPatch:
    box = patches.FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.018",
        facecolor=color,
        edgecolor="#111827",
        linewidth=0.8,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        color=text_color,
        fontsize=8.5,
        linespacing=1.2,
    )
    return box


def _arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = "#111827",
    linestyle: str = "-",
) -> None:
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={
            "arrowstyle": "-|>",
            "lw": 1.1,
            "color": color,
            "linestyle": linestyle,
            "shrinkA": 2,
            "shrinkB": 2,
        },
    )


def _figure_mechanism_schematic(output_dir: Path) -> None:
    _set_style()
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    _plot_box(
        ax,
        (0.05, 0.68),
        0.22,
        0.16,
        "task-relevant\nirreversible process",
        color="#E8F1FB",
    )
    _plot_box(
        ax,
        (0.05, 0.29),
        0.22,
        0.16,
        "nuisance\nirreversible process",
        color="#FBE9EA",
    )
    _plot_box(ax, (0.39, 0.52), 0.22, 0.18, "observed\nsequence", color="#F3F4F6")
    _plot_box(ax, (0.72, 0.62), 0.22, 0.16, "stable task\nmechanism", color="#E8F7F1")
    _plot_box(
        ax,
        (0.72, 0.27),
        0.22,
        0.16,
        "spurious arrow\nshortcut",
        color="#FFF1D6",
    )

    _arrow(ax, (0.27, 0.76), (0.39, 0.62), color=PALETTE["core"])
    _arrow(ax, (0.27, 0.37), (0.39, 0.58), color=PALETTE["spur"])
    _arrow(ax, (0.61, 0.61), (0.72, 0.70), color=PALETTE["core"])
    _arrow(ax, (0.61, 0.57), (0.72, 0.35), color=PALETTE["spur"])
    _arrow(ax, (0.83, 0.62), (0.83, 0.44), color="#6B7280", linestyle="--")

    ax.text(0.16, 0.90, "core sets label", ha="center", color=PALETTE["core"], fontsize=8)
    ax.text(
        0.16,
        0.18,
        "correlated in train/IID,\nreversed or removed in OOD",
        ha="center",
        color=PALETTE["spur"],
        fontsize=7.6,
    )
    ax.text(
        0.50,
        0.38,
        "counterfactual: keep core,\nresample nuisance",
        ha="center",
        color=PALETTE["cf"],
        fontsize=8,
    )
    ax.text(
        0.83,
        0.17,
        "desired behavior: use stable mechanism;\ndo not trust the strongest arrow by default",
        ha="center",
        color="#374151",
        fontsize=8,
    )
    fig.tight_layout()
    _save_figure(
        fig,
        output_dir / "mechanism_schematic.png",
        {
            "kind": "mechanism_schematic",
            "description": "Conceptual schematic for the spurious-arrow setup.",
        },
    )


def _compact_ink_config(config: dict[str, Any], sample_n: int) -> dict[str, Any]:
    cfg = deepcopy(config)
    cfg.setdefault("splits", {})
    for split, mode in {
        "train": "correlated",
        "val_iid": "correlated",
        "iid_test": "correlated",
        "ood_test": "reversed",
    }.items():
        cfg["splits"][split] = {"n_sequences": int(sample_n), "spurious_mode": mode}
    return cfg


def _select_visual_example(split: Mapping[str, Any], label: int = 1) -> int:
    y = np.asarray(split["y"])
    idxs = np.flatnonzero(y == label)
    if idxs.size == 0:
        idxs = np.arange(y.shape[0])
    core_score = np.asarray(split["core_score"])
    return int(idxs[np.argmax(np.abs(core_score[idxs]))])


def _robust_limits(arrays: list[np.ndarray]) -> tuple[float, float]:
    flat = np.concatenate([np.asarray(arr).reshape(-1) for arr in arrays])
    lo = float(np.percentile(flat, 0.5))
    hi = float(np.percentile(flat, 99.7))
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


def _annotate_source_and_flow(
    ax: plt.Axes,
    source: np.ndarray | None,
    flow: float | None,
    *,
    color: str = "white",
) -> None:
    if source is not None:
        ax.scatter(
            [float(source[0])],
            [float(source[1])],
            marker="x",
            s=35,
            linewidths=1.4,
            color=color,
            zorder=4,
        )
    if flow is not None:
        x0, x1 = (0.18, 0.36) if flow > 0 else (0.36, 0.18)
        ax.annotate(
            "",
            xy=(x1, 0.12),
            xytext=(x0, 0.12),
            xycoords="axes fraction",
            arrowprops={"arrowstyle": "-|>", "lw": 1.4, "color": color},
        )


def _figure_ink_decomposition(config_path: Path, output_dir: Path, sample_n: int) -> None:
    config = _compact_ink_config(load_config(config_path), sample_n)
    splits = generate_splits_from_config(config, return_fields=True)
    frames = [0, 5, 10, 15]
    _set_style()
    fig = plt.figure(figsize=(7.25, 8.45))
    outer = gridspec.GridSpec(2, 1, hspace=0.30, figure=fig)
    row_labels = [
        "observed x",
        "core only",
        "nuisance only",
        "counterfactual x_cf",
        "|x - x_cf|",
    ]
    examples = [("train", "train/IID shortcut"), ("ood_test", "OOD reversal")]
    source_rows: list[dict[str, Any]] = []

    for block, (split_name, title) in enumerate(examples):
        split = splits[split_name]
        idx = _select_visual_example(split, label=1)
        obs = split["x"][idx].reshape(split["metadata"]["obs_shape"])
        obs_cf = split["x_cf"][idx].reshape(split["metadata"]["obs_shape"])
        core = split["core_field"][idx]
        spur = split["spurious_field"][idx]
        delta = np.abs(obs - obs_cf)
        source = split["core_source"][idx]
        flow = float(split["spurious_dynamic_stat"][idx])
        cf_flow = float(split["spurious_cf_dynamic_stat"][idx])
        frames_clipped = [min(frame, obs.shape[0] - 1) for frame in frames]
        vmin, vmax = _robust_limits([obs, obs_cf, core, spur])
        dmin, dmax = _robust_limits([delta])
        inner = gridspec.GridSpecFromSubplotSpec(
            len(row_labels),
            len(frames_clipped),
            subplot_spec=outer[block],
            wspace=0.03,
            hspace=0.08,
        )
        for row_idx, label in enumerate(row_labels):
            for col_idx, frame in enumerate(frames_clipped):
                ax = fig.add_subplot(inner[row_idx, col_idx])
                arr = {
                    "observed x": obs,
                    "core only": core,
                    "nuisance only": spur,
                    "counterfactual x_cf": obs_cf,
                    "|x - x_cf|": delta,
                }[label]
                if label == "|x - x_cf|":
                    ax.imshow(arr[frame], cmap="viridis", vmin=dmin, vmax=dmax, interpolation="nearest")
                else:
                    ax.imshow(arr[frame], cmap="magma", vmin=vmin, vmax=vmax, interpolation="nearest")
                if row_idx == 0:
                    ax.set_title(f"t={frame}")
                if col_idx == 0:
                    ax.set_ylabel(label, rotation=0, ha="right", va="center", labelpad=42)
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                if label in {"observed x", "core only", "counterfactual x_cf"}:
                    _annotate_source_and_flow(ax, source, None, color="white")
                if label == "nuisance only":
                    _annotate_source_and_flow(ax, None, flow, color="white")
                if label == "counterfactual x_cf":
                    _annotate_source_and_flow(ax, None, cf_flow, color="cyan")
        split_meta = split["metadata"]
        fig.text(
            0.5,
            [0.982, 0.515][block],
            (
                f"{title}: y={int(split['y'][idx])}, core source x={source[0]:.1f}, "
                f"nuisance flow={flow:+.2f}, cf flow={cf_flow:+.2f}"
            ),
            ha="center",
            va="top",
            fontsize=9.5,
            color="#111827",
        )
        source_rows.append(
            {
                "split": split_name,
                "index": idx,
                "label": int(split["y"][idx]),
                "core_source": [float(source[0]), float(source[1])],
                "spurious_flow": flow,
                "spurious_cf_flow": cf_flow,
                "corr_y_spurious_dynamic_stat": split_meta["spurious"][
                    "corr_y_spurious_dynamic_stat"
                ],
                "corr_y_spurious_cf_dynamic_stat": split_meta["counterfactual"][
                    "corr_y_spurious_cf_dynamic_stat"
                ],
            }
        )
    _save_figure(
        fig,
        output_dir / "ink_advection_diffusion_decomposition.png",
        {
            "kind": "ink_advection_diffusion_decomposition",
            "config_path": str(config_path),
            "sample_n_per_split": int(sample_n),
            "examples": source_rows,
            "note": (
                "Figure is generated from the benchmark generator with full-study "
                "physics parameters and reduced split counts for visualization only."
            ),
        },
    )


def prepare_paper_figures(
    *,
    result_root: str | Path = DEFAULT_RESULT_ROOT,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    ink_config: str | Path = DEFAULT_INK_CONFIG,
    sample_n: int = 192,
) -> dict[str, Any]:
    result_root = Path(result_root)
    output_dir = Path(output_dir)
    ink_config = Path(ink_config)
    output_dir.mkdir(parents=True, exist_ok=True)
    _figure_mechanism_schematic(output_dir)
    _figure_ink_decomposition(ink_config, output_dir, sample_n)
    _figure_results_summary(result_root, output_dir)
    manifest = {
        "schema_version": 1,
        "output_dir": str(output_dir),
        "result_root": str(result_root),
        "ink_config": str(ink_config),
        "figures": [
            "mechanism_schematic.png",
            "ink_advection_diffusion_decomposition.png",
            "results_summary.png",
        ],
    }
    save_json(output_dir / "paper_figures_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper-oriented final-study figures.")
    parser.add_argument("--result-root", default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ink-config", default=DEFAULT_INK_CONFIG)
    parser.add_argument("--sample-n", type=int, default=192)
    args = parser.parse_args()
    manifest = prepare_paper_figures(
        result_root=args.result_root,
        output_dir=args.output_dir,
        ink_config=args.ink_config,
        sample_n=args.sample_n,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
