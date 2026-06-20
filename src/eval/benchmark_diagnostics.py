"""Smoke diagnostics for irreversible source inference."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data.irreversible_source_inference import (
    IrreversibleSourceConfig,
    IrreversibleSourceSplit,
    generate_irreversible_source_splits,
    load_config,
    safe_corr,
)


GATE_TARGETS = {
    "final_frame_core_oracle_accuracy": ("<=", 0.72),
    "full_sequence_core_oracle_accuracy": (">=", 0.80),
    "core_only_ood_accuracy": (">=", 0.80),
    "nuisance_only_iid_accuracy": (">=", 0.80),
    "mixed_feature_probe_ood_gap": (">=", 0.20),
    "abs_corr_y_nuisance_arrow_train": (">=", 0.70),
    "static_feature_accuracy": ("<=", 0.65),
    "core_forward_reverse_arrow_accuracy": (">=", 0.80),
    "nuisance_forward_reverse_arrow_accuracy": (">=", 0.80),
}


def run_diagnostics(
    config: IrreversibleSourceConfig,
    out_dir: str | Path | None = None,
) -> dict[str, Any]:
    splits = generate_irreversible_source_splits(config)
    metrics: dict[str, Any] = {}

    metrics.update(dynamic_correlation_metrics(splits))

    train = splits["train"]
    iid = splits["iid_test"]
    ood = splits["ood_test"]

    final_core_model = fit_classifier(core_shape_features(train.core_only[:, -1:]), train.y)
    metrics["final_frame_core_oracle_accuracy"] = evaluate_classifier(
        final_core_model, core_shape_features(iid.core_only[:, -1:]), iid.y
    )

    core_model = fit_classifier(core_shape_features(train.core_only), train.y)
    metrics["full_sequence_core_oracle_accuracy"] = evaluate_classifier(
        core_model, core_shape_features(iid.core_only), iid.y
    )
    metrics["core_only_ood_accuracy"] = evaluate_classifier(
        core_model, core_shape_features(ood.core_only), ood.y
    )

    nuisance_model = fit_classifier(motion_arrow_features(train.nuisance_only), train.y)
    metrics["nuisance_only_iid_accuracy"] = evaluate_classifier(
        nuisance_model, motion_arrow_features(iid.nuisance_only), iid.y
    )
    metrics["nuisance_only_ood_accuracy"] = evaluate_classifier(
        nuisance_model, motion_arrow_features(ood.nuisance_only), ood.y
    )

    mixed_model = fit_classifier(mixed_dynamic_features(train.mixed), train.y)
    metrics["mixed_feature_probe_iid_accuracy"] = evaluate_classifier(
        mixed_model, mixed_dynamic_features(iid.mixed), iid.y
    )
    metrics["mixed_feature_probe_ood_accuracy"] = evaluate_classifier(
        mixed_model, mixed_dynamic_features(ood.mixed), ood.y
    )
    metrics["mixed_feature_probe_ood_gap"] = (
        metrics["mixed_feature_probe_iid_accuracy"]
        - metrics["mixed_feature_probe_ood_accuracy"]
    )
    reversed_model = fit_classifier(mixed_dynamic_features(time_reverse(train.mixed)), train.y)
    metrics["time_reversed_iid_accuracy"] = evaluate_classifier(
        reversed_model, mixed_dynamic_features(time_reverse(iid.mixed)), iid.y
    )
    metrics["time_reversed_ood_accuracy"] = evaluate_classifier(
        reversed_model, mixed_dynamic_features(time_reverse(ood.mixed)), ood.y
    )
    static_model = fit_classifier(static_features(train.mixed), train.y)
    metrics["static_feature_accuracy"] = evaluate_classifier(
        static_model, static_features(iid.mixed), iid.y
    )
    final_nuisance_model = fit_classifier(final_frame(train.nuisance_only), train.y)
    metrics["final_nuisance_frame_iid_accuracy"] = evaluate_classifier(
        final_nuisance_model, final_frame(iid.nuisance_only), iid.y
    )
    metrics["final_nuisance_frame_ood_accuracy"] = evaluate_classifier(
        final_nuisance_model, final_frame(ood.nuisance_only), ood.y
    )
    metrics["final_nuisance_frame_ood_gap"] = (
        metrics["final_nuisance_frame_iid_accuracy"]
        - metrics["final_nuisance_frame_ood_accuracy"]
    )

    metrics["core_forward_reverse_arrow_accuracy"] = forward_reverse_accuracy(
        train.core_only, mixed_dynamic_features
    )
    metrics["nuisance_forward_reverse_arrow_accuracy"] = forward_reverse_accuracy(
        train.nuisance_only, motion_arrow_features
    )
    metrics["counterfactual_core_residual_max_abs"] = counterfactual_core_residual_max_abs(train)
    metrics["counterfactual_preserves_core"] = (
        metrics["counterfactual_core_residual_max_abs"] < 1e-6
    )
    metrics["counterfactual_changed_fraction"] = float(
        np.mean(train.nuisance_direction != train.counterfactual_direction)
    )
    if config.counterfactual_mode == "randomized":
        metrics["counterfactual_changes_nuisance"] = bool(
            metrics["counterfactual_changed_fraction"] > 0.35
        )
    else:
        metrics["counterfactual_changes_nuisance"] = bool(
            metrics["counterfactual_changed_fraction"] > 0.95
        )

    gate = evaluate_gate(metrics, config)
    diagnostics = {
        "config": asdict(config),
        "metrics": round_floats(metrics),
        "gate": gate,
        "metadata": {name: split.metadata for name, split in splits.items()},
    }

    if out_dir is not None:
        write_outputs(Path(out_dir), diagnostics)
    return diagnostics


def dynamic_correlation_metrics(splits: dict[str, IrreversibleSourceSplit]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for name, split in splits.items():
        corr = safe_corr(split.y.astype(float), split.nuisance_direction.astype(float))
        realized_arrow = motion_arrow_features(split.nuisance_only)[:, 0]
        cf_corr = safe_corr(
            split.y.astype(float),
            split.counterfactual_direction.astype(float),
        )
        realized_corr = safe_corr(split.y.astype(float), realized_arrow.astype(float))
        metrics[f"corr_y_nuisance_arrow_{name}"] = corr
        metrics[f"abs_corr_y_nuisance_arrow_{name}"] = abs(corr)
        metrics[f"auc_y_from_nuisance_arrow_{name}"] = safe_auc(
            split.y, split.nuisance_direction
        )
        metrics[f"corr_y_realized_nuisance_motion_{name}"] = realized_corr
        metrics[f"abs_corr_y_realized_nuisance_motion_{name}"] = abs(realized_corr)
        metrics[f"auc_y_from_realized_nuisance_motion_{name}"] = safe_auc(
            split.y, realized_arrow
        )
        metrics[f"corr_y_counterfactual_arrow_{name}"] = cf_corr
        metrics[f"abs_corr_y_counterfactual_arrow_{name}"] = abs(cf_corr)
        metrics[f"mean_nuisance_arrow_y0_{name}"] = float(
            split.nuisance_direction[split.y == 0].mean()
        )
        metrics[f"mean_nuisance_arrow_y1_{name}"] = float(
            split.nuisance_direction[split.y == 1].mean()
        )
    metrics["abs_corr_y_nuisance_arrow_train"] = metrics["abs_corr_y_nuisance_arrow_train"]
    metrics["abs_corr_y_nuisance_arrow_ood"] = metrics["abs_corr_y_nuisance_arrow_ood_test"]
    metrics["abs_corr_y_realized_nuisance_motion_train"] = metrics[
        "abs_corr_y_realized_nuisance_motion_train"
    ]
    return metrics


def fit_classifier(x: np.ndarray, y: np.ndarray) -> Any:
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, solver="liblinear", random_state=0),
    )
    model.fit(x, y)
    return model


def evaluate_classifier(model: Any, x: np.ndarray, y: np.ndarray) -> float:
    return float(accuracy_score(y, model.predict(x)))


def flatten_sequence(x: np.ndarray) -> np.ndarray:
    return x.reshape(x.shape[0], -1)


def final_frame(x: np.ndarray) -> np.ndarray:
    return x[:, -1].reshape(x.shape[0], -1)


def core_view(x: np.ndarray) -> np.ndarray:
    if x.ndim == 5:
        return x[:, :, 0]
    return x


def nuisance_view(x: np.ndarray) -> np.ndarray:
    if x.ndim == 5:
        return x[:, :, -1]
    return x


def core_shape_features(x: np.ndarray) -> np.ndarray:
    """Translation-invariant anisotropy features for diffused source shape."""
    x = core_view(x)
    mass = positive_mass(x)
    row_grid, col_grid = coordinate_grids(x.shape[-2], x.shape[-1])
    row_center, col_center = circular_center(mass, row_grid, col_grid)
    row_dist, col_dist = centered_distances(row_grid, col_grid, row_center, col_center)
    total = mass.sum(axis=(2, 3)).clip(min=1e-8)
    var_row = (mass * row_dist[:, :, :, None] ** 2).sum(axis=(2, 3)) / total
    var_col = (mass * col_dist[:, :, None, :] ** 2).sum(axis=(2, 3)) / total
    anisotropy = var_col - var_row
    max_value = mass.max(axis=(2, 3))
    return np.concatenate([anisotropy, var_row, var_col, max_value], axis=1)


def motion_arrow_features(x: np.ndarray) -> np.ndarray:
    """Direction-sensitive center-of-mass motion features."""
    x = nuisance_view(x)
    mass = positive_mass(x)
    row_grid, col_grid = coordinate_grids(x.shape[-2], x.shape[-1])
    row_center, col_center = circular_center(mass, row_grid, col_grid)
    col_unwrapped = np.unwrap(col_center * 2.0 * np.pi / x.shape[-1], axis=1)
    row_unwrapped = np.unwrap(row_center * 2.0 * np.pi / x.shape[-2], axis=1)
    t = np.arange(x.shape[1], dtype=np.float64)
    t = t - t.mean()
    denom = float(np.sum(t**2))
    col_slope = (col_unwrapped * t[None, :]).sum(axis=1) / denom
    row_slope = (row_unwrapped * t[None, :]).sum(axis=1) / denom
    diffs = np.diff(col_unwrapped, axis=1)
    total = mass.sum(axis=(2, 3))
    total_slope = (total * t[None, :]).sum(axis=1) / denom
    return np.concatenate(
        [
            col_slope[:, None],
            row_slope[:, None],
            diffs.mean(axis=1, keepdims=True),
            diffs.std(axis=1, keepdims=True),
            total.mean(axis=1, keepdims=True),
            total.std(axis=1, keepdims=True),
            total_slope[:, None],
        ],
        axis=1,
    )


def mixed_dynamic_features(x: np.ndarray) -> np.ndarray:
    return np.concatenate([core_shape_features(x), motion_arrow_features(x)], axis=1)


def positive_mass(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    min_per_sample = x.reshape(x.shape[0], -1).min(axis=1)[:, None, None, None]
    return x - min_per_sample + 1e-8


def coordinate_grids(rows: int, cols: int) -> tuple[np.ndarray, np.ndarray]:
    row_grid = np.arange(rows, dtype=np.float64)
    col_grid = np.arange(cols, dtype=np.float64)
    return row_grid, col_grid


def circular_center(
    mass: np.ndarray,
    row_grid: np.ndarray,
    col_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    row_angles = 2.0 * np.pi * row_grid / len(row_grid)
    col_angles = 2.0 * np.pi * col_grid / len(col_grid)
    row_complex = (mass * np.exp(1j * row_angles)[None, None, :, None]).sum(axis=(2, 3))
    col_complex = (mass * np.exp(1j * col_angles)[None, None, None, :]).sum(axis=(2, 3))
    row_center = np.mod(np.angle(row_complex), 2.0 * np.pi) * len(row_grid) / (2.0 * np.pi)
    col_center = np.mod(np.angle(col_complex), 2.0 * np.pi) * len(col_grid) / (2.0 * np.pi)
    return row_center, col_center


def centered_distances(
    row_grid: np.ndarray,
    col_grid: np.ndarray,
    row_center: np.ndarray,
    col_center: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rows = len(row_grid)
    cols = len(col_grid)
    row_raw = row_grid[None, None, :] - row_center[:, :, None]
    col_raw = col_grid[None, None, :] - col_center[:, :, None]
    row_dist = (row_raw + rows / 2.0) % rows - rows / 2.0
    col_dist = (col_raw + cols / 2.0) % cols - cols / 2.0
    return row_dist, col_dist


def time_reverse(x: np.ndarray) -> np.ndarray:
    return x[:, ::-1].copy()


def static_features(x: np.ndarray) -> np.ndarray:
    final = x[:, -1]
    if x.ndim == 4:
        total = x.sum(axis=(2, 3))
        mean = x.mean(axis=(2, 3))
        std = x.std(axis=(2, 3))
    elif x.ndim == 5:
        total = x.sum(axis=(3, 4)).reshape(x.shape[0], -1)
        mean = x.mean(axis=(3, 4)).reshape(x.shape[0], -1)
        std = x.std(axis=(3, 4)).reshape(x.shape[0], -1)
    else:
        raise ValueError(f"expected 4D or 5D input, got shape {x.shape}")
    return np.concatenate(
        [
            final.reshape(x.shape[0], -1),
            total,
            mean,
            std,
        ],
        axis=1,
    )


def forward_reverse_accuracy(x: np.ndarray, feature_fn: Any) -> float:
    n = x.shape[0]
    forward = feature_fn(x)
    reverse = feature_fn(x[:, ::-1].copy())
    y = np.concatenate([np.ones(n, dtype=np.int64), np.zeros(n, dtype=np.int64)])
    features = np.concatenate([forward, reverse], axis=0)
    idx = np.arange(len(y))
    train_idx = idx[idx % 2 == 0]
    test_idx = idx[idx % 2 == 1]
    model = fit_classifier(features[train_idx], y[train_idx])
    return evaluate_classifier(model, features[test_idx], y[test_idx])


def counterfactual_core_residual_max_abs(split: IrreversibleSourceSplit) -> float:
    if split.metadata.get("disable_nuisance"):
        return float(np.max(np.abs(split.mixed - split.counterfactual)))
    core_residual = split.mixed - split.counterfactual
    nuisance_residual = split.nuisance_only - split.nuisance_counterfactual
    if core_residual.ndim == 5:
        core_channel_delta = float(np.max(np.abs(core_residual[:, :, 0])))
        nuisance_channel_delta = core_residual[:, :, 1] / split.metadata["nuisance_scale"]
        nuisance_delta_error = float(np.max(np.abs(nuisance_channel_delta - nuisance_residual)))
        return max(core_channel_delta, nuisance_delta_error)
    # Since mixed and counterfactual reuse the same core and observation noise,
    # their difference must be exactly explained by nuisance replacement up to
    # the known nuisance scale.
    scaled = core_residual / split.metadata["nuisance_scale"]
    return float(np.max(np.abs(scaled - nuisance_residual)))


def safe_auc(y: np.ndarray, score: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y, score))
    except ValueError:
        return 0.5


def evaluate_gate(metrics: dict[str, Any], config: IrreversibleSourceConfig) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    passed = True
    targets = dict(GATE_TARGETS)
    if config.ood_mode == "reversed":
        targets["corr_y_nuisance_arrow_ood_test"] = ("<=", -0.70)
        targets["corr_y_realized_nuisance_motion_ood_test"] = ("<=", -0.55)
        targets["nuisance_only_ood_accuracy"] = ("<=", 0.40)
    elif config.ood_mode == "randomized":
        targets["abs_corr_y_nuisance_arrow_ood_test"] = ("<=", 0.20)
        targets["abs_corr_y_realized_nuisance_motion_ood_test"] = ("<=", 0.30)
        targets["nuisance_only_ood_accuracy"] = ("<=", 0.65)
    elif config.ood_mode == "partial_shift":
        target = config.partial_shift_target_correlation
        metrics["partial_shift_corr_error"] = abs(
            float(metrics["corr_y_nuisance_arrow_ood_test"]) - target
        )
        targets["partial_shift_corr_error"] = ("<=", 0.20)
        targets["nuisance_only_ood_accuracy"] = ("<=", 0.75)
    else:
        raise ValueError(f"unknown ood_mode {config.ood_mode!r}")
    if config.benchmark_variant == "endpoint_matched":
        targets["final_nuisance_frame_iid_accuracy"] = ("<=", 0.65)
        targets["final_nuisance_frame_ood_gap"] = ("<=", 0.15)

    for key, (op, threshold) in targets.items():
        value = float(metrics[key])
        if op == ">=":
            ok = value >= threshold
        elif op == "<=":
            ok = value <= threshold
        else:
            raise ValueError(op)
        checks[key] = {"value": value, "op": op, "threshold": threshold, "passed": ok}
        passed = passed and ok
    for key in ("counterfactual_preserves_core", "counterfactual_changes_nuisance"):
        ok = bool(metrics[key])
        checks[key] = {"value": ok, "op": "is", "threshold": True, "passed": ok}
        passed = passed and ok
    if config.counterfactual_mode == "randomized":
        value = float(metrics["abs_corr_y_counterfactual_arrow_train"])
        ok = value <= 0.30
        checks["abs_corr_y_counterfactual_arrow_train"] = {
            "value": value,
            "op": "<=",
            "threshold": 0.30,
            "passed": ok,
        }
        passed = passed and ok
    return {"passed": passed, "checks": checks}


def round_floats(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: round_floats(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [round_floats(v) for v in obj]
    if isinstance(obj, float):
        return round(obj, 6)
    return obj


def write_outputs(out_dir: Path, diagnostics: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "diagnostics.json").open("w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2)
    with (out_dir / "benchmark_gate.json").open("w", encoding="utf-8") as f:
        json.dump(diagnostics["gate"], f, indent=2)
    with (out_dir / "candidate_trials.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"config": diagnostics["config"], "gate": diagnostics["gate"]}) + "\n")
    write_smoke_report(out_dir / "smoke_report.md", diagnostics)


def write_smoke_report(path: Path, diagnostics: dict[str, Any]) -> None:
    gate = diagnostics["gate"]
    metrics = diagnostics["metrics"]
    lines = [
        "# Smoke Benchmark Report",
        "",
        f"Overall gate passed: `{gate['passed']}`",
        "",
        "## Gate Checks",
        "",
        "| Metric | Value | Rule | Pass |",
        "|---|---:|---|---|",
    ]
    for key, check in gate["checks"].items():
        lines.append(
            f"| `{key}` | `{check['value']}` | `{check['op']} {check['threshold']}` | `{check['passed']}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This smoke run is a benchmark diagnostic, not a method claim.",
            "",
            "Key metrics:",
            "",
            f"- final-frame core oracle: `{metrics['final_frame_core_oracle_accuracy']}`",
            f"- full-sequence core oracle: `{metrics['full_sequence_core_oracle_accuracy']}`",
            f"- nuisance-only IID/OOD: `{metrics['nuisance_only_iid_accuracy']}` / `{metrics['nuisance_only_ood_accuracy']}`",
            f"- mixed feature probe IID/OOD: `{metrics['mixed_feature_probe_iid_accuracy']}` / `{metrics['mixed_feature_probe_ood_accuracy']}`",
            f"- mixed feature-probe OOD gap: `{metrics['mixed_feature_probe_ood_gap']}`",
            f"- final nuisance frame IID/OOD: `{metrics['final_nuisance_frame_iid_accuracy']}` / `{metrics['final_nuisance_frame_ood_accuracy']}`",
            f"- realized nuisance-motion corr train/OOD: `{metrics['corr_y_realized_nuisance_motion_train']}` / `{metrics['corr_y_realized_nuisance_motion_ood_test']}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    diagnostics = run_diagnostics(config, args.out)
    print(json.dumps(diagnostics["gate"], indent=2))


if __name__ == "__main__":
    main()
