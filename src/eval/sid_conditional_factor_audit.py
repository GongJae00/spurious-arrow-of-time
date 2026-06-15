"""Conditional SID factor audit.

This audit complements ``sid_factor_audit``. Raw factor probes can be confounded
when label, core dynamics, and spurious dynamics are correlated in IID data.
Here we train probes on ``val_iid`` and report both raw and residualized
factor-target evidence on IID/OOD splits. The audit never selects checkpoints or
hyperparameters from OOD data.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.eval.sid_factor_audit import SID_FACTORS
from src.eval.sid_factor_audit import _encode_factors
from src.eval.sid_factor_audit import _load_json
from src.eval.sid_factor_audit import _load_sid_model
from src.eval.sid_factor_audit import _resolve_run_dir
from src.eval.sid_factor_audit import _target_arrays
from src.train.common import save_json
from src.utils.hardware import get_device


CONDITIONAL_AUDIT_SCHEMA = "sid_conditional_factor_audit_v1"
EVAL_SPLITS = ("iid_test", "ood_test")
TARGETS = ("label", "core_dynamic", "spurious_dynamic")
REPRESENTATIONS = (*SID_FACTORS, "task_rep")
DEFAULT_BENCHMARK_MANIFESTS = {
    "sta": "sta/manifest.json",
    "ink_advection_diffusion": "ink_advection_diffusion/manifest.json",
}


def orientation_free_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    """Return AUC ignoring arbitrary orientation of a binary probe score."""
    y_true = np.asarray(y_true, dtype=np.int64)
    score = np.asarray(score, dtype=np.float64)
    if np.unique(y_true).shape[0] < 2:
        return float("nan")
    auc = float(roc_auc_score(y_true, score))
    return max(auc, 1.0 - auc)


def _as_2d(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        return values[:, None]
    return values


def _with_intercept(values: np.ndarray) -> np.ndarray:
    values = _as_2d(values)
    return np.concatenate([np.ones((values.shape[0], 1)), values], axis=1)


def fit_residualizer(train_controls: np.ndarray, train_x: np.ndarray) -> np.ndarray:
    """Fit least-squares coefficients mapping controls to x."""
    controls = _with_intercept(train_controls)
    x = _as_2d(train_x)
    coef, *_ = np.linalg.lstsq(controls, x, rcond=None)
    return coef


def apply_residualizer(controls: np.ndarray, x: np.ndarray, coef: np.ndarray) -> np.ndarray:
    """Remove the linear component predicted from controls."""
    controls_i = _with_intercept(controls)
    x_2d = _as_2d(x)
    return x_2d - controls_i @ coef


def residualize_against_controls(
    train_x: np.ndarray,
    train_controls: np.ndarray,
    eval_x: np.ndarray,
    eval_controls: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    coef = fit_residualizer(train_controls, train_x)
    return (
        apply_residualizer(train_controls, train_x, coef),
        apply_residualizer(eval_controls, eval_x, coef),
    )


def _linear_probe_metrics(
    train_x: np.ndarray,
    train_y: np.ndarray,
    eval_x: np.ndarray,
    eval_y: np.ndarray,
) -> dict[str, float]:
    if np.unique(train_y).shape[0] < 2 or np.unique(eval_y).shape[0] < 2:
        return {
            "accuracy": float("nan"),
            "orientation_free_auc": float("nan"),
        }
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=500, C=1.0, solver="lbfgs"),
    )
    clf.fit(train_x, train_y)
    accuracy = float(clf.score(eval_x, eval_y))
    score = clf.predict_proba(eval_x)[:, 1]
    return {
        "accuracy": accuracy,
        "orientation_free_auc": orientation_free_auc(eval_y, score),
    }


def _continuous_target(split: Mapping[str, Any], target: str) -> np.ndarray:
    if target == "label":
        return np.asarray(split["y"], dtype=np.float64)
    if target == "core_dynamic":
        return np.asarray(split["core_dynamic_stat"], dtype=np.float64)
    if target == "spurious_dynamic":
        return np.asarray(split["spurious_dynamic_stat"], dtype=np.float64)
    raise ValueError(f"unknown target {target!r}")


def _controls_for_target(
    train_split: Mapping[str, Any],
    eval_split: Mapping[str, Any],
    target: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if target == "label":
        names = ["core_dynamic_stat", "spurious_dynamic_stat"]
        train = np.column_stack(
            [
                _continuous_target(train_split, "core_dynamic"),
                _continuous_target(train_split, "spurious_dynamic"),
            ]
        )
        eval_ = np.column_stack(
            [
                _continuous_target(eval_split, "core_dynamic"),
                _continuous_target(eval_split, "spurious_dynamic"),
            ]
        )
        return train, eval_, names
    if target == "core_dynamic":
        names = ["y", "spurious_dynamic_stat"]
        train = np.column_stack(
            [
                _continuous_target(train_split, "label"),
                _continuous_target(train_split, "spurious_dynamic"),
            ]
        )
        eval_ = np.column_stack(
            [
                _continuous_target(eval_split, "label"),
                _continuous_target(eval_split, "spurious_dynamic"),
            ]
        )
        return train, eval_, names
    if target == "spurious_dynamic":
        names = ["y", "core_dynamic_stat"]
        train = np.column_stack(
            [
                _continuous_target(train_split, "label"),
                _continuous_target(train_split, "core_dynamic"),
            ]
        )
        eval_ = np.column_stack(
            [
                _continuous_target(eval_split, "label"),
                _continuous_target(eval_split, "core_dynamic"),
            ]
        )
        return train, eval_, names
    raise ValueError(f"unknown target {target!r}")


def _task_rep(factors: Mapping[str, np.ndarray]) -> np.ndarray:
    return np.concatenate([factors["z_rev"], factors["z_ir_task"]], axis=1)


def _factor_representations(factors: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    reps = {factor: factors[factor] for factor in SID_FACTORS}
    reps["task_rep"] = _task_rep(factors)
    return reps


def _finite_mean(values: Iterable[float]) -> float:
    arr = np.asarray([value for value in values if math.isfinite(float(value))], dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    return float(arr.mean())


def _summarize_runs(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metric_values: dict[str, list[float]] = {}
    for row in rows:
        for split_name, split_metrics in row["metrics"].items():
            for rep_name, rep_metrics in split_metrics.items():
                for target_name, target_metrics in rep_metrics.items():
                    for metric_name, value in target_metrics.items():
                        if isinstance(value, (int, float)) and not isinstance(value, bool):
                            key = f"{split_name}.{rep_name}.{target_name}.{metric_name}"
                            metric_values.setdefault(key, []).append(float(value))
    return {
        key: {
            "mean": _finite_mean(values),
            "n": len(values),
        }
        for key, values in sorted(metric_values.items())
    }


def audit_sid_run_conditional(
    run_dir: str | Path,
    *,
    checkpoint: str = "best.pt",
    probe_train_split: str = "val_iid",
    eval_splits: tuple[str, ...] = EVAL_SPLITS,
    device: torch.device | None = None,
    save_local: bool = True,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    device = device or torch.device("cpu")
    model, config, splits = _load_sid_model(run_dir, checkpoint, device)
    pooling = str(config.get("model", {}).get("pooling", "last"))
    train_split = splits[probe_train_split]
    train_reps = _factor_representations(
        _encode_factors(model, train_split["x"], pooling, device)
    )

    metrics: dict[str, Any] = {}
    target_metadata: dict[str, Any] = {}

    for split_name in eval_splits:
        eval_split = splits[split_name]
        eval_reps = _factor_representations(
            _encode_factors(model, eval_split["x"], pooling, device)
        )
        split_metrics: dict[str, Any] = {}
        for rep_name in REPRESENTATIONS:
            rep_metrics: dict[str, Any] = {}
            for target in TARGETS:
                train_y, eval_y, meta = _target_arrays(train_split, eval_split, target)
                target_metadata.setdefault(target, meta)
                raw = _linear_probe_metrics(
                    train_reps[rep_name],
                    train_y,
                    eval_reps[rep_name],
                    eval_y,
                )
                train_controls, eval_controls, control_names = _controls_for_target(
                    train_split,
                    eval_split,
                    target,
                )
                train_resid, eval_resid = residualize_against_controls(
                    train_reps[rep_name],
                    train_controls,
                    eval_reps[rep_name],
                    eval_controls,
                )
                residualized = _linear_probe_metrics(
                    train_resid,
                    train_y,
                    eval_resid,
                    eval_y,
                )
                rep_metrics[target] = {
                    "raw_accuracy": raw["accuracy"],
                    "raw_orientation_free_auc": raw["orientation_free_auc"],
                    "residualized_accuracy": residualized["accuracy"],
                    "residualized_orientation_free_auc": residualized[
                        "orientation_free_auc"
                    ],
                    "controls": control_names,
                }
            split_metrics[rep_name] = rep_metrics
        metrics[split_name] = split_metrics

    report = {
        "schema": CONDITIONAL_AUDIT_SCHEMA,
        "passed": True,
        "method": "sid",
        "run_dir": str(run_dir),
        "checkpoint": checkpoint,
        "probe_train_split": probe_train_split,
        "eval_splits": list(eval_splits),
        "benchmark_name": splits["train"]["metadata"].get("benchmark_name"),
        "probe_capacity": "standardized_logistic_regression_C1_max_iter500",
        "target_metadata": target_metadata,
        "metrics": metrics,
        "interpretation_lock": (
            "Conditional probes are diagnostics for mechanism factorization. "
            "They do not select checkpoints and do not upgrade SID success claims."
        ),
    }
    if save_local:
        save_json(run_dir / "sid_conditional_factor_audit.json", report)
    return report


def _manifest_sid_runs(manifest_path: Path) -> list[Path]:
    manifest = _load_json(manifest_path)
    return [
        _resolve_run_dir(manifest_path, str(row["run_dir"]))
        for row in manifest.get("runs", [])
        if row.get("method") == "sid" and row.get("status") == "success"
    ]


def audit_result_root(
    result_root: str | Path,
    *,
    checkpoint: str = "best.pt",
    benchmarks: tuple[str, ...] | None = None,
    max_runs_per_benchmark: int | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    result_root = Path(result_root)
    selected = benchmarks or tuple(DEFAULT_BENCHMARK_MANIFESTS)
    benchmark_reports: dict[str, Any] = {}
    all_rows: list[dict[str, Any]] = []
    for benchmark in selected:
        rel_manifest = DEFAULT_BENCHMARK_MANIFESTS.get(benchmark)
        if rel_manifest is None:
            raise ValueError(f"unknown benchmark {benchmark!r}")
        manifest_path = result_root / rel_manifest
        if not manifest_path.exists():
            raise FileNotFoundError(manifest_path)
        run_dirs = _manifest_sid_runs(manifest_path)
        if max_runs_per_benchmark is not None:
            run_dirs = run_dirs[:max_runs_per_benchmark]
        rows = [
            audit_sid_run_conditional(
                run_dir,
                checkpoint=checkpoint,
                device=device,
            )
            for run_dir in run_dirs
        ]
        benchmark_reports[benchmark] = {
            "manifest": str(manifest_path),
            "n_runs": len(rows),
            "runs": rows,
            "aggregate": _summarize_runs(rows),
        }
        all_rows.extend(rows)
    return {
        "schema": CONDITIONAL_AUDIT_SCHEMA,
        "passed": True,
        "result_root": str(result_root),
        "checkpoint": checkpoint,
        "probe_train_split": "val_iid",
        "eval_splits": list(EVAL_SPLITS),
        "benchmarks": benchmark_reports,
        "aggregate": _summarize_runs(all_rows),
        "interpretation_lock": (
            "Use this audit to distinguish OOD robustness from mechanism "
            "factorization. Do not use it for OOD model selection."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run conditional SID factor audits.")
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument("--benchmark", action="append", choices=tuple(DEFAULT_BENCHMARK_MANIFESTS))
    parser.add_argument("--max-runs-per-benchmark", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    args = parser.parse_args()

    device = get_device() if args.device == "auto" else torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is not available")

    report = audit_result_root(
        args.result_root,
        checkpoint=args.checkpoint,
        benchmarks=tuple(args.benchmark) if args.benchmark else None,
        max_runs_per_benchmark=args.max_runs_per_benchmark,
        device=device,
    )
    save_json(args.output, report)
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "benchmarks": {
                    key: value["n_runs"]
                    for key, value in report["benchmarks"].items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
