"""ITM mechanism-level probe and counterfactual audit.

This audit is the mechanism counterpart to the SID factor audit. It checks the
claim that ITM learns a task-relevant transition mechanism that is stable under
nuisance-arrow interventions, while a separate nuisance mechanism captures the
spurious dynamic statistic.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.models.encoders import pool_sequence
from src.models.itm import ITMModel
from src.train.common import build_supervised_model, generate_splits_from_config, save_json
from src.utils.hardware import get_device


ITM_MECHANISM_AUDIT_SCHEMA = "itm_mechanism_audit_v1"
ITM_REPRESENTATIONS = ("z_core", "z_spur", "task_rep", "spur_rep")
ITM_TARGETS = ("label", "core_dynamic", "spurious_dynamic")
ITM_EVAL_SPLITS = ("iid_test", "ood_test")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_itm_model(
    run_dir: Path,
    checkpoint: str,
    device: torch.device,
) -> tuple[ITMModel, dict[str, Any], dict[str, dict[str, Any]]]:
    config = _load_json(run_dir / "resolved_config.json")
    splits = generate_splits_from_config(config)
    input_dim = int(splits["train"]["x"].shape[-1])
    model = build_supervised_model("itm", config, input_dim)
    if not isinstance(model, ITMModel):
        raise TypeError(f"expected ITMModel, got {type(model).__name__}")
    try:
        payload = torch.load(run_dir / checkpoint, map_location=device, weights_only=True)
    except TypeError:
        payload = torch.load(run_dir / checkpoint, map_location=device)
    model.load_state_dict(payload["model"])
    model.to(device)
    model.eval()
    return model, config, splits


@torch.inference_mode()
def _encode_itm_representations(
    model: ITMModel,
    x: np.ndarray,
    pooling: str,
    device: torch.device,
) -> dict[str, np.ndarray]:
    x_t = torch.as_tensor(x, dtype=torch.float32, device=device)
    out = model(x_t)
    return {
        "z_core": pool_sequence(out["z_core"], pooling).detach().cpu().numpy(),
        "z_spur": pool_sequence(out["z_spur"], pooling).detach().cpu().numpy(),
        "task_rep": out["task_rep"].detach().cpu().numpy(),
        "spur_rep": out["spur_rep"].detach().cpu().numpy(),
    }


@torch.inference_mode()
def _encode_itm_deltas(
    model: ITMModel,
    x: np.ndarray,
    device: torch.device,
) -> dict[str, np.ndarray]:
    x_t = torch.as_tensor(x, dtype=torch.float32, device=device)
    out = model(x_t)
    return {
        "core_delta": out["core_delta"].mean(dim=1).detach().cpu().numpy(),
        "spur_delta": out["spur_delta"].mean(dim=1).detach().cpu().numpy(),
    }


def _binary_targets_from_train(
    train_values: np.ndarray,
    eval_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    train_values = np.asarray(train_values, dtype=np.float64)
    eval_values = np.asarray(eval_values, dtype=np.float64)
    threshold = float(np.median(train_values))
    y_train = (train_values > threshold).astype(np.int64)
    y_eval = (eval_values > threshold).astype(np.int64)
    if np.unique(y_train).shape[0] < 2:
        order = np.argsort(train_values, kind="mergesort")
        y_train = np.zeros_like(y_train)
        y_train[order[len(order) // 2 :]] = 1
        threshold = float(train_values[order[len(order) // 2]])
        y_eval = (eval_values >= threshold).astype(np.int64)
    return y_train, y_eval, threshold


def _target_arrays(
    train_split: Mapping[str, Any],
    eval_split: Mapping[str, Any],
    target: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if target == "label":
        return (
            np.asarray(train_split["y"], dtype=np.int64),
            np.asarray(eval_split["y"], dtype=np.int64),
            {"target_source": "y"},
        )
    if target == "core_dynamic":
        y_train, y_eval, threshold = _binary_targets_from_train(
            np.asarray(train_split["core_dynamic_stat"]),
            np.asarray(eval_split["core_dynamic_stat"]),
        )
        return y_train, y_eval, {"target_source": "core_dynamic_stat", "threshold": threshold}
    if target == "spurious_dynamic":
        y_train, y_eval, threshold = _binary_targets_from_train(
            np.asarray(train_split["spurious_dynamic_stat"]),
            np.asarray(eval_split["spurious_dynamic_stat"]),
        )
        return y_train, y_eval, {"target_source": "spurious_dynamic_stat", "threshold": threshold}
    raise ValueError(f"unknown target {target!r}")


def _continuous_target(split: Mapping[str, Any], target: str) -> np.ndarray:
    if target == "label":
        return np.asarray(split["y"], dtype=np.float64)
    if target == "core_dynamic":
        return np.asarray(split["core_dynamic_stat"], dtype=np.float64)
    if target == "spurious_dynamic":
        return np.asarray(split["spurious_dynamic_stat"], dtype=np.float64)
    raise ValueError(f"unknown target {target!r}")


def _as_2d(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        return values[:, None]
    return values


def _with_intercept(values: np.ndarray) -> np.ndarray:
    values = _as_2d(values)
    return np.concatenate([np.ones((values.shape[0], 1)), values], axis=1)


def _residualize_against_controls(
    train_x: np.ndarray,
    train_controls: np.ndarray,
    eval_x: np.ndarray,
    eval_controls: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    controls = _with_intercept(train_controls)
    x = _as_2d(train_x)
    coef, *_ = np.linalg.lstsq(controls, x, rcond=None)
    return (
        x - controls @ coef,
        _as_2d(eval_x) - _with_intercept(eval_controls) @ coef,
    )


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


def _orientation_free_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    if np.unique(y_true).shape[0] < 2:
        return float("nan")
    auc = float(roc_auc_score(y_true, score))
    return max(auc, 1.0 - auc)


def _linear_probe_metrics(
    train_x: np.ndarray,
    train_y: np.ndarray,
    eval_x: np.ndarray,
    eval_y: np.ndarray,
) -> dict[str, float]:
    if np.unique(train_y).shape[0] < 2 or np.unique(eval_y).shape[0] < 2:
        return {"accuracy": float("nan"), "orientation_free_auc": float("nan")}
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=500, C=1.0, solver="lbfgs"),
    )
    clf.fit(train_x, train_y)
    score = clf.predict_proba(eval_x)[:, 1]
    return {
        "accuracy": float(clf.score(eval_x, eval_y)),
        "orientation_free_auc": _orientation_free_auc(eval_y, score),
    }


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((np.asarray(a) - np.asarray(b)) ** 2))


def _best_rep(metrics: Mapping[str, float], split: str, target: str, metric: str) -> str | None:
    values = {
        rep: float(metrics[f"{split}_{rep}_{target}_{metric}"])
        for rep in ITM_REPRESENTATIONS
        if f"{split}_{rep}_{target}_{metric}" in metrics
        and math.isfinite(float(metrics[f"{split}_{rep}_{target}_{metric}"]))
    }
    if not values:
        return None
    return max(values, key=values.get)


def _role_alignment(
    metrics: Mapping[str, Any],
    *,
    min_task_core_auc: float,
    min_spur_auc: float,
    max_task_spur_auc: float,
    max_core_cf_mse: float,
    min_spur_cf_mse: float,
) -> dict[str, Any]:
    failures: list[str] = []
    role: dict[str, Any] = {
        "min_task_core_auc": min_task_core_auc,
        "min_spur_auc": min_spur_auc,
        "max_task_spur_auc": max_task_spur_auc,
        "max_core_cf_mse": max_core_cf_mse,
        "min_spur_cf_mse": min_spur_cf_mse,
    }
    for split in ITM_EVAL_SPLITS:
        task_label = float(metrics.get(f"{split}_task_rep_label_residualized_auc", float("nan")))
        task_core = float(
            metrics.get(f"{split}_task_rep_core_dynamic_residualized_auc", float("nan"))
        )
        task_spur = float(
            metrics.get(f"{split}_task_rep_spurious_dynamic_residualized_auc", float("nan"))
        )
        spur_spur = float(
            metrics.get(f"{split}_spur_rep_spurious_dynamic_residualized_auc", float("nan"))
        )
        core_cf = float(metrics.get(f"{split}_core_delta_cf_mse", float("nan")))
        spur_cf = float(metrics.get(f"{split}_spur_delta_cf_mse", float("nan")))

        checks = {
            f"{split}_task_rep_label_auc_ok": math.isfinite(task_label)
            and task_label >= min_task_core_auc,
            f"{split}_task_rep_core_auc_ok": math.isfinite(task_core)
            and task_core >= min_task_core_auc,
            f"{split}_task_rep_spurious_auc_low": (not math.isfinite(task_spur))
            or task_spur <= max_task_spur_auc,
            f"{split}_spur_rep_spurious_auc_ok": math.isfinite(spur_spur)
            and spur_spur >= min_spur_auc,
            f"{split}_core_delta_cf_stable": math.isfinite(core_cf)
            and core_cf <= max_core_cf_mse,
            f"{split}_spur_delta_cf_sensitive": math.isfinite(spur_cf)
            and spur_cf >= min_spur_cf_mse,
        }
        role.update(checks)
        role[f"{split}_label_best_rep"] = _best_rep(metrics, split, "label", "residualized_auc")
        role[f"{split}_core_dynamic_best_rep"] = _best_rep(
            metrics, split, "core_dynamic", "residualized_auc"
        )
        role[f"{split}_spurious_dynamic_best_rep"] = _best_rep(
            metrics, split, "spurious_dynamic", "residualized_auc"
        )
        role[f"{split}_task_rep_label_residualized_auc"] = task_label
        role[f"{split}_task_rep_core_dynamic_residualized_auc"] = task_core
        role[f"{split}_task_rep_spurious_dynamic_residualized_auc"] = task_spur
        role[f"{split}_spur_rep_spurious_dynamic_residualized_auc"] = spur_spur
        role[f"{split}_core_delta_cf_mse"] = core_cf
        role[f"{split}_spur_delta_cf_mse"] = spur_cf
        for key, passed in checks.items():
            if not passed:
                failures.append(key)
    role["mechanism_claim_ready"] = not failures
    role["mechanism_alignment_failures"] = failures
    return role


def audit_itm_run(
    run_dir: str | Path,
    *,
    checkpoint: str = "best.pt",
    probe_train_split: str = "val_iid",
    eval_splits: tuple[str, ...] = ITM_EVAL_SPLITS,
    device: torch.device | None = None,
    save_local: bool = True,
    min_task_core_auc: float = 0.60,
    min_spur_auc: float = 0.60,
    max_task_spur_auc: float = 0.58,
    max_core_cf_mse: float = 0.08,
    min_spur_cf_mse: float = 0.002,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    device = device or torch.device("cpu")
    model, config, splits = _load_itm_model(run_dir, checkpoint, device)
    pooling = str(config.get("model", {}).get("pooling", "last"))
    train_split = splits[probe_train_split]
    train_reps = _encode_itm_representations(model, train_split["x"], pooling, device)

    metrics: dict[str, float | str | bool | None] = {}
    target_metadata: dict[str, Any] = {}
    task_head_excludes_spur = model.task_head_input_dim == 2 * model.core_dim
    metrics["task_head_excludes_spur_mechanism"] = bool(task_head_excludes_spur)

    for split_name in eval_splits:
        split = splits[split_name]
        eval_reps = _encode_itm_representations(model, split["x"], pooling, device)
        deltas = _encode_itm_deltas(model, split["x"], device)
        deltas_cf = _encode_itm_deltas(model, split["x_cf"], device)
        metrics[f"{split_name}_core_delta_cf_mse"] = _mse(
            deltas["core_delta"], deltas_cf["core_delta"]
        )
        metrics[f"{split_name}_spur_delta_cf_mse"] = _mse(
            deltas["spur_delta"], deltas_cf["spur_delta"]
        )
        metrics[f"{split_name}_spur_to_core_cf_mse_ratio"] = (
            float(metrics[f"{split_name}_spur_delta_cf_mse"])
            / (float(metrics[f"{split_name}_core_delta_cf_mse"]) + 1e-8)
        )
        for target in ITM_TARGETS:
            y_train, y_eval, meta = _target_arrays(train_split, split, target)
            target_metadata.setdefault(target, meta)
            train_controls, eval_controls, control_names = _controls_for_target(
                train_split, split, target
            )
            for rep in ITM_REPRESENTATIONS:
                raw = _linear_probe_metrics(
                    train_reps[rep], y_train, eval_reps[rep], y_eval
                )
                train_resid, eval_resid = _residualize_against_controls(
                    train_reps[rep],
                    train_controls,
                    eval_reps[rep],
                    eval_controls,
                )
                residualized = _linear_probe_metrics(
                    train_resid, y_train, eval_resid, y_eval
                )
                metrics[f"{split_name}_{rep}_{target}_probe_accuracy"] = raw["accuracy"]
                metrics[f"{split_name}_{rep}_{target}_orientation_free_auc"] = raw[
                    "orientation_free_auc"
                ]
                metrics[f"{split_name}_{rep}_{target}_residualized_accuracy"] = (
                    residualized["accuracy"]
                )
                metrics[f"{split_name}_{rep}_{target}_residualized_auc"] = (
                    residualized["orientation_free_auc"]
                )
                metrics[f"{split_name}_{rep}_{target}_residual_controls"] = ",".join(
                    control_names
                )

    role_alignment = _role_alignment(
        metrics,
        min_task_core_auc=min_task_core_auc,
        min_spur_auc=min_spur_auc,
        max_task_spur_auc=max_task_spur_auc,
        max_core_cf_mse=max_core_cf_mse,
        min_spur_cf_mse=min_spur_cf_mse,
    )
    metrics["mechanism_claim_ready"] = bool(role_alignment["mechanism_claim_ready"])
    report = {
        "schema": ITM_MECHANISM_AUDIT_SCHEMA,
        "passed": True,
        "method": "itm",
        "run_dir": str(run_dir.resolve()),
        "checkpoint": checkpoint,
        "probe_train_split": probe_train_split,
        "eval_splits": list(eval_splits),
        "benchmark_name": splits["train"]["metadata"].get("benchmark_name"),
        "probe_capacity": "standardized_logistic_regression_C1_max_iter500",
        "target_metadata": target_metadata,
        "metrics": metrics,
        "role_alignment": role_alignment,
        "interpretation_lock": (
            "This audit tests ITM mechanism evidence. Positive paper language "
            "requires mechanism_claim_ready in addition to task metrics."
        ),
    }
    if save_local:
        save_json(run_dir / "itm_mechanism_audit.json", report)
    return report


def _resolve_run_dir(manifest_path: Path, run_dir: str) -> Path:
    path = Path(run_dir)
    if path.is_absolute():
        return path.resolve()
    candidates = [
        (Path.cwd() / path).resolve(),
        (manifest_path.parent / path).resolve(),
        (manifest_path.parent.parent / path).resolve(),
    ]
    return next((candidate for candidate in candidates if candidate.exists()), candidates[0])


def audit_manifest(
    manifest_path: str | Path,
    *,
    output: str | Path | None = None,
    device: torch.device | None = None,
    checkpoint: str = "best.pt",
) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    manifest = _load_json(manifest_path)
    device = device or torch.device("cpu")
    rows: list[dict[str, Any]] = []
    for row in manifest.get("runs", []):
        if row.get("status") != "success" or row.get("method") != "itm":
            continue
        run_dir = _resolve_run_dir(manifest_path, str(row.get("run_dir", "")))
        audit = audit_itm_run(
            run_dir,
            checkpoint=checkpoint,
            device=device,
            save_local=True,
        )
        rows.append({"run_dir": str(run_dir), "audit": audit})
    ready_runs = sum(
        1
        for row in rows
        if row["audit"].get("role_alignment", {}).get("mechanism_claim_ready") is True
    )
    summary = {
        "schema": ITM_MECHANISM_AUDIT_SCHEMA,
        "passed": bool(rows) and all(row["audit"].get("passed") is True for row in rows),
        "manifest": str(manifest_path),
        "checkpoint": checkpoint,
        "n_runs": len(rows),
        "mechanism_claim_ready_runs": ready_runs,
        "mechanism_claim_ready": bool(rows) and ready_runs == len(rows),
        "runs": rows,
    }
    if output is not None:
        save_json(output, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit ITM mechanism evidence.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    report = audit_manifest(
        args.manifest,
        output=args.output,
        checkpoint=args.checkpoint,
        device=get_device(args.device),
    )
    print(json.dumps({"passed": report["passed"], "n_runs": report["n_runs"]}, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
