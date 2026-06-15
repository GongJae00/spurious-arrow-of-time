"""SID factor-level probe and counterfactual audit."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.models.encoders import pool_sequence
from src.models.sid import SIDModel
from src.train.common import build_supervised_model, generate_splits_from_config, save_json
from src.utils.hardware import get_device


SID_FACTORS = ("z_rev", "z_ir_task", "z_ir_spur")
TARGETS = ("label", "core_dynamic", "spurious_dynamic")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_config(run_dir: Path) -> dict[str, Any]:
    return _load_json(run_dir / "resolved_config.json")


def _load_sid_model(
    run_dir: Path,
    checkpoint: str,
    device: torch.device,
) -> tuple[SIDModel, dict[str, Any], dict[str, dict[str, Any]]]:
    config = _load_config(run_dir)
    splits = generate_splits_from_config(config)
    input_dim = int(splits["train"]["x"].shape[-1])
    model = build_supervised_model("sid", config, input_dim)
    if not isinstance(model, SIDModel):
        raise TypeError(f"expected SIDModel, got {type(model).__name__}")
    try:
        payload = torch.load(run_dir / checkpoint, map_location=device, weights_only=True)
    except TypeError:
        payload = torch.load(run_dir / checkpoint, map_location=device)
    missing, unexpected = model.load_state_dict(payload["model"], strict=False)
    if missing or unexpected:
        print(f"[sid_factor_audit] load_state_dict non-strict: missing={len(missing)}, unexpected={len(unexpected)} (expected for old checkpoints)")
    model.to(device)
    model.eval()
    return model, config, splits


@torch.inference_mode()
def _encode_factors(
    model: SIDModel,
    x: np.ndarray,
    pooling: str,
    device: torch.device,
) -> dict[str, np.ndarray]:
    x_t = torch.as_tensor(x, dtype=torch.float32, device=device)
    out = model(x_t)
    factors = out["factors"]
    return {
        key: pool_sequence(factors[key], pooling).detach().cpu().numpy()
        for key in SID_FACTORS
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
        return (
            y_train,
            y_eval,
            {"target_source": "spurious_dynamic_stat", "threshold": threshold},
        )
    raise ValueError(f"unknown target {target!r}")


def _linear_probe_accuracy(
    train_x: np.ndarray,
    train_y: np.ndarray,
    eval_x: np.ndarray,
    eval_y: np.ndarray,
) -> float:
    if np.unique(train_y).shape[0] < 2 or np.unique(eval_y).shape[0] < 2:
        return float("nan")
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=500, C=1.0, solver="lbfgs"),
    )
    clf.fit(train_x, train_y)
    return float(clf.score(eval_x, eval_y))


def _finite_or_nan(value: float) -> bool:
    return math.isfinite(value) or math.isnan(value)


def _factor_cf_mse(
    factors: Mapping[str, np.ndarray],
    factors_cf: Mapping[str, np.ndarray],
) -> dict[str, float]:
    return {
        f"{key}_cf_mse": float(np.mean((factors[key] - factors_cf[key]) ** 2))
        for key in SID_FACTORS
    }


def _best_factor(metrics: Mapping[str, float], split: str, target: str) -> str | None:
    values = {
        factor: float(metrics[f"{split}_{factor}_{target}_probe_accuracy"])
        for factor in SID_FACTORS
        if f"{split}_{factor}_{target}_probe_accuracy" in metrics
        and math.isfinite(float(metrics[f"{split}_{factor}_{target}_probe_accuracy"]))
    }
    if not values:
        return None
    return max(values, key=values.get)


def _role_alignment_metrics(
    metrics: dict[str, float | str | bool | None],
    min_probe_acc: float = 0.60,
    max_task_rep_spur_acc: float = 0.55,
) -> dict[str, Any]:
    """Enhanced role alignment per full SID spec.

    Requires:
    - correct argmax factor for label/core on z_ir_task, spurious on z_ir_spur (both splits)
    - the *expected* factor's probe acc >= min_probe_acc
    - task_rep (rev + ir_task) spurious probe acc <= max_task_rep_spur_acc (adversary cannot recover well)
    """
    alignment: dict[str, Any] = {}
    failures: list[str] = []
    expected = {
        "label": "z_ir_task",
        "core_dynamic": "z_ir_task",
        "spurious_dynamic": "z_ir_spur",
    }
    for split in ("iid_test", "ood_test"):
        for target, expected_factor in expected.items():
            key = f"{split}_{target}_best_factor"
            observed = metrics.get(key)
            factor_key = f"{split}_{expected_factor}_{target}_probe_accuracy"
            acc = float(metrics.get(factor_key, float("nan")))
            passed_argmax = observed == expected_factor
            passed_acc = math.isfinite(acc) and acc >= min_probe_acc
            passed = passed_argmax and passed_acc
            metric_key = f"{split}_{target}_best_factor_is_{expected_factor}"
            alignment[metric_key] = bool(passed_argmax)
            alignment[f"{split}_{target}_expected_factor_acc"] = acc
            alignment[f"{split}_{target}_expected_factor_acc_ok"] = passed_acc
            if not passed:
                reason = f"{split}.{target}: expected {expected_factor} (acc={acc:.3f}<{min_probe_acc})" if not passed_acc else f"{split}.{target}: expected {expected_factor}, observed {observed}"
                failures.append(reason)

        # Adversary check on task_rep for spurious (spec: "adversary cannot recover spurious statistic from z_rev + z_ir_task above threshold")
        task_rep_key = f"{split}_task_rep_spurious_dynamic_probe_accuracy"
        task_rep_acc = float(metrics.get(task_rep_key, float("nan")))
        alignment[f"{split}_task_rep_spurious_acc"] = task_rep_acc
        task_rep_ok = (not math.isfinite(task_rep_acc)) or (task_rep_acc <= max_task_rep_spur_acc)
        alignment[f"{split}_task_rep_spurious_low"] = task_rep_ok
        if not task_rep_ok:
            failures.append(f"{split}.task_rep_spurious: acc={task_rep_acc:.3f} > {max_task_rep_spur_acc} (adversary can still recover)")
    alignment["decomposition_role_claim_ready"] = not failures
    alignment["decomposition_role_alignment_failures"] = failures
    alignment["role_min_probe_acc_threshold"] = min_probe_acc
    alignment["role_max_task_rep_spur_acc"] = max_task_rep_spur_acc
    return alignment


def audit_sid_run(
    run_dir: str | Path,
    *,
    checkpoint: str = "best.pt",
    probe_train_split: str = "val_iid",
    eval_splits: tuple[str, ...] = ("iid_test", "ood_test"),
    device: torch.device | None = None,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    device = device or torch.device("cpu")
    model, config, splits = _load_sid_model(run_dir, checkpoint, device)
    pooling = str(config.get("model", {}).get("pooling", "last"))
    train_split = splits[probe_train_split]
    train_factors = _encode_factors(model, train_split["x"], pooling, device)
    metrics: dict[str, float | str | bool | None] = {}
    target_metadata: dict[str, Any] = {}

    task_head_excludes_spur = model.task_head_input_dim == model.z_rev_dim + model.z_ir_task_dim
    metrics["task_head_excludes_z_ir_spur"] = bool(task_head_excludes_spur)

    for split_name in eval_splits:
        split = splits[split_name]
        eval_factors = _encode_factors(model, split["x"], pooling, device)
        eval_factors_cf = _encode_factors(model, split["x_cf"], pooling, device)
        for key, value in _factor_cf_mse(eval_factors, eval_factors_cf).items():
            metrics[f"{split_name}_{key}"] = value
        for target in TARGETS:
            y_train, y_eval, meta = _target_arrays(train_split, split, target)
            target_metadata.setdefault(target, meta)
            for factor in SID_FACTORS:
                metrics[f"{split_name}_{factor}_{target}_probe_accuracy"] = (
                    _linear_probe_accuracy(
                        train_factors[factor],
                        y_train,
                        eval_factors[factor],
                        y_eval,
                    )
                )

            # Explicit task_rep = rev + ir_task probe on spurious (for adversary check)
            # Per spec: "adversary cannot recover spurious statistic from z_rev + z_ir_task above threshold"
            task_rep_train = np.concatenate([train_factors["z_rev"], train_factors["z_ir_task"]], axis=1)
            task_rep_eval = np.concatenate([eval_factors["z_rev"], eval_factors["z_ir_task"]], axis=1)
            if target == "spurious_dynamic":
                metrics[f"{split_name}_task_rep_{target}_probe_accuracy"] = _linear_probe_accuracy(
                    task_rep_train, y_train, task_rep_eval, y_eval
                )

        for target in TARGETS:
            metrics[f"{split_name}_{target}_best_factor"] = _best_factor(
                metrics, split_name, target
            )

        # task_rep spurious probe (lower is better for adversary prevention)
        if f"{split_name}_task_rep_spurious_dynamic_probe_accuracy" not in metrics:
            metrics[f"{split_name}_task_rep_spurious_dynamic_probe_accuracy"] = float("nan")

    role_alignment = _role_alignment_metrics(metrics, min_probe_acc=0.60)
    metrics.update(
        {
            key: value
            for key, value in role_alignment.items()
            if key != "decomposition_role_alignment_failures"
        }
    )

    numeric_values = [value for value in metrics.values() if isinstance(value, (int, float))]
    passed = bool(task_head_excludes_spur) and all(
        _finite_or_nan(float(value)) for value in numeric_values
    )
    report = {
        "passed": passed,
        "method": "sid",
        "run_dir": str(run_dir),
        "checkpoint": checkpoint,
        "probe_train_split": probe_train_split,
        "eval_splits": list(eval_splits),
        "benchmark_name": splits["train"]["metadata"].get("benchmark_name"),
        "probe_capacity": "standardized_logistic_regression_C1_max_iter500",
        "target_metadata": target_metadata,
        "metrics": metrics,
        "role_alignment": role_alignment,
        "interpretation_lock": (
            "This audit records factor-level evidence for SID decomposition claims. "
            "It is not by itself a positive SID success claim. "
            "Use decomposition_role_claim_ready before claiming that factors aligned "
            "with reversible, task-irreversible, and spurious-irreversible roles."
        ),
    }
    save_json(run_dir / "sid_factor_audit.json", report)
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


def audit_sid_manifest(
    manifest_json: str | Path,
    *,
    output: str | Path | None = None,
    checkpoint: str = "best.pt",
    device: torch.device | None = None,
) -> dict[str, Any]:
    manifest_path = Path(manifest_json)
    manifest = _load_json(manifest_path)
    sid_runs = [
        row
        for row in manifest.get("runs", [])
        if row.get("method") == "sid" and row.get("status") == "success"
    ]
    if not sid_runs:
        raise ValueError(f"manifest has no successful SID runs: {manifest_path}")
    rows = []
    for row in sid_runs:
        run_dir = _resolve_run_dir(manifest_path, str(row["run_dir"]))
        rows.append(
            {
                "run_dir": str(run_dir),
                "audit": audit_sid_run(
                    run_dir,
                    checkpoint=checkpoint,
                    device=device,
                ),
            }
        )
    passed = all(row["audit"]["passed"] for row in rows)
    summary = {
        "passed": passed,
        "manifest": str(manifest_path),
        "checkpoint": checkpoint,
        "n_runs": len(rows),
        "runs": rows,
    }
    output_path = Path(output) if output is not None else manifest_path.parent / "sid_factor_audit.json"
    save_json(output_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit SID factor decomposition probes.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run-dir")
    group.add_argument("--manifest")
    parser.add_argument("--output", default=None)
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    device = get_device() if args.device == "auto" else torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is not available")
    if args.run_dir:
        report = audit_sid_run(args.run_dir, checkpoint=args.checkpoint, device=device)
        output = Path(args.output) if args.output else Path(args.run_dir) / "sid_factor_audit.json"
        save_json(output, report)
        passed = report["passed"]
        count = 1
    else:
        summary = audit_sid_manifest(
            args.manifest,
            output=args.output,
            checkpoint=args.checkpoint,
            device=device,
        )
        passed = summary["passed"]
        count = summary["n_runs"]
    print(json.dumps({"passed": passed, "n_runs": count}, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
