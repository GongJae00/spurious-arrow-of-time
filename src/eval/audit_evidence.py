"""Audit final evidence roots.

This audit is stricter than the smoke audit. It is intended for the
non-smoke multi-seed result root that may feed paper assets.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Mapping

from src.eval.audit_smoke import EXPECTED_BENCHMARKS, REQUIRED_METHODS
from src.train.common import save_json


EVIDENCE_AUDIT_SCHEMA_VERSION = "evidence_audit_itm_mechanism_v1"
SID_FACTORS = ("z_rev", "z_ir_task", "z_ir_spur")
SID_AUDIT_TARGETS = ("label", "core_dynamic", "spurious_dynamic")
SID_AUDIT_EVAL_SPLITS = ("iid_test", "ood_test")
ITM_REPRESENTATIONS = ("z_core", "z_spur", "task_rep", "spur_rep")
ITM_AUDIT_TARGETS = ("label", "core_dynamic", "spurious_dynamic")
ITM_AUDIT_EVAL_SPLITS = ("iid_test", "ood_test")
REQUIRED_SID_FACTOR_METRICS = {
    "task_head_excludes_z_ir_spur",
    "decomposition_role_claim_ready",
    *{
        f"{split}_{factor}_cf_mse"
        for split in SID_AUDIT_EVAL_SPLITS
        for factor in SID_FACTORS
    },
    *{
        f"{split}_{factor}_{target}_probe_accuracy"
        for split in SID_AUDIT_EVAL_SPLITS
        for factor in SID_FACTORS
        for target in SID_AUDIT_TARGETS
    },
    *{
        f"{split}_{target}_best_factor"
        for split in SID_AUDIT_EVAL_SPLITS
        for target in SID_AUDIT_TARGETS
    },
    *{
        f"{split}_{target}_best_factor_is_z_ir_task"
        for split in SID_AUDIT_EVAL_SPLITS
        for target in ("label", "core_dynamic")
    },
    *{
        f"{split}_spurious_dynamic_best_factor_is_z_ir_spur"
        for split in SID_AUDIT_EVAL_SPLITS
    },
    *{
        f"{split}_task_rep_spurious_dynamic_probe_accuracy"
        for split in SID_AUDIT_EVAL_SPLITS
    },
    *{
        f"{split}_task_rep_spurious_low"
        for split in SID_AUDIT_EVAL_SPLITS
    },
}
REQUIRED_ITM_MECHANISM_METRICS = {
    "task_head_excludes_spur_mechanism",
    "mechanism_claim_ready",
    *{
        f"{split}_{rep}_{target}_{metric}"
        for split in ITM_AUDIT_EVAL_SPLITS
        for rep in ITM_REPRESENTATIONS
        for target in ITM_AUDIT_TARGETS
        for metric in (
            "probe_accuracy",
            "orientation_free_auc",
            "residualized_accuracy",
            "residualized_auc",
        )
    },
    *{
        f"{split}_{name}"
        for split in ITM_AUDIT_EVAL_SPLITS
        for name in (
            "core_delta_cf_mse",
            "spur_delta_cf_mse",
            "spur_to_core_cf_mse_ratio",
        )
    },
}
DIAGNOSTIC_FILES = {
    "sta": "diagnostics/sta_benchmark_diagnostics.json",
    "ink_advection_diffusion": "diagnostics/ink_advection_diffusion_diagnostics.json",
}
REQUIRED_DIAGNOSTIC_GATES = {
    "sta": {
        "same_mixing_matrix",
        "train_threshold_reused",
        "core_oracle_high_iid",
        "core_oracle_high_ood",
        "spurious_rule_high_iid",
        "spurious_rule_breaks_ood",
    },
    "ink_advection_diffusion": {
        "train_threshold_reused",
        "mass_conservation",
        "nonnegative_concentration",
        "spread_increase",
        "entropy_increase",
        "visible_signal",
        "core_oracle_high_iid",
        "core_oracle_high_ood",
        "spurious_rule_high_iid",
        "spurious_rule_breaks_ood",
        "dynamic_spurious_corr_train",
        "dynamic_spurious_corr_iid",
        "dynamic_spurious_corr_ood_reversed",
        "counterfactual_preserves_core_and_label",
        "counterfactual_changes_spurious_flow",
    },
}
FORBIDDEN_TUNING_SPLITS = {"ood", "ood_test", "iid_test", "test"}
SELECTION_VALUE_KEYS = {
    "best_metric",
    "best_split",
    "calibration_metric",
    "calibration_split",
    "checkpoint_metric",
    "checkpoint_selection_metric",
    "checkpoint_selection_split",
    "checkpoint_split",
    "chosen_metric",
    "chosen_split",
    "early_stopping_metric",
    "hyperparameter_selection_metric",
    "hyperparameter_selection_split",
    "model_selection_metric",
    "model_selection_split",
    "monitor_metric",
    "monitor_split",
    "pretraining_split",
    "reference_metric",
    "reference_split",
    "score_metric",
    "score_split",
    "selection_metric",
    "selection_source",
    "selection_sources",
    "selection_split",
    "selector_metric",
    "selector_split",
    "task_guard_selection_metric",
    "task_guard_selection_split",
    "tuning_metric",
    "tuning_split",
    "validation_metric",
    "validation_split",
}
SELECTED_CHECKPOINT_KEYS = {
    "selected_checkpoint",
    "unguarded_selected_checkpoint",
}
REQUIRED_CHECKPOINT_KEYS_BY_METHOD = {
    "erm": ("selected_checkpoint", "unguarded_selected_checkpoint"),
    "ib": ("selected_checkpoint", "unguarded_selected_checkpoint"),
    "ep_min": ("selected_checkpoint", "unguarded_selected_checkpoint"),
    "ep_max": ("selected_checkpoint", "unguarded_selected_checkpoint"),
    "sib": ("selected_checkpoint", "unguarded_selected_checkpoint"),
    "sid": ("selected_checkpoint", "unguarded_selected_checkpoint"),
    "itm": ("selected_checkpoint", "unguarded_selected_checkpoint"),
    "ocp_style": (
        "frozen_encoder_selected_checkpoint",
        "fine_tuned_encoder_selected_checkpoint",
    ),
    "lens_like_arrow_classifier": (
        "frozen_encoder_selected_checkpoint",
        "fine_tuned_encoder_selected_checkpoint",
    ),
}
EXPECTED_FULL_CONFIGS = {
    "sta": "configs/sta_full.yaml",
    "ink_advection_diffusion": "configs/ink_advection_diffusion_full.yaml",
}
EXPECTED_FULL_SPLITS = {
    "train": 10_000,
    "val_iid": 2_000,
    "iid_test": 5_000,
    "ood_test": 5_000,
}
PREFLIGHT_SOURCE_PATHS = (
    "src",
    "configs/sta_full.yaml",
    "configs/ink_advection_diffusion_full.yaml",
    "experiments/full_suite.sh",
    "experiments/preflight.sh",
    "experiments/wait_for_preflight_and_run.sh",
    "experiments/finalize_paper.sh",
    "RESEARCH.md",
    "pyproject.toml",
    "requirements.txt",
)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _check(condition: bool, name: str, detail: str, checks: list[dict[str, Any]]) -> bool:
    checks.append({"name": name, "passed": bool(condition), "detail": detail})
    return bool(condition)


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


def _is_smoke_path(root: Path) -> bool:
    return any("smoke" in part.lower() for part in root.parts)


def _selection_split_is_safe(value: Any) -> bool:
    if value is None:
        return True
    return str(value).lower() not in FORBIDDEN_TUNING_SPLITS


def _selection_value_is_safe(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (list, tuple, set)):
        return all(_selection_value_is_safe(item) for item in value)
    if isinstance(value, Mapping):
        return all(_selection_value_is_safe(item) for item in value.values())
    text = str(value).lower()
    return not any(token in text for token in FORBIDDEN_TUNING_SPLITS)


def _selection_related_mismatches(
    data: Mapping[str, Any],
    *,
    prefix: str = "",
) -> list[str]:
    mismatches: list[str] = []
    for key, value in data.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        key_lower = str(key).lower()
        if isinstance(value, Mapping):
            mismatches.extend(_selection_related_mismatches(value, prefix=path))
            continue
        if key_lower == "uses_ood_test" and value is True:
            mismatches.append(f"{path}={value}")
        if key_lower in SELECTION_VALUE_KEYS and not _selection_value_is_safe(value):
            mismatches.append(f"{path}={value}")
        if key_lower in SELECTED_CHECKPOINT_KEYS and not _selection_value_is_safe(value):
            mismatches.append(f"{path}={value}")
    return mismatches


def _metadata_no_ood_tuning(metadata: Mapping[str, Any]) -> tuple[bool, str]:
    setpoint = metadata.get("setpoint", {})
    task_guard = metadata.get("task_guard", {})
    arrow_calibration = metadata.get("arrow_calibration", {})
    checks = {
        "setpoint_selection_split": _selection_split_is_safe(
            setpoint.get("selection_split")
        ),
        "task_guard_selection_split": _selection_split_is_safe(
            task_guard.get("selection_split")
        ),
        "arrow_calibration_uses_ood_test": arrow_calibration.get("uses_ood_test") is not True,
    }
    recursive_mismatches = _selection_related_mismatches(metadata)
    return (
        all(checks.values()) and not recursive_mismatches,
        f"{checks}, forbidden_selection_values={recursive_mismatches[:20]}",
    )


def _resolved_config_no_ood_tuning(resolved_config: Mapping[str, Any]) -> tuple[bool, str]:
    mismatches = _selection_related_mismatches(resolved_config)
    return not mismatches, f"forbidden_selection_values={mismatches[:20]}"


def _metrics_no_ood_tuning(metrics: Mapping[str, Any]) -> tuple[bool, str]:
    mismatches = _selection_related_mismatches(metrics)
    return not mismatches, f"forbidden_selection_values={mismatches[:20]}"


def _numeric_items(data: Mapping[str, Any]) -> dict[str, float]:
    return {
        key: float(value)
        for key, value in data.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }


def _aggregate_method_mismatches(
    *,
    aggregate_row: Mapping[str, Any],
    metric_rows: list[Mapping[str, float]],
    tolerance: float = 1e-9,
) -> list[str]:
    mismatches: list[str] = []
    if aggregate_row.get("n_runs") != len(metric_rows):
        mismatches.append(
            f"n_runs expected {len(metric_rows)} observed {aggregate_row.get('n_runs')}"
        )
    metric_names = sorted({metric for row in metric_rows for metric in row})
    for metric in metric_names:
        values = [float(row[metric]) for row in metric_rows if metric in row]
        expected_mean = mean(values)
        expected_std = stdev(values) if len(values) > 1 else 0.0
        observed_mean = aggregate_row.get(f"{metric}_mean")
        observed_std = aggregate_row.get(f"{metric}_std")
        if not isinstance(observed_mean, (int, float)) or math.fabs(
            float(observed_mean) - expected_mean
        ) > tolerance:
            mismatches.append(
                f"{metric}_mean expected {expected_mean} observed {observed_mean}"
            )
        if not isinstance(observed_std, (int, float)) or math.fabs(
            float(observed_std) - expected_std
        ) > tolerance:
            mismatches.append(
                f"{metric}_std expected {expected_std} observed {observed_std}"
            )
    return mismatches


def _aggregate_method_payloads(aggregate: Mapping[str, Any]) -> Mapping[str, Any]:
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


def _diagnostic_schema_mismatches(
    diagnostic: Mapping[str, Any],
    *,
    benchmark_name: str,
    allow_smoke: bool,
) -> list[str]:
    mismatches: list[str] = []
    if diagnostic.get("schema_version") != 1:
        mismatches.append(f"schema_version={diagnostic.get('schema_version')}")
    if diagnostic.get("benchmark_name") != benchmark_name:
        mismatches.append(
            f"benchmark_name={diagnostic.get('benchmark_name')} expected={benchmark_name}"
        )
    splits = diagnostic.get("splits", {})
    if not isinstance(splits, Mapping):
        return [*mismatches, "splits is not a mapping"]
    expected_splits = set(EXPECTED_FULL_SPLITS)
    observed_splits = set(splits)
    if observed_splits != expected_splits:
        mismatches.append(f"splits={sorted(observed_splits)} expected={sorted(expected_splits)}")
    for split_name, minimum_n in EXPECTED_FULL_SPLITS.items():
        split = splits.get(split_name)
        if not isinstance(split, Mapping):
            mismatches.append(f"{split_name} missing")
            continue
        n_sequences = split.get("n_sequences")
        length_l = split.get("length_L")
        n_transitions = split.get("n_transitions")
        if not isinstance(n_sequences, int) or isinstance(n_sequences, bool):
            mismatches.append(f"{split_name}.n_sequences invalid: {n_sequences}")
        elif not allow_smoke and n_sequences < minimum_n:
            mismatches.append(
                f"{split_name}.n_sequences={n_sequences} below required {minimum_n}"
            )
        if not isinstance(length_l, int) or isinstance(length_l, bool) or length_l <= 1:
            mismatches.append(f"{split_name}.length_L invalid: {length_l}")
        if (
            not isinstance(n_transitions, int)
            or isinstance(n_transitions, bool)
            or not isinstance(length_l, int)
            or n_transitions != length_l - 1
        ):
            mismatches.append(
                f"{split_name}.n_transitions={n_transitions} length_L={length_l}"
            )
        class_balance = split.get("class_balance", {})
        class_n = class_balance.get("n") if isinstance(class_balance, Mapping) else None
        if isinstance(n_sequences, int) and class_n != n_sequences:
            mismatches.append(
                f"{split_name}.class_balance.n={class_n} n_sequences={n_sequences}"
            )
        threshold_source = split.get("label_threshold_source")
        expected_sources = {"local", "local_calibration"} if split_name == "train" else {"train"}
        if threshold_source not in expected_sources:
            mismatches.append(
                f"{split_name}.label_threshold_source={threshold_source} "
                f"expected={sorted(expected_sources)}"
            )
    return mismatches


def _latest_source_mtime(paths: tuple[str, ...] = PREFLIGHT_SOURCE_PATHS) -> float | None:
    latest: float | None = None
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            continue
        candidates = path.rglob("*") if path.is_dir() else [path]
        for candidate in candidates:
            if not candidate.is_file():
                continue
            if "__pycache__" in candidate.parts or candidate.suffix in {".pyc", ".pyo"}:
                continue
            mtime = candidate.stat().st_mtime
            latest = mtime if latest is None else max(latest, mtime)
    return latest


def _audit_preflight_artifact(
    *,
    root: Path,
    preflight_path: Path,
    min_seeds: int,
    min_epochs: int,
    checks: list[dict[str, Any]],
) -> dict[str, Any]:
    if not _check(
        preflight_path.exists(),
        "preflight_exists",
        str(preflight_path),
        checks,
    ):
        return {}
    try:
        payload = _load_json(preflight_path)
        _check(True, "preflight_valid_json", str(preflight_path), checks)
    except (OSError, json.JSONDecodeError) as exc:
        _check(False, "preflight_valid_json", str(exc), checks)
        return {}

    environment = payload.get("environment", {})
    preflight_out = environment.get("out")
    preflight_methods = environment.get("methods", [])
    require_cuda_for_full_run = environment.get("require_cuda_for_full_run")
    launch_authorization = environment.get("launch_authorization")
    _check(
        payload.get("pass") is True and payload.get("launch_recommended") is True,
        "preflight_passed",
        (
            f"pass={payload.get('pass')}, "
            f"launch_recommended={payload.get('launch_recommended')}"
        ),
        checks,
    )
    _check(
        preflight_out is not None and Path(str(preflight_out)).resolve() == root.resolve(),
        "preflight_matches_result_root",
        f"preflight_out={preflight_out}, result_root={root}",
        checks,
    )
    _check(
        int(environment.get("min_seeds", 0)) >= int(min_seeds),
        "preflight_seed_threshold",
        f"preflight_min_seeds={environment.get('min_seeds')}, audit_min_seeds={min_seeds}",
        checks,
    )
    _check(
        int(environment.get("epochs", 0)) >= int(min_epochs),
        "preflight_epoch_threshold",
        f"preflight_epochs={environment.get('epochs')}, audit_min_epochs={min_epochs}",
        checks,
    )
    _check(
        sorted(preflight_methods) == sorted(REQUIRED_METHODS),
        "preflight_required_methods",
        f"preflight_methods={preflight_methods}, required={list(REQUIRED_METHODS)}",
        checks,
    )
    _check(
        str(require_cuda_for_full_run) in {"0", "1"},
        "preflight_require_cuda_recorded",
        f"require_cuda_for_full_run={require_cuda_for_full_run}",
        checks,
    )
    _check(
        launch_authorization in {"final_cuda_launch", "maintenance_or_diagnostic"},
        "preflight_launch_authorization_recorded",
        f"launch_authorization={launch_authorization}",
        checks,
    )
    latest_source_mtime = _latest_source_mtime()
    if latest_source_mtime is not None:
        preflight_mtime = preflight_path.stat().st_mtime
        _check(
            preflight_mtime + 1e-6 >= latest_source_mtime,
            "preflight_not_older_than_code_sources",
            (
                f"preflight_mtime={preflight_mtime}, "
                f"latest_source_mtime={latest_source_mtime}"
            ),
            checks,
        )
    return {
        "path": str(preflight_path),
        "pass": payload.get("pass"),
        "launch_recommended": payload.get("launch_recommended"),
        "out": preflight_out,
        "device": environment.get("device"),
        "require_cuda_for_full_run": require_cuda_for_full_run,
        "launch_authorization": launch_authorization,
        "min_seeds": environment.get("min_seeds"),
        "epochs": environment.get("epochs"),
        "methods": preflight_methods,
    }


def _metrics_have_final_eval(metrics: Mapping[str, Any], method: str) -> bool:
    if method in {"ocp_style", "lens_like_arrow_classifier"}:
        return any(
            all(
                f"{prefix}{key}" in metrics
                for key in ("iid_test_accuracy", "ood_test_accuracy", "ood_gap")
            )
            for prefix in ("frozen_encoder_", "fine_tuned_encoder_")
        )
    return all(key in metrics for key in ("iid_test_accuracy", "ood_test_accuracy", "ood_gap"))


def _missing_checkpoint_files(
    *,
    run_dir: Path,
    metrics: Mapping[str, Any],
    method: str,
) -> list[str]:
    missing: list[str] = []
    for key in REQUIRED_CHECKPOINT_KEYS_BY_METHOD.get(method, ()):
        value = metrics.get(key)
        if not value:
            missing.append(f"{key}=<missing>")
            continue
        checkpoint_path = run_dir / str(value)
        if not checkpoint_path.is_file():
            missing.append(f"{key}={value}")
    if not (run_dir / "final.pt").is_file():
        missing.append("final.pt")
    return missing


def _jsonl_has_records(path: Path) -> tuple[bool, str]:
    if not path.is_file():
        return False, f"missing {path.name}"
    try:
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        for line in lines:
            json.loads(line)
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"invalid {path.name}: {exc}"
    if not lines:
        return False, f"empty {path.name}"
    return True, f"{path.name} records={len(lines)}"


def _training_log_mismatches(run_dir: Path, method: str) -> list[str]:
    paths = [run_dir / "metrics.jsonl"]
    if method in {"ocp_style", "lens_like_arrow_classifier"}:
        paths.extend(
            [
                run_dir / "frozen_encoder_metrics.jsonl",
                run_dir / "fine_tuned_encoder_metrics.jsonl",
            ]
        )
    mismatches: list[str] = []
    for path in paths:
        ok, detail = _jsonl_has_records(path)
        if not ok:
            mismatches.append(detail)
    return mismatches


def _epoch_range_mismatches(metrics: Mapping[str, Any], method: str) -> list[str]:
    mismatches: list[str] = []

    def check_epoch(key: str, completed_key: str) -> None:
        value = metrics.get(key)
        completed = metrics.get(completed_key)
        if value is None:
            return
        if not isinstance(value, int) or isinstance(value, bool):
            mismatches.append(f"{key} is not int: {value}")
            return
        if not isinstance(completed, int) or isinstance(completed, bool) or completed <= 0:
            mismatches.append(f"{completed_key} invalid: {completed}")
            return
        if value < 0 or value >= completed:
            mismatches.append(f"{key}={value} outside [0,{completed})")

    if method in {"ocp_style", "lens_like_arrow_classifier"}:
        for prefix in ("frozen_encoder", "fine_tuned_encoder"):
            completed_key = f"{prefix}_epochs_completed"
            completed = metrics.get(completed_key)
            if not isinstance(completed, int) or isinstance(completed, bool) or completed <= 0:
                mismatches.append(f"{completed_key} invalid: {completed}")
            check_epoch(f"{prefix}_best_epoch", completed_key)
        return mismatches

    completed = metrics.get("epochs_completed")
    if not isinstance(completed, int) or isinstance(completed, bool) or completed <= 0:
        mismatches.append(f"epochs_completed invalid: {completed}")
    for key in ("selected_epoch", "best_epoch", "unguarded_best_epoch", "task_guard_best_epoch"):
        check_epoch(key, "epochs_completed")
    n_eligible = metrics.get("task_guard_n_eligible_epochs")
    if n_eligible is not None:
        if not isinstance(n_eligible, int) or isinstance(n_eligible, bool) or n_eligible < 0:
            mismatches.append(f"task_guard_n_eligible_epochs invalid: {n_eligible}")
        elif isinstance(completed, int) and n_eligible > completed:
            mismatches.append(
                f"task_guard_n_eligible_epochs={n_eligible} exceeds epochs_completed={completed}"
            )
    return mismatches


def audit_evidence(
    root: str | Path,
    *,
    min_seeds: int = 5,
    min_epochs: int = 10,
    allow_smoke: bool = False,
    preflight_path: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(root)
    checks: list[dict[str, Any]] = []
    run_count = 0
    seed_summary: dict[str, dict[str, list[int]]] = {}
    preflight_environment_summary: dict[str, Any] | None = None

    _check(root.exists(), "root_exists", str(root), checks)
    _check(
        allow_smoke or not _is_smoke_path(root),
        "non_smoke_result_root",
        f"root={root}, allow_smoke={allow_smoke}",
        checks,
    )
    if preflight_path is not None:
        preflight_environment_summary = _audit_preflight_artifact(
            root=root,
            preflight_path=Path(preflight_path),
            min_seeds=min_seeds,
            min_epochs=min_epochs,
            checks=checks,
        )

    for suite_name, benchmark_name in EXPECTED_BENCHMARKS.items():
        suite_root = root / suite_name
        manifest_path = suite_root / "manifest.json"
        aggregate_path = suite_root / "aggregate.json"
        diagnostic_path = root / DIAGNOSTIC_FILES[suite_name]
        sid_factor_path = suite_root / "sid_factor_audit.json"
        itm_mechanism_path = suite_root / "itm_mechanism_audit.json"

        _check(
            diagnostic_path.exists(),
            f"{suite_name}_diagnostic_exists",
            str(diagnostic_path),
            checks,
        )
        if diagnostic_path.exists():
            diagnostic = _load_json(diagnostic_path)
            _check(
                diagnostic.get("pass") is True,
                f"{suite_name}_diagnostic_passed",
                f"pass={diagnostic.get('pass')}",
                checks,
            )
            diagnostic_schema_mismatches = _diagnostic_schema_mismatches(
                diagnostic,
                benchmark_name=benchmark_name,
                allow_smoke=allow_smoke,
            )
            _check(
                not diagnostic_schema_mismatches,
                f"{suite_name}_diagnostic_schema",
                f"mismatches={diagnostic_schema_mismatches[:20]}",
                checks,
            )
            quality = diagnostic.get("quality_gates", {})
            required_gates = REQUIRED_DIAGNOSTIC_GATES[suite_name]
            quality_keys = set(quality) if isinstance(quality, dict) else set()
            missing_gates = sorted(required_gates.difference(quality_keys))
            _check(
                not missing_gates,
                f"{suite_name}_diagnostic_required_quality_gates_present",
                f"missing={missing_gates}",
                checks,
            )
            if isinstance(quality, dict) and quality:
                _check(
                    all(value is True for value in quality.values()),
                    f"{suite_name}_diagnostic_quality_gates",
                    f"quality_gates={quality}",
                    checks,
                )
        _check(manifest_path.exists(), f"{suite_name}_manifest_exists", str(manifest_path), checks)
        _check(aggregate_path.exists(), f"{suite_name}_aggregate_exists", str(aggregate_path), checks)
        _check(
            sid_factor_path.exists(),
            f"{suite_name}_sid_factor_audit_exists",
            str(sid_factor_path),
            checks,
        )
        if sid_factor_path.exists():
            sid_factor = _load_json(sid_factor_path)
            _check(
                sid_factor.get("passed") is True,
                f"{suite_name}_sid_factor_audit_passed",
                f"passed={sid_factor.get('passed')}, n_runs={sid_factor.get('n_runs')}",
                checks,
            )
            sid_factor_runs = sid_factor.get("runs", [])
            _check(
                isinstance(sid_factor_runs, list)
                and len(sid_factor_runs) >= int(min_seeds)
                and sid_factor.get("n_runs") == len(sid_factor_runs),
                f"{suite_name}_sid_factor_audit_min_runs",
                (
                    f"n_runs={sid_factor.get('n_runs')}, "
                    f"actual_runs={len(sid_factor_runs) if isinstance(sid_factor_runs, list) else None}, "
                    f"min_seeds={min_seeds}"
                ),
                checks,
            )
            if isinstance(sid_factor_runs, list):
                for index, sid_row in enumerate(sid_factor_runs):
                    audit = sid_row.get("audit", {}) if isinstance(sid_row, dict) else {}
                    metrics = audit.get("metrics", {}) if isinstance(audit, dict) else {}
                    missing_metrics = sorted(REQUIRED_SID_FACTOR_METRICS.difference(metrics))
                    eval_splits = audit.get("eval_splits", [])
                    sid_run_dir = _resolve_run_dir(
                        sid_factor_path,
                        str(sid_row.get("run_dir", "")),
                    )
                    local_sid_audit_path = sid_run_dir / "sid_factor_audit.json"
                    _check(
                        audit.get("passed") is True,
                        f"{suite_name}_sid_factor_run_{index}_passed",
                        f"passed={audit.get('passed')}",
                        checks,
                    )
                    _check(
                        audit.get("method") == "sid",
                        f"{suite_name}_sid_factor_run_{index}_method",
                        f"method={audit.get('method')}",
                        checks,
                    )
                    _check(
                        audit.get("benchmark_name") == benchmark_name,
                        f"{suite_name}_sid_factor_run_{index}_benchmark",
                        (
                            f"benchmark_name={audit.get('benchmark_name')}, "
                            f"expected={benchmark_name}"
                        ),
                        checks,
                    )
                    _check(
                        str(Path(str(audit.get("run_dir", ""))).resolve())
                        == str(sid_run_dir.resolve()),
                        f"{suite_name}_sid_factor_run_{index}_run_dir_identity",
                        f"audit.run_dir={audit.get('run_dir')}, row.run_dir={sid_run_dir}",
                        checks,
                    )
                    _check(
                        local_sid_audit_path.exists(),
                        f"{suite_name}_sid_factor_run_{index}_local_audit_exists",
                        str(local_sid_audit_path),
                        checks,
                    )
                    if local_sid_audit_path.exists():
                        local_sid_audit = _load_json(local_sid_audit_path)
                        _check(
                            local_sid_audit == audit,
                            f"{suite_name}_sid_factor_run_{index}_local_audit_matches_summary",
                            str(local_sid_audit_path),
                            checks,
                        )
                    checkpoint = str(audit.get("checkpoint", ""))
                    _check(
                        bool(checkpoint) and (sid_run_dir / checkpoint).is_file(),
                        f"{suite_name}_sid_factor_run_{index}_checkpoint_exists",
                        f"checkpoint={checkpoint}, run_dir={sid_run_dir}",
                        checks,
                    )
                    _check(
                        audit.get("probe_train_split") == "val_iid",
                        f"{suite_name}_sid_factor_run_{index}_probe_train_split",
                        f"probe_train_split={audit.get('probe_train_split')}",
                        checks,
                    )
                    _check(
                        sorted(eval_splits) == sorted(SID_AUDIT_EVAL_SPLITS),
                        f"{suite_name}_sid_factor_run_{index}_eval_splits",
                        f"eval_splits={eval_splits}",
                        checks,
                    )
                    _check(
                        not missing_metrics,
                        f"{suite_name}_sid_factor_run_{index}_required_metrics",
                        f"missing={missing_metrics[:20]}",
                        checks,
                    )
                    _check(
                        metrics.get("task_head_excludes_z_ir_spur") is True,
                        f"{suite_name}_sid_factor_run_{index}_task_head_lock",
                        (
                            "task_head_excludes_z_ir_spur="
                            f"{metrics.get('task_head_excludes_z_ir_spur')}"
                        ),
                        checks,
                    )
                    role_alignment = audit.get("role_alignment", {})
                    role_claim_metric = metrics.get("decomposition_role_claim_ready")
                    _check(
                        isinstance(role_alignment, Mapping)
                        and role_alignment.get("decomposition_role_claim_ready")
                        == role_claim_metric
                        and isinstance(
                            role_alignment.get("decomposition_role_alignment_failures"),
                            list,
                        ),
                        f"{suite_name}_sid_factor_run_{index}_role_alignment_schema",
                        (
                            "role_alignment_ready="
                            f"{role_alignment.get('decomposition_role_claim_ready') if isinstance(role_alignment, Mapping) else None}, "
                            f"metric_ready={role_claim_metric}, "
                            "failures="
                            f"{role_alignment.get('decomposition_role_alignment_failures') if isinstance(role_alignment, Mapping) else None}"
                        ),
                        checks,
                    )
                    target_metadata = audit.get("target_metadata", {})
                    _check(
                        isinstance(target_metadata, Mapping)
                        and all(target in target_metadata for target in SID_AUDIT_TARGETS),
                        f"{suite_name}_sid_factor_run_{index}_target_metadata",
                        f"target_metadata_keys={sorted(target_metadata) if isinstance(target_metadata, Mapping) else None}",
                        checks,
                    )
        _check(
            itm_mechanism_path.exists(),
            f"{suite_name}_itm_mechanism_audit_exists",
            str(itm_mechanism_path),
            checks,
        )
        if itm_mechanism_path.exists():
            itm_mechanism = _load_json(itm_mechanism_path)
            _check(
                itm_mechanism.get("passed") is True,
                f"{suite_name}_itm_mechanism_audit_passed",
                (
                    f"passed={itm_mechanism.get('passed')}, "
                    f"n_runs={itm_mechanism.get('n_runs')}, "
                    f"ready_runs={itm_mechanism.get('mechanism_claim_ready_runs')}"
                ),
                checks,
            )
            itm_runs = itm_mechanism.get("runs", [])
            _check(
                isinstance(itm_runs, list)
                and len(itm_runs) >= int(min_seeds)
                and itm_mechanism.get("n_runs") == len(itm_runs),
                f"{suite_name}_itm_mechanism_audit_min_runs",
                (
                    f"n_runs={itm_mechanism.get('n_runs')}, "
                    f"actual_runs={len(itm_runs) if isinstance(itm_runs, list) else None}, "
                    f"min_seeds={min_seeds}"
                ),
                checks,
            )
            if isinstance(itm_runs, list):
                for index, itm_row in enumerate(itm_runs):
                    audit = itm_row.get("audit", {}) if isinstance(itm_row, dict) else {}
                    metrics = audit.get("metrics", {}) if isinstance(audit, dict) else {}
                    missing_metrics = sorted(REQUIRED_ITM_MECHANISM_METRICS.difference(metrics))
                    eval_splits = audit.get("eval_splits", [])
                    itm_run_dir = _resolve_run_dir(
                        itm_mechanism_path,
                        str(itm_row.get("run_dir", "")),
                    )
                    local_itm_audit_path = itm_run_dir / "itm_mechanism_audit.json"
                    _check(
                        audit.get("passed") is True,
                        f"{suite_name}_itm_mechanism_run_{index}_passed",
                        f"passed={audit.get('passed')}",
                        checks,
                    )
                    _check(
                        audit.get("method") == "itm",
                        f"{suite_name}_itm_mechanism_run_{index}_method",
                        f"method={audit.get('method')}",
                        checks,
                    )
                    _check(
                        audit.get("benchmark_name") == benchmark_name,
                        f"{suite_name}_itm_mechanism_run_{index}_benchmark",
                        (
                            f"benchmark_name={audit.get('benchmark_name')}, "
                            f"expected={benchmark_name}"
                        ),
                        checks,
                    )
                    _check(
                        str(Path(str(audit.get("run_dir", ""))).resolve())
                        == str(itm_run_dir.resolve()),
                        f"{suite_name}_itm_mechanism_run_{index}_run_dir_identity",
                        f"audit.run_dir={audit.get('run_dir')}, row.run_dir={itm_run_dir}",
                        checks,
                    )
                    _check(
                        local_itm_audit_path.exists(),
                        f"{suite_name}_itm_mechanism_run_{index}_local_audit_exists",
                        str(local_itm_audit_path),
                        checks,
                    )
                    if local_itm_audit_path.exists():
                        local_itm_audit = _load_json(local_itm_audit_path)
                        _check(
                            local_itm_audit == audit,
                            f"{suite_name}_itm_mechanism_run_{index}_local_audit_matches_summary",
                            str(local_itm_audit_path),
                            checks,
                        )
                    checkpoint = str(audit.get("checkpoint", ""))
                    _check(
                        bool(checkpoint) and (itm_run_dir / checkpoint).is_file(),
                        f"{suite_name}_itm_mechanism_run_{index}_checkpoint_exists",
                        f"checkpoint={checkpoint}, run_dir={itm_run_dir}",
                        checks,
                    )
                    _check(
                        audit.get("probe_train_split") == "val_iid",
                        f"{suite_name}_itm_mechanism_run_{index}_probe_train_split",
                        f"probe_train_split={audit.get('probe_train_split')}",
                        checks,
                    )
                    _check(
                        sorted(eval_splits) == sorted(ITM_AUDIT_EVAL_SPLITS),
                        f"{suite_name}_itm_mechanism_run_{index}_eval_splits",
                        f"eval_splits={eval_splits}",
                        checks,
                    )
                    _check(
                        not missing_metrics,
                        f"{suite_name}_itm_mechanism_run_{index}_required_metrics",
                        f"missing={missing_metrics[:20]}",
                        checks,
                    )
                    _check(
                        metrics.get("task_head_excludes_spur_mechanism") is True,
                        f"{suite_name}_itm_mechanism_run_{index}_task_head_lock",
                        (
                            "task_head_excludes_spur_mechanism="
                            f"{metrics.get('task_head_excludes_spur_mechanism')}"
                        ),
                        checks,
                    )
                    role_alignment = audit.get("role_alignment", {})
                    role_claim_metric = metrics.get("mechanism_claim_ready")
                    _check(
                        isinstance(role_alignment, Mapping)
                        and role_alignment.get("mechanism_claim_ready")
                        == role_claim_metric
                        and isinstance(
                            role_alignment.get("mechanism_alignment_failures"),
                            list,
                        ),
                        f"{suite_name}_itm_mechanism_run_{index}_role_alignment_schema",
                        (
                            "role_alignment_ready="
                            f"{role_alignment.get('mechanism_claim_ready') if isinstance(role_alignment, Mapping) else None}, "
                            f"metric_ready={role_claim_metric}, "
                            "failures="
                            f"{role_alignment.get('mechanism_alignment_failures') if isinstance(role_alignment, Mapping) else None}"
                        ),
                        checks,
                    )
                    target_metadata = audit.get("target_metadata", {})
                    _check(
                        isinstance(target_metadata, Mapping)
                        and all(target in target_metadata for target in ITM_AUDIT_TARGETS),
                        f"{suite_name}_itm_mechanism_run_{index}_target_metadata",
                        f"target_metadata_keys={sorted(target_metadata) if isinstance(target_metadata, Mapping) else None}",
                        checks,
                    )
        if not manifest_path.exists():
            continue
        manifest = _load_json(manifest_path)
        aggregate = _load_json(aggregate_path) if aggregate_path.exists() else {}
        runs = manifest.get("runs", [])
        run_count += len(runs)
        methods = sorted({row.get("method") for row in runs})
        _check(
            methods == sorted(REQUIRED_METHODS),
            f"{suite_name}_required_methods_present",
            f"methods={methods}",
            checks,
        )
        failed = [row for row in runs if row.get("status") != "success"]
        _check(not failed, f"{suite_name}_no_failed_runs", f"failed={len(failed)}", checks)

        seeds_by_method: dict[str, set[int]] = {method: set() for method in REQUIRED_METHODS}
        for row in runs:
            method = str(row.get("method"))
            seed = row.get("seed")
            if method in seeds_by_method and seed is not None and row.get("status") == "success":
                seeds_by_method[method].add(int(seed))
        seed_summary[suite_name] = {
            method: sorted(seeds)
            for method, seeds in sorted(seeds_by_method.items())
        }
        insufficient = {
            method: sorted(seeds)
            for method, seeds in seeds_by_method.items()
            if len(seeds) < int(min_seeds)
        }
        _check(
            not insufficient,
            f"{suite_name}_min_seed_count",
            f"min_seeds={min_seeds}, insufficient={insufficient}",
            checks,
        )

        metric_rows_by_method: dict[str, list[dict[str, float]]] = {
            method: [] for method in REQUIRED_METHODS
        }
        for row in runs:
            method = str(row.get("method"))
            run_dir = _resolve_run_dir(manifest_path, str(row.get("run_dir", "")))
            metadata_path = run_dir / "metadata.json"
            metrics_path = run_dir / "final_metrics.json"
            resolved_config_path = run_dir / "resolved_config.json"
            if not _check(
                metadata_path.exists(),
                f"{suite_name}_{method}_metadata_exists",
                str(metadata_path),
                checks,
            ):
                continue
            if not _check(
                metrics_path.exists(),
                f"{suite_name}_{method}_metrics_exists",
                str(metrics_path),
                checks,
            ):
                continue
            if not _check(
                resolved_config_path.exists(),
                f"{suite_name}_{method}_resolved_config_exists",
                str(resolved_config_path),
                checks,
            ):
                continue
            metadata = _load_json(metadata_path)
            metrics = _load_json(metrics_path)
            resolved_config = _load_json(resolved_config_path)
            _check(
                row.get("seed") == metadata.get("seed") == resolved_config.get("seed"),
                f"{suite_name}_{method}_seed_identity",
                (
                    f"manifest_seed={row.get('seed')}, "
                    f"metadata_seed={metadata.get('seed')}, "
                    f"resolved_config_seed={resolved_config.get('seed')}"
                ),
                checks,
            )
            _check(
                metadata.get("run_id") == run_dir.name,
                f"{suite_name}_{method}_run_id_matches_run_dir",
                f"metadata.run_id={metadata.get('run_id')}, run_dir.name={run_dir.name}",
                checks,
            )
            _check(
                row.get("config_hash") == metadata.get("config_hash"),
                f"{suite_name}_{method}_config_hash_identity",
                (
                    f"manifest_config_hash={row.get('config_hash')}, "
                    f"metadata_config_hash={metadata.get('config_hash')}"
                ),
                checks,
            )
            manifest_metrics = row.get("metrics")
            _check(
                isinstance(manifest_metrics, Mapping) and dict(manifest_metrics) == metrics,
                f"{suite_name}_{method}_manifest_metrics_match_final_metrics",
                (
                    f"seed={row.get('seed')}, run_dir={run_dir}, "
                    f"manifest_metrics_type={type(manifest_metrics).__name__}"
                ),
                checks,
            )
            if method in metric_rows_by_method:
                metric_rows_by_method[method].append(_numeric_items(metrics))
            config_path = str(
                metadata.get("config_path")
                or resolved_config.get("run", {}).get("config_path", "")
            )
            experiment_name = str(resolved_config.get("experiment", ""))
            if not allow_smoke:
                _check(
                    "smoke" not in config_path.lower()
                    and "smoke" not in experiment_name.lower(),
                    f"{suite_name}_{method}_non_smoke_config",
                    f"config_path={config_path}, experiment={experiment_name}",
                    checks,
                )
                _check(
                    Path(config_path).as_posix() == EXPECTED_FULL_CONFIGS[suite_name],
                    f"{suite_name}_{method}_expected_full_config",
                    (
                        f"config_path={config_path}, "
                        f"expected={EXPECTED_FULL_CONFIGS[suite_name]}"
                    ),
                    checks,
                )
                split_cfg = resolved_config.get("splits", {})
                split_sizes = {
                    split_name: int(
                        split_cfg.get(split_name, {}).get("n_sequences", 0)
                    )
                    for split_name in EXPECTED_FULL_SPLITS
                }
                _check(
                    all(
                        split_sizes[split_name] >= minimum
                        for split_name, minimum in EXPECTED_FULL_SPLITS.items()
                    ),
                    f"{suite_name}_{method}_full_split_sizes",
                    f"observed={split_sizes}, required={EXPECTED_FULL_SPLITS}",
                    checks,
                )
            observed_benchmark = metadata.get("dataset_metadata", {}).get("train", {}).get(
                "benchmark_name"
            )
            _check(
                observed_benchmark == benchmark_name,
                f"{suite_name}_{method}_benchmark_metadata",
                f"observed={observed_benchmark}, expected={benchmark_name}",
                checks,
            )
            _check(
                metadata.get("method") == method,
                f"{suite_name}_{method}_method_metadata",
                f"metadata.method={metadata.get('method')}, manifest.method={method}",
                checks,
            )
            training_epochs = int(resolved_config.get("training", {}).get("epochs", 0))
            _check(
                training_epochs >= int(min_epochs),
                f"{suite_name}_{method}_min_epochs_configured",
                f"epochs={training_epochs}, min_epochs={min_epochs}",
                checks,
            )
            if method in {"erm", "ib", "ep_min", "ep_max", "sib", "sid", "itm"}:
                completed = int(metrics.get("epochs_completed", 0))
                _check(
                    completed > 0,
                    f"{suite_name}_{method}_epochs_completed_logged",
                    f"epochs_completed={completed}",
                    checks,
                )
            epoch_mismatches = _epoch_range_mismatches(metrics, method)
            _check(
                not epoch_mismatches,
                f"{suite_name}_{method}_epoch_ranges_valid",
                f"mismatches={epoch_mismatches}",
                checks,
            )
            log_mismatches = _training_log_mismatches(run_dir, method)
            _check(
                not log_mismatches,
                f"{suite_name}_{method}_training_logs_present",
                f"mismatches={log_mismatches}",
                checks,
            )
            _check(
                _metrics_have_final_eval(metrics, method),
                f"{suite_name}_{method}_final_eval_metrics",
                f"metric_keys={sorted(metrics)[:20]}...",
                checks,
            )
            missing_checkpoints = _missing_checkpoint_files(
                run_dir=run_dir,
                metrics=metrics,
                method=method,
            )
            _check(
                not missing_checkpoints,
                f"{suite_name}_{method}_checkpoint_files_exist",
                f"missing={missing_checkpoints}",
                checks,
            )
            no_ood_tuning, tuning_detail = _metadata_no_ood_tuning(metadata)
            _check(
                no_ood_tuning,
                f"{suite_name}_{method}_no_ood_tuning_metadata",
                tuning_detail,
                checks,
            )
            no_config_ood_tuning, config_tuning_detail = _resolved_config_no_ood_tuning(
                resolved_config
            )
            _check(
                no_config_ood_tuning,
                f"{suite_name}_{method}_no_ood_tuning_resolved_config",
                config_tuning_detail,
                checks,
            )
            no_metric_ood_tuning, metric_tuning_detail = _metrics_no_ood_tuning(metrics)
            _check(
                no_metric_ood_tuning,
                f"{suite_name}_{method}_no_ood_tuning_metrics",
                metric_tuning_detail,
                checks,
            )
            if method in {"ocp_style", "lens_like_arrow_classifier"}:
                _check(
                    metadata.get("pretraining_split") == "train",
                    f"{suite_name}_{method}_pretraining_train_only",
                    f"pretraining_split={metadata.get('pretraining_split')}",
                    checks,
                )
                _check(
                    metadata.get("transductive") is False,
                    f"{suite_name}_{method}_not_transductive",
                    f"transductive={metadata.get('transductive')}",
                    checks,
                )
            if method in {"sib", "sid", "itm"}:
                has_cf = any(key.endswith("cf_prediction_consistency") for key in metrics)
                _check(
                    has_cf,
                    f"{suite_name}_{method}_counterfactual_metrics",
                    "requires cf_prediction_consistency metric",
                    checks,
                )
            if method == "itm":
                training_cfg = resolved_config.get("training", {})
                itm_schedule_cfg = resolved_config.get("itm_schedule", {})
                task_warmup = max(int(itm_schedule_cfg.get("task_warmup_epochs", 5)), 0)
                transition_warmup = max(
                    int(itm_schedule_cfg.get("transition_warmup_epochs", 5)), 0
                )
                ramp = max(int(itm_schedule_cfg.get("regularizer_ramp_epochs", 10)), 0)
                itm_required_epochs = max(task_warmup + transition_warmup + ramp, 1)
                configured_checkpoint_floor = int(
                    training_cfg.get("min_epoch_for_checkpoint_selection", 0)
                )
                configured_early_floor = int(
                    training_cfg.get("min_epochs_before_early_stopping", 0)
                )
                schedule_required = int(
                    metrics.get("itm_schedule_required_epochs", 0) or 0
                )
                selected_epoch = int(metrics.get("selected_epoch", -1))
                completed = int(metrics.get("epochs_completed", 0))
                selected_progress = float(
                    metrics.get("itm_selected_schedule_progress", 0.0) or 0.0
                )
                selected_transition_progress = float(
                    metrics.get(
                        "itm_selected_transition_schedule_progress", 0.0
                    )
                    or 0.0
                )
                _check(
                    schedule_required == itm_required_epochs,
                    f"{suite_name}_{method}_itm_schedule_required_logged",
                    (
                        f"logged={schedule_required}, "
                        f"expected={itm_required_epochs}"
                    ),
                    checks,
                )
                _check(
                    configured_checkpoint_floor >= itm_required_epochs,
                    f"{suite_name}_{method}_itm_checkpoint_floor_configured",
                    (
                        f"min_epoch_for_checkpoint_selection={configured_checkpoint_floor}, "
                        f"itm_schedule_required_epochs={itm_required_epochs}"
                    ),
                    checks,
                )
                _check(
                    configured_early_floor >= itm_required_epochs,
                    f"{suite_name}_{method}_itm_early_stop_floor_configured",
                    (
                        f"min_epochs_before_early_stopping={configured_early_floor}, "
                        f"itm_schedule_required_epochs={itm_required_epochs}"
                    ),
                    checks,
                )
                _check(
                    completed >= itm_required_epochs,
                    f"{suite_name}_{method}_itm_completed_schedule",
                    f"epochs_completed={completed}, itm_schedule_required_epochs={itm_required_epochs}",
                    checks,
                )
                _check(
                    selected_epoch + 1 >= itm_required_epochs,
                    f"{suite_name}_{method}_itm_selected_checkpoint_after_schedule",
                    f"selected_epoch={selected_epoch}, itm_schedule_required_epochs={itm_required_epochs}",
                    checks,
                )
                _check(
                    bool(metrics.get("itm_schedule_floor_satisfied")) is True
                    and selected_progress >= 1.0
                    and selected_transition_progress >= 1.0,
                    f"{suite_name}_{method}_itm_schedule_floor_satisfied",
                    (
                        f"itm_schedule_floor_satisfied={metrics.get('itm_schedule_floor_satisfied')}, "
                        f"itm_selected_schedule_progress={selected_progress}, "
                        "itm_selected_transition_schedule_progress="
                        f"{selected_transition_progress}"
                    ),
                    checks,
                )
                itm_arrow_delta_values = {
                    key: value
                    for key, value in metrics.items()
                    if "_cf_delta_arrow_" in key and value is not None
                }
                itm_arrow_availability_values = {
                    key: value
                    for key, value in metrics.items()
                    if key.endswith("_cf_arrow_metrics_available")
                }
                _check(
                    not itm_arrow_delta_values,
                    f"{suite_name}_{method}_arrow_delta_metrics_unavailable",
                    (
                        "ITM has no forward/reverse latent arrow dynamics; "
                        f"numeric placeholders={itm_arrow_delta_values}"
                    ),
                    checks,
                )
                _check(
                    bool(itm_arrow_availability_values)
                    and all(value is False for value in itm_arrow_availability_values.values()),
                    f"{suite_name}_{method}_arrow_metric_availability_logged",
                    f"availability={itm_arrow_availability_values}",
                    checks,
                )
            if method == "sid":
                training_cfg = resolved_config.get("training", {})
                configured_checkpoint_floor = int(
                    training_cfg.get("min_epoch_for_checkpoint_selection", 0)
                )
                configured_early_floor = int(
                    training_cfg.get("min_epochs_before_early_stopping", 0)
                )
                schedule_required = int(metrics.get("sid_schedule_required_epochs", 0) or 0)
                selected_epoch = int(metrics.get("selected_epoch", -1))
                completed = int(metrics.get("epochs_completed", 0))
                selected_progress = float(metrics.get("sid_selected_schedule_progress", 0.0) or 0.0)
                _check(
                    schedule_required > 0,
                    f"{suite_name}_{method}_sid_schedule_required_logged",
                    f"sid_schedule_required_epochs={schedule_required}",
                    checks,
                )
                _check(
                    configured_checkpoint_floor >= schedule_required,
                    f"{suite_name}_{method}_sid_checkpoint_floor_configured",
                    (
                        f"min_epoch_for_checkpoint_selection={configured_checkpoint_floor}, "
                        f"sid_schedule_required_epochs={schedule_required}"
                    ),
                    checks,
                )
                _check(
                    configured_early_floor >= schedule_required,
                    f"{suite_name}_{method}_sid_early_stop_floor_configured",
                    (
                        f"min_epochs_before_early_stopping={configured_early_floor}, "
                        f"sid_schedule_required_epochs={schedule_required}"
                    ),
                    checks,
                )
                _check(
                    completed >= schedule_required,
                    f"{suite_name}_{method}_sid_completed_schedule",
                    f"epochs_completed={completed}, sid_schedule_required_epochs={schedule_required}",
                    checks,
                )
                _check(
                    selected_epoch + 1 >= schedule_required,
                    f"{suite_name}_{method}_sid_selected_checkpoint_after_schedule",
                    f"selected_epoch={selected_epoch}, sid_schedule_required_epochs={schedule_required}",
                    checks,
                )
                _check(
                    bool(metrics.get("sid_schedule_floor_satisfied")) is True
                    and selected_progress >= 1.0,
                    f"{suite_name}_{method}_sid_schedule_floor_satisfied",
                    (
                        f"sid_schedule_floor_satisfied={metrics.get('sid_schedule_floor_satisfied')}, "
                        f"sid_selected_schedule_progress={selected_progress}"
                    ),
                    checks,
                )

        _check(
            aggregate.get("n_skipped_metric_files") == 0
            and aggregate.get("skipped_metric_files") == [],
            f"{suite_name}_aggregate_no_skipped_metric_files",
            (
                f"n_skipped_metric_files={aggregate.get('n_skipped_metric_files')}, "
                f"skipped_metric_files={aggregate.get('skipped_metric_files')}"
            ),
            checks,
        )
        aggregate_payload = _aggregate_method_payloads(aggregate)
        for method in REQUIRED_METHODS:
            aggregate_row = aggregate_payload.get(method, {})
            method_rows = metric_rows_by_method[method]
            _check(
                isinstance(aggregate_row, Mapping),
                f"{suite_name}_{method}_aggregate_method_row_exists",
                f"aggregate_row_type={type(aggregate_row).__name__}",
                checks,
            )
            if isinstance(aggregate_row, Mapping):
                mismatches = _aggregate_method_mismatches(
                    aggregate_row=aggregate_row,
                    metric_rows=method_rows,
                )
                _check(
                    not mismatches,
                    f"{suite_name}_{method}_aggregate_matches_run_metrics",
                    f"mismatches={mismatches[:10]}",
                    checks,
                )

    failed_checks = [check["name"] for check in checks if not check["passed"]]
    passed = not failed_checks
    return {
        "schema_version": EVIDENCE_AUDIT_SCHEMA_VERSION,
        "passed": passed,
        "root": str(root),
        "min_seeds": int(min_seeds),
        "min_epochs": int(min_epochs),
        "allow_smoke": bool(allow_smoke),
        "preflight_path": None if preflight_path is None else str(preflight_path),
        "preflight_environment_summary": preflight_environment_summary,
        "required_methods": list(REQUIRED_METHODS),
        "expected_benchmarks": EXPECTED_BENCHMARKS,
        "run_count": run_count,
        "seed_summary": seed_summary,
        "n_checks": len(checks),
        "n_failed": len(failed_checks),
        "failed_checks": failed_checks,
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit final evidence roots.")
    parser.add_argument("--root", default="results/full_run")
    parser.add_argument("--output", default=None)
    parser.add_argument("--min-seeds", type=int, default=5)
    parser.add_argument("--min-epochs", type=int, default=25)
    parser.add_argument("--preflight-path", default=None)
    parser.add_argument("--allow-smoke", action="store_true")
    args = parser.parse_args()
    report = audit_evidence(
        args.root,
        min_seeds=args.min_seeds,
        min_epochs=args.min_epochs,
        allow_smoke=args.allow_smoke,
        preflight_path=args.preflight_path,
    )
    output = Path(args.output) if args.output else Path(args.root) / "evidence_audit.json"
    save_json(output, report)
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "run_count": report["run_count"],
                "n_checks": report["n_checks"],
                "n_failed": report["n_failed"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
