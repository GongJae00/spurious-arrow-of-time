"""Scientific validation gates for STA-Bench experiment outputs."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.train.common import save_json

VALIDATION_SCHEMA_VERSION = 2


@dataclass
class Check:
    name: str
    passed: bool
    detail: str
    severity: str = "error"

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "detail": self.detail,
            "severity": self.severity,
        }


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _add(checks: list[Check], name: str, passed: bool, detail: str, severity: str = "error") -> None:
    checks.append(Check(name=name, passed=bool(passed), detail=detail, severity=severity))


def _iter_run_dirs(root: Path) -> list[tuple[Path, str | None]]:
    return sorted((path.parent, None) for path in root.rglob("final_metrics.json"))


def _resolve_manifest_run_dir(root: Path, run_dir: str) -> Path:
    path = Path(run_dir)
    if path.is_absolute():
        return path.resolve()
    candidates = [
        (Path.cwd() / path).resolve(),
        (root / path).resolve(),
        (root.parent / path).resolve(),
    ]
    return next((candidate for candidate in candidates if candidate.exists()), candidates[0])


def _manifest_run_dirs(
    root: Path,
) -> tuple[list[tuple[Path, str | None]], list[dict[str, Any]], list[Path], bool]:
    """Return successful run dirs plus manifest integrity details.

    If a manifest exists, validation must use it as the authoritative run list
    instead of scanning arbitrary old final_metrics files under the root.
    """

    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        return _iter_run_dirs(root), [], [], False

    manifest = _load_json(manifest_path)
    successful: list[tuple[Path, str | None]] = []
    failed: list[dict[str, Any]] = []
    manifested_dirs: set[Path] = set()
    for row in manifest.get("runs", []):
        run_dir = row.get("run_dir")
        if not run_dir:
            continue
        resolved = _resolve_manifest_run_dir(root, str(run_dir))
        manifested_dirs.add(resolved)
        if row.get("status", "success") == "success":
            method = row.get("method")
            successful.append((resolved, str(method) if method else None))
        elif row.get("status") == "failed":
            failed.append(row)

    all_metrics = {path.parent.resolve() for path in root.rglob("final_metrics.json")}
    stale_metrics = sorted(all_metrics - manifested_dirs)
    return sorted(successful), failed, stale_metrics, True


def _is_randomized_control_context(
    run_dir: Path,
    dataset: dict[str, Any],
    config: dict[str, Any] | None,
) -> bool:
    config = config or {}
    if "negative_controls" in str(run_dir):
        return True
    if config.get("experiment") == "ablation_negative_controls":
        return True
    ablation = config.get("ablation") or {}
    if ablation.get("name") == "no_spurious_correlation":
        return True
    if config.get("diagnostic") == "randomized_labels":
        return True
    modes = {
        data.get("spurious", {}).get("spurious_mode")
        for data in dataset.values()
        if isinstance(data, dict)
    }
    return bool(modes) and modes == {"randomized"}


def _is_static_spurious_control_context(run_dir: Path, config: dict[str, Any] | None) -> bool:
    config = config or {}
    ablation = config.get("ablation") or {}
    return (
        ablation.get("name") == "static_spurious_control"
        or config.get("experiment") == "static_spurious_control"
        or "static_spurious_control" in str(run_dir)
    )


def _effective_initial_state_mode(process: dict[str, Any]) -> str | None:
    return process.get("effective_initial_state_mode", process.get("initial_state_mode"))


def _validate_dataset_metadata(
    checks: list[Check],
    run_dir: Path,
    metadata: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> None:
    dataset = metadata.get("dataset_metadata", {})
    required_splits = {"train", "val_iid", "iid_test", "ood_test"}
    _add(
        checks,
        f"{run_dir}:dataset_splits",
        set(dataset) == required_splits,
        f"splits={sorted(dataset)}",
    )
    if set(dataset) != required_splits:
        return

    hashes = {
        split: data.get("observation", {}).get("mixing_matrix_hash")
        for split, data in dataset.items()
    }
    _add(
        checks,
        f"{run_dir}:same_mixing_matrix",
        len(set(hashes.values())) == 1 and None not in hashes.values(),
        f"hashes={hashes}",
    )

    thresholds = {
        split: data.get("core", {}).get("label_threshold")
        for split, data in dataset.items()
    }
    _add(
        checks,
        f"{run_dir}:train_threshold_reused",
        len(set(thresholds.values())) == 1 and None not in thresholds.values(),
        f"thresholds={thresholds}",
    )
    forbidden_eval_fallback = {
        split: data.get("core", {}).get("label_balance_fallback_used")
        for split, data in dataset.items()
        if split != "train"
    }
    _add(
        checks,
        f"{run_dir}:eval_label_balance_fallback_disabled",
        all(value is False for value in forbidden_eval_fallback.values()),
        f"eval_fallback={forbidden_eval_fallback}",
    )
    core_initial_modes = {
        split: _effective_initial_state_mode(data.get("core", {}))
        for split, data in dataset.items()
    }
    spurious_initial_modes = {
        split: _effective_initial_state_mode(data.get("spurious", {}))
        for split, data in dataset.items()
    }
    static_control_context = _is_static_spurious_control_context(run_dir, config)
    core_stationary = all(value == "uniform_stationary" for value in core_initial_modes.values())
    if static_control_context:
        spurious_stationary_ok = all(
            value in {"uniform_stationary", "sector_conditioned", "provided"}
            for value in spurious_initial_modes.values()
        )
    else:
        spurious_stationary_ok = all(
            value == "uniform_stationary" for value in spurious_initial_modes.values()
        )
    _add(
        checks,
        f"{run_dir}:stationary_initialization_lock",
        core_stationary and spurious_stationary_ok,
        (
            f"core_initial_modes={core_initial_modes}, "
            f"spurious_initial_modes={spurious_initial_modes}, "
            f"static_control_context={static_control_context}"
        ),
    )

    train = dataset["train"]
    core_ep = train.get("core", {}).get("analytic_ep")
    spur_ep = train.get("spurious", {}).get("analytic_ep")
    is_negative_control = "negative_controls" in str(run_dir) or "sigma_s_less_than_sigma_c" in str(run_dir)
    sweep = (config or {}).get("sweep", {})
    is_ep_ratio_sweep = sweep.get("type") == "ep_ratio"
    _add(
        checks,
        f"{run_dir}:main_trap_spurious_ep_ge_core_ep",
        is_negative_control
        or is_ep_ratio_sweep
        or (spur_ep is not None and core_ep is not None and spur_ep > core_ep),
        (
            f"core_ep={core_ep}, spur_ep={spur_ep}, negative_control={is_negative_control}, "
            f"ep_ratio_sweep={is_ep_ratio_sweep}"
        ),
        severity="warning" if (is_negative_control or is_ep_ratio_sweep) else "error",
    )
    if is_ep_ratio_sweep and core_ep is not None and spur_ep is not None:
        expected_ratio = float(spur_ep) / float(core_ep) if float(core_ep) > 0 else float("inf")
        actual_ratio = sweep.get("actual_ratio")
        _add(
            checks,
            f"{run_dir}:ep_ratio_actual_axis",
            actual_ratio is not None and abs(float(actual_ratio) - expected_ratio) < 1e-6,
            f"logged_actual_ratio={actual_ratio}, metadata_ratio={expected_ratio}, sweep={sweep}",
        )

    corr = {
        split: data.get("spurious", {}).get("corr_y_spurious_dynamic_stat")
        for split, data in dataset.items()
    }
    spurious_type = train.get("spurious", {}).get("spurious_correlation_type")
    if spurious_type == "initial_sector_static_control":
        static_context = _is_static_spurious_control_context(run_dir, config)
        _add(
            checks,
            f"{run_dir}:dynamic_spurious_correlation",
            static_context,
            "static_spurious_control diagnostic; dynamic-correlation gate not applied",
            severity="warning" if static_context else "error",
        )
    elif not all(value is not None for value in corr.values()):
        _add(
            checks,
            f"{run_dir}:dynamic_spurious_correlation",
            False,
            f"missing dynamic spurious correlation metadata; corr={corr}",
        )
    else:
        train_like = [corr["train"], corr["val_iid"], corr["iid_test"]]
        ood = corr["ood_test"]
        strength = train.get("spurious", {}).get("spurious_label_correlation_strength")
        if strength is None:
            min_abs_corr = 0.25
        else:
            min_abs_corr = max(0.15, min(0.25, 0.5 * abs(float(strength))))
        train_like_nontrivial = all(abs(value) >= min_abs_corr for value in train_like)
        ood_breaks = abs(ood) <= 0.25 or (corr["train"] * ood < 0)
        randomized_control = all(abs(value) <= 0.35 for value in corr.values())
        randomized_context = _is_randomized_control_context(run_dir, dataset, config)
        _add(
            checks,
            f"{run_dir}:dynamic_spurious_correlation",
            (randomized_context and randomized_control)
            or (train_like_nontrivial and ood_breaks),
            (
                f"corr={corr}, randomized_control={randomized_control}, "
                f"randomized_context={randomized_context}, min_abs_corr={min_abs_corr}"
            ),
        )

    cf = train.get("counterfactual", {})
    _add(
        checks,
        f"{run_dir}:counterfactual_mode_logged",
        "spurious_cf_mode" in cf and "corr_y_spurious_cf_dynamic_stat" in cf,
        f"counterfactual={cf}",
    )


def _validate_final_metrics(checks: list[Check], run_dir: Path, metrics: dict[str, Any]) -> None:
    if {"iid_test_accuracy", "ood_test_accuracy", "ood_gap"}.issubset(metrics):
        expected = float(metrics["iid_test_accuracy"]) - float(metrics["ood_test_accuracy"])
        observed = float(metrics["ood_gap"])
        _add(
            checks,
            f"{run_dir}:ood_gap_definition",
            abs(expected - observed) < 1e-9,
            f"expected={expected}, observed={observed}",
        )
    for prefix in ("frozen_encoder", "fine_tuned_encoder"):
        keys = {
            f"{prefix}_iid_test_accuracy",
            f"{prefix}_ood_test_accuracy",
            f"{prefix}_ood_gap",
        }
        if keys.issubset(metrics):
            expected = float(metrics[f"{prefix}_iid_test_accuracy"]) - float(
                metrics[f"{prefix}_ood_test_accuracy"]
            )
            observed = float(metrics[f"{prefix}_ood_gap"])
            _add(
                checks,
                f"{run_dir}:{prefix}_ood_gap_definition",
                abs(expected - observed) < 1e-9,
                f"expected={expected}, observed={observed}",
            )


def _validate_reproducibility_metadata(
    checks: list[Check],
    run_dir: Path,
    metadata: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> None:
    required_top = {
        "run_id",
        "timestamp_utc",
        "method",
        "config_hash",
        "seed",
        "device",
        "parameter_count",
        "git",
        "hardware",
        "data_loader",
        "deterministic",
    }
    missing_top = sorted(required_top - set(metadata))
    _add(
        checks,
        f"{run_dir}:reproducibility_metadata_schema",
        not missing_top,
        f"missing={missing_top}",
    )
    hardware = metadata.get("hardware", {})
    required_hardware = {
        "hostname",
        "platform",
        "python",
        "torch_version",
        "cuda_available",
        "device",
        "tf32_matmul_allowed",
        "tf32_cudnn_allowed",
    }
    missing_hardware = sorted(required_hardware - set(hardware))
    _add(
        checks,
        f"{run_dir}:hardware_metadata_schema",
        isinstance(hardware, dict) and not missing_hardware,
        f"missing={missing_hardware}",
    )
    git = metadata.get("git", {})
    _add(
        checks,
        f"{run_dir}:git_metadata_schema",
        isinstance(git, dict) and {"commit", "dirty", "status_short"}.issubset(git),
        f"git_keys={sorted(git) if isinstance(git, dict) else git}",
    )
    data_loader = metadata.get("data_loader", {})
    _add(
        checks,
        f"{run_dir}:data_loader_metadata_schema",
        isinstance(data_loader, dict)
        and {"num_workers", "pin_memory", "persistent_workers", "worker_seed"}.issubset(data_loader),
        f"data_loader={data_loader}",
    )
    if config is not None:
        expected_seed = config.get("seed")
        _add(
            checks,
            f"{run_dir}:metadata_seed_matches_config",
            expected_seed is None or int(metadata.get("seed", -1)) == int(expected_seed),
            f"metadata_seed={metadata.get('seed')}, config_seed={expected_seed}",
        )


def _validate_setpoint(checks: list[Check], run_dir: Path, metadata: dict[str, Any]) -> None:
    setpoint = metadata.get("setpoint")
    if setpoint is None:
        return
    mode = setpoint.get("mode")
    allowed = {
        "analytic_direct",
        "oracle_core_reference",
        "val_iid_sweep",
        "fixed_grid",
        "calibrated_val_reference",
    }
    _add(
        checks,
        f"{run_dir}:setpoint_mode_logged",
        mode in allowed and "sigma_target" in setpoint,
        f"setpoint={setpoint}",
    )
    if mode == "oracle_core_reference":
        _add(
            checks,
            f"{run_dir}:oracle_setpoint_transparency",
            setpoint.get("oracle_assisted") is True,
            f"setpoint={setpoint}",
        )
        estimated = setpoint.get("target_source") == "estimated_oracle_core_reference"
        required_reference = {
            "reference_epochs",
            "reference_batch_size",
            "reference_lr",
            "estimated_sigma_target",
        }
        missing_reference = sorted(required_reference - set(setpoint))
        _add(
            checks,
            f"{run_dir}:oracle_reference_metadata",
            (not estimated) or not missing_reference,
            f"estimated={estimated}, missing={missing_reference}, setpoint={setpoint}",
        )


def _load_last_jsonl(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    rows = [line for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        return None
    return json.loads(rows[-1])


def _validate_training_diagnostics(
    checks: list[Check],
    run_dir: Path,
    metadata: dict[str, Any],
) -> None:
    if metadata.get("method") != "sib":
        return
    new_diagnostic_schema = "arrow_calibration" in metadata
    last = _load_last_jsonl(run_dir / "metrics.jsonl")
    _add(
        checks,
        f"{run_dir}:sib_metrics_jsonl_present",
        last is not None,
        str(run_dir / "metrics.jsonl"),
        severity="error" if new_diagnostic_schema else "warning",
    )
    if last is None:
        return
    task_only_fast_path = last.get("sib_task_only_training_fast_path") is True
    task_only_required = {
        "loss_task",
        "loss_task_cf",
        "loss_forward_nll",
        "loss_reverse_nll",
        "loss_cf_arrow_total",
        "loss_cf_arrow_step",
        "loss_setpoint",
        "weighted_loss_task",
        "weighted_loss_task_cf",
        "weighted_loss_dynamics",
        "weighted_loss_cf_arrow_total",
        "weighted_loss_cf_arrow_step",
        "weighted_loss_setpoint",
        "loss_component_ratio_regularizer_to_task_abs",
        "train_prediction_entropy",
        "val_iid_prediction_entropy",
        "val_iid_cf_accuracy",
        "val_iid_cf_delta_arrow_total_calibrated",
        "val_iid_cf_delta_arrow_step_calibrated",
        "val_iid_cf_prediction_consistency",
        "near_chance_task_detector",
        "constant_prediction_detector",
        "high_prediction_entropy_detector",
        "sigma_zero_collapse_detector",
        "loss_regularizer_dominance_detector",
        "dynamics_likelihood_explosion_detector",
        "latent_norm_collapse_detector",
        "sib_task_only_training_fast_path",
    }
    dynamics_required = {
        "sigma_total_mean",
        "sigma_total_std",
        "sigma_total_min",
        "sigma_total_max",
        "sigma_per_step_mean",
        "sigma_per_step_std",
        "sigma_steps_mean",
        "sigma_steps_std",
        "sigma_per_step_abs_max",
        "latent_norm_mean",
        "latent_norm_std",
        "raw_sigma_per_step_mean",
        "raw_sigma_per_step_std",
        "calibrated_sigma_per_step_mean",
        "calibrated_sigma_per_step_std",
        "calibrated_sigma_steps_mean",
        "calibrated_sigma_steps_std",
        "calibrated_sigma_per_step_abs_max",
        "calibrated_latent_norm_mean",
        "calibrated_latent_norm_std",
        "forward_logvar_mean",
        "reverse_logvar_mean",
    }
    required = task_only_required if task_only_fast_path else task_only_required | dynamics_required
    missing = sorted(required - set(last))
    _add(
        checks,
        f"{run_dir}:sib_required_diagnostics_logged",
        not missing,
        f"task_only_fast_path={task_only_fast_path}, missing={missing}",
        severity="error" if new_diagnostic_schema else "warning",
    )
    if task_only_fast_path:
        _add(
            checks,
            f"{run_dir}:sib_task_only_fast_path_skips_training_dynamics",
            all(
                float(last.get(name, 1.0)) == 0.0
                for name in (
                    "loss_task_cf",
                    "loss_forward_nll",
                    "loss_reverse_nll",
                    "loss_cf_arrow_total",
                    "loss_cf_arrow_step",
                    "loss_setpoint",
                )
            ),
            "task-only SIB path must not train inactive CF/dynamics/arrow/setpoint losses",
        )
    for detector in (
        "near_chance_task_detector",
        "constant_prediction_detector",
        "high_prediction_entropy_detector",
        "sigma_zero_collapse_detector",
        "loss_regularizer_dominance_detector",
        "dynamics_likelihood_explosion_detector",
        "latent_norm_collapse_detector",
    ):
        if detector in last:
            _add(
                checks,
                f"{run_dir}:{detector}",
                last.get(detector) is not True,
                f"{detector}={last.get(detector)}",
                severity="warning",
            )


def _validate_ep_sanity(ep_sanity_dir: Path) -> list[Check]:
    checks: list[Check] = []
    direct = ep_sanity_dir / "ep_sanity_direct.json"
    decoded = ep_sanity_dir / "ep_sanity_decoded_transition.json"
    latent = ep_sanity_dir / "ep_sanity_latent.json"
    for path, name in ((direct, "ep_sanity_1A_direct"), (decoded, "ep_sanity_1B_decoded_transition")):
        if not path.exists():
            _add(checks, name, False, f"missing {path}")
            continue
        payload = _load_json(path)
        gate = payload.get("gate", {})
        _add(checks, name, gate.get("pass") is True, f"gate={gate}")
        has_calibration_fields = {
            "R0_sigma_per_step_raw_mean",
            "R0_sigma_per_step_calibrated_mean",
            "sigma_calibration",
        }.issubset(payload)
        _add(
            checks,
            f"{name}:raw_and_calibrated_sigma_reported",
            has_calibration_fields,
            f"keys={sorted(payload)}",
            severity="warning",
        )
    if latent.exists():
        payload = _load_json(latent)
        gate = payload.get("gate", {})
        _add(
            checks,
            "ep_sanity_1B_latent_classifier_diagnostic",
            gate.get("pass") is True,
            f"gate={gate}",
            severity="warning",
        )
        has_calibration_fields = {
            "R0_sigma_per_step_raw_mean",
            "R0_sigma_per_step_calibrated_mean",
            "sigma_calibration",
            "calibrated_gate",
        }.issubset(payload)
        _add(
            checks,
            "ep_sanity_1B_latent_classifier_diagnostic:raw_and_calibrated_sigma_reported",
            has_calibration_fields,
            f"keys={sorted(payload)}",
            severity="warning",
        )
    return checks


def _validate_setpoint_selection(checks: list[Check], root: Path) -> None:
    path = root / "setpoint_selection.json"
    if not path.exists():
        return
    payload = _load_json(path)
    _add(
        checks,
        "setpoint_selection_no_ood",
        payload.get("uses_ood_test") is False and payload.get("selection_split") == "val_iid",
        f"selection={payload}",
    )


def validate_results(
    root: str | Path,
    *,
    ep_sanity_dir: str | Path | None = None,
    required_methods: list[str] | None = None,
    output: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(root).resolve()
    checks: list[Check] = []
    run_dirs, failed_manifest_runs, stale_metric_dirs, has_manifest = _manifest_run_dirs(root)
    _add(checks, "final_metrics_present", bool(run_dirs), f"n_run_dirs={len(run_dirs)}")
    if has_manifest:
        missing_methods = [
            str(run_dir)
            for run_dir, method in run_dirs
            if method is None
        ]
        _add(
            checks,
            "manifest_no_failed_runs",
            not failed_manifest_runs,
            f"n_failed={len(failed_manifest_runs)}",
        )
        _add(
            checks,
            "manifest_no_untracked_final_metrics",
            not stale_metric_dirs,
            f"stale_run_dirs={[str(path) for path in stale_metric_dirs[:20]]}",
        )
        _add(
            checks,
            "manifest_success_methods_logged",
            not missing_methods,
            f"missing_method_run_dirs={missing_methods[:20]}",
        )

    methods_seen: set[str] = set()
    for run_dir, manifest_method in run_dirs:
        metrics_path = run_dir / "final_metrics.json"
        _add(checks, f"{run_dir}:final_metrics_present", metrics_path.exists(), str(metrics_path))
        if not metrics_path.exists():
            continue
        methods_seen.add(manifest_method or run_dir.parent.name)
        metrics = _load_json(metrics_path)
        metadata_path = run_dir / "metadata.json"
        _add(checks, f"{run_dir}:metadata_present", metadata_path.exists(), str(metadata_path))
        _validate_final_metrics(checks, run_dir, metrics)
        if metadata_path.exists():
            metadata = _load_json(metadata_path)
            config_path = run_dir / "resolved_config.json"
            config = _load_json(config_path) if config_path.exists() else None
            _validate_reproducibility_metadata(checks, run_dir, metadata, config)
            _validate_dataset_metadata(checks, run_dir, metadata, config)
            _validate_setpoint(checks, run_dir, metadata)
            _validate_training_diagnostics(checks, run_dir, metadata)

    if required_methods:
        missing = sorted(set(required_methods) - methods_seen)
        _add(
            checks,
            "required_methods_present",
            not missing,
            f"missing={missing}, seen={sorted(methods_seen)}",
        )

    manifest = root / "manifest.json"
    aggregate = root / "aggregate.json"
    _add(checks, "manifest_present", manifest.exists(), str(manifest), severity="warning")
    _add(checks, "aggregate_present", aggregate.exists(), str(aggregate), severity="warning")
    _validate_setpoint_selection(checks, root)

    if ep_sanity_dir is not None:
        checks.extend(_validate_ep_sanity(Path(ep_sanity_dir)))

    error_checks = [check for check in checks if check.severity == "error"]
    warning_checks = [check for check in checks if check.severity == "warning"]
    result = {
        "root": str(root),
        "validation_schema_version": VALIDATION_SCHEMA_VERSION,
        "pass": all(check.passed for check in error_checks),
        "n_checks": len(checks),
        "n_errors_failed": sum(1 for check in error_checks if not check.passed),
        "n_warnings_failed": sum(1 for check in warning_checks if not check.passed),
        "checks": [check.as_dict() for check in checks],
    }
    if output is not None:
        save_json(output, result)
    return result


def validation_summary(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "root": result["root"],
        "pass": result["pass"],
        "n_checks": result["n_checks"],
        "n_errors_failed": result["n_errors_failed"],
        "n_warnings_failed": result["n_warnings_failed"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate scientific gates for result outputs.")
    parser.add_argument("root")
    parser.add_argument("--ep-sanity-dir", default=None)
    parser.add_argument("--required-methods", nargs="*", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print only pass/fail counts while still writing the full output file.",
    )
    args = parser.parse_args()
    result = validate_results(
        args.root,
        ep_sanity_dir=args.ep_sanity_dir,
        required_methods=args.required_methods,
        output=args.output,
    )
    printable = validation_summary(result) if args.summary else result
    print(json.dumps(printable, indent=2, sort_keys=True))
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
