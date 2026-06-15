import json
import os
from pathlib import Path
from statistics import mean, stdev

from src.eval.audit_evidence import audit_evidence
from src.eval.audit_evidence import PREFLIGHT_SOURCE_PATHS
from src.eval.audit_evidence import EVIDENCE_AUDIT_SCHEMA_VERSION
from src.eval.audit_evidence import REQUIRED_ITM_MECHANISM_METRICS
from src.eval.audit_evidence import REQUIRED_DIAGNOSTIC_GATES
from src.eval.audit_evidence import REQUIRED_SID_FACTOR_METRICS
from src.eval.audit_smoke import REQUIRED_METHODS
from src.train.common import save_json


EXPECTED_CONFIGS = {
    "sta": "configs/sta_full.yaml",
    "ink_advection_diffusion": "configs/ink_advection_diffusion_full.yaml",
}
FULL_SPLITS = {
    "train": {"n_sequences": 10_000, "spurious_mode": "correlated"},
    "val_iid": {"n_sequences": 2_000, "spurious_mode": "correlated"},
    "iid_test": {"n_sequences": 5_000, "spurious_mode": "correlated"},
    "ood_test": {"n_sequences": 5_000, "spurious_mode": "reversed"},
}


def _sid_factor_metrics() -> dict:
    metrics = {}
    for key in REQUIRED_SID_FACTOR_METRICS:
        if key in {
            "task_head_excludes_z_ir_spur",
            "decomposition_role_claim_ready",
        } or "_best_factor_is_" in key or key.endswith("_task_rep_spurious_low"):
            metrics[key] = True
        elif key.endswith("_best_factor"):
            metrics[key] = "z_ir_task"
        else:
            metrics[key] = 0.5
    return metrics


def _sid_factor_audit(run_dirs: list[Path], *, benchmark: str) -> dict:
    rows = []
    for run_dir in run_dirs:
        audit = {
            "passed": True,
            "method": "sid",
            "run_dir": str(run_dir),
            "checkpoint": "best.pt",
            "benchmark_name": benchmark,
            "probe_train_split": "val_iid",
            "eval_splits": ["iid_test", "ood_test"],
            "target_metadata": {
                "label": {"target_source": "y"},
                "core_dynamic": {"target_source": "core_dynamic_stat", "threshold": 0.0},
                "spurious_dynamic": {
                    "target_source": "spurious_dynamic_stat",
                    "threshold": 0.0,
                },
            },
            "metrics": _sid_factor_metrics(),
            "role_alignment": {
                "decomposition_role_claim_ready": True,
                "decomposition_role_alignment_failures": [],
            },
        }
        save_json(run_dir / "sid_factor_audit.json", audit)
        rows.append({"run_dir": str(run_dir), "audit": audit})
    return {
        "passed": True,
        "manifest": "manifest.json",
        "checkpoint": "best.pt",
        "n_runs": len(rows),
        "runs": rows,
    }


def _itm_mechanism_metrics() -> dict:
    metrics = {}
    for key in REQUIRED_ITM_MECHANISM_METRICS:
        if key in {"task_head_excludes_spur_mechanism", "mechanism_claim_ready"}:
            metrics[key] = True
        elif key.endswith("_residual_controls"):
            metrics[key] = "y,core_dynamic_stat"
        elif key.endswith("_core_delta_cf_mse"):
            metrics[key] = 0.01
        elif key.endswith("_spur_delta_cf_mse"):
            metrics[key] = 0.02
        elif key.endswith("_spur_to_core_cf_mse_ratio"):
            metrics[key] = 2.0
        elif "_task_rep_spurious_dynamic_residualized_auc" in key:
            metrics[key] = 0.52
        else:
            metrics[key] = 0.75
    return metrics


def _itm_mechanism_audit(run_dirs: list[Path], *, benchmark: str) -> dict:
    rows = []
    for run_dir in run_dirs:
        audit = {
            "schema": "itm_mechanism_audit_v1",
            "passed": True,
            "method": "itm",
            "run_dir": str(run_dir.resolve()),
            "checkpoint": "best.pt",
            "benchmark_name": benchmark,
            "probe_train_split": "val_iid",
            "eval_splits": ["iid_test", "ood_test"],
            "target_metadata": {
                "label": {"target_source": "y"},
                "core_dynamic": {"target_source": "core_dynamic_stat", "threshold": 0.0},
                "spurious_dynamic": {
                    "target_source": "spurious_dynamic_stat",
                    "threshold": 0.0,
                },
            },
            "metrics": _itm_mechanism_metrics(),
            "role_alignment": {
                "mechanism_claim_ready": True,
                "mechanism_alignment_failures": [],
            },
        }
        save_json(run_dir / "itm_mechanism_audit.json", audit)
        rows.append({"run_dir": str(run_dir), "audit": audit})
    return {
        "schema": "itm_mechanism_audit_v1",
        "passed": True,
        "manifest": "manifest.json",
        "checkpoint": "best.pt",
        "n_runs": len(rows),
        "mechanism_claim_ready_runs": len(rows),
        "mechanism_claim_ready": True,
        "runs": rows,
    }


def _numeric_items(row: dict) -> dict[str, float]:
    return {
        key: float(value)
        for key, value in row.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }


def _aggregate_from_rows(rows_by_method: dict[str, list[dict]]) -> dict:
    aggregate = {"n_skipped_metric_files": 0, "skipped_metric_files": []}
    for method, rows in rows_by_method.items():
        numeric_rows = [_numeric_items(row) for row in rows]
        summary = {"n_runs": len(numeric_rows)}
        metric_names = sorted({metric for row in numeric_rows for metric in row})
        for metric in metric_names:
            values = [row[metric] for row in numeric_rows if metric in row]
            summary[f"{metric}_mean"] = mean(values)
            summary[f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0
        aggregate[method] = summary
    return aggregate


def _write_evidence_fixture(root: Path, *, seeds: tuple[int, ...], epochs: int) -> None:
    diagnostics = root / "diagnostics"
    diagnostics.mkdir(parents=True)
    diagnostic_files = {
        "sta": ("sta_benchmark_diagnostics.json", "sta_bench"),
        "ink_advection_diffusion": (
            "ink_advection_diffusion_diagnostics.json",
            "ink_advection_diffusion",
        ),
    }
    diagnostic_splits = {
        split_name: {
            "n_sequences": cfg["n_sequences"],
            "length_L": 16,
            "n_transitions": 15,
            "class_balance": {"n": cfg["n_sequences"], "n0": cfg["n_sequences"] // 2, "n1": cfg["n_sequences"] - cfg["n_sequences"] // 2},
            "label_threshold_source": "local" if split_name == "train" else "train",
        }
        for split_name, cfg in FULL_SPLITS.items()
    }
    for suite, (filename, benchmark_name) in diagnostic_files.items():
        save_json(
            diagnostics / filename,
            {
                "schema_version": 1,
                "benchmark_name": benchmark_name,
                "splits": diagnostic_splits,
                "pass": True,
                "quality_gates": {
                    gate: True for gate in REQUIRED_DIAGNOSTIC_GATES[suite]
                },
            },
        )

    for suite, benchmark in {
        "sta": "sta_bench",
        "ink_advection_diffusion": "ink_advection_diffusion",
    }.items():
        suite_root = root / suite
        runs = []
        sid_run_dirs = []
        rows_by_method = {method: [] for method in REQUIRED_METHODS}
        for seed in seeds:
            for method in REQUIRED_METHODS:
                run_dir = suite_root / method / f"seed{seed}"
                run_dir.mkdir(parents=True)
                config_hash = f"{suite}_{method}_{seed}"
                metadata = {
                    "run_id": run_dir.name,
                    "method": method,
                    "seed": seed,
                    "config_path": EXPECTED_CONFIGS[suite],
                    "config_hash": config_hash,
                    "dataset_metadata": {"train": {"benchmark_name": benchmark}},
                    "setpoint": {"selection_split": None},
                    "task_guard": {"selection_split": "val_iid"},
                    "arrow_calibration": {"uses_ood_test": False},
                }
                if method in {"ocp_style", "lens_like_arrow_classifier"}:
                    metadata.update({"pretraining_split": "train", "transductive": False})
                save_json(run_dir / "metadata.json", metadata)
                metrics = (
                    {
                        "frozen_encoder_iid_test_accuracy": 0.5,
                        "frozen_encoder_ood_test_accuracy": 0.5,
                        "frozen_encoder_ood_gap": 0.0,
                        "frozen_encoder_selected_checkpoint": "frozen_encoder_best.pt",
                        "frozen_encoder_best_epoch": 0,
                        "frozen_encoder_epochs_completed": epochs,
                        "fine_tuned_encoder_iid_test_accuracy": 0.5,
                        "fine_tuned_encoder_ood_test_accuracy": 0.5,
                        "fine_tuned_encoder_ood_gap": 0.0,
                        "fine_tuned_encoder_selected_checkpoint": (
                            "fine_tuned_encoder_best.pt"
                        ),
                        "fine_tuned_encoder_best_epoch": 0,
                        "fine_tuned_encoder_epochs_completed": epochs,
                    }
                    if method in {"ocp_style", "lens_like_arrow_classifier"}
                    else {
                        "iid_test_accuracy": 0.5,
                        "ood_test_accuracy": 0.5,
                        "ood_gap": 0.0,
                        "epochs_completed": epochs,
                        "selected_checkpoint": "best.pt",
                        "unguarded_selected_checkpoint": "best.pt",
                        "selected_epoch": 0,
                        "best_epoch": 0,
                        "unguarded_best_epoch": 0,
                    }
                )
                if method in {"sib", "sid", "itm"}:
                    metrics["iid_test_cf_prediction_consistency"] = 1.0
                if method == "itm":
                    metrics["selected_epoch"] = epochs - 1
                    metrics["best_epoch"] = epochs - 1
                    metrics["unguarded_best_epoch"] = epochs - 1
                    metrics["iid_test_cf_arrow_metrics_available"] = False
                    metrics["iid_test_cf_delta_arrow_total"] = None
                    metrics["iid_test_cf_delta_arrow_step"] = None
                    metrics["iid_test_cf_delta_arrow_total_calibrated"] = None
                    metrics["iid_test_cf_delta_arrow_step_calibrated"] = None
                    metrics["itm_schedule_required_epochs"] = epochs
                    metrics["itm_selected_schedule_progress"] = 1.0
                    metrics["itm_selected_transition_schedule_progress"] = 1.0
                    metrics["itm_schedule_floor_satisfied"] = True
                if method == "sid":
                    metrics.update(
                        {
                            "selected_epoch": epochs - 1,
                            "best_epoch": epochs - 1,
                            "unguarded_best_epoch": epochs - 1,
                            "sid_schedule_required_epochs": epochs,
                            "sid_selected_schedule_progress": 1.0,
                            "sid_selected_dynamics_progress": 1.0,
                            "sid_schedule_floor_satisfied": True,
                            "checkpoint_selection_floor_satisfied": True,
                            "early_stopping_floor_satisfied": True,
                            "min_epoch_for_checkpoint_selection": epochs,
                            "min_epochs_before_early_stopping": epochs,
                        }
                    )
                (run_dir / "final.pt").write_text("dummy checkpoint")
                (run_dir / "metrics.jsonl").write_text(json.dumps({"epoch": 0}) + "\n")
                if method in {"ocp_style", "lens_like_arrow_classifier"}:
                    (run_dir / "frozen_encoder_best.pt").write_text("dummy checkpoint")
                    (run_dir / "fine_tuned_encoder_best.pt").write_text(
                        "dummy checkpoint"
                    )
                    (run_dir / "frozen_encoder_metrics.jsonl").write_text(
                        json.dumps({"epoch": 0}) + "\n"
                    )
                    (run_dir / "fine_tuned_encoder_metrics.jsonl").write_text(
                        json.dumps({"epoch": 0}) + "\n"
                    )
                else:
                    (run_dir / "best.pt").write_text("dummy checkpoint")
                save_json(run_dir / "final_metrics.json", metrics)
                rows_by_method[method].append(metrics)
                save_json(
                    run_dir / "resolved_config.json",
                    {
                        "seed": seed,
                        "experiment": f"{suite}_full",
                        "run": {"config_path": EXPECTED_CONFIGS[suite]},
                        "splits": FULL_SPLITS,
                        "training": {
                            "epochs": epochs,
                            "min_epoch_for_checkpoint_selection": epochs,
                            "min_epochs_before_early_stopping": epochs,
                        },
                        "itm_schedule": {
                            "task_warmup_epochs": 0,
                            "transition_warmup_epochs": 0,
                            "regularizer_ramp_epochs": epochs,
                        },
                    },
                )
                runs.append(
                    {
                        "method": method,
                        "status": "success",
                        "seed": seed,
                        "run_dir": str(run_dir),
                        "config_hash": config_hash,
                        "metrics": metrics,
                    }
                )
                if method == "sid":
                    sid_run_dirs.append(run_dir)
        save_json(suite_root / "manifest.json", {"runs": runs})
        save_json(suite_root / "aggregate.json", _aggregate_from_rows(rows_by_method))
        save_json(
            suite_root / "sid_factor_audit.json",
            _sid_factor_audit(sid_run_dirs, benchmark=benchmark),
        )
        itm_run_dirs = [
            suite_root / "itm" / f"seed{seed}"
            for seed in seeds
        ]
        save_json(
            suite_root / "itm_mechanism_audit.json",
            _itm_mechanism_audit(itm_run_dirs, benchmark=benchmark),
        )


def _write_preflight(path: Path, *, root: Path, seeds: int = 2, epochs: int = 3) -> None:
    save_json(
        path,
        {
            "pass": True,
            "launch_recommended": True,
            "environment": {
                "out": str(root),
                "device": "cpu",
                "min_seeds": seeds,
                "epochs": epochs,
                "methods": list(REQUIRED_METHODS),
                "require_cuda_for_full_run": "0",
                "launch_authorization": "maintenance_or_diagnostic",
            },
        },
    )


def test_evidence_audit_passes_complete_fixture(tmp_path):
    root = tmp_path / "full_run"
    _write_evidence_fixture(root, seeds=(0, 1), epochs=3)
    preflight = tmp_path / "preflight.json"
    _write_preflight(preflight, root=root)

    report = audit_evidence(
        root,
        min_seeds=2,
        min_epochs=3,
        preflight_path=preflight,
    )

    assert report["passed"] is True
    assert report["schema_version"] == EVIDENCE_AUDIT_SCHEMA_VERSION
    assert report["run_count"] == 36
    assert report["n_checks"] == len(report["checks"])
    assert report["n_failed"] == 0
    assert report["failed_checks"] == []
    assert report["preflight_environment_summary"]["launch_authorization"] == (
        "maintenance_or_diagnostic"
    )


def test_evidence_audit_rejects_preflight_without_authorization_schema(
    tmp_path,
):
    root = tmp_path / "full_run"
    _write_evidence_fixture(root, seeds=(0, 1), epochs=3)
    preflight = tmp_path / "preflight.json"
    _write_preflight(preflight, root=root)
    payload = json.loads(preflight.read_text())
    payload["environment"].pop("require_cuda_for_full_run")
    payload["environment"].pop("launch_authorization")
    save_json(preflight, payload)

    report = audit_evidence(
        root,
        min_seeds=2,
        min_epochs=3,
        preflight_path=preflight,
    )

    failed = {check["name"] for check in report["checks"] if not check["passed"]}
    assert report["passed"] is False
    assert "preflight_require_cuda_recorded" in failed
    assert "preflight_launch_authorization_recorded" in failed


def test_evidence_source_freshness_includes_finalizer():
    assert "experiments/finalize_paper.sh" in PREFLIGHT_SOURCE_PATHS


def test_evidence_required_diagnostic_gates_are_explicit():
    assert "mass_conservation" in REQUIRED_DIAGNOSTIC_GATES["ink_advection_diffusion"]
    assert "entropy_increase" in REQUIRED_DIAGNOSTIC_GATES["ink_advection_diffusion"]
    assert (
        "counterfactual_changes_spurious_flow"
        in REQUIRED_DIAGNOSTIC_GATES["ink_advection_diffusion"]
    )
    assert "same_mixing_matrix" in REQUIRED_DIAGNOSTIC_GATES["sta"]
    assert "task_head_excludes_z_ir_spur" in REQUIRED_SID_FACTOR_METRICS
    assert "iid_test_z_ir_task_label_probe_accuracy" in REQUIRED_SID_FACTOR_METRICS
    assert "ood_test_z_ir_spur_spurious_dynamic_probe_accuracy" in REQUIRED_SID_FACTOR_METRICS


def test_evidence_audit_rejects_smoke_root_and_low_seed_count(tmp_path):
    root = tmp_path / "full_run_smoke"
    _write_evidence_fixture(root, seeds=(0,), epochs=3)

    report = audit_evidence(root, min_seeds=2, min_epochs=3)

    assert report["passed"] is False
    assert report["n_failed"] > 0
    assert set(report["failed_checks"]) == {
        check["name"] for check in report["checks"] if not check["passed"]
    }
    failed = {check["name"] for check in report["checks"] if not check["passed"]}
    assert "non_smoke_result_root" in failed
    assert "sta_min_seed_count" in failed


def test_evidence_audit_rejects_missing_required_diagnostic_gate(tmp_path):
    root = tmp_path / "full_run"
    _write_evidence_fixture(root, seeds=(0, 1), epochs=3)
    diagnostic_path = root / "diagnostics" / "ink_advection_diffusion_diagnostics.json"
    diagnostic = json.loads(diagnostic_path.read_text())
    diagnostic["quality_gates"].pop("mass_conservation")
    save_json(diagnostic_path, diagnostic)

    report = audit_evidence(root, min_seeds=2, min_epochs=3)
    failed = {check["name"] for check in report["checks"] if not check["passed"]}

    assert report["passed"] is False
    assert "ink_advection_diffusion_diagnostic_required_quality_gates_present" in failed


def test_evidence_audit_rejects_diagnostic_schema_mismatch(tmp_path):
    root = tmp_path / "full_run"
    _write_evidence_fixture(root, seeds=(0, 1), epochs=3)
    diagnostic_path = root / "diagnostics" / "sta_benchmark_diagnostics.json"
    diagnostic = json.loads(diagnostic_path.read_text())
    diagnostic["benchmark_name"] = "wrong"
    save_json(diagnostic_path, diagnostic)

    report = audit_evidence(root, min_seeds=2, min_epochs=3)
    failed = {check["name"] for check in report["checks"] if not check["passed"]}

    assert report["passed"] is False
    assert "sta_diagnostic_schema" in failed


def test_evidence_audit_rejects_diagnostic_transition_mismatch(tmp_path):
    root = tmp_path / "full_run"
    _write_evidence_fixture(root, seeds=(0, 1), epochs=3)
    diagnostic_path = root / "diagnostics" / "ink_advection_diffusion_diagnostics.json"
    diagnostic = json.loads(diagnostic_path.read_text())
    diagnostic["splits"]["iid_test"]["n_transitions"] = 999
    save_json(diagnostic_path, diagnostic)

    report = audit_evidence(root, min_seeds=2, min_epochs=3)
    failed = {check["name"] for check in report["checks"] if not check["passed"]}

    assert report["passed"] is False
    assert "ink_advection_diffusion_diagnostic_schema" in failed


def test_evidence_audit_rejects_incomplete_sid_factor_audit(tmp_path):
    root = tmp_path / "full_run"
    _write_evidence_fixture(root, seeds=(0, 1), epochs=3)
    sid_factor_path = root / "sta" / "sid_factor_audit.json"
    sid_factor = json.loads(sid_factor_path.read_text())
    sid_factor["runs"][0]["audit"]["metrics"].pop("iid_test_z_ir_task_label_probe_accuracy")
    save_json(sid_factor_path, sid_factor)

    report = audit_evidence(root, min_seeds=2, min_epochs=3)
    failed = {check["name"] for check in report["checks"] if not check["passed"]}

    assert report["passed"] is False
    assert "sta_sid_factor_run_0_required_metrics" in failed


def test_evidence_audit_rejects_missing_sid_role_alignment_schema(tmp_path):
    root = tmp_path / "full_run"
    _write_evidence_fixture(root, seeds=(0, 1), epochs=3)
    sid_factor_path = root / "sta" / "sid_factor_audit.json"
    sid_factor = json.loads(sid_factor_path.read_text())
    sid_factor["runs"][0]["audit"].pop("role_alignment")
    save_json(sid_factor_path, sid_factor)
    local_path = root / "sta" / "sid" / "seed0" / "sid_factor_audit.json"
    local = json.loads(local_path.read_text())
    local.pop("role_alignment")
    save_json(local_path, local)

    report = audit_evidence(root, min_seeds=2, min_epochs=3)
    failed = {check["name"] for check in report["checks"] if not check["passed"]}

    assert report["passed"] is False
    assert "sta_sid_factor_run_0_role_alignment_schema" in failed


def test_evidence_audit_allows_negative_sid_role_alignment_result(tmp_path):
    root = tmp_path / "full_run"
    _write_evidence_fixture(root, seeds=(0, 1), epochs=3)
    sid_factor_path = root / "sta" / "sid_factor_audit.json"
    sid_factor = json.loads(sid_factor_path.read_text())
    audit = sid_factor["runs"][0]["audit"]
    audit["metrics"]["decomposition_role_claim_ready"] = False
    audit["role_alignment"] = {
        "decomposition_role_claim_ready": False,
        "decomposition_role_alignment_failures": [
            "iid_test.label: expected z_ir_task, observed z_rev"
        ],
    }
    save_json(sid_factor_path, sid_factor)
    local_path = root / "sta" / "sid" / "seed0" / "sid_factor_audit.json"
    save_json(local_path, audit)

    report = audit_evidence(root, min_seeds=2, min_epochs=3)

    assert report["passed"] is True
    assert any(
        check["name"] == "sta_sid_factor_run_0_role_alignment_schema"
        and check["passed"] is True
        for check in report["checks"]
    )


def test_evidence_audit_rejects_missing_local_sid_factor_audit(tmp_path):
    root = tmp_path / "full_run"
    _write_evidence_fixture(root, seeds=(0, 1), epochs=3)
    local_path = root / "sta" / "sid" / "seed0" / "sid_factor_audit.json"
    local_path.unlink()

    report = audit_evidence(root, min_seeds=2, min_epochs=3)
    failed = {check["name"] for check in report["checks"] if not check["passed"]}

    assert report["passed"] is False
    assert "sta_sid_factor_run_0_local_audit_exists" in failed


def test_evidence_audit_rejects_stale_local_sid_factor_audit(tmp_path):
    root = tmp_path / "full_run"
    _write_evidence_fixture(root, seeds=(0, 1), epochs=3)
    local_path = root / "sta" / "sid" / "seed0" / "sid_factor_audit.json"
    local = json.loads(local_path.read_text())
    local["metrics"]["iid_test_z_ir_task_label_probe_accuracy"] = 0.123
    save_json(local_path, local)

    report = audit_evidence(root, min_seeds=2, min_epochs=3)
    failed = {check["name"] for check in report["checks"] if not check["passed"]}

    assert report["passed"] is False
    assert "sta_sid_factor_run_0_local_audit_matches_summary" in failed


def test_evidence_audit_rejects_sid_factor_run_dir_mismatch(tmp_path):
    root = tmp_path / "full_run"
    _write_evidence_fixture(root, seeds=(0, 1), epochs=3)
    summary_path = root / "sta" / "sid_factor_audit.json"
    summary = json.loads(summary_path.read_text())
    summary["runs"][0]["audit"]["run_dir"] = str(root / "other")
    save_json(summary_path, summary)

    report = audit_evidence(root, min_seeds=2, min_epochs=3)
    failed = {check["name"] for check in report["checks"] if not check["passed"]}

    assert report["passed"] is False
    assert "sta_sid_factor_run_0_run_dir_identity" in failed


def test_evidence_audit_rejects_stale_aggregate_metrics(tmp_path):
    root = tmp_path / "full_run"
    _write_evidence_fixture(root, seeds=(0, 1), epochs=3)
    aggregate_path = root / "sta" / "aggregate.json"
    aggregate = json.loads(aggregate_path.read_text())
    aggregate["erm"]["iid_test_accuracy_mean"] = 0.123
    save_json(aggregate_path, aggregate)

    report = audit_evidence(root, min_seeds=2, min_epochs=3)
    failed = {check["name"] for check in report["checks"] if not check["passed"]}

    assert report["passed"] is False
    assert "sta_erm_aggregate_matches_run_metrics" in failed


def test_evidence_audit_rejects_stale_manifest_metrics(tmp_path):
    root = tmp_path / "full_run"
    _write_evidence_fixture(root, seeds=(0, 1), epochs=3)
    manifest_path = root / "sta" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["runs"][0]["metrics"]["iid_test_accuracy"] = 0.123
    stale_method = manifest["runs"][0]["method"]
    save_json(manifest_path, manifest)

    report = audit_evidence(root, min_seeds=2, min_epochs=3)
    failed = {check["name"] for check in report["checks"] if not check["passed"]}

    assert report["passed"] is False
    assert f"sta_{stale_method}_manifest_metrics_match_final_metrics" in failed


def test_evidence_audit_rejects_seed_identity_mismatch(tmp_path):
    root = tmp_path / "full_run"
    _write_evidence_fixture(root, seeds=(0, 1), epochs=3)
    resolved_path = root / "sta" / "sid" / "seed0" / "resolved_config.json"
    resolved = json.loads(resolved_path.read_text())
    resolved["seed"] = 99
    save_json(resolved_path, resolved)

    report = audit_evidence(root, min_seeds=2, min_epochs=3)
    failed = {check["name"] for check in report["checks"] if not check["passed"]}

    assert report["passed"] is False
    assert "sta_sid_seed_identity" in failed


def test_evidence_audit_rejects_run_id_mismatch(tmp_path):
    root = tmp_path / "full_run"
    _write_evidence_fixture(root, seeds=(0, 1), epochs=3)
    metadata_path = root / "sta" / "sid" / "seed0" / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["run_id"] = "other_run"
    save_json(metadata_path, metadata)

    report = audit_evidence(root, min_seeds=2, min_epochs=3)
    failed = {check["name"] for check in report["checks"] if not check["passed"]}

    assert report["passed"] is False
    assert "sta_sid_run_id_matches_run_dir" in failed


def test_evidence_audit_rejects_config_hash_mismatch(tmp_path):
    root = tmp_path / "full_run"
    _write_evidence_fixture(root, seeds=(0, 1), epochs=3)
    manifest_path = root / "sta" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["runs"][0]["config_hash"] = "different"
    stale_method = manifest["runs"][0]["method"]
    save_json(manifest_path, manifest)

    report = audit_evidence(root, min_seeds=2, min_epochs=3)
    failed = {check["name"] for check in report["checks"] if not check["passed"]}

    assert report["passed"] is False
    assert f"sta_{stale_method}_config_hash_identity" in failed


def test_evidence_audit_rejects_missing_selected_checkpoint(tmp_path):
    root = tmp_path / "full_run"
    _write_evidence_fixture(root, seeds=(0, 1), epochs=3)
    checkpoint_path = root / "sta" / "sid" / "seed0" / "best.pt"
    checkpoint_path.unlink()

    report = audit_evidence(root, min_seeds=2, min_epochs=3)
    failed = {check["name"] for check in report["checks"] if not check["passed"]}

    assert report["passed"] is False
    assert "sta_sid_checkpoint_files_exist" in failed


def test_evidence_audit_rejects_missing_itm_selected_checkpoint(tmp_path):
    root = tmp_path / "full_run"
    _write_evidence_fixture(root, seeds=(0, 1), epochs=3)
    checkpoint_path = root / "sta" / "itm" / "seed0" / "best.pt"
    checkpoint_path.unlink()

    report = audit_evidence(root, min_seeds=2, min_epochs=3)
    failed = {check["name"] for check in report["checks"] if not check["passed"]}

    assert report["passed"] is False
    assert "sta_itm_checkpoint_files_exist" in failed


def test_evidence_audit_rejects_numeric_itm_arrow_delta_placeholders(tmp_path):
    root = tmp_path / "full_run"
    _write_evidence_fixture(root, seeds=(0, 1), epochs=3)
    metrics_path = root / "sta" / "itm" / "seed0" / "final_metrics.json"
    metrics = json.loads(metrics_path.read_text())
    metrics["iid_test_cf_delta_arrow_total"] = 0.0
    metrics.pop("iid_test_cf_arrow_metrics_available", None)
    save_json(metrics_path, metrics)
    manifest_path = root / "sta" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for row in manifest["runs"]:
        if row["method"] == "itm" and row["seed"] == 0:
            row["metrics"] = metrics
    save_json(manifest_path, manifest)

    report = audit_evidence(root, min_seeds=2, min_epochs=3)
    failed = {check["name"] for check in report["checks"] if not check["passed"]}

    assert report["passed"] is False
    assert "sta_itm_arrow_delta_metrics_unavailable" in failed
    assert "sta_itm_arrow_metric_availability_logged" in failed


def test_evidence_audit_rejects_missing_arrow_protocol_checkpoint(tmp_path):
    root = tmp_path / "full_run"
    _write_evidence_fixture(root, seeds=(0, 1), epochs=3)
    checkpoint_path = root / "sta" / "ocp_style" / "seed0" / "frozen_encoder_best.pt"
    checkpoint_path.unlink()

    report = audit_evidence(root, min_seeds=2, min_epochs=3)
    failed = {check["name"] for check in report["checks"] if not check["passed"]}

    assert report["passed"] is False
    assert "sta_ocp_style_checkpoint_files_exist" in failed


def test_evidence_audit_rejects_selected_epoch_out_of_range(tmp_path):
    root = tmp_path / "full_run"
    _write_evidence_fixture(root, seeds=(0, 1), epochs=3)
    metrics_path = root / "sta" / "sid" / "seed0" / "final_metrics.json"
    metrics = json.loads(metrics_path.read_text())
    metrics["selected_epoch"] = 99
    save_json(metrics_path, metrics)
    manifest_path = root / "sta" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for row in manifest["runs"]:
        if row["method"] == "sid" and row["seed"] == 0:
            row["metrics"] = metrics
    save_json(manifest_path, manifest)

    report = audit_evidence(root, min_seeds=2, min_epochs=3)
    failed = {check["name"] for check in report["checks"] if not check["passed"]}

    assert report["passed"] is False
    assert "sta_sid_epoch_ranges_valid" in failed


def test_evidence_audit_rejects_missing_training_log(tmp_path):
    root = tmp_path / "full_run"
    _write_evidence_fixture(root, seeds=(0, 1), epochs=3)
    log_path = root / "sta" / "sid" / "seed0" / "metrics.jsonl"
    log_path.unlink()

    report = audit_evidence(root, min_seeds=2, min_epochs=3)
    failed = {check["name"] for check in report["checks"] if not check["passed"]}

    assert report["passed"] is False
    assert "sta_sid_training_logs_present" in failed


def test_evidence_audit_rejects_missing_arrow_protocol_log(tmp_path):
    root = tmp_path / "full_run"
    _write_evidence_fixture(root, seeds=(0, 1), epochs=3)
    log_path = root / "sta" / "ocp_style" / "seed0" / "frozen_encoder_metrics.jsonl"
    log_path.unlink()

    report = audit_evidence(root, min_seeds=2, min_epochs=3)
    failed = {check["name"] for check in report["checks"] if not check["passed"]}

    assert report["passed"] is False
    assert "sta_ocp_style_training_logs_present" in failed


def test_evidence_audit_rejects_ood_tuning_metadata(tmp_path):
    root = tmp_path / "full_run"
    _write_evidence_fixture(root, seeds=(0, 1), epochs=3)
    metadata_path = root / "sta" / "sid" / "seed0" / "metadata.json"
    metadata = {
        **json.loads(metadata_path.read_text()),
        "setpoint": {"selection_split": "ood_test"},
    }
    save_json(metadata_path, metadata)

    report = audit_evidence(root, min_seeds=2, min_epochs=3)

    assert report["passed"] is False
    assert any(
        check["name"] == "sta_sid_no_ood_tuning_metadata" and check["passed"] is False
        for check in report["checks"]
    )


def test_evidence_audit_rejects_broader_ood_tuning_metadata_keys(tmp_path):
    root = tmp_path / "full_run"
    _write_evidence_fixture(root, seeds=(0, 1), epochs=3)
    metadata_path = root / "sta" / "sid" / "seed0" / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["selector"] = {
        "model_selection_split": "ood_test",
        "nested": {"uses_ood_test": True},
    }
    save_json(metadata_path, metadata)

    report = audit_evidence(root, min_seeds=2, min_epochs=3)

    assert report["passed"] is False
    failed_details = {
        check["detail"]
        for check in report["checks"]
        if check["name"] == "sta_sid_no_ood_tuning_metadata"
        and check["passed"] is False
    }
    assert any("model_selection_split=ood_test" in detail for detail in failed_details)
    assert any("uses_ood_test=True" in detail for detail in failed_details)


def test_evidence_audit_rejects_ood_tuning_resolved_config(tmp_path):
    root = tmp_path / "full_run"
    _write_evidence_fixture(root, seeds=(0, 1), epochs=3)
    resolved_path = root / "sta" / "sid" / "seed0" / "resolved_config.json"
    resolved = json.loads(resolved_path.read_text())
    resolved["training"]["early_stopping_metric"] = "ood_test_accuracy"
    save_json(resolved_path, resolved)

    report = audit_evidence(root, min_seeds=2, min_epochs=3)

    assert report["passed"] is False
    assert any(
        check["name"] == "sta_sid_no_ood_tuning_resolved_config"
        and check["passed"] is False
        for check in report["checks"]
    )


def test_evidence_audit_rejects_broader_ood_tuning_resolved_config_keys(tmp_path):
    root = tmp_path / "full_run"
    _write_evidence_fixture(root, seeds=(0, 1), epochs=3)
    resolved_path = root / "sta" / "sid" / "seed0" / "resolved_config.json"
    resolved = json.loads(resolved_path.read_text())
    resolved["training"]["validation_split"] = "ood_test"
    resolved["training"]["monitor_metric"] = "iid_test_accuracy"
    save_json(resolved_path, resolved)

    report = audit_evidence(root, min_seeds=2, min_epochs=3)

    assert report["passed"] is False
    failed_details = {
        check["detail"]
        for check in report["checks"]
        if check["name"] == "sta_sid_no_ood_tuning_resolved_config"
        and check["passed"] is False
    }
    assert any("validation_split=ood_test" in detail for detail in failed_details)
    assert any("monitor_metric=iid_test_accuracy" in detail for detail in failed_details)


def test_evidence_audit_rejects_ood_tuning_metrics(tmp_path):
    root = tmp_path / "full_run"
    _write_evidence_fixture(root, seeds=(0, 1), epochs=3)
    metrics_path = root / "sta" / "sid" / "seed0" / "final_metrics.json"
    metrics = json.loads(metrics_path.read_text())
    metrics["task_guard_selection_metric"] = "ood_test_accuracy"
    save_json(metrics_path, metrics)

    report = audit_evidence(root, min_seeds=2, min_epochs=3)

    assert report["passed"] is False
    assert any(
        check["name"] == "sta_sid_no_ood_tuning_metrics" and check["passed"] is False
        for check in report["checks"]
    )


def test_evidence_audit_rejects_sid_checkpoint_before_schedule(tmp_path):
    root = tmp_path / "full_run"
    _write_evidence_fixture(root, seeds=(0, 1), epochs=3)
    metrics_path = root / "sta" / "sid" / "seed0" / "final_metrics.json"
    metrics = json.loads(metrics_path.read_text())
    metrics["selected_epoch"] = 0
    metrics["sid_selected_schedule_progress"] = 0.0
    metrics["sid_schedule_floor_satisfied"] = False
    save_json(metrics_path, metrics)

    report = audit_evidence(root, min_seeds=2, min_epochs=3)

    assert report["passed"] is False
    assert any(
        check["name"] == "sta_sid_sid_selected_checkpoint_after_schedule"
        and check["passed"] is False
        for check in report["checks"]
    )
    assert any(
        check["name"] == "sta_sid_sid_schedule_floor_satisfied"
        and check["passed"] is False
        for check in report["checks"]
    )


def test_evidence_audit_rejects_broader_ood_tuning_metric_keys(tmp_path):
    root = tmp_path / "full_run"
    _write_evidence_fixture(root, seeds=(0, 1), epochs=3)
    metrics_path = root / "sta" / "sid" / "seed0" / "final_metrics.json"
    metrics = json.loads(metrics_path.read_text())
    metrics["checkpoint_metric"] = "ood_test_accuracy"
    metrics["selector_split"] = "iid_test"
    save_json(metrics_path, metrics)

    report = audit_evidence(root, min_seeds=2, min_epochs=3)

    assert report["passed"] is False
    failed_details = {
        check["detail"]
        for check in report["checks"]
        if check["name"] == "sta_sid_no_ood_tuning_metrics"
        and check["passed"] is False
    }
    assert any("checkpoint_metric=ood_test_accuracy" in detail for detail in failed_details)
    assert any("selector_split=iid_test" in detail for detail in failed_details)


def test_evidence_audit_rejects_failed_preflight_artifact(tmp_path):
    root = tmp_path / "full_run"
    _write_evidence_fixture(root, seeds=(0, 1), epochs=3)
    preflight = tmp_path / "preflight.json"
    _write_preflight(preflight, root=root)
    payload = json.loads(preflight.read_text())
    payload["launch_recommended"] = False
    save_json(preflight, payload)

    report = audit_evidence(
        root,
        min_seeds=2,
        min_epochs=3,
        preflight_path=preflight,
    )

    failed = {check["name"] for check in report["checks"] if not check["passed"]}
    assert "preflight_passed" in failed


def test_evidence_audit_rejects_mismatched_preflight_root(tmp_path):
    root = tmp_path / "full_run"
    _write_evidence_fixture(root, seeds=(0, 1), epochs=3)
    preflight = tmp_path / "preflight.json"
    _write_preflight(preflight, root=tmp_path / "alternate_full_run")

    report = audit_evidence(
        root,
        min_seeds=2,
        min_epochs=3,
        preflight_path=preflight,
    )

    failed = {check["name"] for check in report["checks"] if not check["passed"]}
    assert "preflight_matches_result_root" in failed


def test_evidence_audit_rejects_smoke_config_in_non_smoke_root(tmp_path):
    root = tmp_path / "full_run"
    _write_evidence_fixture(root, seeds=(0, 1), epochs=3)
    preflight = tmp_path / "preflight.json"
    _write_preflight(preflight, root=root)
    metadata_path = root / "sta" / "erm" / "seed0" / "metadata.json"
    resolved_path = root / "sta" / "erm" / "seed0" / "resolved_config.json"
    metadata = json.loads(metadata_path.read_text())
    resolved = json.loads(resolved_path.read_text())
    metadata["config_path"] = "configs/sta_smoke.yaml"
    resolved["run"]["config_path"] = "configs/sta_smoke.yaml"
    resolved["experiment"] = "sta_smoke"
    save_json(metadata_path, metadata)
    save_json(resolved_path, resolved)

    report = audit_evidence(
        root,
        min_seeds=2,
        min_epochs=3,
        preflight_path=preflight,
    )

    failed = {check["name"] for check in report["checks"] if not check["passed"]}
    assert "sta_erm_non_smoke_config" in failed
    assert "sta_erm_expected_full_config" in failed


def test_evidence_audit_rejects_tiny_splits_in_non_smoke_root(tmp_path):
    root = tmp_path / "full_run"
    _write_evidence_fixture(root, seeds=(0, 1), epochs=3)
    preflight = tmp_path / "preflight.json"
    _write_preflight(preflight, root=root)
    resolved_path = root / "ink_advection_diffusion" / "sid" / "seed0" / "resolved_config.json"
    resolved = json.loads(resolved_path.read_text())
    resolved["splits"]["train"]["n_sequences"] = 96
    save_json(resolved_path, resolved)

    report = audit_evidence(
        root,
        min_seeds=2,
        min_epochs=3,
        preflight_path=preflight,
    )

    failed = {check["name"] for check in report["checks"] if not check["passed"]}
    assert "ink_advection_diffusion_sid_full_split_sizes" in failed


def test_evidence_audit_rejects_stale_preflight(tmp_path):
    root = tmp_path / "full_run"
    _write_evidence_fixture(root, seeds=(0, 1), epochs=3)
    preflight = tmp_path / "preflight.json"
    _write_preflight(preflight, root=root)
    os.utime(preflight, (1, 1))

    report = audit_evidence(
        root,
        min_seeds=2,
        min_epochs=3,
        preflight_path=preflight,
    )

    failed = {check["name"] for check in report["checks"] if not check["passed"]}
    assert "preflight_not_older_than_code_sources" in failed
