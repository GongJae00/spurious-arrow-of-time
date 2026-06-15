import json
from pathlib import Path
from statistics import mean, stdev

import pytest

from src.eval.audit_evidence import REQUIRED_DIAGNOSTIC_GATES
from src.eval.audit_evidence import REQUIRED_ITM_MECHANISM_METRICS
from src.eval.audit_evidence import REQUIRED_SID_FACTOR_METRICS
from src.eval.audit_smoke import REQUIRED_METHODS
from src.eval.prepare_paper_assets import prepare_paper_assets
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
            metrics[key] = "z_ir_spur" if "spurious_dynamic" in key else "z_ir_task"
        elif key.endswith("_cf_mse"):
            metrics[key] = 0.3 if "z_ir_spur" in key else 0.1
        else:
            metrics[key] = 0.5
    return metrics


def _itm_mechanism_metrics() -> dict:
    metrics = {}
    for key in REQUIRED_ITM_MECHANISM_METRICS:
        if key in {"task_head_excludes_spur_mechanism", "mechanism_claim_ready"}:
            metrics[key] = True
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


def _write_result_root(root: Path, *, smoke: bool = False) -> Path:
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
            "class_balance": {
                "n": cfg["n_sequences"],
                "n0": cfg["n_sequences"] // 2,
                "n1": cfg["n_sequences"] - cfg["n_sequences"] // 2,
            },
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
        itm_run_dirs = []
        rows_by_method = {method: [] for method in REQUIRED_METHODS}
        for method in REQUIRED_METHODS:
            run_dir = suite_root / method / "seed0"
            run_dir.mkdir(parents=True)
            seed = 0
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
                metrics = {
                    "frozen_encoder_iid_test_accuracy": 0.5,
                    "frozen_encoder_ood_test_accuracy": 0.4,
                    "frozen_encoder_ood_gap": 0.1,
                    "frozen_encoder_selected_checkpoint": "frozen_encoder_best.pt",
                    "frozen_encoder_best_epoch": 0,
                    "frozen_encoder_epochs_completed": 1,
                    "fine_tuned_encoder_iid_test_accuracy": 0.6,
                    "fine_tuned_encoder_ood_test_accuracy": 0.5,
                    "fine_tuned_encoder_ood_gap": 0.1,
                    "fine_tuned_encoder_selected_checkpoint": (
                        "fine_tuned_encoder_best.pt"
                    ),
                    "fine_tuned_encoder_best_epoch": 0,
                    "fine_tuned_encoder_epochs_completed": 1,
                }
            else:
                metrics = {
                    "iid_test_accuracy": 0.6,
                    "ood_test_accuracy": 0.5,
                    "ood_gap": 0.1,
                    "epochs_completed": 1,
                    "selected_checkpoint": "best.pt",
                    "unguarded_selected_checkpoint": "best.pt",
                    "selected_epoch": 0,
                    "best_epoch": 0,
                    "unguarded_best_epoch": 0,
                }
            if method in {"sib", "sid", "itm"}:
                metrics["iid_test_cf_prediction_consistency"] = 1.0
            if method == "itm":
                metrics["iid_test_cf_arrow_metrics_available"] = False
                metrics["iid_test_cf_delta_arrow_total"] = None
                metrics["iid_test_cf_delta_arrow_step"] = None
                metrics["iid_test_cf_delta_arrow_total_calibrated"] = None
                metrics["iid_test_cf_delta_arrow_step_calibrated"] = None
                metrics["itm_schedule_required_epochs"] = 1
                metrics["itm_selected_schedule_progress"] = 1.0
                metrics["itm_selected_transition_schedule_progress"] = 1.0
                metrics["itm_schedule_floor_satisfied"] = True
            if method == "sid":
                metrics.update(
                    {
                        "selected_epoch": 0,
                        "best_epoch": 0,
                        "unguarded_best_epoch": 0,
                        "sid_schedule_required_epochs": 1,
                        "sid_selected_schedule_progress": 1.0,
                        "sid_selected_dynamics_progress": 1.0,
                        "sid_schedule_floor_satisfied": True,
                        "checkpoint_selection_floor_satisfied": True,
                        "early_stopping_floor_satisfied": True,
                        "min_epoch_for_checkpoint_selection": 1,
                        "min_epochs_before_early_stopping": 1,
                    }
                )
            (run_dir / "final.pt").write_text("dummy checkpoint")
            (run_dir / "metrics.jsonl").write_text(json.dumps({"epoch": 0}) + "\n")
            if method in {"ocp_style", "lens_like_arrow_classifier"}:
                (run_dir / "frozen_encoder_best.pt").write_text("dummy checkpoint")
                (run_dir / "fine_tuned_encoder_best.pt").write_text("dummy checkpoint")
                (run_dir / "frozen_encoder_metrics.jsonl").write_text(
                    json.dumps({"epoch": 0}) + "\n"
                )
                (run_dir / "fine_tuned_encoder_metrics.jsonl").write_text(
                    json.dumps({"epoch": 0}) + "\n"
                )
            else:
                (run_dir / "best.pt").write_text("dummy checkpoint")
            save_json(run_dir / "metadata.json", metadata)
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
                        "epochs": 1,
                        "min_epoch_for_checkpoint_selection": (
                            1 if method in {"sid", "itm"} else 0
                        ),
                        "min_epochs_before_early_stopping": (
                            1 if method in {"sid", "itm"} else 0
                        ),
                    },
                    "itm_schedule": {
                        "task_warmup_epochs": 0,
                        "transition_warmup_epochs": 0,
                        "regularizer_ramp_epochs": 1,
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
            if method == "itm":
                itm_run_dirs.append(run_dir)
        save_json(suite_root / "manifest.json", {"runs": runs})
        save_json(suite_root / "aggregate.json", _aggregate_from_rows(rows_by_method))
        sid_rows = []
        for sid_run_dir in sid_run_dirs:
            audit = {
                "passed": True,
                "method": "sid",
                "run_dir": str(sid_run_dir),
                "checkpoint": "best.pt",
                "benchmark_name": benchmark,
                "probe_train_split": "val_iid",
                "eval_splits": ["iid_test", "ood_test"],
                "target_metadata": {
                    "label": {"target_source": "y"},
                    "core_dynamic": {
                        "target_source": "core_dynamic_stat",
                        "threshold": 0.0,
                    },
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
            save_json(sid_run_dir / "sid_factor_audit.json", audit)
            sid_rows.append({"run_dir": str(sid_run_dir), "audit": audit})
        save_json(
            suite_root / "sid_factor_audit.json",
            {
                "passed": True,
                "manifest": "manifest.json",
                "checkpoint": "best.pt",
                "n_runs": len(sid_rows),
                "runs": sid_rows,
            },
        )
        itm_rows = []
        for itm_run_dir in itm_run_dirs:
            audit = {
                "schema": "itm_mechanism_audit_v1",
                "passed": True,
                "method": "itm",
                "run_dir": str(itm_run_dir.resolve()),
                "checkpoint": "best.pt",
                "benchmark_name": benchmark,
                "probe_train_split": "val_iid",
                "eval_splits": ["iid_test", "ood_test"],
                "target_metadata": {
                    "label": {"target_source": "y"},
                    "core_dynamic": {
                        "target_source": "core_dynamic_stat",
                        "threshold": 0.0,
                    },
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
            save_json(itm_run_dir / "itm_mechanism_audit.json", audit)
            itm_rows.append({"run_dir": str(itm_run_dir), "audit": audit})
        save_json(
            suite_root / "itm_mechanism_audit.json",
            {
                "schema": "itm_mechanism_audit_v1",
                "passed": True,
                "manifest": "manifest.json",
                "checkpoint": "best.pt",
                "n_runs": len(itm_rows),
                "mechanism_claim_ready_runs": len(itm_rows),
                "mechanism_claim_ready": True,
                "runs": itm_rows,
            },
        )
    save_json(
        root / "evidence_audit.json",
        {
            "passed": True,
            "run_count": 24,
            "min_seeds": 1,
            "min_epochs": 1,
            "allow_smoke": smoke,
        },
    )
    return root


def _write_preflight(path: Path, *, root: Path) -> Path:
    save_json(
        path,
        {
            "pass": True,
            "launch_recommended": True,
            "environment": {
                "out": str(root),
                "device": "cpu",
                "min_seeds": 1,
                "epochs": 1,
                "methods": list(REQUIRED_METHODS),
                "require_cuda_for_full_run": "0",
                "launch_authorization": "maintenance_or_diagnostic",
            },
        },
    )
    return path


def test_prepare_paper_assets_writes_logged_tables(tmp_path):
    root = _write_result_root(tmp_path / "full_run")
    preflight = _write_preflight(root / "preflight.json", root=root)
    output = tmp_path / "paper_assets"

    manifest = prepare_paper_assets(
        root,
        output_dir=output,
        min_seeds=1,
        min_epochs=1,
    )

    assert manifest["evidence_audit_passed"] is True
    assert manifest["evidence_audit_n_checks"] > 0
    assert manifest["evidence_audit_n_failed"] == 0
    assert manifest["evidence_audit_failed_checks"] == []
    assert manifest["preflight_path"] == str(preflight)
    assert manifest["preflight_device"] == "cpu"
    assert manifest["preflight_require_cuda_for_full_run"] == "0"
    assert manifest["preflight_launch_authorization"] == (
        "maintenance_or_diagnostic"
    )
    assert manifest["preflight_environment_summary"]["device"] == "cpu"
    assert manifest["final_manuscript_ready"] is True
    assert manifest["positive_primary_success_claim_ready"] is False
    assert manifest["positive_sid_success_claim_ready"] is False
    assert manifest["primary_method"] == "itm"
    assert manifest["primary_label"] == "ITM"
    assert manifest["claim_mode"] == "diagnostic_or_negative_evidence"
    assert manifest["result_claim_mode"] == "diagnostic_or_negative_evidence"
    assert manifest["paper_submission_scope"] == "diagnostic_or_negative_evidence_candidate"
    assert manifest["final_manuscript_ready_without_warnings"] is None
    assert manifest["hardware_execution_context_ready"] is None
    assert manifest["operational_warning_names"] == []
    assert (output / "results_summary.tex").exists()
    assert (output / "tables" / "main_metrics_table.tex").exists()
    assert (output / "tables" / "itm_mechanism_audit_table.tex").exists()
    assert (output / "tables" / "sid_factor_audit_table.tex").exists()
    table = (output / "tables" / "main_metrics_table.tex").read_text()
    assert "ink\\_advection\\_diffusion" in table
    assert "sid" in table
    summary = (output / "results_summary.tex").read_text()
    assert "Positive primary claim allowed" in summary
    assert "Positive primary success claim ready" in summary
    assert "diagnostic\\_or\\_negative\\_evidence\\_candidate" in summary
    assert "final\\_submission\\_ready" in summary
    assert "warning-free hardware-execution claim" in summary
    assert "Claim mode" in summary
    assert "Generated Claim-Gate Interpretation" in summary
    assert "ITM Mechanism Audit" in summary
    assert "ITM mechanism claim ready" in summary
    assert "result\\_interpretation.json" in summary
    assert "\\texttt{ink\\_\\allowbreak{}advection\\_\\allowbreak{}diffusion}" in summary
    assert "claim gate" in summary
    assert "SID diagnostic factor role claim ready" in summary
    assert "Gate reasons" in summary
    readme = (output / "README.md").read_text()
    assert "Readiness fields are intentionally separated" in readme
    assert "final_manuscript_ready: true" in readme
    assert "positive_primary_success_claim_ready: false" in readme
    assert "paper_submission_scope: diagnostic_or_negative_evidence_candidate" in readme
    assert "final mode" in readme
    assert "require_final_manuscript_ready=true" in readme
    assert "generated_manifest_final_manuscript_ready=true" in readme
    assert "hardware execution context is" in readme
    assert "warning-free" in readme
    assert "Preflight authorization provenance" in readme
    assert "launch_authorization: maintenance_or_diagnostic" in readme
    assert "Benchmark claim gate summary" in readme
    assert "claim_gate_passed=" in readme
    assert "ITM mechanism claim ready=" in readme
    assert "SID diagnostic factor role claim ready=" in readme
    assert "SID factor-role diagnostics are auxiliary" in readme


def test_prepare_paper_assets_rejects_smoke_without_override(tmp_path):
    root = _write_result_root(tmp_path / "smoke_run", smoke=True)

    with pytest.raises(ValueError, match="smoke"):
        prepare_paper_assets(root, output_dir=tmp_path / "paper_assets")


def test_prepare_paper_assets_allows_smoke_as_nonfinal_diagnostic(tmp_path):
    root = _write_result_root(tmp_path / "smoke_run", smoke=True)

    manifest = prepare_paper_assets(
        root,
        output_dir=tmp_path / "paper_assets",
        min_seeds=1,
        min_epochs=1,
        allow_smoke=True,
    )

    saved = json.loads((tmp_path / "paper_assets" / "paper_assets_manifest.json").read_text())
    assert manifest["final_manuscript_ready"] is False
    assert manifest["primary_method"] == "itm"
    assert manifest["claim_mode"] == "diagnostic_or_negative_evidence"
    assert manifest["paper_submission_scope"] == "diagnostic_or_negative_evidence_candidate"
    assert saved["allow_smoke"] is True
    assert "itm_mechanism_audit_table" in saved["generated_files"]
