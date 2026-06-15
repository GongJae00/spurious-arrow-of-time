import json
from pathlib import Path
from statistics import mean, stdev

from src.eval.audit_evidence import REQUIRED_DIAGNOSTIC_GATES
from src.eval.audit_evidence import REQUIRED_ITM_MECHANISM_METRICS
from src.eval.audit_evidence import REQUIRED_SID_FACTOR_METRICS
from src.eval.audit_smoke import REQUIRED_METHODS
from src.eval.interpret_results import interpret_results
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


def _write_fixture(
    root: Path,
    *,
    sid_iid: float,
    sid_ood: float,
    baseline_iid: float,
    baseline_ood: float,
    seeds: tuple[int, ...] = (0, 1),
    epochs: int = 3,
    nested_aggregate: bool = False,
) -> None:
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
            for seed in seeds:
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
                    metrics = {
                        "frozen_encoder_iid_test_accuracy": baseline_iid,
                        "frozen_encoder_ood_test_accuracy": baseline_ood,
                        "frozen_encoder_ood_gap": baseline_iid - baseline_ood,
                        "frozen_encoder_selected_checkpoint": "frozen_encoder_best.pt",
                        "frozen_encoder_best_epoch": 0,
                        "frozen_encoder_epochs_completed": epochs,
                        "fine_tuned_encoder_iid_test_accuracy": baseline_iid,
                        "fine_tuned_encoder_ood_test_accuracy": baseline_ood,
                        "fine_tuned_encoder_ood_gap": baseline_iid - baseline_ood,
                        "fine_tuned_encoder_selected_checkpoint": (
                            "fine_tuned_encoder_best.pt"
                        ),
                        "fine_tuned_encoder_best_epoch": 0,
                        "fine_tuned_encoder_epochs_completed": epochs,
                    }
                else:
                    iid = sid_iid if method == "itm" else baseline_iid
                    ood = sid_ood if method == "itm" else baseline_ood
                    metrics = {
                        "iid_test_accuracy": iid,
                        "ood_test_accuracy": ood,
                        "ood_gap": iid - ood,
                        "epochs_completed": epochs,
                        "selected_checkpoint": "best.pt",
                        "unguarded_selected_checkpoint": "best.pt",
                        "selected_epoch": 0,
                        "best_epoch": 0,
                        "unguarded_best_epoch": 0,
                    }
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
                            "epochs": epochs,
                            "min_epoch_for_checkpoint_selection": (
                                epochs if method in {"sid", "itm"} else 0
                            ),
                            "min_epochs_before_early_stopping": (
                                epochs if method in {"sid", "itm"} else 0
                            ),
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
                if method == "itm":
                    itm_run_dirs.append(run_dir)
        if nested_aggregate:
            aggregate = {
                "n_skipped_metric_files": 0,
                "skipped_metric_files": [],
                "by_condition": {f"{suite}": _aggregate_from_rows(rows_by_method)},
            }
        else:
            aggregate = _aggregate_from_rows(rows_by_method)
        save_json(suite_root / "manifest.json", {"runs": runs})
        save_json(suite_root / "aggregate.json", aggregate)
        save_json(
            suite_root / "sid_factor_audit.json",
            _sid_factor_audit(sid_run_dirs, benchmark=benchmark),
        )
        save_json(
            suite_root / "itm_mechanism_audit.json",
            _itm_mechanism_audit(itm_run_dirs, benchmark=benchmark),
        )


def test_interpret_results_allows_qualified_positive_claim(tmp_path):
    root = tmp_path / "full_run"
    _write_fixture(root, sid_iid=0.72, sid_ood=0.70, baseline_iid=0.75, baseline_ood=0.62)

    report = interpret_results(root, min_seeds=2, min_epochs=3)

    assert report["positive_primary_claim_allowed"] is True
    assert report["positive_primary_success_claim_ready"] is True
    assert report["positive_sid_claim_allowed"] is False
    assert report["claim_mode"] == "qualified_positive_primary_claim"
    assert report["paper_submission_scope"] == "positive_primary_claim_candidate"
    assert (root / "result_interpretation.json").exists()
    assert (root / "result_interpretation.md").exists()


def test_interpret_results_blocks_weak_sid_result(tmp_path):
    root = tmp_path / "full_run"
    _write_fixture(root, sid_iid=0.62, sid_ood=0.58, baseline_iid=0.70, baseline_ood=0.64)

    report = interpret_results(root, min_seeds=2, min_epochs=3)

    assert report["positive_primary_claim_allowed"] is False
    assert report["positive_primary_success_claim_ready"] is False
    assert report["claim_mode"] == "diagnostic_or_negative_evidence"
    assert report["paper_submission_scope"] == "diagnostic_or_negative_evidence_candidate"
    markdown = (root / "result_interpretation.md").read_text()
    assert "positive_primary_success_claim_ready" in markdown
    assert "paper_submission_scope" in markdown
    assert "paper_assets_manifest.json" in markdown
    reasons = [
        reason
        for gate in report["benchmark_gates"]
        for reason in gate["reasons"]
    ]
    assert any("OOD accuracy is below threshold" in reason for reason in reasons)
    assert any("OOD improvement is insufficient" in reason for reason in reasons)


def test_interpret_results_does_not_use_sid_role_alignment_as_primary_gate(tmp_path):
    root = tmp_path / "full_run"
    _write_fixture(root, sid_iid=0.72, sid_ood=0.70, baseline_iid=0.75, baseline_ood=0.62)
    for sid_factor_path in root.glob("*/sid_factor_audit.json"):
        sid_factor = json.loads(sid_factor_path.read_text())
        for row in sid_factor["runs"]:
            audit = row["audit"]
            audit["metrics"]["decomposition_role_claim_ready"] = False
            audit["role_alignment"] = {
                "decomposition_role_claim_ready": False,
                "decomposition_role_alignment_failures": [
                    "iid_test.label: expected z_ir_task, observed z_rev"
                ],
            }
            save_json(Path(row["run_dir"]) / "sid_factor_audit.json", audit)
        save_json(sid_factor_path, sid_factor)

    report = interpret_results(root, min_seeds=2, min_epochs=3)

    assert report["positive_primary_claim_allowed"] is True
    reasons = [
        reason
        for gate in report["benchmark_gates"]
        for reason in gate["reasons"]
    ]
    assert not any("factor role alignment claim is not ready" in reason for reason in reasons)
    assert all(gate["sid_factor_role_claim_ready"] is False for gate in report["benchmark_gates"])


def test_interpret_results_reads_nested_aggregate_and_blocks_smoke(tmp_path):
    root = tmp_path / "smoke_run"
    _write_fixture(
        root,
        sid_iid=0.72,
        sid_ood=0.70,
        baseline_iid=0.75,
        baseline_ood=0.62,
        seeds=(0,),
        epochs=1,
        nested_aggregate=True,
    )

    report = interpret_results(root, min_seeds=1, min_epochs=1, allow_smoke=True)

    assert report["evidence_audit_passed"] is True
    assert report["positive_primary_claim_allowed"] is False
    assert report["benchmark_gates"][0]["aggregate_condition"] in {
        "sta",
        "ink_advection_diffusion",
    }
