import torch

from src.eval.audit_smoke import audit_smoke
from src.eval.sid_factor_audit import audit_sid_run
from src.train.common import load_config, save_json, train_supervised_method


def test_sid_factor_audit_writes_factor_probe_metrics(tmp_path):
    config = load_config("configs/sid_ink_advection_diffusion_smoke.yaml")
    run_dir = tmp_path / "sid_run"
    train_supervised_method("sid", config, run_dir, torch.device("cpu"))

    report = audit_sid_run(run_dir, device=torch.device("cpu"))

    assert report["passed"] is True
    assert report["benchmark_name"] == "ink_advection_diffusion"
    assert report["metrics"]["task_head_excludes_z_ir_spur"] is True
    assert "iid_test_z_ir_task_label_probe_accuracy" in report["metrics"]
    assert "iid_test_z_ir_spur_spurious_dynamic_probe_accuracy" in report["metrics"]
    assert "iid_test_z_ir_spur_cf_mse" in report["metrics"]
    assert "ood_test_spurious_dynamic_best_factor" in report["metrics"]
    assert "decomposition_role_claim_ready" in report["metrics"]
    assert "iid_test_label_best_factor_is_z_ir_task" in report["metrics"]
    assert "ood_test_spurious_dynamic_best_factor_is_z_ir_spur" in report["metrics"]
    assert "decomposition_role_alignment_failures" in report["role_alignment"]
    assert (run_dir / "sid_factor_audit.json").exists()


def test_smoke_audit_requires_factor_and_mechanism_artifacts(tmp_path):
    root = tmp_path / "audit_root"
    methods = [
        "erm",
        "ib",
        "ep_min",
        "ep_max",
        "ocp_style",
        "lens_like_arrow_classifier",
        "sib",
        "sid",
        "itm",
    ]
    for suite, benchmark in {
        "sta": "sta_bench",
        "ink_advection_diffusion": "ink_advection_diffusion",
    }.items():
        suite_root = root / suite
        runs = []
        for method in methods:
            run_dir = suite_root / method / "run"
            run_dir.mkdir(parents=True)
            save_json(
                run_dir / "metadata.json",
                {
                    "method": method,
                    "dataset_metadata": {"train": {"benchmark_name": benchmark}},
                },
            )
            metrics = (
                {
                    "frozen_encoder_iid_test_accuracy": 0.5,
                    "frozen_encoder_ood_test_accuracy": 0.5,
                    "frozen_encoder_ood_gap": 0.0,
                }
                if method in {"ocp_style", "lens_like_arrow_classifier"}
                else {
                    "iid_test_accuracy": 0.5,
                    "ood_test_accuracy": 0.5,
                    "ood_gap": 0.0,
                }
            )
            if method in {"sib", "sid", "itm"}:
                metrics["iid_test_cf_prediction_consistency"] = 1.0
            save_json(run_dir / "final_metrics.json", metrics)
            runs.append({"method": method, "status": "success", "run_dir": str(run_dir)})
        save_json(suite_root / "manifest.json", {"runs": runs})
        save_json(suite_root / "sid_factor_audit.json", {"passed": True, "n_runs": 1})
        save_json(
            suite_root / "itm_mechanism_audit.json",
            {"passed": True, "n_runs": 1, "mechanism_claim_ready_runs": 1},
        )

    report = audit_smoke(root)

    assert report["passed"] is True
    assert report["run_count"] == 18
    assert any(check["name"] == "sta_sid_factor_audit_passed" for check in report["checks"])
    assert any(check["name"] == "sta_itm_mechanism_audit_passed" for check in report["checks"])

    (root / "sta" / "sid_factor_audit.json").unlink()
    failed = audit_smoke(root)
    assert failed["passed"] is False
    assert any(
        check["name"] == "sta_sid_factor_audit_exists" and check["passed"] is False
        for check in failed["checks"]
    )
    save_json(root / "sta" / "sid_factor_audit.json", {"passed": True, "n_runs": 1})
    (root / "sta" / "itm_mechanism_audit.json").unlink()
    failed = audit_smoke(root)
    assert failed["passed"] is False
    assert any(
        check["name"] == "sta_itm_mechanism_audit_exists" and check["passed"] is False
        for check in failed["checks"]
    )
