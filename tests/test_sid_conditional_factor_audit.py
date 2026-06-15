import numpy as np
import torch

from src.eval.sid_conditional_factor_audit import CONDITIONAL_AUDIT_SCHEMA
from src.eval.sid_conditional_factor_audit import apply_residualizer
from src.eval.sid_conditional_factor_audit import audit_result_root
from src.eval.sid_conditional_factor_audit import audit_sid_run_conditional
from src.eval.sid_conditional_factor_audit import fit_residualizer
from src.eval.sid_conditional_factor_audit import orientation_free_auc
from src.eval.sid_conditional_factor_audit import residualize_against_controls
from src.train.common import load_config, save_json, train_supervised_method


def test_orientation_free_auc_handles_flipped_scores():
    y = np.array([0, 0, 1, 1])
    good_score = np.array([0.1, 0.2, 0.8, 0.9])
    flipped_score = -good_score

    assert orientation_free_auc(y, good_score) == 1.0
    assert orientation_free_auc(y, flipped_score) == 1.0


def test_residualized_metric_removes_linear_confounder():
    rng = np.random.default_rng(0)
    controls = rng.normal(size=(64, 2))
    x = np.column_stack(
        [
            2.0 * controls[:, 0] - controls[:, 1],
            -0.5 * controls[:, 0] + 0.25 * controls[:, 1],
        ]
    )

    coef = fit_residualizer(controls, x)
    residual = apply_residualizer(controls, x, coef)

    assert np.max(np.abs(residual)) < 1e-10


def test_residualize_against_controls_train_eval_shapes():
    rng = np.random.default_rng(1)
    train_controls = rng.normal(size=(16, 2))
    eval_controls = rng.normal(size=(8, 2))
    train_x = rng.normal(size=(16, 3))
    eval_x = rng.normal(size=(8, 3))

    train_resid, eval_resid = residualize_against_controls(
        train_x,
        train_controls,
        eval_x,
        eval_controls,
    )

    assert train_resid.shape == train_x.shape
    assert eval_resid.shape == eval_x.shape


def test_sid_conditional_audit_schema_and_no_ood_selection(tmp_path):
    config = load_config("configs/sid_ink_advection_diffusion_smoke.yaml")
    run_dir = tmp_path / "sid_run"
    train_supervised_method("sid", config, run_dir, torch.device("cpu"))

    report = audit_sid_run_conditional(run_dir, device=torch.device("cpu"))

    assert report["schema"] == CONDITIONAL_AUDIT_SCHEMA
    assert report["passed"] is True
    assert report["probe_train_split"] == "val_iid"
    assert report["eval_splits"] == ["iid_test", "ood_test"]
    assert "ood_test" not in report["probe_train_split"]
    assert "iid_test" in report["metrics"]
    assert "z_ir_task" in report["metrics"]["iid_test"]
    assert "spurious_dynamic" in report["metrics"]["iid_test"]["z_ir_spur"]
    target_metrics = report["metrics"]["iid_test"]["z_ir_spur"]["spurious_dynamic"]
    assert "raw_orientation_free_auc" in target_metrics
    assert "residualized_orientation_free_auc" in target_metrics
    assert "controls" in target_metrics
    assert (run_dir / "sid_conditional_factor_audit.json").exists()


def test_sid_conditional_result_root_summary(tmp_path):
    config = load_config("configs/sid_ink_advection_diffusion_smoke.yaml")
    run_dir = tmp_path / "root" / "ink_advection_diffusion" / "sid" / "seed0"
    train_supervised_method("sid", config, run_dir, torch.device("cpu"))
    save_json(
        tmp_path / "root" / "ink_advection_diffusion" / "manifest.json",
        {
            "runs": [
                {
                    "method": "sid",
                    "status": "success",
                    "run_dir": str(run_dir),
                }
            ]
        },
    )

    report = audit_result_root(
        tmp_path / "root",
        benchmarks=("ink_advection_diffusion",),
        device=torch.device("cpu"),
    )

    assert report["schema"] == CONDITIONAL_AUDIT_SCHEMA
    assert report["passed"] is True
    assert report["benchmarks"]["ink_advection_diffusion"]["n_runs"] == 1
    assert report["aggregate"]
