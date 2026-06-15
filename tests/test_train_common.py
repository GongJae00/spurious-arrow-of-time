import json

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.train.common import (
    load_config,
    _resolve_setpoint,
    _scheduled_sib_weights,
    build_supervised_model,
    evaluate_counterfactual_sensitivity,
    generate_splits_from_config,
    make_run_dir,
    train_arrow_pretraining_method,
    train_supervised_method,
)
from src.models.itm import ITMModel
from src.models.sid import SIDModel


def _tiny_config(seed: int = 0):
    return {
        "seed": seed,
        "splits": {
            "train": {"n_sequences": 32, "spurious_mode": "correlated"},
            "val_iid": {"n_sequences": 16, "spurious_mode": "correlated"},
            "iid_test": {"n_sequences": 16, "spurious_mode": "correlated"},
            "ood_test": {"n_sequences": 16, "spurious_mode": "reversed"},
        },
        "data": {
            "length": 8,
            "n_core_states": 8,
            "n_spur_states": 8,
            "p_core": 0.35,
            "q_core": 0.25,
            "p_spur": 0.45,
            "q_spur": 0.15,
            "obs_dim": 8,
            "label_mode": "core_net_displacement_median_threshold",
            "spurious_correlation_type": "drift_direction",
        },
        "observation": {
            "core_scale": 1.0,
            "spur_scale": 1.0,
            "normalize_mixing_columns": True,
            "noise_std": 0.1,
        },
        "counterfactual": {"spurious_cf_mode": "randomized", "reuse_noise": True},
        "model": {"hidden_dim": 8, "latent_dim": 4, "pooling": "last"},
        "dynamics": {"hidden_dim": 8, "min_logvar": -6.0, "max_logvar": 2.0},
        "training": {
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "batch_size": 16,
            "epochs": 1,
            "grad_clip_norm": 1.0,
        },
        "loss_weights": {
            "lambda_f": 1.0,
            "lambda_r": 1.0,
            "eta_total": 1.0,
            "eta_step": 0.1,
            "rho": 1.0,
            "lambda_cf_task": 1.0,
            "beta_latent": 0.0,
        },
        "setpoint": {"mode": "fixed_grid", "fixed_target": 0.0},
    }


def test_train_sib_writes_metrics_and_uses_val_iid_checkpoint(tmp_path):
    config = _tiny_config(seed=30)
    metrics = train_supervised_method("sib", config, tmp_path, torch.device("cpu"))
    metadata = json.loads((tmp_path / "metadata.json").read_text())
    assert "val_iid_accuracy" in metrics
    assert "ood_gap" in metrics
    assert "iid_test_cf_delta_arrow_total" in metrics
    assert "iid_test_cf_delta_arrow_step" in metrics
    assert "iid_test_cf_delta_arrow_total_calibrated" in metrics
    assert "iid_test_cf_delta_arrow_step_calibrated" in metrics
    assert "iid_test_cf_prediction_consistency" in metrics
    assert "iid_test_cf_probability_l1_drift" in metrics
    assert metrics["selected_checkpoint"] == "best.pt"
    assert (tmp_path / "best.pt").exists()
    assert (tmp_path / "final_metrics.json").exists()
    last = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()][-1]
    assert "loss_cf_arrow_total" in last
    assert "loss_cf_arrow_step" in last
    assert "loss_task_cf" in last
    assert "forward_logvar_min" in last
    assert "reverse_logvar_max" in last
    assert "sigma_per_step_abs_max" in last
    assert "weighted_loss_task" in last
    assert "weighted_loss_dynamics" in last
    assert "loss_component_ratio_regularizer_to_task_abs" in last
    assert "raw_sigma_per_step_mean" in last
    assert "calibrated_sigma_per_step_mean" in last
    assert "calibrated_latent_norm_mean" in last
    assert "train_prediction_entropy" in last
    assert "val_iid_prediction_entropy" in last
    assert "val_iid_cf_accuracy" in last
    assert "val_iid_cf_delta_arrow_total_calibrated" in last
    assert "val_iid_cf_delta_arrow_step_calibrated" in last
    assert "near_chance_task_detector" in last
    assert "loss_regularizer_dominance_detector" in last
    assert "sigma_threshold_exceeded" in last
    assert metadata["run_id"] == tmp_path.name
    assert metadata["method"] == "sib"
    assert metadata["seed"] == 30
    assert metadata["device"] == "cpu"
    assert metadata["hardware"]["device"] == "cpu"
    assert "torch_version" in metadata["hardware"]
    assert {"commit", "dirty", "status_short"}.issubset(metadata["git"])
    assert metadata["data_loader"]["num_workers"] == 0
    assert metadata["data_loader"]["pin_memory"] is False
    assert metadata["data_loader"]["worker_seed"] == 30
    assert metadata["dynamics_training"]["train_on_counterfactual"] is False
    assert metadata["arrow_calibration"]["mode"] == "none"
    assert last["dynamics_train_on_counterfactual"] is False


def test_generate_splits_routes_current_benchmarks():
    sta_config = _tiny_config(seed=1)
    ink_config = {
        "benchmark_name": "ink_advection_diffusion",
        "seed": 1,
        "splits": {
            "train": {"n_sequences": 16, "spurious_mode": "correlated"},
            "val_iid": {"n_sequences": 8, "spurious_mode": "correlated"},
            "iid_test": {"n_sequences": 8, "spurious_mode": "correlated"},
            "ood_test": {"n_sequences": 8, "spurious_mode": "reversed"},
        },
        "data": {"length": 6, "grid_size": 16, "dt": 0.1},
        "observation": {"noise_std": 0.03},
        "model": {"hidden_dim": 8, "latent_dim": 4},
        "training": {"optimizer": "adamw"},
    }

    sta = generate_splits_from_config(sta_config)
    ink = generate_splits_from_config(ink_config)

    assert sta["train"]["metadata"]["benchmark_name"] == "sta_bench"
    assert ink["train"]["metadata"]["benchmark_name"] == "ink_advection_diffusion"


def test_generate_splits_rejects_retired_benchmarks():
    config = _tiny_config(seed=2)
    config["benchmark_name"] = "retired_benchmark"
    with pytest.raises(ValueError, match="sta_bench.*ink_advection_diffusion"):
        generate_splits_from_config(config)


def test_build_supervised_model_sid_returns_sid_model():
    model = build_supervised_model(
        "sid",
        {
            "model": {"hidden_dim": 8, "latent_dim": 4, "pooling": "last"},
            "sid": {"z_rev_dim": 3, "z_ir_task_dim": 4, "z_ir_spur_dim": 5},
            "dynamics": {"hidden_dim": 8, "fixed_variance": True},
        },
        input_dim=8,
    )

    assert isinstance(model, SIDModel)
    assert model.z_rev_dim == 3
    assert model.z_ir_task_dim == 4
    assert model.z_ir_spur_dim == 5


def test_build_supervised_model_itm_returns_itm_model():
    model = build_supervised_model(
        "itm",
        {
            "model": {"hidden_dim": 8, "latent_dim": 4, "pooling": "last"},
            "itm": {"core_dim": 3, "spur_dim": 5},
            "dynamics": {"hidden_dim": 8},
        },
        input_dim=8,
    )

    assert isinstance(model, ITMModel)
    assert model.core_dim == 3
    assert model.spur_dim == 5
    assert model.task_head_input_dim == 6


def test_counterfactual_sensitivity_marks_arrow_metrics_unavailable_without_dynamics():
    class NoArrowDynamicsModel(torch.nn.Module):
        def forward(self, x):
            score = x.mean(dim=(1, 2))
            return {"logits": torch.stack([-score, score], dim=-1)}

    x = torch.randn(6, 4, 3)
    y = torch.randint(0, 2, (6,))
    x_cf = x + 0.1 * torch.randn_like(x)
    loader = DataLoader(TensorDataset(x, y, x_cf), batch_size=3)

    metrics = evaluate_counterfactual_sensitivity(
        NoArrowDynamicsModel(),
        loader,
        torch.device("cpu"),
    )

    assert metrics["cf_arrow_metrics_available"] is False
    assert metrics["cf_delta_arrow_total"] is None
    assert metrics["cf_delta_arrow_step"] is None
    assert metrics["cf_delta_arrow_total_calibrated"] is None
    assert metrics["cf_delta_arrow_step_calibrated"] is None
    assert 0.0 <= metrics["cf_prediction_consistency"] <= 1.0


def test_train_sid_on_ink_advection_diffusion_smoke_writes_factor_metrics(tmp_path):
    config = load_config("configs/sid_ink_advection_diffusion_smoke.yaml")
    metrics = train_supervised_method("sid", config, tmp_path, torch.device("cpu"))
    metadata = json.loads((tmp_path / "metadata.json").read_text())
    last = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()][-1]

    assert metadata["method"] == "sid"
    assert metadata["dataset_metadata"]["train"]["benchmark_name"] == "ink_advection_diffusion"
    assert metadata["setpoint"]["mode"] == "fixed_grid"
    assert metadata["sid_loss_normalization"]["log_gradient_norms"] is True
    assert "val_iid_accuracy" in metrics
    assert "ood_gap" in metrics
    assert "iid_test_cf_prediction_consistency" in metrics
    assert "loss_rev_cf_invariance" in last
    assert "loss_task_ir_cf_invariance" in last
    assert "loss_spur_ir_cf_sensitivity" in last
    assert "loss_task_forward_nll" in last
    assert "loss_spur_reverse_nll" in last
    assert "z_ir_spur_cf_mse" in last
    assert "task_loss_fraction" in last
    assert "dynamics_loss_fraction" in last
    assert "role_loss_fraction" in last
    assert "arrow_alignment_loss_fraction" in last
    assert "sid_grad_norm_encoder" in last
    assert "sid_grad_norm_spur_adversary_head" in last


def test_train_sid_respects_schedule_checkpoint_floor(tmp_path):
    config = _tiny_config(seed=43)
    config["sid"] = {
        "z_rev_dim": 3,
        "z_ir_task_dim": 3,
        "z_ir_spur_dim": 3,
        "z_resid_dim": 0,
        "spur_sensitivity_margin": 0.02,
    }
    config["dynamics"] = {**config["dynamics"], "fixed_variance": True}
    config["sid_schedule"] = {
        "task_warmup_epochs": 0,
        "dynamics_warmup_epochs": 0,
        "regularizer_ramp_epochs": 3,
    }
    config["training"] = {
        **config["training"],
        "epochs": 4,
        "patience": 1,
        "min_epoch_for_checkpoint_selection": 3,
        "min_epochs_before_early_stopping": 3,
    }
    config["loss_weights"] = {
        **config["loss_weights"],
        "lambda_rev_cf": 0.25,
        "lambda_task_ir_cf": 0.25,
        "lambda_spur_sens": 0.25,
        "lambda_spur_adv": 0.1,
        "lambda_core_preserve": 1.0,
        "lambda_spur_capture": 0.2,
        "lambda_arrow_decomp": 0.1,
        "lambda_anti_collapse": 0.05,
        "lambda_task_forward": 1.0,
        "lambda_task_reverse": 1.0,
        "lambda_spur_forward": 1.0,
        "lambda_spur_reverse": 1.0,
    }

    metrics = train_supervised_method("sid", config, tmp_path, torch.device("cpu"))
    records = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]

    assert metrics["epochs_completed"] >= 3
    assert metrics["selected_epoch"] >= 2
    assert metrics["checkpoint_selection_floor_satisfied"] is True
    assert metrics["sid_schedule_floor_satisfied"] is True
    assert records[0]["checkpoint_selection_eligible"] is False
    assert any(record["checkpoint_selection_eligible"] is True for record in records)


def test_train_sib_can_train_dynamics_on_counterfactual_pairs(tmp_path):
    config = _tiny_config(seed=37)
    config["dynamics"] = {**config["dynamics"], "train_on_counterfactual": True}
    train_supervised_method("sib", config, tmp_path, torch.device("cpu"))
    metadata = json.loads((tmp_path / "metadata.json").read_text())
    last = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()][-1]
    assert metadata["dynamics_training"]["train_on_counterfactual"] is True
    assert last["dynamics_train_on_counterfactual"] is True
    assert "loss_forward_nll_cf" in last
    assert "loss_reverse_nll_cf" in last


def test_train_sib_task_only_uses_pure_task_fast_path(tmp_path):
    config = _tiny_config(seed=39)
    config["loss_weights"] = {
        **config["loss_weights"],
        "lambda_cf_task": 0.0,
        "lambda_f": 0.0,
        "lambda_r": 0.0,
        "eta_total": 0.0,
        "eta_step": 0.0,
        "rho": 0.0,
    }
    train_supervised_method("sib", config, tmp_path, torch.device("cpu"))
    last = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()][-1]

    assert last["sib_task_only_training_fast_path"] is True
    assert last["dynamics_train_on_counterfactual"] is False
    assert last["loss_forward_nll"] == 0.0
    assert last["loss_reverse_nll"] == 0.0
    assert last["loss_task_cf"] == 0.0
    assert last["loss_cf_arrow_total"] == 0.0
    assert last["loss_cf_arrow_step"] == 0.0
    assert last["loss_setpoint"] == 0.0
    assert last["loss"] == last["loss_task"]


def test_train_sib_logs_schedule_and_batch_reverse_calibration(tmp_path):
    config = _tiny_config(seed=40)
    config["arrow_calibration"] = {"mode": "batch_reverse_centering"}
    config["loss_normalization"] = {"normalize_cf_total_by_transitions": True}
    config["sib_schedule"] = {
        "task_warmup_epochs": 0,
        "dynamics_warmup_epochs": 0,
        "regularizer_ramp_epochs": 2,
        "eta_total_start": 0.0,
        "eta_step_start": 0.0,
        "rho_start": 0.0,
    }
    config["task_guard"] = {
        "enabled": True,
        "min_val_iid_accuracy": 0.0,
        "selection_metric": "val_iid_accuracy_then_cf_stability",
    }
    metrics = train_supervised_method("sib", config, tmp_path, torch.device("cpu"))
    metadata = json.loads((tmp_path / "metadata.json").read_text())
    last = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()][-1]
    assert metadata["arrow_calibration"]["mode"] == "batch_reverse_centering"
    assert metadata["loss_normalization"]["normalize_cf_total_by_transitions"] is True
    assert "arrow_calibration_offset" in last
    assert last["arrow_calibration_mode"] == "batch_reverse_centering"
    assert last["effective_eta_total"] <= config["loss_weights"]["eta_total"]
    assert "task_guard_failed" in metrics
    assert "val_iid_cf_delta_arrow_total_calibrated" in metrics
    assert "unguarded_val_iid_cf_delta_arrow_total_calibrated" in metrics
    assert metrics["selected_checkpoint"] == "task_guard_best.pt"
    assert metrics["task_guard_failed"] is False
    assert metrics["task_guard_n_eligible_epochs"] >= 1
    assert "unguarded_val_iid_accuracy" in metrics
    assert (tmp_path / "task_guard_best.pt").exists()


def test_sib_schedule_can_warm_up_dynamics_losses():
    weights = {
        "lambda_f": 1.0,
        "lambda_r": 1.0,
        "eta_total": 1.0,
        "eta_step": 0.1,
        "rho": 1.0,
    }
    schedule = {
        "task_warmup_epochs": 2,
        "dynamics_warmup_epochs": 3,
        "regularizer_ramp_epochs": 4,
        "lambda_f_start": 0.0,
        "lambda_r_start": 0.0,
        "eta_total_start": 0.0,
        "eta_step_start": 0.0,
        "rho_start": 0.0,
    }

    epoch0 = _scheduled_sib_weights(weights, schedule, 0)
    assert epoch0["lambda_f"] == 0.0
    assert epoch0["lambda_r"] == 0.0
    assert epoch0["eta_total"] == 0.0
    assert epoch0["eta_step"] == 0.0
    assert epoch0["rho"] == 0.0
    assert epoch0["dynamics_schedule_progress"] == 0.0
    assert epoch0["dynamics_schedule_warmup_active"] is True

    epoch2 = _scheduled_sib_weights(weights, schedule, 2)
    assert 0.0 < epoch2["lambda_f"] < 1.0
    assert 0.0 < epoch2["lambda_r"] < 1.0
    assert epoch2["eta_total"] == 0.0
    assert epoch2["rho"] == 0.0

    epoch4 = _scheduled_sib_weights(weights, schedule, 4)
    assert epoch4["lambda_f"] == 1.0
    assert epoch4["lambda_r"] == 1.0
    assert epoch4["eta_total"] == 0.0
    assert epoch4["dynamics_schedule_warmup_active"] is False

    epoch5 = _scheduled_sib_weights(weights, schedule, 5)
    assert epoch5["lambda_f"] == 1.0
    assert epoch5["lambda_r"] == 1.0
    assert 0.0 < epoch5["eta_total"] < 1.0
    assert 0.0 < epoch5["rho"] < 1.0


def test_train_ep_min_logs_dynamics_losses(tmp_path):
    config = _tiny_config(seed=31)
    train_supervised_method("ep_min", config, tmp_path, torch.device("cpu"))
    last = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()][-1]
    assert "loss_forward_nll" in last
    assert "loss_reverse_nll" in last
    assert "loss_sigma_per_step" in last
    assert "loss_sigma_per_step_abs_mean" in last
    assert "loss_ep_regularizer" in last
    assert last["ep_regularizer_mode"] == "abs_mean"
    assert "forward_logvar_mean" in last
    assert "reverse_logvar_mean" in last
    assert "sigma_per_step_abs_max" in last
    assert "sigma_threshold_exceeded" in last


def test_model_config_num_layers_dropout_propagate():
    config = _tiny_config(seed=35)
    config["model"] = {
        **config["model"],
        "num_layers": 2,
        "dropout": 0.25,
        "bidirectional": False,
    }
    model = build_supervised_model("erm", config, input_dim=8)
    assert model.encoder.gru.num_layers == 2
    assert model.encoder.gru.dropout == 0.25


def test_bidirectional_config_fails_loudly():
    config = _tiny_config(seed=36)
    config["model"] = {**config["model"], "bidirectional": True}
    try:
        build_supervised_model("sib", config, input_dim=8)
    except ValueError as exc:
        assert "bidirectional" in str(exc)
    else:
        raise AssertionError("bidirectional=true must fail loudly or be implemented")


def test_unsupported_encoder_and_optimizer_fail_loudly(tmp_path):
    config = _tiny_config(seed=38)
    config["model"] = {**config["model"], "encoder": "transformer"}
    try:
        train_supervised_method("erm", config, tmp_path / "encoder", torch.device("cpu"))
    except ValueError as exc:
        assert "unsupported model.encoder" in str(exc)
    else:
        raise AssertionError("unsupported model.encoder must fail loudly")

    config = _tiny_config(seed=39)
    config["training"] = {**config["training"], "optimizer": "sgd"}
    try:
        train_supervised_method("erm", config, tmp_path / "optimizer", torch.device("cpu"))
    except ValueError as exc:
        assert "unsupported training.optimizer" in str(exc)
    else:
        raise AssertionError("unsupported training.optimizer must fail loudly")


def test_make_run_dir_allocates_unique_attempt_suffix(tmp_path, monkeypatch):
    monkeypatch.setattr("time.strftime", lambda _fmt: "20260101_000000")
    config = {"seed": 0}
    first = make_run_dir(tmp_path, "exp", "erm", config, seed=0, overwrite=False)
    second = make_run_dir(tmp_path, "exp", "erm", config, seed=0, overwrite=False)
    assert first != second
    assert second.name.endswith("_attempt1")


def test_generate_splits_rejects_nonstationary_initialization_outside_static_control():
    config = _tiny_config(seed=41)
    config["data"]["initial_state_mode"] = {
        "core": "uniform_stationary",
        "spurious": "sector_conditioned",
    }
    with pytest.raises(ValueError, match="diagnostic-only"):
        generate_splits_from_config(config)


def test_generate_splits_rejects_static_initial_sector_as_main_trap():
    config = _tiny_config(seed=42)
    config["data"]["spurious_correlation_type"] = "initial_sector_static_control"
    with pytest.raises(ValueError, match="main spurious_arrow_trap"):
        generate_splits_from_config(config)


def test_generate_splits_allows_static_initial_sector_control_ablation():
    config = _tiny_config(seed=43)
    config["experiment"] = "ablation_negative_controls"
    config["ablation"] = {"name": "static_spurious_control"}
    config["data"]["spurious_correlation_type"] = "initial_sector_static_control"
    splits = generate_splits_from_config(config)
    assert (
        splits["train"]["metadata"]["spurious"]["effective_initial_state_mode"]
        == "sector_conditioned"
    )


def test_generate_splits_ood_shift_type_overrides_base_ood_split_mode():
    config = _tiny_config(seed=44)
    config["data"]["ood_shift_type"] = "randomized"
    config["splits"]["ood_test"]["spurious_mode"] = "reversed"
    splits = generate_splits_from_config(config)
    assert splits["ood_test"]["metadata"]["spurious"]["spurious_mode"] == "randomized"
    assert splits["ood_test"]["metadata"]["spurious"]["ood_shift_type"] == "randomized"
    assert (
        splits["ood_test"]["metadata"]["spurious"]["spurious_label_correlation_strength"]
        == 0.0
    )


def test_setpoint_resolver_modes_are_explicit_and_logged():
    config = _tiny_config(seed=33)
    splits = generate_splits_from_config(config)
    input_dim = int(splits["train"]["x"].shape[-1])
    model = build_supervised_model("sib", config, input_dim)

    analytic_cfg = {**config, "setpoint": {"mode": "analytic_direct"}}
    analytic = _resolve_setpoint(analytic_cfg, splits, model, torch.device("cpu"))
    assert analytic["target_source"] == "train_core_analytic_ep"
    assert analytic["sigma_target"] == splits["train"]["metadata"]["core"]["analytic_ep"]

    val_cfg = {**config, "setpoint": {"mode": "val_iid_sweep", "selected_target": 0.123}}
    val_selected = _resolve_setpoint(val_cfg, splits, model, torch.device("cpu"))
    assert val_selected["sigma_target"] == 0.123
    assert val_selected["selection_split"] == "val_iid"
    assert val_selected["target_source"] == "val_iid_selected_target"

    oracle_cfg = {
        **config,
        "setpoint": {
            "mode": "oracle_core_reference",
            "reference_epochs": 1,
            "reference_batch_size": 16,
        },
    }
    oracle = _resolve_setpoint(oracle_cfg, splits, model, torch.device("cpu"))
    assert oracle["oracle_assisted"] is True
    assert oracle["target_source"] == "estimated_oracle_core_reference"
    assert isinstance(oracle["sigma_target"], float)
    assert oracle["reference_epochs"] == 1
    assert oracle["reference_batch_size"] == 16
    assert oracle["reference_lr"] == config["training"]["lr"]
    assert oracle["estimated_sigma_target"] == oracle["sigma_target"]


def test_arrow_pretraining_uses_train_split_only(tmp_path):
    config = _tiny_config(seed=32)
    config["training"]["num_workers"] = 0
    config["training"]["persistent_workers"] = True
    metrics = train_arrow_pretraining_method(
        "lens_like_arrow_classifier", config, tmp_path, torch.device("cpu")
    )
    metadata = json.loads((tmp_path / "metadata.json").read_text())
    assert metrics["pretraining_split"] == "train"
    assert metadata["pretraining_split"] == "train"
    assert metrics["pretraining_objective"] == "forward_reverse"
    assert metadata["pretraining_objective"] == "forward_reverse"
    assert "arrow_train_accuracy" in metrics
    assert metadata["transductive"] is False
    assert metadata["downstream_protocols"] == ["frozen_encoder", "fine_tuned_encoder"]
    assert metadata["data_loader"]["persistent_workers"] is False
    assert metadata["data_loader"]["worker_seed"] == 32
    assert "frozen_encoder_iid_test_accuracy" in metrics
    assert "fine_tuned_encoder_iid_test_accuracy" in metrics
    assert metrics["frozen_encoder_selected_checkpoint"] == "frozen_encoder_best.pt"
    assert metrics["fine_tuned_encoder_selected_checkpoint"] == "fine_tuned_encoder_best.pt"
    assert (tmp_path / "frozen_encoder_best.pt").exists()
    assert (tmp_path / "fine_tuned_encoder_best.pt").exists()


def test_ocp_pretraining_uses_segment_order_objective(tmp_path):
    config = _tiny_config(seed=34)
    metrics = train_arrow_pretraining_method("ocp_style", config, tmp_path, torch.device("cpu"))
    metadata = json.loads((tmp_path / "metadata.json").read_text())
    last = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()][-1]
    assert metrics["pretraining_objective"] == "segment_order"
    assert metadata["pretraining_objective"] == "segment_order"
    assert "order_train_accuracy" in metrics
    assert "order_train_accuracy" in last
    assert "arrow_train_accuracy" not in metrics
