import torch
import torch.nn.functional as F

from src.eval.itm_mechanism_audit import audit_itm_run
from src.losses.itm_loss import itm_anti_collapse_loss, itm_loss
from src.models.itm import ITMModel
from src.models.sid import GradientReversal
from src.train.common import build_supervised_model, generate_splits_from_config, load_config, save_json


def test_itm_model_shapes_and_task_head_lock():
    torch.manual_seed(0)
    model = ITMModel(input_dim=5, hidden_dim=8, core_dim=3, spur_dim=4)
    out = model(torch.randn(7, 9, 5))
    assert out["z_core"].shape == (7, 9, 3)
    assert out["z_spur"].shape == (7, 9, 4)
    assert out["core_delta"].shape == (7, 8, 3)
    assert out["spur_delta"].shape == (7, 8, 4)
    assert out["logits"].shape == (7, 2)
    assert model.task_head_input_dim == 2 * model.core_dim
    first_linear = model.classifier.net[0]
    assert first_linear.in_features == 2 * model.core_dim
    assert first_linear.in_features != 2 * (model.core_dim + model.spur_dim)


def test_itm_task_loss_does_not_update_spurious_projection():
    torch.manual_seed(0)
    model = ITMModel(input_dim=5, hidden_dim=8, core_dim=3, spur_dim=4)
    x = torch.randn(6, 7, 5)
    y = torch.randint(0, 2, (6,))
    out = model(x)
    F.cross_entropy(out["logits"], y).backward()
    assert model.core_proj.weight.grad is not None
    grad = model.spur_proj.weight.grad
    assert grad is None or torch.count_nonzero(grad).item() == 0


def test_itm_loss_components_and_gradients():
    torch.manual_seed(0)
    model = ITMModel(input_dim=5, hidden_dim=8, core_dim=3, spur_dim=4)
    x = torch.randn(6, 7, 5)
    x_cf = x + 0.05 * torch.randn_like(x)
    y = torch.randint(0, 2, (6,))
    core_stat = torch.randn(6)
    spur_stat = torch.randn(6)
    out = model(x)
    out_cf = model(x_cf)
    spur_stat_n = (spur_stat - spur_stat.mean()) / (spur_stat.std(unbiased=False) + 1e-8)
    adv_pred = model.spur_adversary_head(
        GradientReversal.apply(out["task_rep"], 1.0)
    ).squeeze(-1)
    loss_out = itm_loss(
        task_loss=F.cross_entropy(out["logits"], y),
        task_loss_cf=F.cross_entropy(out_cf["logits"], y),
        core_next_pred=out["core_next_pred"],
        core_next_target=out["z_core"][:, 1:],
        spur_next_pred=out["spur_next_pred"],
        spur_next_target=out["z_spur"][:, 1:],
        core_delta=out["core_delta"],
        core_delta_cf=out_cf["core_delta"],
        spur_delta=out["spur_delta"],
        spur_delta_cf=out_cf["spur_delta"],
        core_stat_pred=out["core_stat_pred"],
        core_stat=core_stat,
        spur_stat_pred=out["spurious_stat_pred"],
        spur_stat=spur_stat,
        spur_adversary_loss=F.mse_loss(adv_pred, spur_stat_n),
        anti_collapse_loss=itm_anti_collapse_loss(out["z_core"], out["z_spur"]),
        weights={
            "lambda_cf_task": 1.0,
            "lambda_core_transition": 1.0,
            "lambda_spur_transition": 0.5,
            "lambda_core_mech_cf": 1.0,
            "lambda_spur_mech_sens": 0.5,
            "lambda_core_preserve": 1.0,
            "lambda_spur_capture": 0.2,
            "lambda_spur_adv": 0.1,
            "lambda_anti_collapse": 0.05,
        },
        spur_sensitivity_margin=0.01,
    )
    assert torch.isfinite(loss_out.total)
    assert "core_cf_invariance" in loss_out.components
    assert "spur_cf_sensitivity" in loss_out.components
    assert "core_transition_fit" in loss_out.components
    loss_out.total.backward()
    assert model.core_proj.weight.grad is not None
    assert model.spur_proj.weight.grad is not None


def test_itm_mechanism_audit_reports_required_mechanism_metrics(tmp_path):
    config = load_config("configs/sta_smoke.yaml")
    config["seed"] = 123
    config["splits"] = {
        "train": {"n_sequences": 48, "spurious_mode": "correlated"},
        "val_iid": {"n_sequences": 40, "spurious_mode": "correlated"},
        "iid_test": {"n_sequences": 40, "spurious_mode": "correlated"},
        "ood_test": {"n_sequences": 40, "spurious_mode": "reversed"},
    }
    splits = generate_splits_from_config(config)
    model = build_supervised_model("itm", config, int(splits["train"]["x"].shape[-1]))
    run_dir = tmp_path / "itm_run"
    run_dir.mkdir()
    save_json(run_dir / "resolved_config.json", config)
    torch.save({"model": model.state_dict(), "epoch": 0}, run_dir / "best.pt")

    report = audit_itm_run(run_dir, device=torch.device("cpu"))

    assert report["passed"] is True
    assert report["method"] == "itm"
    assert (run_dir / "itm_mechanism_audit.json").exists()
    metrics = report["metrics"]
    assert metrics["task_head_excludes_spur_mechanism"] is True
    assert "iid_test_task_rep_label_residualized_auc" in metrics
    assert "ood_test_spur_rep_spurious_dynamic_residualized_auc" in metrics
    assert "iid_test_core_delta_cf_mse" in metrics
    assert "ood_test_spur_delta_cf_mse" in metrics
    assert "mechanism_claim_ready" in metrics
