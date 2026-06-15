import torch
import torch.nn.functional as F

from src.losses.sid_loss import sid_anti_collapse_loss, sid_factor_diagnostics, sid_loss
from src.models.sid import GradientReversal, SIDModel


def test_sid_model_factor_shapes_and_task_head_lock():
    torch.manual_seed(0)
    model = SIDModel(
        input_dim=5,
        hidden_dim=8,
        z_rev_dim=3,
        z_ir_task_dim=4,
        z_ir_spur_dim=6,
        z_resid_dim=2,
    )
    out = model(torch.randn(7, 9, 5))
    assert out["z_rev"].shape == (7, 9, 3)
    assert out["z_ir_task"].shape == (7, 9, 4)
    assert out["z_ir_spur"].shape == (7, 9, 6)
    assert out["z_resid"].shape == (7, 9, 2)
    assert out["logits"].shape == (7, 2)
    assert out["core_probe_logits"].shape == (7, 2)
    assert out["spurious_probe_logits"].shape == (7, 2)
    assert model.task_head_input_dim == 7
    first_linear = model.classifier.net[0]
    assert first_linear.in_features == model.z_rev_dim + model.z_ir_task_dim
    assert first_linear.in_features != (
        model.z_rev_dim + model.z_ir_task_dim + model.z_ir_spur_dim
    )


def test_sid_task_loss_does_not_update_spurious_projection():
    torch.manual_seed(0)
    model = SIDModel(input_dim=5, hidden_dim=8, z_rev_dim=3, z_ir_task_dim=4, z_ir_spur_dim=6)
    x = torch.randn(6, 7, 5)
    y = torch.randint(0, 2, (6,))
    out = model(x)
    loss = F.cross_entropy(out["logits"], y)
    loss.backward()
    assert model.z_rev_proj.weight.grad is not None
    assert model.z_ir_task_proj.weight.grad is not None
    grad = model.z_ir_spur_proj.weight.grad
    assert grad is None or torch.count_nonzero(grad).item() == 0


def test_gradient_reversal_flips_representation_gradient_only():
    torch.manual_seed(0)
    model = SIDModel(input_dim=5, hidden_dim=8, z_rev_dim=3, z_ir_task_dim=4, z_ir_spur_dim=6)
    x = torch.randn(6, 7, 5)
    spur = torch.randn(6)

    out = model(x)
    task_rep = model.task_representation(out["factors"])
    task_rep.retain_grad()
    grl_pred = model.spur_adversary_head(GradientReversal.apply(task_rep, 1.0)).squeeze(-1)
    grl_loss = F.mse_loss(grl_pred, spur)
    model.zero_grad()
    grl_loss.backward()
    grl_task_grad = task_rep.grad.detach().clone()
    grl_head_grad = model.spur_adversary_head[-1].weight.grad.detach().clone()

    out_plain = model(x)
    task_rep_plain = model.task_representation(out_plain["factors"])
    task_rep_plain.retain_grad()
    plain_pred = model.spur_adversary_head(task_rep_plain).squeeze(-1)
    plain_loss = F.mse_loss(plain_pred, spur)
    model.zero_grad()
    plain_loss.backward()

    assert torch.allclose(grl_task_grad, -task_rep_plain.grad, atol=1e-6)
    assert torch.allclose(grl_head_grad, model.spur_adversary_head[-1].weight.grad, atol=1e-6)


def test_sid_loss_components_and_gradients():
    torch.manual_seed(0)
    model = SIDModel(input_dim=5, hidden_dim=8, z_rev_dim=3, z_ir_task_dim=4, z_ir_spur_dim=6)
    x = torch.randn(6, 7, 5)
    x_cf = x + 0.05 * torch.randn_like(x)
    y = torch.randint(0, 2, (6,))
    out = model(x)
    out_cf = model(x_cf)
    loss_out = sid_loss(
        task_loss=F.cross_entropy(out["logits"], y),
        task_loss_cf=F.cross_entropy(out_cf["logits"], y),
        factors=out["factors"],
        factors_cf=out_cf["factors"],
        weights={
            "lambda_cf_task": 1.0,
            "lambda_rev_cf": 0.5,
            "lambda_task_ir_cf": 0.5,
            "lambda_spur_sens": 0.5,
            "lambda_anti_collapse": 0.1,
        },
        spur_sensitivity_margin=0.01,
    )
    assert torch.isfinite(loss_out.total)
    assert "rev_cf_invariance" in loss_out.components
    assert "task_ir_cf_invariance" in loss_out.components
    assert "spur_ir_cf_sensitivity" in loss_out.components
    assert "anti_collapse" in loss_out.components
    # Full spec new terms (default 0 when not passed)
    assert "spur_adversary" in loss_out.components
    assert "core_preservation" in loss_out.components
    assert "spur_capture" in loss_out.components
    loss_out.total.backward()
    assert model.z_rev_proj.weight.grad is not None
    assert model.z_ir_task_proj.weight.grad is not None
    assert model.z_ir_spur_proj.weight.grad is not None


def test_sid_spurious_sensitivity_loss_zero_when_cf_delta_exceeds_margin():
    factors = {
        "z_rev": torch.zeros(2, 3, 4),
        "z_ir_task": torch.zeros(2, 3, 4),
        "z_ir_spur": torch.zeros(2, 3, 4),
    }
    factors_cf = {
        "z_rev": torch.zeros(2, 3, 4),
        "z_ir_task": torch.zeros(2, 3, 4),
        "z_ir_spur": torch.ones(2, 3, 4),
    }
    out = sid_loss(
        task_loss=torch.tensor(0.5),
        task_loss_cf=torch.tensor(0.5),
        factors=factors,
        factors_cf=factors_cf,
        weights={},
        spur_sensitivity_margin=0.1,
        anti_collapse_loss=torch.tensor(0.0),
    )
    assert out.components["rev_cf_invariance"].item() == 0.0
    assert out.components["task_ir_cf_invariance"].item() == 0.0
    assert out.components["spur_ir_cf_sensitivity"].item() == 0.0


def test_sid_anti_collapse_and_diagnostics_report_factor_stats():
    collapsed = {
        "z_rev": torch.zeros(2, 3, 4),
        "z_ir_task": torch.zeros(2, 3, 4),
        "z_ir_spur": torch.zeros(2, 3, 4),
    }
    varied = {
        "z_rev": torch.randn(4, 5, 4),
        "z_ir_task": torch.randn(4, 5, 4),
        "z_ir_spur": torch.randn(4, 5, 4),
    }
    assert sid_anti_collapse_loss(collapsed).item() > sid_anti_collapse_loss(varied).item()
    diagnostics = sid_factor_diagnostics(varied, varied)
    assert diagnostics["z_rev_norm_mean"] > 0.0
    assert diagnostics["z_ir_task_std_mean"] > 0.0
    assert diagnostics["z_ir_spur_cf_mse"] == 0.0
