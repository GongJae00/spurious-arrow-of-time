import torch

from src.losses.ib_loss import gaussian_kl_to_standard_normal
from src.losses.sib_loss import sib_loss


def test_sib_counterfactual_losses_zero_when_scores_match():
    sigma_steps = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    sigma_total = sigma_steps.sum(dim=1)
    out = sib_loss(
        task_loss=torch.tensor(0.5),
        task_loss_cf=torch.tensor(0.25),
        forward_nll=torch.tensor(0.1),
        reverse_nll=torch.tensor(0.2),
        sigma_total=sigma_total,
        sigma_total_cf=sigma_total.clone(),
        sigma_steps=sigma_steps,
        sigma_steps_cf=sigma_steps.clone(),
        sigma_target=float((sigma_total / 2).mean()),
        weights={
            "lambda_cf_task": 1.0,
            "lambda_f": 1.0,
            "lambda_r": 1.0,
            "eta_total": 1.0,
            "eta_step": 1.0,
            "rho": 1.0,
        },
    )
    assert out.components["cf_arrow_total"].item() == 0.0
    assert out.components["cf_arrow_step"].item() == 0.0
    assert out.components["setpoint"].item() == 0.0


def test_sib_loss_has_gradients():
    sigma_steps = torch.randn(3, 4, requires_grad=True)
    sigma_steps_cf = torch.randn(3, 4, requires_grad=True)
    sigma_total = sigma_steps.sum(dim=1)
    sigma_total_cf = sigma_steps_cf.sum(dim=1)
    out = sib_loss(
        task_loss=torch.tensor(0.5, requires_grad=True),
        task_loss_cf=torch.tensor(0.25, requires_grad=True),
        forward_nll=torch.tensor(0.1, requires_grad=True),
        reverse_nll=torch.tensor(0.2, requires_grad=True),
        sigma_total=sigma_total,
        sigma_total_cf=sigma_total_cf,
        sigma_steps=sigma_steps,
        sigma_steps_cf=sigma_steps_cf,
        sigma_target=0.0,
        weights={},
    )
    out.total.backward()
    assert sigma_steps.grad is not None
    assert sigma_steps_cf.grad is not None


def test_sib_loss_zero_weights_reduce_to_task_loss():
    task_loss = torch.tensor(0.5, requires_grad=True)
    task_loss_cf = torch.tensor(0.25, requires_grad=True)
    forward_nll = torch.tensor(10.0, requires_grad=True)
    reverse_nll = torch.tensor(11.0, requires_grad=True)
    sigma_steps = torch.randn(3, 4, requires_grad=True)
    sigma_steps_cf = torch.randn(3, 4, requires_grad=True)
    out = sib_loss(
        task_loss=task_loss,
        task_loss_cf=task_loss_cf,
        forward_nll=forward_nll,
        reverse_nll=reverse_nll,
        sigma_total=sigma_steps.sum(dim=1),
        sigma_total_cf=sigma_steps_cf.sum(dim=1),
        sigma_steps=sigma_steps,
        sigma_steps_cf=sigma_steps_cf,
        sigma_target=3.0,
        weights={
            "lambda_cf_task": 0.0,
            "lambda_f": 0.0,
            "lambda_r": 0.0,
            "eta_total": 0.0,
            "eta_step": 0.0,
            "rho": 0.0,
        },
    )

    assert out.total.item() == task_loss.item()
    out.total.backward()
    assert task_loss.grad.item() == 1.0
    assert task_loss_cf.grad.item() == 0.0
    assert forward_nll.grad.item() == 0.0
    assert reverse_nll.grad.item() == 0.0
    assert torch.count_nonzero(sigma_steps.grad).item() == 0
    assert torch.count_nonzero(sigma_steps_cf.grad).item() == 0


def test_gaussian_kl_to_standard_normal_zero_for_standard_normal():
    mu = torch.zeros(5, 3)
    logvar = torch.zeros(5, 3)
    assert gaussian_kl_to_standard_normal(mu, logvar).item() == 0.0
