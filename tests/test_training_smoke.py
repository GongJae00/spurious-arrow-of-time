import torch
import torch.nn.functional as F

from src.losses.arrow_score import dynamics_nlls, latent_arrow_score
from src.losses.sib_loss import sib_loss
from src.models.baselines import ERMGRU, IBGRU
from src.models.sib import SIBModel


def test_erm_gru_one_training_step():
    torch.manual_seed(0)
    model = ERMGRU(input_dim=5, hidden_dim=8, latent_dim=4)
    x = torch.randn(6, 7, 5)
    y = torch.randint(0, 2, (6,))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    out = model(x)
    loss = F.cross_entropy(out["logits"], y)
    loss.backward()
    opt.step()
    assert torch.isfinite(loss)


def test_ib_gru_pooled_bottleneck_outputs_kl_inputs():
    torch.manual_seed(0)
    model = IBGRU(input_dim=5, hidden_dim=8, latent_dim=4)
    out = model(torch.randn(6, 7, 5))
    assert out["mu"].shape == (6, 4)
    assert out["logvar"].shape == (6, 4)
    assert out["logits"].shape == (6, 2)


def test_sib_model_one_training_step_with_counterfactual_loss():
    torch.manual_seed(0)
    model = SIBModel(input_dim=5, hidden_dim=8, latent_dim=4)
    x = torch.randn(6, 7, 5)
    x_cf = x + 0.05 * torch.randn_like(x)
    y = torch.randint(0, 2, (6,))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    out = model(x)
    out_cf = model(x_cf)
    score = latent_arrow_score(out["z"], model.forward_dynamics, model.reverse_dynamics)
    score_cf = latent_arrow_score(out_cf["z"], model.forward_dynamics, model.reverse_dynamics)
    f_nll, r_nll = dynamics_nlls(out["z"], model.forward_dynamics, model.reverse_dynamics)
    loss_out = sib_loss(
        task_loss=F.cross_entropy(out["logits"], y),
        task_loss_cf=F.cross_entropy(out_cf["logits"], y),
        forward_nll=f_nll,
        reverse_nll=r_nll,
        sigma_total=score.sigma_total,
        sigma_total_cf=score_cf.sigma_total,
        sigma_steps=score.sigma_steps,
        sigma_steps_cf=score_cf.sigma_steps,
        sigma_target=0.0,
        weights={},
    )
    loss_out.total.backward()
    opt.step()
    assert torch.isfinite(loss_out.total)
