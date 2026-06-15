import torch

from src.losses.arrow_score import calibrate_arrow_score, dynamics_nlls, latent_arrow_score
from src.models.dynamics import GaussianDynamics, LOG_TWO_PI


def test_latent_arrow_score_shapes_and_finite_values():
    torch.manual_seed(0)
    z = torch.randn(4, 6, 3)
    forward = GaussianDynamics(3, hidden_dim=8, fixed_variance=True)
    reverse = GaussianDynamics(3, hidden_dim=8, fixed_variance=True)
    score = latent_arrow_score(z, forward, reverse)
    assert score.sigma_steps.shape == (4, 5)
    assert score.sigma_total.shape == (4,)
    assert score.sigma_per_step.shape == (4,)
    assert torch.isfinite(score.sigma_steps).all()
    assert torch.allclose(score.sigma_per_step, score.sigma_total / 5)


def test_dynamics_nlls_are_finite_and_differentiable():
    torch.manual_seed(0)
    z = torch.randn(4, 6, 3, requires_grad=True)
    forward = GaussianDynamics(3, hidden_dim=8)
    reverse = GaussianDynamics(3, hidden_dim=8)
    f_nll, r_nll = dynamics_nlls(z, forward, reverse)
    loss = f_nll + r_nll
    loss.backward()
    assert torch.isfinite(loss)
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()


def test_gaussian_dynamics_log_prob_uses_stable_constant():
    model = GaussianDynamics(2, hidden_dim=4, fixed_variance=True)
    z = torch.zeros(3, 2)
    mu, logvar = model(z)
    manual = -0.5 * ((z - mu).pow(2) * torch.exp(-logvar) + logvar + LOG_TWO_PI).sum(dim=-1)
    observed = model.log_prob(z, z)
    assert torch.allclose(observed, manual)


def test_batch_reverse_calibration_centers_without_touching_labels():
    torch.manual_seed(1)
    z = torch.randn(5, 7, 3)
    y = torch.tensor([0, 1, 0, 1, 1])
    y_before = y.clone()
    forward = GaussianDynamics(3, hidden_dim=8, fixed_variance=True)
    reverse = GaussianDynamics(3, hidden_dim=8, fixed_variance=True)
    raw = latent_arrow_score(z, forward, reverse)

    calibrated = calibrate_arrow_score(
        raw,
        mode="batch_reverse_centering",
        z=z,
        forward_model=forward,
        reverse_model=reverse,
    )

    assert torch.equal(y, y_before)
    assert calibrated.metadata["arrow_calibration_mode"] == "batch_reverse_centering"
    assert torch.allclose(calibrated.raw.sigma_total, raw.sigma_total)
    assert abs(float(calibrated.calibrated.sigma_per_step.mean())) <= (
        abs(float(raw.sigma_per_step.mean())) + 1.0
    )
    assert torch.isfinite(calibrated.calibrated.sigma_steps).all()
