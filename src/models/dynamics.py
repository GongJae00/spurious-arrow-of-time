"""Forward/reverse diagonal Gaussian latent dynamics."""

from __future__ import annotations

import math

import torch
from torch import nn

LOG_TWO_PI = math.log(2.0 * math.pi)


class GaussianDynamics(nn.Module):
    """Diagonal Gaussian p(z_next | z_current)."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 64,
        min_logvar: float = -6.0,
        max_logvar: float = 2.0,
        fixed_variance: bool = False,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.min_logvar = min_logvar
        self.max_logvar = max_logvar
        self.fixed_variance = fixed_variance
        out_dim = latent_dim if fixed_variance else 2 * latent_dim
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, z_current: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.net(z_current)
        if self.fixed_variance:
            mu = out
            logvar = torch.zeros_like(mu)
        else:
            mu, raw_logvar = out.chunk(2, dim=-1)
            logvar = raw_logvar.clamp(self.min_logvar, self.max_logvar)
        return mu, logvar

    def log_prob(self, z_next: torch.Tensor, z_current: torch.Tensor) -> torch.Tensor:
        mu, logvar = self.forward(z_current)
        inv_var = torch.exp(-logvar)
        elementwise = -0.5 * ((z_next - mu).pow(2) * inv_var + logvar + LOG_TWO_PI)
        return elementwise.sum(dim=-1)

    def nll(self, z_next: torch.Tensor, z_current: torch.Tensor) -> torch.Tensor:
        return -self.log_prob(z_next, z_current).mean()

    def logvar_summary(self, z_current: torch.Tensor, prefix: str) -> dict[str, float]:
        _, logvar = self.forward(z_current)
        return {
            f"{prefix}_logvar_mean": float(logvar.mean().detach().cpu()),
            f"{prefix}_logvar_std": float(logvar.std(unbiased=False).detach().cpu()),
            f"{prefix}_logvar_min": float(logvar.min().detach().cpu()),
            f"{prefix}_logvar_max": float(logvar.max().detach().cpu()),
        }
