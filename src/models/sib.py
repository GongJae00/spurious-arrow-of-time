"""Selective Irreversibility Bottleneck model container."""

from __future__ import annotations

import torch
from torch import nn

from src.models.classifiers import MLPClassifier
from src.models.dynamics import GaussianDynamics
from src.models.encoders import GRUEncoder, pool_sequence


class SIBModel(nn.Module):
    """Encoder + task head + forward/reverse latent dynamics."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        latent_dim: int = 16,
        n_classes: int = 2,
        pooling: str = "last",
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,
        min_logvar: float = -6.0,
        max_logvar: float = 2.0,
        fixed_variance: bool = False,
        dynamics_hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.pooling = pooling
        self.encoder = GRUEncoder(
            input_dim,
            hidden_dim,
            latent_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        self.classifier = MLPClassifier(latent_dim, n_classes, hidden_dim)
        dynamics_hidden_dim = hidden_dim if dynamics_hidden_dim is None else int(dynamics_hidden_dim)
        self.forward_dynamics = GaussianDynamics(
            latent_dim, dynamics_hidden_dim, min_logvar, max_logvar, fixed_variance
        )
        self.reverse_dynamics = GaussianDynamics(
            latent_dim, dynamics_hidden_dim, min_logvar, max_logvar, fixed_variance
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def logits_from_z(self, z: torch.Tensor) -> torch.Tensor:
        return self.classifier(pool_sequence(z, self.pooling))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.encode(x)
        return {"z": z, "logits": self.logits_from_z(z)}
