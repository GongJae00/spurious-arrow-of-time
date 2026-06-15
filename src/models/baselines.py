"""Baseline model containers."""

from __future__ import annotations

import torch
from torch import nn

from src.models.classifiers import MLPClassifier
from src.models.encoders import GRUEncoder, pool_sequence


class ERMGRU(nn.Module):
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

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.encoder(x)
        return {"z": z, "logits": self.classifier(pool_sequence(z, self.pooling))}


class IBGRU(nn.Module):
    """IB-GRU with default stochastic bottleneck on pooled representation."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        latent_dim: int = 16,
        n_classes: int = 2,
        pooling: str = "last",
        min_logvar: float = -6.0,
        max_logvar: float = 2.0,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.pooling = pooling
        self.min_logvar = min_logvar
        self.max_logvar = max_logvar
        self.encoder = GRUEncoder(
            input_dim,
            hidden_dim,
            latent_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        self.posterior = nn.Linear(latent_dim, 2 * latent_dim)
        self.classifier = MLPClassifier(latent_dim, n_classes, hidden_dim)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h_seq = self.encoder(x)
        h = pool_sequence(h_seq, self.pooling)
        mu, raw_logvar = self.posterior(h).chunk(2, dim=-1)
        logvar = raw_logvar.clamp(self.min_logvar, self.max_logvar)
        std = torch.exp(0.5 * logvar)
        z = mu + torch.randn_like(std) * std if self.training else mu
        logits = self.classifier(z)
        return {"z_sequence": h_seq, "z": z, "mu": mu, "logvar": logvar, "logits": logits}


class ArrowClassifier(nn.Module):
    """Binary sequence classifier for order/arrow pretraining objectives."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        latent_dim: int = 16,
        pooling: str = "last",
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,
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
        self.classifier = MLPClassifier(latent_dim, 2, hidden_dim)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.encoder(x)
        return {"z": z, "logits": self.classifier(pool_sequence(z, self.pooling))}
