"""Sequence encoders."""

from __future__ import annotations

import torch
from torch import nn


class GRUEncoder(nn.Module):
    """Causal GRU encoder returning a latent sequence [B, L, latent_dim]."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        latent_dim: int = 16,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        if bidirectional:
            raise ValueError("bidirectional encoders are disabled by default for causal models")
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.proj = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.gru(x)
        return self.proj(h)


def pool_sequence(z: torch.Tensor, mode: str = "last") -> torch.Tensor:
    if z.ndim != 3:
        raise ValueError("z must have shape [B, L, D]")
    if mode == "last":
        return z[:, -1]
    if mode == "mean":
        return z.mean(dim=1)
    raise ValueError("pooling mode must be 'last' or 'mean'")
