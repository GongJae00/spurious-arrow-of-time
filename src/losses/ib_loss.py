"""Information bottleneck losses."""

from __future__ import annotations

import torch


def gaussian_kl_to_standard_normal(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Mean KL(q=N(mu,diag(var)) || N(0,I)) over batch."""

    kl = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar)
    return kl.sum(dim=-1).mean()
