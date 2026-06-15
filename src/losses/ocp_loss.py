"""Order/arrow classification losses."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def binary_arrow_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Binary CE for forward/reversed labels."""

    if logits.ndim == 2 and logits.shape[-1] == 2:
        return F.cross_entropy(logits, labels.long())
    return F.binary_cross_entropy_with_logits(logits.squeeze(-1), labels.float())
