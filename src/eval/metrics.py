"""Evaluation metrics shared by all methods."""

from __future__ import annotations

import numpy as np
import torch


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    return float((preds == labels).float().mean().detach().cpu())


def ood_gap(iid_test_accuracy: float, ood_test_accuracy: float) -> float:
    """Main OOD gap: iid_test_accuracy - ood_test_accuracy."""

    return float(iid_test_accuracy - ood_test_accuracy)


def counterfactual_prediction_metrics(
    probs: torch.Tensor,
    probs_cf: torch.Tensor,
) -> dict[str, float]:
    pred = probs.argmax(dim=-1)
    pred_cf = probs_cf.argmax(dim=-1)
    consistency = (pred == pred_cf).float().mean()
    drift = (probs - probs_cf).abs().sum(dim=-1).mean()
    return {
        "prediction_consistency": float(consistency.detach().cpu()),
        "probability_l1_drift": float(drift.detach().cpu()),
    }


def arrow_summary(
    sigma_total: torch.Tensor,
    sigma_per_step: torch.Tensor,
    prefix: str = "sigma",
) -> dict[str, float]:
    total = sigma_total.detach().cpu()
    per_step = sigma_per_step.detach().cpu()
    return {
        f"{prefix}_total_mean": float(total.mean()),
        f"{prefix}_total_std": float(total.std()),
        f"{prefix}_total_min": float(total.min()),
        f"{prefix}_total_max": float(total.max()),
        f"{prefix}_per_step_mean": float(per_step.mean()),
        f"{prefix}_per_step_std": float(per_step.std()),
        f"{prefix}_per_step_min": float(per_step.min()),
        f"{prefix}_per_step_max": float(per_step.max()),
    }


def reverse_sequence(x: np.ndarray | torch.Tensor, time_dim: int = 1):
    """Reverse only the temporal axis."""

    if isinstance(x, torch.Tensor):
        return torch.flip(x, dims=(time_dim,))
    return np.flip(x, axis=time_dim).copy()
