"""Selective Irreversibility Bottleneck loss."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch


@dataclass
class SIBLossOutput:
    total: torch.Tensor
    components: dict[str, torch.Tensor]


def _weight(weights: Mapping[str, float], name: str, default: float) -> float:
    return float(weights.get(name, default))


def sib_loss(
    task_loss: torch.Tensor,
    task_loss_cf: torch.Tensor,
    forward_nll: torch.Tensor,
    reverse_nll: torch.Tensor,
    sigma_total: torch.Tensor,
    sigma_total_cf: torch.Tensor,
    sigma_steps: torch.Tensor,
    sigma_steps_cf: torch.Tensor,
    sigma_target: float | torch.Tensor,
    weights: Mapping[str, float],
    loss_normalization: Mapping[str, float | bool] | None = None,
) -> SIBLossOutput:
    """Compute SIB objective with total and per-step CF arrow invariance."""

    if sigma_steps.shape != sigma_steps_cf.shape:
        raise ValueError("sigma_steps and sigma_steps_cf must have identical shapes")
    if sigma_total.shape != sigma_total_cf.shape:
        raise ValueError("sigma_total and sigma_total_cf must have identical shapes")
    n_transitions = sigma_steps.shape[1]
    sigma_per_step = sigma_total / float(n_transitions)
    target = torch.as_tensor(sigma_target, dtype=sigma_per_step.dtype, device=sigma_per_step.device)
    loss_normalization = loss_normalization or {}

    total_delta = sigma_total - sigma_total_cf
    if bool(loss_normalization.get("normalize_cf_total_by_transitions", False)):
        total_delta = total_delta / float(n_transitions)
    cf_arrow_total = total_delta.pow(2).mean()

    step_delta = sigma_steps - sigma_steps_cf
    if bool(loss_normalization.get("normalize_cf_step_by_step_variance", False)):
        step_scale = sigma_steps.detach().std().clamp_min(1e-6)
        step_delta = step_delta / step_scale
    cf_arrow_step = step_delta.pow(2).mean()

    setpoint_delta = sigma_per_step.mean() - target
    if bool(loss_normalization.get("normalize_setpoint_by_reference_scale", False)):
        reference_scale = torch.as_tensor(
            loss_normalization.get("setpoint_reference_scale", 1.0),
            dtype=setpoint_delta.dtype,
            device=setpoint_delta.device,
        ).abs().clamp_min(1e-6)
        setpoint_delta = setpoint_delta / reference_scale
    setpoint = setpoint_delta.pow(2)

    components = {
        "task": task_loss,
        "task_cf": task_loss_cf,
        "forward_nll": forward_nll,
        "reverse_nll": reverse_nll,
        "cf_arrow_total": cf_arrow_total,
        "cf_arrow_step": cf_arrow_step,
        "setpoint": setpoint,
    }
    total = (
        task_loss
        + _weight(weights, "lambda_cf_task", 1.0) * task_loss_cf
        + _weight(weights, "lambda_f", 1.0) * forward_nll
        + _weight(weights, "lambda_r", 1.0) * reverse_nll
        + _weight(weights, "eta_total", 1.0) * cf_arrow_total
        + _weight(weights, "eta_step", 0.1) * cf_arrow_step
        + _weight(weights, "rho", 1.0) * setpoint
    )
    return SIBLossOutput(total=total, components=components)
