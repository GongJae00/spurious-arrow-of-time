"""Latent transition arrow evidence scores."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass
class ArrowScoreOutput:
    sigma_steps: torch.Tensor
    sigma_total: torch.Tensor
    sigma_per_step: torch.Tensor
    forward_log_prob: torch.Tensor
    reverse_log_prob: torch.Tensor


@dataclass
class ArrowCalibrationOutput:
    raw: ArrowScoreOutput
    calibrated: ArrowScoreOutput
    metadata: dict[str, Any]


def latent_arrow_score(
    z: torch.Tensor,
    forward_model: nn.Module,
    reverse_model: nn.Module,
) -> ArrowScoreOutput:
    """Compute transition-level latent arrow evidence.

    Returns total and per-step scores. This is not exact physical entropy
    production unless additional boundary terms are explicitly modeled.
    """

    if z.ndim != 3:
        raise ValueError("z must have shape [B, L, latent_dim]")
    if z.shape[1] < 2:
        raise ValueError("z must contain at least two time points")
    z_t = z[:, :-1]
    z_next = z[:, 1:]
    forward_log_prob = forward_model.log_prob(z_next, z_t)
    reverse_log_prob = reverse_model.log_prob(z_t, z_next)
    sigma_steps = forward_log_prob - reverse_log_prob
    sigma_total = sigma_steps.sum(dim=1)
    sigma_per_step = sigma_total / float(z.shape[1] - 1)
    return ArrowScoreOutput(
        sigma_steps=sigma_steps,
        sigma_total=sigma_total,
        sigma_per_step=sigma_per_step,
        forward_log_prob=forward_log_prob,
        reverse_log_prob=reverse_log_prob,
    )


def _shift_score(score: ArrowScoreOutput, offset_per_step: torch.Tensor) -> ArrowScoreOutput:
    offset = offset_per_step.to(dtype=score.sigma_steps.dtype, device=score.sigma_steps.device)
    sigma_steps = score.sigma_steps - offset
    sigma_total = sigma_steps.sum(dim=1)
    sigma_per_step = sigma_total / float(score.sigma_steps.shape[1])
    return ArrowScoreOutput(
        sigma_steps=sigma_steps,
        sigma_total=sigma_total,
        sigma_per_step=sigma_per_step,
        forward_log_prob=score.forward_log_prob,
        reverse_log_prob=score.reverse_log_prob,
    )


def calibrate_arrow_score(
    score: ArrowScoreOutput,
    *,
    mode: str = "none",
    z: torch.Tensor | None = None,
    forward_model: nn.Module | None = None,
    reverse_model: nn.Module | None = None,
    reference_offset: float | torch.Tensor = 0.0,
) -> ArrowCalibrationOutput:
    """Return raw and calibrated latent arrow scores.

    Calibration changes only the score used by losses/diagnostics; it does not
    change sequence labels, model outputs, or latent states. All modes are
    label-free and must be configured without OOD-test performance.
    """

    mode = str(mode)
    if mode == "none":
        offset = torch.zeros((), dtype=score.sigma_per_step.dtype, device=score.sigma_per_step.device)
        source = "none"
    elif mode in {"reversible_centering", "validation_reference_centering"}:
        offset = torch.as_tensor(
            reference_offset,
            dtype=score.sigma_per_step.dtype,
            device=score.sigma_per_step.device,
        )
        source = "configured_reference_offset"
    elif mode == "batch_reverse_centering":
        if z is None or forward_model is None or reverse_model is None:
            raise ValueError(
                "batch_reverse_centering requires z, forward_model, and reverse_model"
            )
        reversed_score = latent_arrow_score(
            torch.flip(z, dims=[1]),
            forward_model,
            reverse_model,
        )
        offset = 0.5 * (
            score.sigma_per_step.mean() + reversed_score.sigma_per_step.mean()
        ).detach()
        source = "batch_forward_reverse_pair"
    else:
        raise ValueError(f"unknown arrow calibration mode {mode!r}")

    calibrated = _shift_score(score, offset)
    return ArrowCalibrationOutput(
        raw=score,
        calibrated=calibrated,
        metadata={
            "arrow_calibration_mode": mode,
            "arrow_calibration_offset": float(offset.detach().cpu()),
            "arrow_calibration_source": source,
            "raw_sigma_per_step_mean": float(score.sigma_per_step.mean().detach().cpu()),
            "calibrated_sigma_per_step_mean": float(
                calibrated.sigma_per_step.mean().detach().cpu()
            ),
        },
    )


def dynamics_nlls(
    z: torch.Tensor,
    forward_model: nn.Module,
    reverse_model: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor]:
    if z.ndim != 3 or z.shape[1] < 2:
        raise ValueError("z must have shape [B, L, latent_dim] with L >= 2")
    z_t = z[:, :-1]
    z_next = z[:, 1:]
    return forward_model.nll(z_next, z_t), reverse_model.nll(z_t, z_next)
