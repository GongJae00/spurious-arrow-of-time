"""Selective Irreversibility Decomposition loss utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn.functional as F


SID_FACTOR_KEYS = ("z_rev", "z_ir_task", "z_ir_spur")


@dataclass
class SIDLossOutput:
    total: torch.Tensor
    components: dict[str, torch.Tensor]
    weighted_components: dict[str, torch.Tensor]


def _weight(weights: Mapping[str, float], name: str, default: float) -> float:
    return float(weights.get(name, default))


def _zero_like(reference: torch.Tensor) -> torch.Tensor:
    return reference.new_tensor(0.0)


def _mse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.shape != b.shape:
        raise ValueError(f"factor shapes must match, got {tuple(a.shape)} and {tuple(b.shape)}")
    return (a - b).pow(2).mean()


def sid_anti_collapse_loss(
    factors: Mapping[str, torch.Tensor],
    *,
    min_std: float = 0.05,
) -> torch.Tensor:
    """Penalize collapsed factor representations with too little batch/time variance."""

    penalties = []
    for key in SID_FACTOR_KEYS:
        z = factors[key]
        if z.ndim != 3:
            raise ValueError(f"{key} must have shape [B, L, D]")
        std = z.float().reshape(-1, z.shape[-1]).std(dim=0, unbiased=False).mean()
        penalties.append(F.relu(z.new_tensor(float(min_std)) - std).pow(2))
    return torch.stack(penalties).mean()


def sid_factor_diagnostics(
    factors: Mapping[str, torch.Tensor],
    factors_cf: Mapping[str, torch.Tensor] | None = None,
) -> dict[str, float]:
    """Return scalar factor norms/deltas for logging and collapse checks."""

    out: dict[str, float] = {}
    for key in SID_FACTOR_KEYS:
        z = factors[key].detach().float()
        flat = z.reshape(-1, z.shape[-1])
        out[f"{key}_norm_mean"] = float(z.norm(dim=-1).mean().cpu())
        out[f"{key}_std_mean"] = float(flat.std(dim=0, unbiased=False).mean().cpu())
        if factors_cf is not None:
            out[f"{key}_cf_mse"] = float(_mse(factors[key], factors_cf[key]).detach().cpu())
    return out


def sid_loss(
    *,
    task_loss: torch.Tensor,
    task_loss_cf: torch.Tensor,
    factors: Mapping[str, torch.Tensor],
    factors_cf: Mapping[str, torch.Tensor],
    weights: Mapping[str, float],
    spur_sensitivity_margin: float = 0.1,
    spurious_adversary_loss: torch.Tensor | None = None,
    core_preservation_loss: torch.Tensor | None = None,
    spur_capture_loss: torch.Tensor | None = None,
    arrow_decomposition_loss: torch.Tensor | None = None,
    anti_collapse_loss: torch.Tensor | None = None,
) -> SIDLossOutput:
    """Compute the SID objective from factorized representations.

    The task loss is assumed to come from a head that only sees `z_rev` and
    `z_ir_task`. This function keeps counterfactual invariance/sensitivity and
    loss accounting explicit.
    """

    for key in SID_FACTOR_KEYS:
        if key not in factors or key not in factors_cf:
            raise ValueError(f"missing SID factor {key!r}")

    rev_cf = _mse(factors["z_rev"], factors_cf["z_rev"])
    task_ir_cf = _mse(factors["z_ir_task"], factors_cf["z_ir_task"])
    spur_delta = _mse(factors["z_ir_spur"], factors_cf["z_ir_spur"])
    spur_sensitivity = F.relu(
        task_loss.new_tensor(float(spur_sensitivity_margin)) - spur_delta
    )
    if anti_collapse_loss is None:
        anti_collapse_loss = sid_anti_collapse_loss(factors)
    components = {
        "task": task_loss,
        "cf_task": task_loss_cf,
        "rev_cf_invariance": rev_cf,
        "task_ir_cf_invariance": task_ir_cf,
        "spur_ir_cf_sensitivity": spur_sensitivity,
        "spur_ir_cf_delta": spur_delta.detach(),
        "spur_adversary": (
            _zero_like(task_loss) if spurious_adversary_loss is None else spurious_adversary_loss
        ),
        "core_preservation": (
            _zero_like(task_loss) if core_preservation_loss is None else core_preservation_loss
        ),
        "spur_capture": _zero_like(task_loss) if spur_capture_loss is None else spur_capture_loss,
        "arrow_decomposition": (
            _zero_like(task_loss) if arrow_decomposition_loss is None else arrow_decomposition_loss
        ),
        "anti_collapse": anti_collapse_loss,
    }
    weighted_components = {
        "task": components["task"],
        "cf_task": _weight(weights, "lambda_cf_task", 1.0) * components["cf_task"],
        "rev_cf_invariance": _weight(weights, "lambda_rev_cf", 0.5)
        * components["rev_cf_invariance"],
        "task_ir_cf_invariance": _weight(weights, "lambda_task_ir_cf", 0.5)
        * components["task_ir_cf_invariance"],
        "spur_ir_cf_sensitivity": _weight(weights, "lambda_spur_sens", 0.5)
        * components["spur_ir_cf_sensitivity"],
        "spur_adversary": _weight(weights, "lambda_spur_adv", 0.1)
        * components["spur_adversary"],
        "core_preservation": _weight(weights, "lambda_core_preserve", 1.0)
        * components["core_preservation"],
        "spur_capture": _weight(weights, "lambda_spur_capture", 0.2)
        * components["spur_capture"],
        "arrow_decomposition": _weight(weights, "lambda_arrow_decomp", 0.5)
        * components["arrow_decomposition"],
        "anti_collapse": _weight(weights, "lambda_anti_collapse", 0.1)
        * components["anti_collapse"],
    }
    total = torch.stack([value for value in weighted_components.values()]).sum()
    return SIDLossOutput(
        total=total,
        components=components,
        weighted_components=weighted_components,
    )
