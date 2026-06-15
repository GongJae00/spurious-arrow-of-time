"""Loss utilities for invariant transition mechanism learning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn.functional as F


@dataclass
class ITMLossOutput:
    total: torch.Tensor
    components: dict[str, torch.Tensor]
    weighted_components: dict[str, torch.Tensor]


def _weight(weights: Mapping[str, float], name: str, default: float) -> float:
    return float(weights.get(name, default))


def _mse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.shape != b.shape:
        raise ValueError(f"tensor shapes must match, got {tuple(a.shape)} and {tuple(b.shape)}")
    return (a - b).pow(2).mean()


def _standardize_batch(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    return (x - x.mean()) / (x.std(unbiased=False) + 1e-8)


def itm_anti_collapse_loss(
    z_core: torch.Tensor,
    z_spur: torch.Tensor,
    *,
    min_std: float = 0.05,
) -> torch.Tensor:
    penalties = []
    for z in (z_core, z_spur):
        flat = z.float().reshape(-1, z.shape[-1])
        std = flat.std(dim=0, unbiased=False).mean()
        penalties.append(F.relu(z.new_tensor(float(min_std)) - std).pow(2))
    return torch.stack(penalties).mean()


def itm_loss(
    *,
    task_loss: torch.Tensor,
    task_loss_cf: torch.Tensor,
    core_next_pred: torch.Tensor,
    core_next_target: torch.Tensor,
    spur_next_pred: torch.Tensor,
    spur_next_target: torch.Tensor,
    core_delta: torch.Tensor,
    core_delta_cf: torch.Tensor,
    spur_delta: torch.Tensor,
    spur_delta_cf: torch.Tensor,
    core_stat_pred: torch.Tensor | None,
    core_stat: torch.Tensor | None,
    spur_stat_pred: torch.Tensor | None,
    spur_stat: torch.Tensor | None,
    spur_adversary_loss: torch.Tensor | None,
    anti_collapse_loss: torch.Tensor,
    weights: Mapping[str, float],
    spur_sensitivity_margin: float = 0.1,
) -> ITMLossOutput:
    """Compute the ITM objective.

    The objective fits transition mechanisms, makes the core transition stable
    under nuisance counterfactuals, encourages the spurious mechanism to change
    under the same intervention, and uses controlled benchmark statistics only
    as auxiliary role pressure.
    """

    core_transition_fit = _mse(core_next_pred, core_next_target)
    spur_transition_fit = _mse(spur_next_pred, spur_next_target)
    core_cf_invariance = _mse(core_delta, core_delta_cf)
    spur_cf_delta = _mse(spur_delta, spur_delta_cf)
    spur_cf_sensitivity = F.relu(
        task_loss.new_tensor(float(spur_sensitivity_margin)) - spur_cf_delta
    )
    zero = task_loss.new_tensor(0.0)
    core_preservation = (
        zero
        if core_stat_pred is None or core_stat is None
        else F.mse_loss(core_stat_pred, _standardize_batch(core_stat))
    )
    spur_capture = (
        zero
        if spur_stat_pred is None or spur_stat is None
        else F.mse_loss(spur_stat_pred, _standardize_batch(spur_stat))
    )
    spur_adversary = zero if spur_adversary_loss is None else spur_adversary_loss

    components = {
        "task": task_loss,
        "cf_task": task_loss_cf,
        "core_transition_fit": core_transition_fit,
        "spur_transition_fit": spur_transition_fit,
        "core_cf_invariance": core_cf_invariance,
        "spur_cf_sensitivity": spur_cf_sensitivity,
        "spur_cf_delta": spur_cf_delta.detach(),
        "core_preservation": core_preservation,
        "spur_capture": spur_capture,
        "spur_adversary": spur_adversary,
        "anti_collapse": anti_collapse_loss,
    }
    weighted_components = {
        "task": components["task"],
        "cf_task": _weight(weights, "lambda_cf_task", 1.0) * components["cf_task"],
        "core_transition_fit": _weight(weights, "lambda_core_transition", 1.0)
        * components["core_transition_fit"],
        "spur_transition_fit": _weight(weights, "lambda_spur_transition", 0.5)
        * components["spur_transition_fit"],
        "core_cf_invariance": _weight(weights, "lambda_core_mech_cf", 1.0)
        * components["core_cf_invariance"],
        "spur_cf_sensitivity": _weight(weights, "lambda_spur_mech_sens", 0.5)
        * components["spur_cf_sensitivity"],
        "core_preservation": _weight(weights, "lambda_core_preserve", 1.0)
        * components["core_preservation"],
        "spur_capture": _weight(weights, "lambda_spur_capture", 0.2)
        * components["spur_capture"],
        "spur_adversary": _weight(weights, "lambda_spur_adv", 0.1)
        * components["spur_adversary"],
        "anti_collapse": _weight(weights, "lambda_anti_collapse", 0.05)
        * components["anti_collapse"],
    }
    total = torch.stack([value for value in weighted_components.values()]).sum()
    return ITMLossOutput(total=total, components=components, weighted_components=weighted_components)
