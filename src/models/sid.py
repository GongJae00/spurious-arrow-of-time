"""Selective Irreversibility Decomposition model."""

from __future__ import annotations

import torch
from torch import nn
from torch.autograd import Function

from src.models.classifiers import MLPClassifier
from src.models.dynamics import GaussianDynamics
from src.models.encoders import pool_sequence

class GradientReversal(Function):
    """Gradient Reversal Layer for adversarial training.

    Forward: identity.
    Backward: negate gradient * alpha.
    Used so that the spur_adversary_head is trained to predict spurious from task_rep
    (minimize its loss), while the task_rep (encoder) is trained to fool it (maximize the head's loss).
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output.neg() * ctx.alpha, None


class SIDModel(nn.Module):
    """Factorized sequence encoder for reversible/task/spurious irreversibility.

    The deployable task head is intentionally restricted to `z_rev` and
    `z_ir_task`. The nuisance factor `z_ir_spur` is exposed only for diagnostics,
    adversaries, and oracle-assisted analyses.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        z_rev_dim: int = 16,
        z_ir_task_dim: int = 16,
        z_ir_spur_dim: int = 16,
        z_resid_dim: int = 0,
        n_classes: int = 2,
        pooling: str = "last",
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,
        min_logvar: float = -6.0,
        max_logvar: float = 2.0,
        fixed_variance: bool = True,
        dynamics_hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        if bidirectional:
            raise ValueError("bidirectional encoders are disabled for causal SID")
        if min(z_rev_dim, z_ir_task_dim, z_ir_spur_dim) <= 0:
            raise ValueError("SID factor dimensions must be positive")
        if z_resid_dim < 0:
            raise ValueError("z_resid_dim must be non-negative")
        self.pooling = pooling
        self.z_rev_dim = int(z_rev_dim)
        self.z_ir_task_dim = int(z_ir_task_dim)
        self.z_ir_spur_dim = int(z_ir_spur_dim)
        self.z_resid_dim = int(z_resid_dim)
        self.task_head_input_dim = self.z_rev_dim + self.z_ir_task_dim

        self.encoder = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.z_rev_proj = nn.Linear(hidden_dim, self.z_rev_dim)
        self.z_ir_task_proj = nn.Linear(hidden_dim, self.z_ir_task_dim)
        self.z_ir_spur_proj = nn.Linear(hidden_dim, self.z_ir_spur_dim)
        self.z_resid_proj = (
            nn.Linear(hidden_dim, self.z_resid_dim) if self.z_resid_dim > 0 else None
        )

        self.classifier = MLPClassifier(self.task_head_input_dim, n_classes, hidden_dim)
        self.spurious_probe = MLPClassifier(self.z_ir_spur_dim, n_classes, hidden_dim)
        self.core_probe = MLPClassifier(self.z_ir_task_dim, n_classes, hidden_dim)

        # Aux heads for full SID decomposition losses (core preservation, spur capture, spur adversary on task rep).
        # These enforce the role semantics during training on controlled benchmarks.
        # They are auxiliary; the deployable task head never uses z_ir_spur or these heads directly.
        self.core_preservation_head = nn.Sequential(
            nn.Linear(self.z_ir_task_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.spur_capture_head = nn.Sequential(
            nn.Linear(self.z_ir_spur_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.spur_adversary_head = nn.Sequential(
            nn.Linear(self.task_head_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        dynamics_hidden_dim = hidden_dim if dynamics_hidden_dim is None else int(dynamics_hidden_dim)
        self.task_forward_dynamics = GaussianDynamics(
            self.z_ir_task_dim,
            dynamics_hidden_dim,
            min_logvar,
            max_logvar,
            fixed_variance,
        )
        self.task_reverse_dynamics = GaussianDynamics(
            self.z_ir_task_dim,
            dynamics_hidden_dim,
            min_logvar,
            max_logvar,
            fixed_variance,
        )
        self.spur_forward_dynamics = GaussianDynamics(
            self.z_ir_spur_dim,
            dynamics_hidden_dim,
            min_logvar,
            max_logvar,
            fixed_variance,
        )
        self.spur_reverse_dynamics = GaussianDynamics(
            self.z_ir_spur_dim,
            dynamics_hidden_dim,
            min_logvar,
            max_logvar,
            fixed_variance,
        )

    def encode(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h, _ = self.encoder(x)
        factors = {
            "z_rev": self.z_rev_proj(h),
            "z_ir_task": self.z_ir_task_proj(h),
            "z_ir_spur": self.z_ir_spur_proj(h),
        }
        if self.z_resid_proj is not None:
            factors["z_resid"] = self.z_resid_proj(h)
        return factors

    def task_representation(self, factors: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat(
            [
                pool_sequence(factors["z_rev"], self.pooling),
                pool_sequence(factors["z_ir_task"], self.pooling),
            ],
            dim=-1,
        )

    def spurious_representation(self, factors: dict[str, torch.Tensor]) -> torch.Tensor:
        return pool_sequence(factors["z_ir_spur"], self.pooling)

    def task_logits_from_factors(self, factors: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.classifier(self.task_representation(factors))

    def diagnostic_logits_from_factors(
        self,
        factors: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        return {
            "core_probe_logits": self.core_probe(pool_sequence(factors["z_ir_task"], self.pooling)),
            "spurious_probe_logits": self.spurious_probe(
                self.spurious_representation(factors)
            ),
        }

    # --- Aux decomposition heads for full SID (used only in training losses for role enforcement) ---
    def core_preservation_pred(self, factors: dict[str, torch.Tensor]) -> torch.Tensor:
        """Regression prediction of core stat from z_ir_task (for L_core_preservation)."""
        return self.core_preservation_head(pool_sequence(factors["z_ir_task"], self.pooling)).squeeze(-1)

    def spur_capture_pred(self, factors: dict[str, torch.Tensor]) -> torch.Tensor:
        """Regression prediction of spurious stat from z_ir_spur (for L_spur_capture)."""
        return self.spur_capture_head(pool_sequence(factors["z_ir_spur"], self.pooling)).squeeze(-1)

    def spur_adversary_pred(self, factors: dict[str, torch.Tensor]) -> torch.Tensor:
        """Regression prediction of spurious stat from task rep (rev+ir_task) for L_spur_adversary."""
        return self.spur_adversary_head(self.task_representation(factors)).squeeze(-1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        factors = self.encode(x)
        return {
            **factors,
            "factors": factors,
            "logits": self.task_logits_from_factors(factors),
            **self.diagnostic_logits_from_factors(factors),
        }
