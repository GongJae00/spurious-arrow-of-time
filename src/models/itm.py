"""Invariant Transition Mechanism model."""

from __future__ import annotations

import torch
from torch import nn

from src.models.classifiers import MLPClassifier
from src.models.encoders import pool_sequence


class ResidualTransition(nn.Module):
    """Predict a residual transition in latent space."""

    def __init__(self, latent_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, z_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        delta = self.net(z_t)
        return z_t + delta, delta


class ITMModel(nn.Module):
    """Decompose transition mechanisms rather than only representation slots.

    `z_core` is the candidate invariant, task-relevant transition mechanism.
    `z_spur` is the candidate nuisance transition mechanism. The task head only
    sees the core mechanism representation. Nuisance information is handled by
    auxiliary heads and counterfactual intervention losses.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        core_dim: int = 16,
        spur_dim: int = 16,
        n_classes: int = 2,
        pooling: str = "last",
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,
        transition_hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        if bidirectional:
            raise ValueError("bidirectional encoders are disabled for causal ITM")
        if min(core_dim, spur_dim) <= 0:
            raise ValueError("ITM mechanism dimensions must be positive")
        self.pooling = pooling
        self.core_dim = int(core_dim)
        self.spur_dim = int(spur_dim)
        transition_hidden_dim = hidden_dim if transition_hidden_dim is None else int(
            transition_hidden_dim
        )

        self.encoder = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.core_proj = nn.Linear(hidden_dim, self.core_dim)
        self.spur_proj = nn.Linear(hidden_dim, self.spur_dim)
        self.core_transition = ResidualTransition(self.core_dim, transition_hidden_dim)
        self.spur_transition = ResidualTransition(self.spur_dim, transition_hidden_dim)

        self.task_head_input_dim = 2 * self.core_dim
        self.classifier = MLPClassifier(self.task_head_input_dim, n_classes, hidden_dim)
        self.core_preservation_head = nn.Sequential(
            nn.Linear(self.task_head_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.spur_capture_head = nn.Sequential(
            nn.Linear(2 * self.spur_dim, hidden_dim),
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

    def encode(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h, _ = self.encoder(x)
        return {
            "z_core": self.core_proj(h),
            "z_spur": self.spur_proj(h),
        }

    def mechanisms(self, factors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        z_core = factors["z_core"]
        z_spur = factors["z_spur"]
        core_next, core_delta = self.core_transition(z_core[:, :-1])
        spur_next, spur_delta = self.spur_transition(z_spur[:, :-1])
        return {
            "core_next_pred": core_next,
            "core_delta": core_delta,
            "spur_next_pred": spur_next,
            "spur_delta": spur_delta,
        }

    def task_representation_from_parts(
        self,
        factors: dict[str, torch.Tensor],
        mechanisms: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        core_state = pool_sequence(factors["z_core"], self.pooling)
        core_delta = mechanisms["core_delta"].mean(dim=1)
        return torch.cat([core_state, core_delta], dim=-1)

    def spurious_representation_from_parts(
        self,
        factors: dict[str, torch.Tensor],
        mechanisms: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        spur_state = pool_sequence(factors["z_spur"], self.pooling)
        spur_delta = mechanisms["spur_delta"].mean(dim=1)
        return torch.cat([spur_state, spur_delta], dim=-1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        factors = self.encode(x)
        mechanisms = self.mechanisms(factors)
        task_rep = self.task_representation_from_parts(factors, mechanisms)
        spur_rep = self.spurious_representation_from_parts(factors, mechanisms)
        return {
            **factors,
            **mechanisms,
            "factors": factors,
            "mechanisms": mechanisms,
            "task_rep": task_rep,
            "spur_rep": spur_rep,
            "logits": self.classifier(task_rep),
            "core_stat_pred": self.core_preservation_head(task_rep).squeeze(-1),
            "spurious_stat_pred": self.spur_capture_head(spur_rep).squeeze(-1),
        }
