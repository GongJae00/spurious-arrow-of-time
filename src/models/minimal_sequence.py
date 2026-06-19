"""Minimal raw-tensor neural models for irreversible source experiments."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ModelOutput:
    logits: torch.Tensor
    representation: torch.Tensor


class FinalFrameMLP(nn.Module):
    """Classify from the raw final frame only."""

    def __init__(self, grid_size: int, hidden_dim: int = 64, dropout: float = 0.0) -> None:
        super().__init__()
        input_dim = grid_size * grid_size
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(hidden_dim, 2)

    def forward(self, x: torch.Tensor) -> ModelOutput:
        final_frame = x[:, -1]
        representation = self.net(final_frame)
        return ModelOutput(logits=self.classifier(representation), representation=representation)


class SequenceGRU(nn.Module):
    """Classify from a raw image sequence after flattening each frame."""

    def __init__(
        self,
        grid_size: int,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        frame_dim = grid_size * grid_size
        gru_dropout = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=frame_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=gru_dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, 2)

    def forward(self, x: torch.Tensor) -> ModelOutput:
        batch, length, rows, cols = x.shape
        frames = x.reshape(batch, length, rows * cols)
        _, hidden = self.gru(frames)
        representation = self.dropout(hidden[-1])
        return ModelOutput(logits=self.classifier(representation), representation=representation)


class SequenceCNNGRU(nn.Module):
    """Encode each frame with a small CNN, then model time with a GRU."""

    def __init__(
        self,
        grid_size: int,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        gru_dropout = dropout if num_layers > 1 else 0.0
        self.frame_encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )
        frame_dim = 16 * 4 * 4
        self.frame_projection = nn.Sequential(
            nn.Linear(frame_dim, hidden_dim),
            nn.ReLU(),
        )
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=gru_dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, 2)

    def forward(self, x: torch.Tensor) -> ModelOutput:
        batch, length, rows, cols = x.shape
        frames = x.reshape(batch * length, 1, rows, cols)
        encoded = self.frame_projection(self.frame_encoder(frames))
        encoded = encoded.reshape(batch, length, -1)
        _, hidden = self.gru(encoded)
        representation = self.dropout(hidden[-1])
        return ModelOutput(logits=self.classifier(representation), representation=representation)


def build_model(
    model_type: str,
    grid_size: int,
    hidden_dim: int = 64,
    num_layers: int = 1,
    dropout: float = 0.0,
) -> nn.Module:
    if model_type == "final_frame_mlp":
        return FinalFrameMLP(grid_size=grid_size, hidden_dim=hidden_dim, dropout=dropout)
    if model_type == "sequence_gru":
        return SequenceGRU(
            grid_size=grid_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
    if model_type == "sequence_cnn_gru":
        return SequenceCNNGRU(
            grid_size=grid_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
    raise ValueError(f"unknown model_type {model_type!r}")


def parameter_count(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)
