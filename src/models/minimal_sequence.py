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

    def __init__(
        self,
        grid_size: int,
        hidden_dim: int = 64,
        dropout: float = 0.0,
        input_channels: int = 1,
    ) -> None:
        super().__init__()
        input_dim = input_channels * grid_size * grid_size
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
        input_channels: int = 1,
    ) -> None:
        super().__init__()
        frame_dim = input_channels * grid_size * grid_size
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
        if x.ndim == 4:
            batch, length, rows, cols = x.shape
            frames = x.reshape(batch, length, rows * cols)
        elif x.ndim == 5:
            batch, length, channels, rows, cols = x.shape
            frames = x.reshape(batch, length, channels * rows * cols)
        else:
            raise ValueError(f"expected 4D or 5D sequence input, got shape {tuple(x.shape)}")
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
        input_channels: int = 1,
    ) -> None:
        super().__init__()
        gru_dropout = dropout if num_layers > 1 else 0.0
        self.frame_encoder = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )
        frame_dim = 32 * 4 * 4
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
        if x.ndim == 4:
            batch, length, rows, cols = x.shape
            frames = x.reshape(batch * length, 1, rows, cols)
        elif x.ndim == 5:
            batch, length, channels, rows, cols = x.shape
            frames = x.reshape(batch * length, channels, rows, cols)
        else:
            raise ValueError(f"expected 4D or 5D sequence input, got shape {tuple(x.shape)}")
        encoded = self.frame_projection(self.frame_encoder(frames))
        encoded = encoded.reshape(batch, length, -1)
        _, hidden = self.gru(encoded)
        representation = self.dropout(hidden[-1])
        return ModelOutput(logits=self.classifier(representation), representation=representation)


class _CNNFrameEncoder(nn.Module):
    """Shared per-frame CNN encoder matching SequenceCNNGRU.

    Encodes an image tensor of shape ``[N, C, H, W]`` into ``[N, hidden_dim]``.
    All expanded temporal backbones reuse this encoder so that architecture
    comparisons vary only in how time is modeled, not in frame features.
    """

    def __init__(self, hidden_dim: int, input_channels: int) -> None:
        super().__init__()
        self.frame_encoder = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )
        self.frame_projection = nn.Sequential(
            nn.Linear(32 * 4 * 4, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-frame features of shape ``[batch, length, hidden_dim]``."""
        if x.ndim == 4:
            batch, length, rows, cols = x.shape
            frames = x.reshape(batch * length, 1, rows, cols)
        elif x.ndim == 5:
            batch, length, channels, rows, cols = x.shape
            frames = x.reshape(batch * length, channels, rows, cols)
        else:
            raise ValueError(f"expected 4D or 5D sequence input, got shape {tuple(x.shape)}")
        encoded = self.frame_projection(self.frame_encoder(frames))
        return encoded.reshape(batch, length, -1)


class SequenceCNNLSTM(nn.Module):
    """CNN frame encoder followed by an LSTM temporal backbone."""

    def __init__(
        self,
        grid_size: int,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
        input_channels: int = 1,
    ) -> None:
        super().__init__()
        self.encoder = _CNNFrameEncoder(hidden_dim, input_channels)
        rnn_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=rnn_dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, 2)

    def forward(self, x: torch.Tensor) -> ModelOutput:
        encoded = self.encoder(x)
        _, (hidden, _) = self.lstm(encoded)
        representation = self.dropout(hidden[-1])
        return ModelOutput(logits=self.classifier(representation), representation=representation)


class _TemporalBlock(nn.Module):
    """Dilated causal 1D conv block with residual connection."""

    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.pad = padding
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def _causal(self, y: torch.Tensor) -> torch.Tensor:
        # Trim the right padding so the output stays causal and length-preserving.
        return y[:, :, : -self.pad] if self.pad > 0 else y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.dropout(self.relu(self._causal(self.conv1(x))))
        y = self.dropout(self.relu(self._causal(self.conv2(y))))
        return self.relu(x + y)


class SequenceCNNTCN(nn.Module):
    """CNN frame encoder followed by a dilated temporal convolutional network."""

    def __init__(
        self,
        grid_size: int,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
        input_channels: int = 1,
    ) -> None:
        super().__init__()
        self.encoder = _CNNFrameEncoder(hidden_dim, input_channels)
        depth = max(2, num_layers + 1)
        self.blocks = nn.ModuleList(
            [
                _TemporalBlock(hidden_dim, kernel_size=3, dilation=2**i, dropout=dropout)
                for i in range(depth)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, 2)

    def forward(self, x: torch.Tensor) -> ModelOutput:
        encoded = self.encoder(x)  # [B, L, H]
        y = encoded.transpose(1, 2)  # [B, H, L]
        for block in self.blocks:
            y = block(y)
        representation = self.dropout(y[:, :, -1])  # last timestep summary
        return ModelOutput(logits=self.classifier(representation), representation=representation)


class SequenceCNNTransformer(nn.Module):
    """CNN frame encoder followed by a Transformer encoder over time."""

    def __init__(
        self,
        grid_size: int,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
        input_channels: int = 1,
        max_len: int = 64,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.encoder = _CNNFrameEncoder(hidden_dim, input_channels)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_len, hidden_dim))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        heads = num_heads if hidden_dim % num_heads == 0 else 1
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=2 * hidden_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=max(2, num_layers + 1))
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, 2)

    def forward(self, x: torch.Tensor) -> ModelOutput:
        encoded = self.encoder(x)  # [B, L, H]
        length = encoded.shape[1]
        encoded = encoded + self.pos_embedding[:, :length]
        hidden = self.transformer(encoded)
        representation = self.dropout(hidden.mean(dim=1))  # mean-pool over time
        return ModelOutput(logits=self.classifier(representation), representation=representation)


class SequenceCNNTemporalPool(nn.Module):
    """CNN frame encoder with order-agnostic temporal (mean+max) pooling."""

    def __init__(
        self,
        grid_size: int,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
        input_channels: int = 1,
    ) -> None:
        super().__init__()
        self.encoder = _CNNFrameEncoder(hidden_dim, input_channels)
        self.head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_dim, 2)

    def forward(self, x: torch.Tensor) -> ModelOutput:
        encoded = self.encoder(x)  # [B, L, H]
        pooled = torch.cat([encoded.mean(dim=1), encoded.max(dim=1).values], dim=1)
        representation = self.head(pooled)
        return ModelOutput(logits=self.classifier(representation), representation=representation)


def build_model(
    model_type: str,
    grid_size: int,
    hidden_dim: int = 64,
    num_layers: int = 1,
    dropout: float = 0.0,
    input_channels: int = 1,
) -> nn.Module:
    if model_type == "final_frame_mlp":
        return FinalFrameMLP(
            grid_size=grid_size,
            hidden_dim=hidden_dim,
            dropout=dropout,
            input_channels=input_channels,
        )
    if model_type == "sequence_gru":
        return SequenceGRU(
            grid_size=grid_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            input_channels=input_channels,
        )
    if model_type == "sequence_cnn_gru":
        return SequenceCNNGRU(
            grid_size=grid_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            input_channels=input_channels,
        )
    if model_type == "sequence_cnn_lstm":
        return SequenceCNNLSTM(
            grid_size=grid_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            input_channels=input_channels,
        )
    if model_type == "sequence_cnn_tcn":
        return SequenceCNNTCN(
            grid_size=grid_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            input_channels=input_channels,
        )
    if model_type == "sequence_cnn_transformer":
        return SequenceCNNTransformer(
            grid_size=grid_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            input_channels=input_channels,
        )
    if model_type == "sequence_cnn_temporal_pool":
        return SequenceCNNTemporalPool(
            grid_size=grid_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            input_channels=input_channels,
        )
    raise ValueError(f"unknown model_type {model_type!r}")


def parameter_count(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)
