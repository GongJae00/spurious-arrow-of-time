from dataclasses import dataclass

import torch
from torch import nn

# CNN+GRU is the main sequence model. Eq. 10 is ŷ = f_θ; Eq. 11 is ERM; Eq. 12 is CF.
# Table A4–A5: LSTM / TCN / Transformer / pool. Table 9: SegGRU.
# The GRU frame encoder is not CNNFrameEncoder (logged init order).


@dataclass(frozen=True)
class ModelOutput:
    logits: torch.Tensor
    representation: torch.Tensor


def _frames(x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    if x.ndim == 4:
        batch, length, rows, cols = x.shape
        return x.reshape(batch * length, 1, rows, cols), batch, length
    batch, length, channels, rows, cols = x.shape
    return x.reshape(batch * length, channels, rows, cols), batch, length


class SequenceCNNGRU(nn.Module):
    def __init__(self, grid_size: int, hidden_dim: int = 64, num_layers: int = 1, dropout: float = 0.0, input_channels: int = 1):
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
        self.frame_projection = nn.Sequential(nn.Linear(32 * 4 * 4, hidden_dim), nn.ReLU())
        self.gru = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, dropout=gru_dropout)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, 2)

    def forward(self, x: torch.Tensor) -> ModelOutput:
        frames, batch, length = _frames(x)
        encoded = self.frame_projection(self.frame_encoder(frames)).reshape(batch, length, -1)
        _, hidden = self.gru(encoded)
        representation = self.dropout(hidden[-1])
        return ModelOutput(logits=self.classifier(representation), representation=representation)


class FinalFrameMLP(nn.Module):
    def __init__(self, grid_size: int, hidden_dim: int = 64, dropout: float = 0.0, input_channels: int = 1, input_dim: int | None = None, num_layers: int = 1):
        super().__init__()
        dim = input_channels * grid_size * grid_size if input_dim is None else input_dim
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(hidden_dim, 2)

    def forward(self, x: torch.Tensor) -> ModelOutput:
        representation = self.net(x[:, -1])
        return ModelOutput(logits=self.classifier(representation), representation=representation)


class CNNFrameEncoder(nn.Module):
    def __init__(self, hidden_dim: int, input_channels: int):
        super().__init__()
        self.frame_encoder = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )
        self.frame_projection = nn.Sequential(nn.Linear(32 * 4 * 4, hidden_dim), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        frames, batch, length = _frames(x)
        return self.frame_projection(self.frame_encoder(frames)).reshape(batch, length, -1)


class SequenceCNNLSTM(nn.Module):
    def __init__(self, grid_size: int, hidden_dim: int = 64, num_layers: int = 1, dropout: float = 0.0, input_channels: int = 1):
        super().__init__()
        self.encoder = CNNFrameEncoder(hidden_dim, input_channels)
        rnn_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, dropout=rnn_dropout)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, 2)

    def forward(self, x: torch.Tensor) -> ModelOutput:
        _, (hidden, _) = self.lstm(self.encoder(x))
        representation = self.dropout(hidden[-1])
        return ModelOutput(logits=self.classifier(representation), representation=representation)


class TemporalBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=self.pad, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=self.pad, dilation=dilation)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def _causal(self, y: torch.Tensor) -> torch.Tensor:
        return y[:, :, : -self.pad] if self.pad > 0 else y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.dropout(self.relu(self._causal(self.conv1(x))))
        y = self.dropout(self.relu(self._causal(self.conv2(y))))
        return self.relu(x + y)


class SequenceCNNTCN(nn.Module):
    def __init__(self, grid_size: int, hidden_dim: int = 64, num_layers: int = 1, dropout: float = 0.0, input_channels: int = 1):
        super().__init__()
        self.encoder = CNNFrameEncoder(hidden_dim, input_channels)
        depth = max(2, num_layers + 1)
        self.blocks = nn.ModuleList([TemporalBlock(hidden_dim, kernel_size=3, dilation=2 ** i, dropout=dropout) for i in range(depth)])
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, 2)

    def forward(self, x: torch.Tensor) -> ModelOutput:
        y = self.encoder(x).transpose(1, 2)
        for block in self.blocks:
            y = block(y)
        representation = self.dropout(y[:, :, -1])
        return ModelOutput(logits=self.classifier(representation), representation=representation)


class SequenceCNNTransformer(nn.Module):
    def __init__(self, grid_size: int, hidden_dim: int = 64, num_layers: int = 1, dropout: float = 0.0, input_channels: int = 1, max_len: int = 64, num_heads: int = 4):
        super().__init__()
        self.encoder = CNNFrameEncoder(hidden_dim, input_channels)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_len, hidden_dim))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=2 * hidden_dim, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=max(2, num_layers + 1))
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, 2)

    def forward(self, x: torch.Tensor) -> ModelOutput:
        encoded = self.encoder(x)
        encoded = encoded + self.pos_embedding[:, : encoded.shape[1]]
        representation = self.dropout(self.transformer(encoded).mean(dim=1))
        return ModelOutput(logits=self.classifier(representation), representation=representation)


class SequenceCNNTemporalPool(nn.Module):
    def __init__(self, grid_size: int, hidden_dim: int = 64, num_layers: int = 1, dropout: float = 0.0, input_channels: int = 1):
        super().__init__()
        self.encoder = CNNFrameEncoder(hidden_dim, input_channels)
        self.head = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.classifier = nn.Linear(hidden_dim, 2)

    def forward(self, x: torch.Tensor) -> ModelOutput:
        encoded = self.encoder(x)
        representation = self.head(torch.cat([encoded.mean(dim=1), encoded.max(dim=1).values], dim=1))
        return ModelOutput(logits=self.classifier(representation), representation=representation)


class SegGRU(nn.Module):
    def __init__(self, channels: int, width: int = 50, hidden_dim: int = 64):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(channels * width, hidden_dim), nn.ReLU())
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, 2)

    def forward(self, x: torch.Tensor) -> ModelOutput:
        b = x.shape[0]
        h = self.enc(x.reshape(b, x.shape[1], -1))
        representation = self.gru(h)[0][:, -1]
        return ModelOutput(logits=self.head(representation), representation=representation)


MODELS = {
    "sequence_cnn_gru": SequenceCNNGRU,
    "final_frame_mlp": FinalFrameMLP,
    "sequence_cnn_lstm": SequenceCNNLSTM,
    "sequence_cnn_tcn": SequenceCNNTCN,
    "sequence_cnn_transformer": SequenceCNNTransformer,
    "sequence_cnn_temporal_pool": SequenceCNNTemporalPool,
}


def build_model(model_type: str, grid_size: int, hidden_dim: int = 64, num_layers: int = 1, dropout: float = 0.0, input_channels: int = 1) -> nn.Module:
    return MODELS[model_type](grid_size=grid_size, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout, input_channels=input_channels)
