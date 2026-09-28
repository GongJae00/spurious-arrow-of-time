import torch

from src.models import SequenceCNNGRU, build_model

# Eq. 10 CNN+GRU logits and the model roster.


def test_sequence_cnn_gru_logits_shape():
    model = SequenceCNNGRU(grid_size=8, hidden_dim=16, input_channels=2)
    x = torch.zeros(4, 5, 2, 8, 8)
    output = model(x)
    assert output.logits.shape == (4, 2)
    assert output.representation.shape == (4, 16)


def test_build_model_roster():
    model = build_model("sequence_cnn_gru", grid_size=8, hidden_dim=8, input_channels=1)
    x = torch.zeros(2, 3, 8, 8)
    assert model(x).logits.shape == (2, 2)
