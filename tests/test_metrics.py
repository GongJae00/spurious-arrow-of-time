import numpy as np
import pytest
import torch

from src.eval.metrics import ood_gap, reverse_sequence
from src.eval.ood_eval import summarize_split_accuracies


def test_ood_gap_uses_iid_test_not_val_iid():
    assert ood_gap(iid_test_accuracy=0.8, ood_test_accuracy=0.5) == pytest.approx(0.3)
    summary = summarize_split_accuracies(1.0, 0.95, 0.8, 0.5)
    assert summary["ood_gap"] == pytest.approx(0.3)
    assert summary["val_iid_accuracy"] == 0.95


def test_reverse_sequence_reverses_only_time_axis_numpy_and_torch():
    x_np = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    rev_np = reverse_sequence(x_np, time_dim=1)
    assert np.array_equal(rev_np[:, 0], x_np[:, -1])
    assert np.array_equal(rev_np[:, -1], x_np[:, 0])

    x_t = torch.arange(2 * 3 * 4).reshape(2, 3, 4)
    rev_t = reverse_sequence(x_t, time_dim=1)
    assert torch.equal(rev_t[:, 0], x_t[:, -1])
    assert torch.equal(rev_t[:, -1], x_t[:, 0])
