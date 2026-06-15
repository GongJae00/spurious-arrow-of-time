"""Probe target construction utilities."""

from __future__ import annotations

import numpy as np

from src.data.biased_ring import net_clockwise_displacement
from src.eval.metrics import reverse_sequence


def make_forward_reverse_probe_sequences(states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    forward = np.asarray(states)
    reverse = reverse_sequence(forward, time_dim=1)
    labels = np.concatenate(
        [np.ones(forward.shape[0], dtype=np.int64), np.zeros(forward.shape[0], dtype=np.int64)]
    )
    sequences = np.concatenate([forward, reverse], axis=0)
    return sequences, labels


def trajectory_displacement_targets(states: np.ndarray, n_states: int) -> np.ndarray:
    return net_clockwise_displacement(states, n_states)
