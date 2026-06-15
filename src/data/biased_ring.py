"""Biased ring Markov chains and analytic transition-level EP utilities.

The analytic quantity here is the steady-state transition-level entropy
production of a uniform stationary biased ring chain. It is not a learned
latent score and not a physical heat measurement in this project.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RingEPSolution:
    """Solution for a biased ring with a requested analytic EP magnitude."""

    p_forward: float
    p_backward: float
    target_ep: float
    actual_ep: float
    abs_error: float
    move_rate: float
    bias: float


def _validate_ring_args(
    n_states: int | None = None,
    p_forward: float | None = None,
    p_backward: float | None = None,
) -> None:
    if n_states is not None and n_states < 3:
        raise ValueError("n_states must be at least 3 for a ring chain")
    if p_forward is None or p_backward is None:
        return
    if not np.isfinite(p_forward) or not np.isfinite(p_backward):
        raise ValueError("transition probabilities must be finite")
    if p_forward <= 0 or p_backward <= 0:
        raise ValueError("p_forward and p_backward must be strictly positive")
    if p_forward + p_backward > 1.0 + 1e-12:
        raise ValueError("p_forward + p_backward must be <= 1")


def ring_entropy_production(p_forward: float, p_backward: float) -> float:
    """Analytic per-transition EP magnitude for a uniform stationary ring.

    Sigma = (p - q) * log(p / q). This is non-negative for p != q because the
    factors have the same sign. The signed direction is stored separately as
    p - q in metadata.
    """

    _validate_ring_args(p_forward=p_forward, p_backward=p_backward)
    if np.isclose(p_forward, p_backward, rtol=0.0, atol=1e-15):
        return 0.0
    return float((p_forward - p_backward) * log(p_forward / p_backward))


def ring_transition_metadata(p_forward: float, p_backward: float) -> dict[str, Any]:
    """Return signed drift and analytic EP metadata for a ring transition."""

    _validate_ring_args(p_forward=p_forward, p_backward=p_backward)
    signed_drift = float(p_forward - p_backward)
    analytic_ep = ring_entropy_production(p_forward, p_backward)
    if signed_drift > 0:
        drift_direction = 1
    elif signed_drift < 0:
        drift_direction = -1
    else:
        drift_direction = 0
    return {
        "p_forward": float(p_forward),
        "p_backward": float(p_backward),
        "signed_drift": signed_drift,
        "analytic_ep": float(analytic_ep),
        "ep_magnitude": float(analytic_ep),
        "drift_direction": drift_direction,
        "move_rate": float(p_forward + p_backward),
    }


def solve_biased_ring_for_ep(
    target_ep: float,
    move_rate: float = 0.6,
    min_prob: float = 1e-4,
) -> RingEPSolution:
    """Find p, q with p + q = move_rate and analytic EP close to target_ep."""

    if not np.isfinite(target_ep) or target_ep < 0:
        raise ValueError("target_ep must be finite and non-negative")
    if not np.isfinite(move_rate) or move_rate <= 0 or move_rate > 1:
        raise ValueError("move_rate must be in (0, 1]")
    if min_prob <= 0 or 2 * min_prob >= move_rate:
        raise ValueError("min_prob must be positive and less than move_rate / 2")

    max_bias = 1.0 - (2.0 * min_prob / move_rate)

    def ep_for_bias(bias: float) -> float:
        p = move_rate * (1.0 + bias) / 2.0
        q = move_rate * (1.0 - bias) / 2.0
        return ring_entropy_production(p, q)

    max_ep = ep_for_bias(max_bias)
    if target_ep > max_ep + 1e-10:
        raise ValueError(
            f"target_ep={target_ep:.6g} exceeds max achievable EP={max_ep:.6g} "
            f"for move_rate={move_rate} and min_prob={min_prob}"
        )

    if target_ep == 0:
        bias = 0.0
    else:
        lo, hi = 0.0, max_bias
        for _ in range(80):
            mid = (lo + hi) / 2.0
            if ep_for_bias(mid) < target_ep:
                lo = mid
            else:
                hi = mid
        bias = (lo + hi) / 2.0

    p_forward = move_rate * (1.0 + bias) / 2.0
    p_backward = move_rate * (1.0 - bias) / 2.0
    actual_ep = ring_entropy_production(p_forward, p_backward)
    return RingEPSolution(
        p_forward=float(p_forward),
        p_backward=float(p_backward),
        target_ep=float(target_ep),
        actual_ep=float(actual_ep),
        abs_error=float(abs(actual_ep - target_ep)),
        move_rate=float(move_rate),
        bias=float(bias),
    )


def sample_biased_ring(
    n_sequences: int,
    length: int,
    n_states: int,
    p_forward: float,
    p_backward: float,
    seed: int,
    initial_state_mode: str = "uniform_stationary",
    initial_states: np.ndarray | None = None,
) -> np.ndarray:
    """Generate biased ring Markov chains.

    `length` is the number of observed time points L. The number of transitions
    is L - 1. Main experiments should use uniform_stationary initialization.
    """

    if n_sequences <= 0:
        raise ValueError("n_sequences must be positive")
    if length <= 0:
        raise ValueError("length must be positive")
    _validate_ring_args(n_states=n_states, p_forward=p_forward, p_backward=p_backward)

    rng = np.random.default_rng(seed)
    states = np.empty((n_sequences, length), dtype=np.int64)

    if initial_state_mode == "uniform_stationary":
        states[:, 0] = rng.integers(0, n_states, size=n_sequences)
    elif initial_state_mode == "provided":
        if initial_states is None:
            raise ValueError("initial_states is required for initial_state_mode='provided'")
        initial_arr = np.asarray(initial_states, dtype=np.int64)
        if initial_arr.shape != (n_sequences,):
            raise ValueError("initial_states must have shape [n_sequences]")
        if np.any(initial_arr < 0) or np.any(initial_arr >= n_states):
            raise ValueError("initial_states contain values outside [0, n_states)")
        states[:, 0] = initial_arr
    elif initial_state_mode == "sector_conditioned":
        # Diagnostic-only static shortcut control. Use lower half by default.
        upper = max(1, n_states // 2)
        states[:, 0] = rng.integers(0, upper, size=n_sequences)
    else:
        raise ValueError(
            "initial_state_mode must be one of "
            "'uniform_stationary', 'provided', 'sector_conditioned'"
        )

    stay_threshold = p_forward + p_backward
    for t in range(1, length):
        draws = rng.random(n_sequences)
        step = np.zeros(n_sequences, dtype=np.int64)
        step[draws < p_forward] = 1
        step[(draws >= p_forward) & (draws < stay_threshold)] = -1
        states[:, t] = (states[:, t - 1] + step) % n_states

    return states


def path_step_deltas(states: np.ndarray, n_states: int) -> np.ndarray:
    """Return signed step deltas for ring paths using path-wise transitions."""

    arr = np.asarray(states)
    if arr.ndim != 2:
        raise ValueError("states must have shape [N, L]")
    forward = (arr[:, :-1] + 1) % n_states
    backward = (arr[:, :-1] - 1) % n_states
    nxt = arr[:, 1:]
    deltas = np.zeros((arr.shape[0], arr.shape[1] - 1), dtype=np.int64)
    deltas[nxt == forward] = 1
    deltas[nxt == backward] = -1
    return deltas


def net_clockwise_displacement(states: np.ndarray, n_states: int) -> np.ndarray:
    """Path-wise net clockwise displacement; never uses (c_T - c_0) mod K."""

    return path_step_deltas(states, n_states).sum(axis=1)
