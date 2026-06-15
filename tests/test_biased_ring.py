import math

import numpy as np
import pytest

from src.data.biased_ring import (
    RingEPSolution,
    net_clockwise_displacement,
    ring_entropy_production,
    ring_transition_metadata,
    sample_biased_ring,
    solve_biased_ring_for_ep,
)


def test_ring_entropy_production_zero_and_positive_for_both_directions():
    assert ring_entropy_production(0.3, 0.3) == 0.0
    cw = ring_entropy_production(0.4, 0.2)
    ccw = ring_entropy_production(0.2, 0.4)
    assert cw > 0
    assert ccw > 0
    assert math.isclose(cw, ccw)


def test_ring_transition_metadata_separates_signed_drift_and_ep_magnitude():
    meta = ring_transition_metadata(0.2, 0.4)
    assert meta["signed_drift"] < 0
    assert meta["drift_direction"] == -1
    assert meta["analytic_ep"] > 0
    assert meta["ep_magnitude"] == meta["analytic_ep"]


def test_sample_biased_ring_shape_values_and_determinism():
    a = sample_biased_ring(32, 12, 8, 0.35, 0.25, seed=7)
    b = sample_biased_ring(32, 12, 8, 0.35, 0.25, seed=7)
    assert a.shape == (32, 12)
    assert np.array_equal(a, b)
    assert a.min() >= 0
    assert a.max() < 8


def test_sample_biased_ring_rejects_zero_backward_probability():
    with pytest.raises(ValueError):
        sample_biased_ring(4, 8, 5, 0.5, 0.0, seed=0)


def test_net_clockwise_displacement_is_path_based_not_endpoint_modulo():
    states = np.array([[7, 0, 1, 0, 7]])
    # Endpoint modulo would be zero, but the path has +1,+1,-1,-1.
    assert net_clockwise_displacement(states, n_states=8)[0] == 0
    states2 = np.array([[7, 0, 1, 2]])
    assert net_clockwise_displacement(states2, n_states=8)[0] == 3


def test_solve_biased_ring_for_ep_returns_structured_solution():
    target = ring_entropy_production(0.4, 0.2)
    solution = solve_biased_ring_for_ep(target, move_rate=0.6)
    assert isinstance(solution, RingEPSolution)
    assert solution.p_forward > solution.p_backward
    assert math.isclose(solution.p_forward + solution.p_backward, 0.6)
    assert solution.abs_error < 1e-8
    assert math.isclose(solution.actual_ep, target, rel_tol=1e-7, abs_tol=1e-8)
