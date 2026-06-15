"""Data generation utilities for the active spurious-arrow benchmarks."""

from src.data.biased_ring import (
    RingEPSolution,
    ring_entropy_production,
    ring_transition_metadata,
    sample_biased_ring,
    solve_biased_ring_for_ep,
)
from src.data.sta_bench import generate_sta_bench, generate_sta_splits
from src.data.ink_advection_diffusion import (
    generate_ink_advection_diffusion_bench,
    generate_ink_advection_diffusion_splits,
)

__all__ = [
    "RingEPSolution",
    "generate_ink_advection_diffusion_bench",
    "generate_ink_advection_diffusion_splits",
    "generate_sta_bench",
    "generate_sta_splits",
    "ring_entropy_production",
    "ring_transition_metadata",
    "sample_biased_ring",
    "solve_biased_ring_for_ep",
]
