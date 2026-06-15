"""Counterfactual nuisance resampling utilities for STA-Bench."""

from __future__ import annotations

import numpy as np


COUNTERFACTUAL_MODES = {
    "randomized",
    "reversed",
    "independent_same_marginal",
    "resample_same_mode",
}


def validate_counterfactual_mode(mode: str) -> None:
    if mode not in COUNTERFACTUAL_MODES:
        raise ValueError(f"unknown counterfactual mode {mode!r}")


def counterfactual_mode_to_spurious_mode(original_mode: str, cf_mode: str) -> str:
    """Map counterfactual sampling config to a spurious split mode."""

    validate_counterfactual_mode(cf_mode)
    if cf_mode == "randomized":
        return "randomized"
    if cf_mode == "reversed":
        return "reversed"
    if cf_mode == "independent_same_marginal":
        return "independent_same_marginal"
    if cf_mode == "resample_same_mode":
        return original_mode
    raise AssertionError(f"unhandled counterfactual mode {cf_mode}")


def reuse_or_resample_noise(
    noise: np.ndarray,
    reuse_noise: bool,
    seed: int,
    noise_std: float,
) -> np.ndarray:
    """Return original noise or a same-shape Gaussian resample."""

    if reuse_noise:
        return noise.copy()
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, noise_std, size=noise.shape).astype(np.float32)
