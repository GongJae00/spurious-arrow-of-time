"""Ink Advection-Diffusion data generation.

This benchmark is intentionally not a full fluid simulation. It is a controlled
2D passive-scalar experiment that preserves the ink-drop intuition more
faithfully than the Gaussian surrogate: concentration is evolved by conservative
advection plus diffusion, with explicit physical sanity metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

from src.data.splits import SPLIT_ORDER, validate_split


SPURIOUS_MODES = {"correlated", "reversed", "randomized"}
SPURIOUS_CF_MODES = {"randomized", "reversed", "independent_same_marginal", "resample_same_mode"}
LABEL_MODES = {"core_source_x_median_threshold", "core_source_quadrant"}


@dataclass(frozen=True)
class InkAdvectionDiffusionSeeds:
    dataset_seed: int
    core_seed: int
    spurious_seed: int
    spurious_cf_seed: int
    noise_seed: int

    @classmethod
    def from_base(cls, seed: int, split_index: int = 0) -> "InkAdvectionDiffusionSeeds":
        base = int(seed) + 30_000 * split_index
        return cls(
            dataset_seed=base,
            core_seed=base + 101,
            spurious_seed=base + 202,
            spurious_cf_seed=base + 303,
            noise_seed=base + 404,
        )

    def as_dict(self) -> dict[str, int]:
        return {
            "dataset_seed": self.dataset_seed,
            "core_seed": self.core_seed,
            "spurious_seed": self.spurious_seed,
            "spurious_cf_seed": self.spurious_cf_seed,
            "noise_seed": self.noise_seed,
        }


def _validate_probability(name: str, value: float) -> float:
    if not np.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return float(value)


def _validate_common(
    n_sequences: int,
    length: int,
    grid_size: int,
    split: str,
    spurious_mode: str,
    label_mode: str,
) -> None:
    if n_sequences <= 0:
        raise ValueError("n_sequences must be positive")
    if length < 2:
        raise ValueError("length L must be at least 2")
    if grid_size < 8:
        raise ValueError("grid_size must be at least 8")
    validate_split(split)
    if spurious_mode not in SPURIOUS_MODES:
        raise ValueError(f"spurious_mode must be one of {sorted(SPURIOUS_MODES)}")
    if label_mode not in LABEL_MODES:
        raise ValueError(f"label_mode must be one of {sorted(LABEL_MODES)}")


def _safe_corr(y: np.ndarray, values: np.ndarray) -> float:
    y_f = y.astype(np.float64)
    v_f = values.astype(np.float64)
    if np.std(y_f) < 1e-12 or np.std(v_f) < 1e-12:
        return float("nan")
    return float(np.corrcoef(y_f, v_f)[0, 1])


def _safe_auc(y: np.ndarray, values: np.ndarray) -> float:
    if np.unique(y).shape[0] < 2 or np.unique(values).shape[0] < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y, values))
    except ValueError:
        return float("nan")


def _mean_by_y(y: np.ndarray, values: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    for cls in (0, 1):
        mask = y == cls
        out[str(cls)] = float(values[mask].mean()) if np.any(mask) else float("nan")
    return out


def _class_balance(y: np.ndarray) -> dict[str, float]:
    return {
        "n": int(y.shape[0]),
        "n0": int((y == 0).sum()),
        "n1": int((y == 1).sum()),
        "p1": float((y == 1).mean()),
    }


def _labels_from_scores(
    scores: np.ndarray,
    threshold: float | None,
    *,
    allow_balance_fallback: bool,
) -> tuple[np.ndarray, float, bool]:
    if threshold is None:
        threshold = float(np.median(scores))
    y = (scores > threshold).astype(np.int64)
    fallback_used = False
    if y.min() == y.max() and allow_balance_fallback:
        order = np.argsort(scores, kind="mergesort")
        y = np.zeros_like(y)
        y[order[len(order) // 2 :]] = 1
        fallback_used = True
    return y, float(threshold), fallback_used


def _sample_sources(n_sequences: int, grid_size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    margin = max(3, grid_size // 6)
    return rng.integers(margin, grid_size - margin, size=(n_sequences, 2)).astype(np.float32)


def _core_scores(sources: np.ndarray, grid_size: int, label_mode: str) -> np.ndarray:
    centered_x = sources[:, 0] - (grid_size - 1) / 2.0
    if label_mode == "core_source_x_median_threshold":
        return centered_x.astype(np.float32)
    if label_mode == "core_source_quadrant":
        centered_y = sources[:, 1] - (grid_size - 1) / 2.0
        return (centered_x + centered_y).astype(np.float32)
    raise ValueError(f"unknown label_mode {label_mode!r}")


def _orientation_from_y(
    y: np.ndarray,
    mode: str,
    strength: float,
    seed: int,
) -> np.ndarray:
    strength = _validate_probability("spurious_label_correlation_strength", strength)
    rng = np.random.default_rng(seed)
    if mode == "randomized":
        return rng.choice(np.array([-1, 1], dtype=np.int64), size=y.shape[0])
    desired = np.where(y == 1, 1, -1)
    if mode == "reversed":
        desired = -desired
    keep_desired = rng.random(y.shape[0]) < ((1.0 + strength) / 2.0)
    return np.where(keep_desired, desired, -desired).astype(np.int64)


def _make_grid(grid_size: int) -> tuple[np.ndarray, np.ndarray]:
    coords = np.arange(grid_size, dtype=np.float32)
    xx, yy = np.meshgrid(coords, coords, indexing="xy")
    return xx.astype(np.float32), yy.astype(np.float32)


def _initial_blob(sources: np.ndarray, grid_size: int, sigma: float) -> np.ndarray:
    if sigma <= 0.0:
        raise ValueError("source_blur_sigma must be positive")
    xx, yy = _make_grid(grid_size)
    dx2 = (xx[None, :, :] - sources[:, 0, None, None]) ** 2
    dy2 = (yy[None, :, :] - sources[:, 1, None, None]) ** 2
    blob = np.exp(-(dx2 + dy2) / (2.0 * sigma**2)).astype(np.float32)
    return _renormalize_mass(blob)


def _renormalize_mass(field: np.ndarray) -> np.ndarray:
    mass = np.maximum(field.sum(axis=(1, 2), keepdims=True), 1e-12)
    return (field / mass).astype(np.float32)


def _diffuse_no_flux(field: np.ndarray, diffusion: float, dt: float, dx: float) -> np.ndarray:
    if diffusion <= 0.0:
        return field
    padded = np.pad(field, ((0, 0), (1, 1), (1, 1)), mode="edge")
    center = padded[:, 1:-1, 1:-1]
    lap = (
        padded[:, 1:-1, 2:]
        + padded[:, 1:-1, :-2]
        + padded[:, 2:, 1:-1]
        + padded[:, :-2, 1:-1]
        - 4.0 * center
    ) / (dx * dx)
    return field + float(diffusion) * float(dt) * lap


def _advect_upwind_no_flux(
    field: np.ndarray,
    velocity_x: np.ndarray,
    velocity_y: np.ndarray,
    dt: float,
    dx: float,
) -> np.ndarray:
    out = field.copy()
    batch, height, width = field.shape
    vx = velocity_x.astype(np.float32)
    vy = velocity_y.astype(np.float32)

    flux_x = np.zeros((batch, height, width + 1), dtype=np.float32)
    pos = vx >= 0.0
    if np.any(pos):
        flux_x[pos, :, 1:width] = vx[pos, None, None] * out[pos, :, : width - 1]
    if np.any(~pos):
        flux_x[~pos, :, 1:width] = vx[~pos, None, None] * out[~pos, :, 1:width]
    out = out - (dt / dx) * (flux_x[:, :, 1:] - flux_x[:, :, :-1])

    flux_y = np.zeros((batch, height + 1, width), dtype=np.float32)
    pos_y = vy >= 0.0
    if np.any(pos_y):
        flux_y[pos_y, 1:height, :] = vy[pos_y, None, None] * out[pos_y, : height - 1, :]
    if np.any(~pos_y):
        flux_y[~pos_y, 1:height, :] = vy[~pos_y, None, None] * out[~pos_y, 1:height, :]
    out = out - (dt / dx) * (flux_y[:, 1:, :] - flux_y[:, :-1, :])
    return out


def _stability_substeps(
    diffusion: float,
    velocity_x: np.ndarray,
    velocity_y: np.ndarray,
    dt: float,
    dx: float,
    safety: float = 0.9,
) -> tuple[int, dict[str, float]]:
    max_vx = float(np.max(np.abs(velocity_x))) if velocity_x.size else 0.0
    max_vy = float(np.max(np.abs(velocity_y))) if velocity_y.size else 0.0
    adv_x = max_vx * dt / dx
    adv_y = max_vy * dt / dx
    diff = float(diffusion) * dt / (dx * dx)
    raw_total = adv_x + adv_y + 4.0 * diff
    substeps = max(1, int(np.ceil(raw_total / max(safety, 1e-6))))
    sub_adv_x = adv_x / substeps
    sub_adv_y = adv_y / substeps
    sub_diff = diff / substeps
    return substeps, {
        "advective_cfl_x": float(sub_adv_x),
        "advective_cfl_y": float(sub_adv_y),
        "diffusion_cfl": float(sub_diff),
        "stability_margin": float(1.0 - (sub_adv_x + sub_adv_y + 4.0 * sub_diff)),
        "raw_cfl_total_before_substeps": float(raw_total),
    }


def _simulate_passive_scalar(
    sources: np.ndarray,
    length: int,
    grid_size: int,
    diffusion: float,
    velocity_x: np.ndarray,
    velocity_y: np.ndarray,
    source_blur_sigma: float,
    dt: float,
    dx: float,
) -> tuple[np.ndarray, dict[str, float]]:
    if diffusion < 0.0:
        raise ValueError("diffusion must be nonnegative")
    if dt <= 0.0 or dx <= 0.0:
        raise ValueError("dt and dx must be positive")
    substeps, stability = _stability_substeps(diffusion, velocity_x, velocity_y, dt, dx)
    sub_dt = dt / substeps
    field = _initial_blob(sources, grid_size, source_blur_sigma)
    fields = np.empty((sources.shape[0], length, grid_size, grid_size), dtype=np.float32)
    fields[:, 0] = field
    for t in range(1, length):
        for _ in range(substeps):
            field = _advect_upwind_no_flux(field, velocity_x, velocity_y, sub_dt, dx)
            field = _diffuse_no_flux(field, diffusion, sub_dt, dx)
            field = np.maximum(field, 0.0).astype(np.float32)
            field = _renormalize_mass(field)
        fields[:, t] = field
    stability["n_substeps"] = int(substeps)
    return fields, stability


def _simulate_observed_passive_scalar(
    sources: np.ndarray,
    length: int,
    grid_size: int,
    diffusion: float,
    velocity_x: np.ndarray,
    velocity_y: np.ndarray,
    source_blur_sigma: float,
    dt: float,
    dx: float,
    pre_observation_steps: int,
) -> tuple[np.ndarray, dict[str, float]]:
    if pre_observation_steps < 0:
        raise ValueError("pre_observation_steps must be nonnegative")
    total_length = int(length) + int(pre_observation_steps)
    fields, stability = _simulate_passive_scalar(
        sources,
        total_length,
        grid_size,
        diffusion,
        velocity_x,
        velocity_y,
        source_blur_sigma,
        dt,
        dx,
    )
    stability["pre_observation_steps"] = int(pre_observation_steps)
    return fields[:, pre_observation_steps:], stability


def _field_entropy(field: np.ndarray) -> np.ndarray:
    flat = field.reshape(field.shape[0], -1).astype(np.float64)
    probs = flat / np.maximum(flat.sum(axis=1, keepdims=True), 1e-12)
    return (-probs * np.log(np.maximum(probs, 1e-12))).sum(axis=1).astype(np.float32)


def _field_spread(field: np.ndarray) -> np.ndarray:
    xx, yy = _make_grid(field.shape[-1])
    probs = field.astype(np.float64)
    probs = probs / np.maximum(probs.sum(axis=(1, 2), keepdims=True), 1e-12)
    mean_x = (probs * xx[None, :, :]).sum(axis=(1, 2))
    mean_y = (probs * yy[None, :, :]).sum(axis=(1, 2))
    var_x = (probs * (xx[None, :, :] - mean_x[:, None, None]) ** 2).sum(axis=(1, 2))
    var_y = (probs * (yy[None, :, :] - mean_y[:, None, None]) ** 2).sum(axis=(1, 2))
    return np.sqrt(np.maximum(var_x + var_y, 0.0)).astype(np.float32)


def _center_of_mass_x(field: np.ndarray) -> np.ndarray:
    xx, _ = _make_grid(field.shape[-1])
    probs = field.astype(np.float64)
    probs = probs / np.maximum(probs.sum(axis=(1, 2), keepdims=True), 1e-12)
    return (probs * xx[None, :, :]).sum(axis=(1, 2)).astype(np.float32)


def _source_recovery_mass(field: np.ndarray, sources: np.ndarray, tolerance: float = 1.5) -> float:
    xx, yy = _make_grid(field.shape[-1])
    probs = field.astype(np.float64)
    probs = probs / np.maximum(probs.sum(axis=(1, 2), keepdims=True), 1e-12)
    dx2 = (xx[None, :, :] - sources[:, 0, None, None]) ** 2
    dy2 = (yy[None, :, :] - sources[:, 1, None, None]) ** 2
    near_source = (dx2 + dy2) <= tolerance**2
    return float((probs * near_source).sum(axis=(1, 2)).mean())


def _source_center_error(field: np.ndarray, sources: np.ndarray) -> float:
    xx, yy = _make_grid(field.shape[-1])
    probs = field.astype(np.float64)
    probs = probs / np.maximum(probs.sum(axis=(1, 2), keepdims=True), 1e-12)
    mean_x = (probs * xx[None, :, :]).sum(axis=(1, 2))
    mean_y = (probs * yy[None, :, :]).sum(axis=(1, 2))
    err = np.sqrt((mean_x - sources[:, 0]) ** 2 + (mean_y - sources[:, 1]) ** 2)
    return float(err.mean())


def _source_peak_error(field: np.ndarray, sources: np.ndarray) -> float:
    height, width = field.shape[-2:]
    flat_idx = field.reshape(field.shape[0], -1).argmax(axis=1)
    row, col = np.unravel_index(flat_idx, (height, width))
    err = np.sqrt(
        (col.astype(np.float32) - sources[:, 0]) ** 2
        + (row.astype(np.float32) - sources[:, 1]) ** 2
    )
    return float(err.mean())


def _source_peak_contrast(field: np.ndarray) -> float:
    flat = field.reshape(field.shape[0], -1).astype(np.float64)
    return float((flat.max(axis=1) / np.maximum(flat.mean(axis=1), 1e-12)).mean())


def _source_visibility_metrics(fields: np.ndarray, sources: np.ndarray) -> dict[str, float]:
    start = fields[:, 0]
    final = fields[:, -1]
    mass_start = _source_recovery_mass(start, sources)
    mass_final = _source_recovery_mass(final, sources)
    peak_start = _source_peak_contrast(start)
    peak_final = _source_peak_contrast(final)
    return {
        "mass_near_source_observed_start": mass_start,
        "mass_near_source_observed_final": mass_final,
        "mass_near_source_final_fraction": mass_final / max(mass_start, 1e-12),
        "center_error_observed_start": _source_center_error(start, sources),
        "center_error_observed_final": _source_center_error(final, sources),
        "peak_error_observed_start": _source_peak_error(start, sources),
        "peak_error_observed_final": _source_peak_error(final, sources),
        "peak_contrast_observed_start": peak_start,
        "peak_contrast_observed_final": peak_final,
        "peak_contrast_final_fraction": peak_final / max(peak_start, 1e-12),
    }


def _mass_stats(fields: np.ndarray) -> dict[str, float]:
    mass = fields.sum(axis=(2, 3)).astype(np.float64)
    initial = np.maximum(mass[:, 0], 1e-12)
    rel = np.abs(mass - initial[:, None]) / initial[:, None]
    return {
        "mass_initial_mean": float(mass[:, 0].mean()),
        "mass_final_mean": float(mass[:, -1].mean()),
        "mass_relative_error_mean": float(rel.mean()),
        "mass_relative_error_max": float(rel.max()),
        "min_concentration": float(fields.min()),
    }


def _physics_metadata(
    fields: np.ndarray,
    stability: dict[str, float],
    *,
    dx: float,
    dt: float,
    diffusion: float,
    boundary: str = "no_flux",
) -> dict[str, float | int | str]:
    return {
        "scheme": "conservative_upwind_advection_explicit_diffusion",
        "boundary": boundary,
        "dx": float(dx),
        "dt": float(dt),
        "diffusion": float(diffusion),
        **stability,
        **_mass_stats(fields),
    }


def _source_mean_by_y(y: np.ndarray, sources: np.ndarray, coord: int) -> dict[str, float]:
    return _mean_by_y(y, sources[:, coord])


def generate_ink_advection_diffusion_bench(
    n_sequences: int,
    length: int,
    grid_size: int,
    split: str,
    spurious_mode: str,
    seed: int,
    *,
    core_diffusion: float = 0.16,
    spurious_diffusion: float = 0.12,
    core_flow_x: float = 0.0,
    core_flow_y: float = 0.0,
    spurious_flow_scale: float = 0.8,
    source_blur_sigma: float = 1.0,
    pre_observation_steps: int = 0,
    dt: float = 0.35,
    dx: float = 1.0,
    observation_noise_std: float = 0.003,
    core_scale: float = 1.0,
    spur_scale: float = 0.9,
    label_mode: str = "core_source_x_median_threshold",
    label_threshold: float | None = None,
    spurious_label_correlation_strength: float = 1.0,
    spurious_cf_mode: str = "randomized",
    reuse_noise: bool = True,
    allow_local_eval_calibration: bool = False,
    return_fields: bool = False,
) -> dict[str, Any]:
    """Generate one Ink Advection-Diffusion split."""

    _validate_common(n_sequences, length, grid_size, split, spurious_mode, label_mode)
    if spurious_cf_mode not in SPURIOUS_CF_MODES:
        raise ValueError(f"spurious_cf_mode must be one of {sorted(SPURIOUS_CF_MODES)}")
    seeds = InkAdvectionDiffusionSeeds.from_base(seed, SPLIT_ORDER.index(split))
    if split != "train" and label_threshold is None and not allow_local_eval_calibration:
        raise ValueError(
            "label_threshold is required for non-train splits. Use "
            "generate_ink_advection_diffusion_splits so evaluation splits reuse the "
            "train threshold."
        )

    core_sources = _sample_sources(n_sequences, grid_size, seeds.core_seed)
    core_score = _core_scores(core_sources, grid_size, label_mode)
    threshold_source = "train" if label_threshold is not None and split != "train" else "local"
    y, threshold, fallback_used = _labels_from_scores(
        core_score,
        label_threshold,
        allow_balance_fallback=label_threshold is None,
    )

    orientation = _orientation_from_y(
        y,
        spurious_mode,
        spurious_label_correlation_strength,
        seeds.spurious_seed + 17,
    )
    spurious_flow_x = orientation.astype(np.float32) * float(spurious_flow_scale)
    spurious_flow_y = np.zeros_like(spurious_flow_x, dtype=np.float32)
    core_vx = np.full(n_sequences, float(core_flow_x), dtype=np.float32)
    core_vy = np.full(n_sequences, float(core_flow_y), dtype=np.float32)

    spurious_sources = _sample_sources(n_sequences, grid_size, seeds.spurious_seed)
    core_fields, core_stability = _simulate_observed_passive_scalar(
        core_sources,
        length,
        grid_size,
        core_diffusion,
        core_vx,
        core_vy,
        source_blur_sigma,
        dt,
        dx,
        pre_observation_steps,
    )
    spurious_fields, spur_stability = _simulate_observed_passive_scalar(
        spurious_sources,
        length,
        grid_size,
        spurious_diffusion,
        spurious_flow_x,
        spurious_flow_y,
        source_blur_sigma,
        dt,
        dx,
        pre_observation_steps,
    )

    if spurious_cf_mode == "resample_same_mode":
        cf_mode = spurious_mode
    elif spurious_cf_mode == "reversed":
        cf_mode = "reversed"
    elif spurious_cf_mode in {"randomized", "independent_same_marginal"}:
        cf_mode = "randomized"
    else:
        raise ValueError(f"unknown spurious_cf_mode {spurious_cf_mode!r}")
    cf_orientation = _orientation_from_y(
        y,
        cf_mode,
        spurious_label_correlation_strength,
        seeds.spurious_cf_seed + 17,
    )
    if spurious_cf_mode == "independent_same_marginal":
        rng = np.random.default_rng(seeds.spurious_cf_seed + 29)
        cf_orientation = orientation[rng.permutation(n_sequences)]
    spurious_cf_flow_x = cf_orientation.astype(np.float32) * float(spurious_flow_scale)
    spurious_cf_flow_y = np.zeros_like(spurious_cf_flow_x, dtype=np.float32)
    spurious_cf_sources = _sample_sources(n_sequences, grid_size, seeds.spurious_cf_seed)
    spurious_cf_fields, spur_cf_stability = _simulate_observed_passive_scalar(
        spurious_cf_sources,
        length,
        grid_size,
        spurious_diffusion,
        spurious_cf_flow_x,
        spurious_cf_flow_y,
        source_blur_sigma,
        dt,
        dx,
        pre_observation_steps,
    )

    rng_noise = np.random.default_rng(seeds.noise_seed)
    noise = rng_noise.normal(
        0.0,
        observation_noise_std,
        size=(n_sequences, length, grid_size, grid_size),
    ).astype(np.float32)
    noise_cf = (
        noise.copy()
        if reuse_noise
        else rng_noise.normal(
            0.0,
            observation_noise_std,
            size=(n_sequences, length, grid_size, grid_size),
        ).astype(np.float32)
    )

    signal = core_scale * core_fields + spur_scale * spurious_fields
    signal_cf = core_scale * core_fields + spur_scale * spurious_cf_fields
    obs = signal + noise
    obs_cf = signal_cf + noise_cf
    x = obs.reshape(n_sequences, length, grid_size * grid_size).astype(np.float32)
    x_cf = obs_cf.reshape(n_sequences, length, grid_size * grid_size).astype(np.float32)

    core_dynamic_stat = core_score.astype(np.float32)
    spurious_dynamic_stat = spurious_flow_x.astype(np.float32)
    spurious_cf_dynamic_stat = spurious_cf_flow_x.astype(np.float32)

    initial_core = core_fields[:, 0]
    final_core = core_fields[:, -1]
    entropy_initial = _field_entropy(initial_core)
    entropy_final = _field_entropy(final_core)
    spread_initial = _field_spread(initial_core)
    spread_final = _field_spread(final_core)
    spurious_entropy_initial = _field_entropy(spurious_fields[:, 0])
    spurious_entropy_final = _field_entropy(spurious_fields[:, -1])
    spurious_spread_initial = _field_spread(spurious_fields[:, 0])
    spurious_spread_final = _field_spread(spurious_fields[:, -1])
    center_start = _center_of_mass_x(spurious_fields[:, 0])
    center_final = _center_of_mass_x(spurious_fields[:, -1])
    signal_std = float(signal.std())
    noise_empirical_std = float(noise.std())
    signal_to_noise = signal_std / max(noise_empirical_std, 1e-12)
    source_visibility = _source_visibility_metrics(core_fields, core_sources)

    core_physics = _physics_metadata(
        core_fields,
        core_stability,
        dx=dx,
        dt=dt,
        diffusion=core_diffusion,
    )
    spur_physics = _physics_metadata(
        spurious_fields,
        spur_stability,
        dx=dx,
        dt=dt,
        diffusion=spurious_diffusion,
    )
    cf_physics = _physics_metadata(
        spurious_cf_fields,
        spur_cf_stability,
        dx=dx,
        dt=dt,
        diffusion=spurious_diffusion,
    )

    metadata = {
        "benchmark_name": "ink_advection_diffusion",
        "benchmark_version": "ink_advection_diffusion",
        "split": split,
        "length_L": int(length),
        "n_transitions": int(length - 1),
        "obs_shape": [int(length), int(grid_size), int(grid_size)],
        "grid_size": int(grid_size),
        "physics": {
            **{f"core_{k}": v for k, v in core_physics.items()},
            **{f"spurious_{k}": v for k, v in spur_physics.items()},
            **{f"spurious_cf_{k}": v for k, v in cf_physics.items()},
        },
        "core": {
            "process_type": "passive_scalar_diffusion",
            "label_mode": label_mode,
            "label_threshold": float(threshold),
            "label_threshold_source": threshold_source,
            "label_balance_fallback_used": bool(fallback_used),
            "class_balance": _class_balance(y),
            "dynamic_stat_name": "core_source_x",
            "corr_y_core_dynamic_stat": _safe_corr(y, core_dynamic_stat),
            "auc_y_from_core_dynamic_stat": _safe_auc(y, core_dynamic_stat),
            "mean_core_dynamic_stat_by_y": _mean_by_y(y, core_dynamic_stat),
            "source_x_mean_by_y": _source_mean_by_y(y, core_sources, 0),
            "source_y_mean_by_y": _source_mean_by_y(y, core_sources, 1),
            "entropy_initial_mean": float(entropy_initial.mean()),
            "entropy_final_mean": float(entropy_final.mean()),
            "spread_initial_mean": float(spread_initial.mean()),
            "spread_final_mean": float(spread_final.mean()),
            "source_recovery_final_mass_near_source": _source_recovery_mass(
                final_core,
                core_sources,
            ),
            **source_visibility,
        },
        "spurious": {
            "process_type": "passive_scalar_advection_diffusion",
            "spurious_mode": spurious_mode,
            "spurious_correlation_type": "flow_direction",
            "spurious_label_correlation_strength": float(spurious_label_correlation_strength),
            "dynamic_stat_name": "spurious_flow_x",
            "flow_x_mean_by_y": _mean_by_y(y, spurious_flow_x),
            "flow_y_mean_by_y": _mean_by_y(y, spurious_flow_y),
            "corr_y_spurious_dynamic_stat": _safe_corr(y, spurious_dynamic_stat),
            "auc_y_from_spurious_dynamic_stat": _safe_auc(y, spurious_dynamic_stat),
            "mean_spurious_dynamic_stat_by_y": _mean_by_y(y, spurious_dynamic_stat),
            "center_of_mass_x_delta_mean": float((center_final - center_start).mean()),
            "center_of_mass_x_delta_by_y": _mean_by_y(y, center_final - center_start),
        },
        "observation": {
            "obs_dim": int(grid_size * grid_size),
            "core_scale": float(core_scale),
            "spur_scale": float(spur_scale),
            "noise_std": float(observation_noise_std),
            "signal_std": signal_std,
            "noise_std_empirical": noise_empirical_std,
            "signal_to_noise_std_ratio": float(signal_to_noise),
            "normalize_mixing_columns": False,
            "mixing_matrix_hash": None,
        },
        "counterfactual": {
            "spurious_cf_mode": spurious_cf_mode,
            "resolved_spurious_cf_mode": cf_mode,
            "reuse_noise": bool(reuse_noise),
            "preserves_y": True,
            "preserves_core_field": True,
            "preserves_core_stat": True,
            "changes_spurious_flow": bool(
                np.mean(np.abs(spurious_dynamic_stat - spurious_cf_dynamic_stat)) > 1e-6
            ),
            "changes_spurious_stat": bool(
                np.mean(np.abs(spurious_dynamic_stat - spurious_cf_dynamic_stat)) > 1e-6
            ),
            "corr_y_spurious_cf_dynamic_stat": _safe_corr(y, spurious_cf_dynamic_stat),
        },
        "ink_advection_diffusion": {
            "grid_size": int(grid_size),
            "core_diffusion": float(core_diffusion),
            "spurious_diffusion": float(spurious_diffusion),
            "core_flow_x": float(core_flow_x),
            "core_flow_y": float(core_flow_y),
            "spurious_flow_scale": float(spurious_flow_scale),
            "source_blur_sigma": float(source_blur_sigma),
            "pre_observation_steps": int(pre_observation_steps),
            "dt": float(dt),
            "dx": float(dx),
            "core_spread_initial_mean": float(spread_initial.mean()),
            "core_spread_final_mean": float(spread_final.mean()),
            "core_entropy_initial_mean": float(entropy_initial.mean()),
            "core_entropy_final_mean": float(entropy_final.mean()),
            "spurious_spread_initial_mean": float(spurious_spread_initial.mean()),
            "spurious_spread_final_mean": float(spurious_spread_final.mean()),
            "spurious_entropy_initial_mean": float(spurious_entropy_initial.mean()),
            "spurious_entropy_final_mean": float(spurious_entropy_final.mean()),
            "source_entropy_initial_mean": float(entropy_initial.mean()),
            "source_entropy_final_mean": float(entropy_final.mean()),
            "source_posterior_spread": float(spread_final.mean()),
            "reverse_source_accuracy": _source_recovery_mass(final_core, core_sources),
            **source_visibility,
            "signal_to_noise_std_ratio": float(signal_to_noise),
        },
        "seeds": seeds.as_dict(),
    }

    result: dict[str, Any] = {
        "x": x,
        "y": y.astype(np.int64),
        "x_cf": x_cf,
        "y_cf": y.astype(np.int64).copy(),
        "core_source": core_sources,
        "spurious_source": spurious_sources,
        "core_score": core_score.astype(np.float32),
        "core_dynamic_stat": core_dynamic_stat,
        "spurious_dynamic_stat": spurious_dynamic_stat,
        "spurious_cf_dynamic_stat": spurious_cf_dynamic_stat,
        "metadata": metadata,
    }
    if return_fields:
        result.update(
            {
                "core_field": core_fields,
                "spurious_field": spurious_fields,
                "spurious_cf_field": spurious_cf_fields,
                "noise": noise,
            }
        )
    return result


def generate_ink_advection_diffusion_splits(
    *,
    n_train: int = 10_000,
    n_val_iid: int = 2_000,
    n_iid_test: int = 5_000,
    n_ood_test: int = 5_000,
    length: int = 16,
    grid_size: int = 32,
    seed: int = 0,
    core_diffusion: float = 0.16,
    spurious_diffusion: float = 0.12,
    core_flow_x: float = 0.0,
    core_flow_y: float = 0.0,
    spurious_flow_scale: float = 0.8,
    source_blur_sigma: float = 1.0,
    pre_observation_steps: int = 0,
    dt: float = 0.35,
    dx: float = 1.0,
    observation_noise_std: float = 0.003,
    core_scale: float = 1.0,
    spur_scale: float = 0.9,
    label_mode: str = "core_source_x_median_threshold",
    spurious_label_correlation_strength: float = 1.0,
    spurious_cf_mode: str = "randomized",
    reuse_noise: bool = True,
    split_spurious_modes: dict[str, str] | None = None,
    return_fields: bool = False,
) -> dict[str, dict[str, Any]]:
    """Generate train/val_iid/iid_test/ood_test splits with train threshold reuse."""

    counts = {
        "train": int(n_train),
        "val_iid": int(n_val_iid),
        "iid_test": int(n_iid_test),
        "ood_test": int(n_ood_test),
    }
    modes = {
        "train": "correlated",
        "val_iid": "correlated",
        "iid_test": "correlated",
        "ood_test": "reversed",
    }
    if split_spurious_modes:
        modes.update(split_spurious_modes)

    splits: dict[str, dict[str, Any]] = {}
    train_threshold: float | None = None
    for split in SPLIT_ORDER:
        data = generate_ink_advection_diffusion_bench(
            n_sequences=counts[split],
            length=length,
            grid_size=grid_size,
            split=split,
            spurious_mode=modes[split],
            seed=seed,
            core_diffusion=core_diffusion,
            spurious_diffusion=spurious_diffusion,
            core_flow_x=core_flow_x,
            core_flow_y=core_flow_y,
            spurious_flow_scale=spurious_flow_scale,
            source_blur_sigma=source_blur_sigma,
            pre_observation_steps=pre_observation_steps,
            dt=dt,
            dx=dx,
            observation_noise_std=observation_noise_std,
            core_scale=core_scale,
            spur_scale=spur_scale,
            label_mode=label_mode,
            label_threshold=train_threshold,
            spurious_label_correlation_strength=spurious_label_correlation_strength,
            spurious_cf_mode=spurious_cf_mode,
            reuse_noise=reuse_noise,
            return_fields=return_fields,
        )
        if split == "train":
            train_threshold = float(data["metadata"]["core"]["label_threshold"])
        splits[split] = data
    return splits
