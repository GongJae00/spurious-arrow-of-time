"""Irreversible source inference with a spurious arrow shortcut.

The core task is to infer a hidden source pattern after diffusion. The nuisance
is an independent directed process whose arrow is correlated with the label in
train/IID and shifted OOD.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import yaml


SPLITS = ("train", "val_iid", "iid_test", "ood_test")


@dataclass(frozen=True)
class IrreversibleSourceConfig:
    grid_size: int = 16
    length: int = 8
    n_train: int = 1024
    n_val_iid: int = 256
    n_iid_test: int = 256
    n_ood_test: int = 256
    seed: int = 0
    diffusion_alpha: float = 0.22
    diffusion_start_step: int = 1
    diffusion_steps_between_frames: int = 8
    core_noise_std: float = 0.014
    core_noise_growth_power: float = 1.0
    observation_noise_std: float = 0.08
    core_scale: float = 0.45
    nuisance_scale: float = 2.8
    nuisance_sigma: float = 1.15
    nuisance_speed: float = 2.0
    nuisance_trail_decay: float = 0.78
    nuisance_correlation: float = 0.97
    benchmark_variant: str = "residue_visible"
    observation_layout: str = "additive"
    ood_mode: str = "reversed"
    partial_shift_target_correlation: float = -0.3
    counterfactual_mode: str = "reversed"
    train_nuisance_mode: str = "correlated"
    random_labels: bool = False
    disable_nuisance: bool = False

    def split_size(self, split: str) -> int:
        sizes = {
            "train": self.n_train,
            "val_iid": self.n_val_iid,
            "iid_test": self.n_iid_test,
            "ood_test": self.n_ood_test,
        }
        if split not in sizes:
            raise ValueError(f"unknown split {split!r}")
        return sizes[split]


@dataclass(frozen=True)
class IrreversibleSourceSplit:
    split: str
    core_only: np.ndarray
    nuisance_only: np.ndarray
    nuisance_counterfactual: np.ndarray
    mixed: np.ndarray
    counterfactual: np.ndarray
    y: np.ndarray
    source_index: np.ndarray
    source_center: np.ndarray
    source_orientation: np.ndarray
    nuisance_direction: np.ndarray
    counterfactual_direction: np.ndarray
    metadata: dict[str, Any]


def load_config(path: str | Path) -> IrreversibleSourceConfig:
    with Path(path).open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return IrreversibleSourceConfig(**raw)


def generate_irreversible_source_splits(
    config: IrreversibleSourceConfig,
) -> dict[str, IrreversibleSourceSplit]:
    return {split: generate_split(config, split) for split in SPLITS}


def generate_split(config: IrreversibleSourceConfig, split: str) -> IrreversibleSourceSplit:
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}")
    if config.benchmark_variant not in {"residue_visible", "endpoint_matched"}:
        raise ValueError(f"unknown benchmark_variant {config.benchmark_variant!r}")
    if config.observation_layout not in {"additive", "two_channel"}:
        raise ValueError(f"unknown observation_layout {config.observation_layout!r}")
    n = config.split_size(split)
    rng = np.random.default_rng(config.seed + 1009 * SPLITS.index(split))
    grid = config.grid_size

    y = balanced_labels(n, rng)
    source_orientation = y.copy()
    if config.random_labels:
        y = balanced_labels(n, rng)
    source_center = rng.integers(0, grid, size=(n, 2), endpoint=False)

    core = build_core_sequences(config, source_center, source_orientation, rng)
    nuisance_direction = sample_nuisance_direction(config, y, split, rng)
    nuisance = build_nuisance_sequences(config, nuisance_direction, rng)

    cf_direction = sample_counterfactual_direction(config, nuisance_direction, rng)
    nuisance_cf = build_nuisance_sequences(config, cf_direction, rng)

    if config.disable_nuisance:
        mixed = compose_core_only_observation(config, core)
        counterfactual = mixed.copy()
    else:
        mixed, counterfactual = compose_observation_pair(
            config=config,
            core=core,
            nuisance=nuisance,
            nuisance_cf=nuisance_cf,
            rng=rng,
        )

    metadata = split_metadata(
        config=config,
        split=split,
        y=y,
        source_center=source_center,
        nuisance_direction=nuisance_direction,
        cf_direction=cf_direction,
    )
    return IrreversibleSourceSplit(
        split=split,
        core_only=core.astype(np.float32),
        nuisance_only=nuisance.astype(np.float32),
        nuisance_counterfactual=nuisance_cf.astype(np.float32),
        mixed=mixed,
        counterfactual=counterfactual,
        y=y.astype(np.int64),
        source_index=(source_center[:, 0] * grid + source_center[:, 1]).astype(np.int64),
        source_center=source_center.astype(np.int64),
        source_orientation=source_orientation.astype(np.int64),
        nuisance_direction=nuisance_direction.astype(np.int64),
        counterfactual_direction=cf_direction.astype(np.int64),
        metadata=metadata,
    )


def balanced_labels(n: int, rng: np.random.Generator) -> np.ndarray:
    y = np.zeros(n, dtype=np.int64)
    y[n // 2 :] = 1
    rng.shuffle(y)
    return y


def build_core_sequences(
    config: IrreversibleSourceConfig,
    centers: np.ndarray,
    orientations: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    n = len(orientations)
    grid = config.grid_size
    total_steps = config.diffusion_start_step + config.diffusion_steps_between_frames * (
        config.length - 1
    )
    frames = np.zeros((n, config.length, grid, grid), dtype=np.float32)
    state = np.zeros((n, grid, grid), dtype=np.float32)
    rows = centers[:, 0]
    cols = centers[:, 1]
    for i, orientation in enumerate(orientations):
        r = rows[i]
        c = cols[i]
        if orientation == 0:
            state[i, r, (c - 1) % grid] = 0.5
            state[i, r, (c + 1) % grid] = 0.5
        else:
            state[i, (r - 1) % grid, c] = 0.5
            state[i, (r + 1) % grid, c] = 0.5

    frame_idx = 0
    for step in range(total_steps + 1):
        if (
            step >= config.diffusion_start_step
            and (step - config.diffusion_start_step) % config.diffusion_steps_between_frames == 0
        ):
            frames[:, frame_idx] = state
            frame_idx += 1
        if step < total_steps:
            state = diffuse_once(state, config.diffusion_alpha)

    if config.core_noise_std > 0:
        time_scale = np.linspace(0.0, 1.0, config.length, dtype=np.float32)
        time_scale = time_scale**config.core_noise_growth_power
        noise = rng.normal(0.0, config.core_noise_std, size=frames.shape)
        frames = frames + noise * time_scale[None, :, None, None]
        frames = np.clip(frames, 0.0, None)
    return frames


def diffuse_once(state: np.ndarray, alpha: float) -> np.ndarray:
    neighbors = (
        np.roll(state, 1, axis=1)
        + np.roll(state, -1, axis=1)
        + np.roll(state, 1, axis=2)
        + np.roll(state, -1, axis=2)
    ) / 4.0
    return (1.0 - alpha) * state + alpha * neighbors


def compose_observation(
    config: IrreversibleSourceConfig,
    core: np.ndarray,
    nuisance: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    if config.observation_layout == "additive":
        noise = rng.normal(0.0, config.observation_noise_std, size=core.shape).astype(np.float32)
        return (config.core_scale * core + config.nuisance_scale * nuisance + noise).astype(
            np.float32
        )
    noise = rng.normal(
        0.0,
        config.observation_noise_std,
        size=(core.shape[0], core.shape[1], 2, core.shape[2], core.shape[3]),
    ).astype(np.float32)
    out = np.zeros_like(noise, dtype=np.float32)
    out[:, :, 0] = config.core_scale * core
    out[:, :, 1] = config.nuisance_scale * nuisance
    return (out + noise).astype(np.float32)


def compose_core_only_observation(
    config: IrreversibleSourceConfig,
    core: np.ndarray,
) -> np.ndarray:
    """Clean no-nuisance upper-bound observation used for diagnostic controls."""
    if config.observation_layout == "additive":
        return core.astype(np.float32)
    out = np.zeros(
        (core.shape[0], core.shape[1], 2, core.shape[2], core.shape[3]),
        dtype=np.float32,
    )
    out[:, :, 0] = config.core_scale * core
    return out


def compose_observation_pair(
    *,
    config: IrreversibleSourceConfig,
    core: np.ndarray,
    nuisance: np.ndarray,
    nuisance_cf: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if config.observation_layout == "additive":
        noise = rng.normal(0.0, config.observation_noise_std, size=core.shape).astype(np.float32)
        mixed = config.core_scale * core + config.nuisance_scale * nuisance + noise
        counterfactual = config.core_scale * core + config.nuisance_scale * nuisance_cf + noise
        return mixed.astype(np.float32), counterfactual.astype(np.float32)

    noise = rng.normal(
        0.0,
        config.observation_noise_std,
        size=(core.shape[0], core.shape[1], 2, core.shape[2], core.shape[3]),
    ).astype(np.float32)
    mixed = np.zeros_like(noise, dtype=np.float32)
    counterfactual = np.zeros_like(noise, dtype=np.float32)
    mixed[:, :, 0] = config.core_scale * core
    mixed[:, :, 1] = config.nuisance_scale * nuisance
    counterfactual[:, :, 0] = config.core_scale * core
    counterfactual[:, :, 1] = config.nuisance_scale * nuisance_cf
    return (mixed + noise).astype(np.float32), (counterfactual + noise).astype(np.float32)


def sample_nuisance_direction(
    config: IrreversibleSourceConfig,
    y: np.ndarray,
    split: str,
    rng: np.random.Generator,
) -> np.ndarray:
    if split == "ood_test":
        if config.ood_mode == "reversed":
            aligned_probability = 1.0 - config.nuisance_correlation
        elif config.ood_mode == "randomized":
            aligned_probability = 0.5
        elif config.ood_mode == "partial_shift":
            aligned_probability = correlation_to_aligned_probability(
                config.partial_shift_target_correlation
            )
        else:
            raise ValueError(f"unknown ood_mode {config.ood_mode!r}")
    elif config.train_nuisance_mode == "correlated":
        aligned_probability = config.nuisance_correlation
    elif config.train_nuisance_mode == "randomized":
        aligned_probability = 0.5
    else:
        raise ValueError(f"unknown train_nuisance_mode {config.train_nuisance_mode!r}")

    aligned = rng.random(len(y)) < aligned_probability
    base = np.where(y == 1, 1, -1)
    direction = np.where(aligned, base, -base)
    return direction.astype(np.int64)


def correlation_to_aligned_probability(correlation: float) -> float:
    if not -1.0 <= correlation <= 1.0:
        raise ValueError("partial_shift_target_correlation must be in [-1, 1]")
    return float((correlation + 1.0) / 2.0)


def sample_counterfactual_direction(
    config: IrreversibleSourceConfig,
    direction: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    if config.counterfactual_mode == "reversed":
        return (-direction).astype(np.int64)
    if config.counterfactual_mode == "randomized":
        return rng.choice(np.array([-1, 1], dtype=np.int64), size=len(direction)).astype(
            np.int64
        )
    if config.counterfactual_mode == "randomized_different":
        cf = rng.choice(np.array([-1, 1], dtype=np.int64), size=len(direction))
        same = cf == direction
        cf[same] *= -1
        return cf.astype(np.int64)
    raise ValueError(f"unknown counterfactual_mode {config.counterfactual_mode!r}")


def build_nuisance_sequences(
    config: IrreversibleSourceConfig,
    direction: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    n = len(direction)
    grid = config.grid_size
    rows = np.arange(grid, dtype=np.float32)[None, :, None]
    cols = np.arange(grid, dtype=np.float32)[None, None, :]
    phases = rng.uniform(0.0, grid, size=n).astype(np.float32)
    if config.benchmark_variant == "endpoint_matched":
        final_cols = rng.uniform(0.0, grid, size=n).astype(np.float32)
        phases = (final_cols - direction * config.nuisance_speed * (config.length - 1)) % grid
    row_centers = rng.uniform(0.0, grid, size=n).astype(np.float32)
    sequences = np.zeros((n, config.length, grid, grid), dtype=np.float32)
    trail = np.zeros((n, grid, grid), dtype=np.float32)
    for t in range(config.length):
        col_center = (phases + direction * config.nuisance_speed * t) % grid
        row_center = row_centers
        col_dist = circular_distance(cols, col_center[:, None, None], grid)
        row_dist = circular_distance(rows, row_center[:, None, None], grid)
        pulse = np.exp(
            -0.5
            * (
                (col_dist / config.nuisance_sigma) ** 2
                + (row_dist / (config.nuisance_sigma * 2.0)) ** 2
            )
        )
        pulse = pulse.astype(np.float32)
        trail = config.nuisance_trail_decay * trail + pulse
        if config.benchmark_variant == "endpoint_matched" and t == config.length - 1:
            sequences[:, t] = pulse
        else:
            sequences[:, t] = trail
    max_per_sample = sequences.reshape(n, -1).max(axis=1).clip(min=1e-8)
    sequences = sequences / max_per_sample[:, None, None, None]
    return sequences.astype(np.float32)


def circular_distance(a: np.ndarray, b: np.ndarray, period: int) -> np.ndarray:
    raw = np.abs(a - b)
    return np.minimum(raw, period - raw)


def split_metadata(
    config: IrreversibleSourceConfig,
    split: str,
    y: np.ndarray,
    source_center: np.ndarray,
    nuisance_direction: np.ndarray,
    cf_direction: np.ndarray,
) -> dict[str, Any]:
    return {
        "split": split,
        "grid_size": config.grid_size,
        "length_L": config.length,
        "n_sequences": int(len(y)),
        "diffusion_alpha": config.diffusion_alpha,
        "diffusion_start_step": config.diffusion_start_step,
        "diffusion_steps_between_frames": config.diffusion_steps_between_frames,
        "core_scale": config.core_scale,
        "nuisance_scale": config.nuisance_scale,
        "nuisance_speed": config.nuisance_speed,
        "nuisance_trail_decay": config.nuisance_trail_decay,
        "nuisance_correlation": config.nuisance_correlation,
        "benchmark_variant": config.benchmark_variant,
        "observation_layout": config.observation_layout,
        "ood_mode": config.ood_mode,
        "partial_shift_target_correlation": config.partial_shift_target_correlation,
        "counterfactual_mode": config.counterfactual_mode,
        "train_nuisance_mode": config.train_nuisance_mode,
        "random_labels": config.random_labels,
        "disable_nuisance": config.disable_nuisance,
        "class_balance": class_balance(y),
        "source_center_mean_by_y": mean_by_y(source_center.astype(np.float64), y),
        "nuisance_direction_mean_by_y": mean_by_y(nuisance_direction.astype(np.float64), y),
        "counterfactual_direction_mean_by_y": mean_by_y(cf_direction.astype(np.float64), y),
        "corr_y_nuisance_arrow": safe_corr(y.astype(np.float64), nuisance_direction.astype(float)),
        "corr_y_counterfactual_arrow": safe_corr(y.astype(np.float64), cf_direction.astype(float)),
        "counterfactual_changed_fraction": float(np.mean(nuisance_direction != cf_direction)),
    }


def class_balance(y: np.ndarray) -> dict[str, float]:
    return {str(cls): float(np.mean(y == cls)) for cls in sorted(np.unique(y).tolist())}


def mean_by_y(values: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for cls in sorted(np.unique(y).tolist()):
        mean = values[y == cls].mean(axis=0)
        out[str(cls)] = np.asarray(mean).round(6).tolist()
    return out


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def with_smaller_sizes(config: IrreversibleSourceConfig, n: int) -> IrreversibleSourceConfig:
    return replace(config, n_train=n, n_val_iid=n, n_iid_test=n, n_ood_test=n)
