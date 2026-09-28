import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Algorithm 2. Core Eq. 6, nuisance Eqs. 7–8, observation Eq. 9. OOD reverses Eq. 4.
# GeneratorConfig defaults are the lab grid. Paper settings are benchmark.PAPER and configs/default.yaml.

SPLITS = ("train", "val_iid", "iid_test", "ood_test")

_REAL_VIDEO_CACHE: dict[str, np.ndarray] = {}


@dataclass(frozen=True)
class GeneratorConfig:
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
    nuisance_motion: str = "translate"
    real_video_cache: str = ""
    real_video_standardize: bool = False
    real_video_blur_sigma: float = 0.0
    real_video_endpoint_roll: bool = False
    core_process: str = "diffusion"
    core_direction_flip_prob: float = 0.0
    advection_shift: int = 1
    background_clutter: float = 0.0
    background_clutter_count: int = 3

    def split_size(self, split: str) -> int:
        return {
            "train": self.n_train,
            "val_iid": self.n_val_iid,
            "iid_test": self.n_iid_test,
            "ood_test": self.n_ood_test,
        }[split]


@dataclass(frozen=True)
class Split:
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
    metadata: dict


def generate_splits(config: GeneratorConfig) -> dict[str, Split]:
    return {split: generate_split(config, split) for split in SPLITS}


def generate_split(config: GeneratorConfig, split: str) -> Split:
    # Algorithm 2. y, core (Eq. 6), d_s (Eq. 4), nuisance (Eqs. 7–8), x (Eq. 9).
    sample_count = config.split_size(split)
    rng = np.random.default_rng(config.seed + 1009 * SPLITS.index(split))
    grid = config.grid_size
    y = balanced_labels(sample_count, rng)
    source_orientation = y.copy()
    source_center = rng.integers(0, grid, size=(sample_count, 2), endpoint=False)

    # OE-Core: directional pulse. Else Eq. 6.
    if config.core_process == "directional_pulse":
        core_direction = (2 * source_orientation - 1).astype(np.int64)
        if config.core_direction_flip_prob > 0:
            flip = rng.random(sample_count) < config.core_direction_flip_prob
            core_direction = np.where(flip, -core_direction, core_direction)
        core = build_nuisance_sequences(config, core_direction, rng)
    else:
        core = build_core_sequences(config, source_center, source_orientation, rng)

    nuisance_direction = sample_nuisance_direction(config, y, split, rng)
    nuisance = build_nuisance_sequences(config, nuisance_direction, rng)

    counterfactual_direction = sample_counterfactual_direction(config, nuisance_direction, rng)
    nuisance_counterfactual = build_nuisance_sequences(config, counterfactual_direction, rng)

    mixed, counterfactual = compose_observation_pair(config, core, nuisance, nuisance_counterfactual, rng)
    metadata = split_metadata(config, y, nuisance_direction, counterfactual_direction)
    return Split(
        split=split,
        core_only=core.astype(np.float32),
        nuisance_only=nuisance.astype(np.float32),
        nuisance_counterfactual=nuisance_counterfactual.astype(np.float32),
        mixed=mixed,
        counterfactual=counterfactual,
        y=y.astype(np.int64),
        source_index=(source_center[:, 0] * grid + source_center[:, 1]).astype(np.int64),
        source_center=source_center.astype(np.int64),
        source_orientation=source_orientation.astype(np.int64),
        nuisance_direction=nuisance_direction.astype(np.int64),
        counterfactual_direction=counterfactual_direction.astype(np.int64),
        metadata=metadata,
    )


def balanced_labels(n: int, rng: np.random.Generator) -> np.ndarray:
    y = np.zeros(n, dtype=np.int64)
    y[n // 2 :] = 1
    rng.shuffle(y)
    return y


def build_core_sequences(config: GeneratorConfig, centers: np.ndarray, orientations: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    sample_count = len(orientations)
    grid = config.grid_size
    total_steps = config.diffusion_start_step + config.diffusion_steps_between_frames * (config.length - 1)
    frames = np.zeros((sample_count, config.length, grid, grid), dtype=np.float32)
    state = np.zeros((sample_count, grid, grid), dtype=np.float32)
    rows = centers[:, 0]
    columns = centers[:, 1]
    for sample_index, orientation in enumerate(orientations):
        row = rows[sample_index]
        column = columns[sample_index]
        if orientation == 0:
            state[sample_index, row, (column - 1) % grid] = 0.5
            state[sample_index, row, (column + 1) % grid] = 0.5
        else:
            state[sample_index, (row - 1) % grid, column] = 0.5
            state[sample_index, (row + 1) % grid, column] = 0.5
    frame_index = 0
    for step in range(total_steps + 1):
        if step >= config.diffusion_start_step and (step - config.diffusion_start_step) % config.diffusion_steps_between_frames == 0:
            frames[:, frame_index] = state
            frame_index += 1
        if step < total_steps:
            state = evolve_core_once(state, config)
    if config.core_noise_std > 0:
        time_scale = np.linspace(0.0, 1.0, config.length, dtype=np.float32)
        time_scale = time_scale ** config.core_noise_growth_power
        noise = rng.normal(0.0, config.core_noise_std, size=frames.shape)
        frames = frames + noise * time_scale[None, :, None, None]
        frames = np.clip(frames, 0.0, None)
    return frames


def diffuse_once(state: np.ndarray, alpha: float) -> np.ndarray:
    # Eq. 6. Paper setting uses α=0.22.
    neighbors = (
        np.roll(state, 1, axis=1)
        + np.roll(state, -1, axis=1)
        + np.roll(state, 1, axis=2)
        + np.roll(state, -1, axis=2)
    ) / 4.0
    return (1.0 - alpha) * state + alpha * neighbors


def evolve_core_once(state: np.ndarray, config: GeneratorConfig) -> np.ndarray:
    if config.core_process == "advection":
        drifted = np.roll(state, config.advection_shift, axis=2)
        return (1.0 - config.diffusion_alpha) * drifted + config.diffusion_alpha * diffuse_once(state, config.diffusion_alpha)
    return diffuse_once(state, config.diffusion_alpha)


def compose_observation_pair(config: GeneratorConfig, core: np.ndarray, nuisance: np.ndarray, nuisance_counterfactual: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    # Eq. 9.
    if config.observation_layout == "additive":
        noise = rng.normal(0.0, config.observation_noise_std, size=core.shape).astype(np.float32)
        mixed = config.core_scale * core + config.nuisance_scale * nuisance + noise
        counterfactual = config.core_scale * core + config.nuisance_scale * nuisance_counterfactual + noise
        if config.background_clutter > 0:
            clutter = config.background_clutter * build_clutter(config, core.shape[0], rng)
            mixed = mixed + clutter
            counterfactual = counterfactual + clutter
        return mixed.astype(np.float32), counterfactual.astype(np.float32)
    noise = rng.normal(0.0, config.observation_noise_std, size=(core.shape[0], core.shape[1], 2, core.shape[2], core.shape[3])).astype(np.float32)
    mixed = np.zeros_like(noise, dtype=np.float32)
    counterfactual = np.zeros_like(noise, dtype=np.float32)
    mixed[:, :, 0] = config.core_scale * core
    mixed[:, :, 1] = config.nuisance_scale * nuisance
    counterfactual[:, :, 0] = config.core_scale * core
    counterfactual[:, :, 1] = config.nuisance_scale * nuisance_counterfactual
    mixed = mixed + noise
    counterfactual = counterfactual + noise
    if config.background_clutter > 0:
        clutter = config.background_clutter * build_clutter(config, core.shape[0], rng)
        mixed = mixed + clutter[:, :, None, :, :]
        counterfactual = counterfactual + clutter[:, :, None, :, :]
    return mixed.astype(np.float32), counterfactual.astype(np.float32)


def build_clutter(config: GeneratorConfig, n: int, rng: np.random.Generator) -> np.ndarray:
    grid = config.grid_size
    rows = np.arange(grid, dtype=np.float32)[None, :, None]
    columns = np.arange(grid, dtype=np.float32)[None, None, :]
    field = np.zeros((n, grid, grid), dtype=np.float32)
    for _ in range(config.background_clutter_count):
        row_center = rng.uniform(0.0, grid, size=n).astype(np.float32)
        column_center = rng.uniform(0.0, grid, size=n).astype(np.float32)
        amplitude = rng.uniform(0.3, 1.0, size=n).astype(np.float32)
        row_distance = circular_distance(rows, row_center[:, None, None], grid)
        column_distance = circular_distance(columns, column_center[:, None, None], grid)
        field += amplitude[:, None, None] * np.exp(-0.5 * ((row_distance / 1.4) ** 2 + (column_distance / 1.4) ** 2))
    return np.repeat(field[:, None, :, :], config.length, axis=1).astype(np.float32)


def sample_nuisance_direction(config: GeneratorConfig, y: np.ndarray, split: str, rng: np.random.Generator) -> np.ndarray:
    # Eq. 4 / Table 4. Train aligned; OOD reversed, randomized, or partial.
    if split == "ood_test":
        aligned_probability = {
            "reversed": 1.0 - config.nuisance_correlation,
            "randomized": 0.5,
            "partial_shift": (config.partial_shift_target_correlation + 1.0) / 2.0,
        }[config.ood_mode]
    else:
        aligned_probability = {"correlated": config.nuisance_correlation, "randomized": 0.5}[config.train_nuisance_mode]
    aligned = rng.random(len(y)) < aligned_probability
    base = np.where(y == 1, 1, -1)
    return np.where(aligned, base, -base).astype(np.int64)


def sample_counterfactual_direction(config: GeneratorConfig, direction: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if config.counterfactual_mode == "randomized":
        return rng.choice(np.array([-1, 1], dtype=np.int64), size=len(direction)).astype(np.int64)
    return (-direction).astype(np.int64)


def _load_real_video_crops(path: str) -> np.ndarray:
    if path not in _REAL_VIDEO_CACHE:
        _REAL_VIDEO_CACHE[path] = np.load(path)["crops"]
    return _REAL_VIDEO_CACHE[path]


def build_real_video_nuisance(config: GeneratorConfig, direction: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    # Table A12. Cached crops. Negative direction plays the clip backward.
    crops = _load_real_video_crops(config.real_video_cache)
    crop_indices = rng.integers(0, len(crops), size=len(direction))
    sequences = crops[crop_indices].astype(np.float32) / 255.0
    reverse = direction < 0
    sequences[reverse] = sequences[reverse, ::-1]
    if config.real_video_endpoint_roll:
        shifts = rng.integers(0, sequences.shape[1], size=len(sequences))
        for shift in np.unique(shifts):
            mask = shifts == shift
            if shift:
                sequences[mask] = np.roll(sequences[mask], int(shift), axis=1)
    if config.real_video_blur_sigma > 0:
        import cv2
        kernel_size = int(2 * round(2 * config.real_video_blur_sigma) + 1)
        flat = sequences.reshape(-1, sequences.shape[2], sequences.shape[3])
        for frame_index in range(len(flat)):
            flat[frame_index] = cv2.GaussianBlur(flat[frame_index], (kernel_size, kernel_size), config.real_video_blur_sigma)
        sequences = flat.reshape(sequences.shape)
    if config.real_video_standardize:
        mean = sequences.reshape(len(sequences), -1).mean(axis=1)[:, None, None, None]
        std = sequences.reshape(len(sequences), -1).std(axis=1)[:, None, None, None]
        sequences = (sequences - mean) / np.clip(std, 1e-6, None)
    return sequences.astype(np.float32)


def build_nuisance_sequences(config: GeneratorConfig, direction: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    # Eqs. 7–8. Pulse plus trail residue γ; γ=0 is OE-Simple, γ=0.78 is FL-Trail.
    if config.nuisance_motion == "real_video":
        return build_real_video_nuisance(config, direction, rng)
    sample_count = len(direction)
    grid = config.grid_size
    rows = np.arange(grid, dtype=np.float32)[None, :, None]
    columns = np.arange(grid, dtype=np.float32)[None, None, :]
    speed = config.nuisance_speed
    length = config.length
    endpoint_matched = config.benchmark_variant == "endpoint_matched"
    motion = config.nuisance_motion
    center = (grid - 1) / 2.0
    row_phases = None
    if motion == "rotate":
        radius = grid * 0.32
        omega = 2.0 * np.pi * speed / grid
        if endpoint_matched:
            final_angle = rng.uniform(0.0, 2.0 * np.pi, size=sample_count).astype(np.float32)
            initial_angle = (final_angle - direction * omega * (length - 1)).astype(np.float32)
        else:
            initial_angle = rng.uniform(0.0, 2.0 * np.pi, size=sample_count).astype(np.float32)
    else:
        phases = rng.uniform(0.0, grid, size=sample_count).astype(np.float32)
        if endpoint_matched:
            final_columns = rng.uniform(0.0, grid, size=sample_count).astype(np.float32)
            phases = (final_columns - direction * speed * (length - 1)) % grid
        row_centers = rng.uniform(0.0, grid, size=sample_count).astype(np.float32)
        if motion == "diagonal":
            row_phases = row_centers
            if endpoint_matched:
                final_rows = rng.uniform(0.0, grid, size=sample_count).astype(np.float32)
                row_phases = (final_rows - direction * speed * (length - 1)) % grid
    row_sigma = config.nuisance_sigma * 2.0 if motion == "translate" else config.nuisance_sigma
    sequences = np.zeros((sample_count, length, grid, grid), dtype=np.float32)
    trail = np.zeros((sample_count, grid, grid), dtype=np.float32)
    for t in range(length):
        if motion == "diagonal":
            column_center = (phases + direction * speed * t) % grid
            row_center = (row_phases + direction * speed * t) % grid
        elif motion == "rotate":
            angle = initial_angle + direction * omega * t
            column_center = (center + radius * np.cos(angle)) % grid
            row_center = (center + radius * np.sin(angle)) % grid
        else:
            column_center = (phases + direction * speed * t) % grid
            row_center = row_centers
        column_distance = circular_distance(columns, column_center[:, None, None], grid)
        row_distance = circular_distance(rows, row_center[:, None, None], grid)
        pulse = np.exp(-0.5 * ((column_distance / config.nuisance_sigma) ** 2 + (row_distance / row_sigma) ** 2)).astype(np.float32)
        trail = config.nuisance_trail_decay * trail + pulse
        sequences[:, t] = pulse if endpoint_matched and t == length - 1 else trail
    max_per_sample = sequences.reshape(sample_count, -1).max(axis=1).clip(min=1e-8)
    return (sequences / max_per_sample[:, None, None, None]).astype(np.float32)


def circular_distance(a: np.ndarray, b: np.ndarray, period: int) -> np.ndarray:
    raw = np.abs(a - b)
    return np.minimum(raw, period - raw)


def split_metadata(config: GeneratorConfig, y: np.ndarray, nuisance_direction: np.ndarray, counterfactual_direction: np.ndarray) -> dict:
    return {
        "benchmark_variant": config.benchmark_variant,
        "observation_layout": config.observation_layout,
        "nuisance_trail_decay": config.nuisance_trail_decay,
        "class_balance": {str(c): float(np.mean(y == c)) for c in sorted(np.unique(y).tolist())},
        "counterfactual_changed_fraction": float(np.mean(nuisance_direction != counterfactual_direction)),
    }


def read_frames(path: str, max_frames: int = 2000) -> np.ndarray:
    import cv2
    capture = cv2.VideoCapture(path)
    frames = []
    while len(frames) < max_frames:
        has_frame, frame = capture.read()
        if not has_frame:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    capture.release()
    return np.stack(frames) if frames else np.zeros((0, 1, 1), np.uint8)


def extract_crops(frames: np.ndarray, grid: int, length: int, t_stride: int, short_side: int, per_clip: int, min_motion: float, rng: np.random.Generator) -> np.ndarray:
    import cv2
    if len(frames) < length * t_stride + 1:
        return np.zeros((0, length, grid, grid), np.uint8)
    height, width = frames.shape[1:]
    scale = short_side / min(height, width)
    frames = np.stack([
        cv2.resize(frame, (max(grid, int(width * scale)), max(grid, int(height * scale))), interpolation=cv2.INTER_AREA)
        for frame in frames
    ])
    resized_height, resized_width = frames.shape[1:]
    crops = []
    attempts = 0
    while len(crops) < per_clip and attempts < per_clip * 20:
        attempts += 1
        start_frame = int(rng.integers(0, len(frames) - length * t_stride))
        start_row = int(rng.integers(0, resized_height - grid + 1))
        start_column = int(rng.integers(0, resized_width - grid + 1))
        clip = frames[start_frame : start_frame + length * t_stride : t_stride, start_row : start_row + grid, start_column : start_column + grid].astype(np.float32)
        if np.abs(np.diff(clip, axis=0)).mean() < min_motion:
            continue
        minimum, maximum = clip.min(), clip.max()
        if maximum - minimum < 8:
            continue
        clip = (clip - minimum) / (maximum - minimum)
        crops.append((clip * 255).astype(np.uint8))
    return np.stack(crops) if crops else np.zeros((0, length, grid, grid), np.uint8)


def build_real_video_cache(src: Path, out: Path, grid: int = 16, length: int = 8, t_stride: int = 3, short_side: int = 48, per_clip: int = 3000, min_motion: float = 4.0, seed: int = 1234) -> np.ndarray:
    # Table A12. Writes the crop cache `build_real_video_nuisance` reads.
    rng = np.random.default_rng(seed)
    all_crops, clip_metadata = [], []
    for path in sorted(src.glob("clip*.webm")) + sorted(src.glob("clip*.mp4")):
        frames = read_frames(str(path))
        crops = extract_crops(frames, grid, length, t_stride, short_side, per_clip, min_motion, rng)
        clip_metadata.append({"clip": path.name, "frames": int(len(frames)), "crops": int(len(crops))})
        if len(crops):
            all_crops.append(crops)
    crops = np.concatenate(all_crops) if all_crops else np.zeros((0,), np.uint8)
    rng.shuffle(crops)
    np.savez_compressed(out, crops=crops)
    out.with_suffix(".json").write_text(json.dumps({"params": {"src": str(src), "out": str(out), "grid": grid, "length": length, "t_stride": t_stride, "short_side": short_side, "per_clip": per_clip, "min_motion": min_motion, "seed": seed}, "clips": clip_metadata}, indent=2), encoding="utf-8")
    return crops


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="data/real_video")
    parser.add_argument("--out", default="data/real_video/cache_g16_L8_s5.npz")
    parser.add_argument("--grid", type=int, default=16)
    parser.add_argument("--length", type=int, default=8)
    parser.add_argument("--t-stride", type=int, default=3)
    parser.add_argument("--short-side", type=int, default=48)
    parser.add_argument("--per-clip", type=int, default=3000)
    parser.add_argument("--min-motion", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=1234)
    arguments = parser.parse_args()
    crops = build_real_video_cache(Path(arguments.src), Path(arguments.out), arguments.grid, arguments.length, arguments.t_stride, arguments.short_side, arguments.per_clip, arguments.min_motion, arguments.seed)
    print(arguments.out, crops.shape)


if __name__ == "__main__":
    main()
