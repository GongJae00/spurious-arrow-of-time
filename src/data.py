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
    n = config.split_size(split)
    rng = np.random.default_rng(config.seed + 1009 * SPLITS.index(split))
    grid = config.grid_size
    y = balanced_labels(n, rng)
    source_orientation = y.copy()
    source_center = rng.integers(0, grid, size=(n, 2), endpoint=False)

    # OE-Core: directional pulse. Else Eq. 6.
    if config.core_process == "directional_pulse":
        core_direction = (2 * source_orientation - 1).astype(np.int64)
        if config.core_direction_flip_prob > 0:
            flip = rng.random(n) < config.core_direction_flip_prob
            core_direction = np.where(flip, -core_direction, core_direction)
        core = build_nuisance_sequences(config, core_direction, rng)
    else:
        core = build_core_sequences(config, source_center, source_orientation, rng)

    nuisance_direction = sample_nuisance_direction(config, y, split, rng)
    nuisance = build_nuisance_sequences(config, nuisance_direction, rng)

    cf_direction = sample_counterfactual_direction(config, nuisance_direction, rng)
    nuisance_cf = build_nuisance_sequences(config, cf_direction, rng)

    mixed, counterfactual = compose_observation_pair(config, core, nuisance, nuisance_cf, rng)
    metadata = split_metadata(config, y, nuisance_direction, cf_direction)
    return Split(
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


def build_core_sequences(config: GeneratorConfig, centers: np.ndarray, orientations: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = len(orientations)
    grid = config.grid_size
    total_steps = config.diffusion_start_step + config.diffusion_steps_between_frames * (config.length - 1)
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
        if step >= config.diffusion_start_step and (step - config.diffusion_start_step) % config.diffusion_steps_between_frames == 0:
            frames[:, frame_idx] = state
            frame_idx += 1
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


def compose_observation_pair(config: GeneratorConfig, core: np.ndarray, nuisance: np.ndarray, nuisance_cf: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    # Eq. 9.
    if config.observation_layout == "additive":
        noise = rng.normal(0.0, config.observation_noise_std, size=core.shape).astype(np.float32)
        mixed = config.core_scale * core + config.nuisance_scale * nuisance + noise
        counterfactual = config.core_scale * core + config.nuisance_scale * nuisance_cf + noise
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
    counterfactual[:, :, 1] = config.nuisance_scale * nuisance_cf
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
    cols = np.arange(grid, dtype=np.float32)[None, None, :]
    field = np.zeros((n, grid, grid), dtype=np.float32)
    for _ in range(config.background_clutter_count):
        rc = rng.uniform(0.0, grid, size=n).astype(np.float32)
        cc = rng.uniform(0.0, grid, size=n).astype(np.float32)
        amp = rng.uniform(0.3, 1.0, size=n).astype(np.float32)
        rd = circular_distance(rows, rc[:, None, None], grid)
        cd = circular_distance(cols, cc[:, None, None], grid)
        field += amp[:, None, None] * np.exp(-0.5 * ((rd / 1.4) ** 2 + (cd / 1.4) ** 2))
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
    crops = _load_real_video_crops(config.real_video_cache)
    idx = rng.integers(0, len(crops), size=len(direction))
    seq = crops[idx].astype(np.float32) / 255.0
    reverse = direction < 0
    seq[reverse] = seq[reverse, ::-1]
    if config.real_video_endpoint_roll:
        shifts = rng.integers(0, seq.shape[1], size=len(seq))
        for s in np.unique(shifts):
            mask = shifts == s
            if s:
                seq[mask] = np.roll(seq[mask], int(s), axis=1)
    if config.real_video_blur_sigma > 0:
        import cv2
        k = int(2 * round(2 * config.real_video_blur_sigma) + 1)
        flat = seq.reshape(-1, seq.shape[2], seq.shape[3])
        for i in range(len(flat)):
            flat[i] = cv2.GaussianBlur(flat[i], (k, k), config.real_video_blur_sigma)
        seq = flat.reshape(seq.shape)
    if config.real_video_standardize:
        mean = seq.reshape(len(seq), -1).mean(axis=1)[:, None, None, None]
        std = seq.reshape(len(seq), -1).std(axis=1)[:, None, None, None]
        seq = (seq - mean) / np.clip(std, 1e-6, None)
    return seq.astype(np.float32)


def build_nuisance_sequences(config: GeneratorConfig, direction: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    # Eqs. 7–8. Pulse plus trail residue γ; γ=0 is Simple OE, γ=0.78 is Trail-FL.
    if config.nuisance_motion == "real_video":
        return build_real_video_nuisance(config, direction, rng)
    n = len(direction)
    grid = config.grid_size
    rows = np.arange(grid, dtype=np.float32)[None, :, None]
    cols = np.arange(grid, dtype=np.float32)[None, None, :]
    speed = config.nuisance_speed
    L = config.length
    ep = config.benchmark_variant == "endpoint_matched"
    motion = config.nuisance_motion
    center = (grid - 1) / 2.0
    row_phases = None
    if motion == "rotate":
        radius = grid * 0.32
        omega = 2.0 * np.pi * speed / grid
        if ep:
            final_angle = rng.uniform(0.0, 2.0 * np.pi, size=n).astype(np.float32)
            angle0 = (final_angle - direction * omega * (L - 1)).astype(np.float32)
        else:
            angle0 = rng.uniform(0.0, 2.0 * np.pi, size=n).astype(np.float32)
    else:
        phases = rng.uniform(0.0, grid, size=n).astype(np.float32)
        if ep:
            final_cols = rng.uniform(0.0, grid, size=n).astype(np.float32)
            phases = (final_cols - direction * speed * (L - 1)) % grid
        row_centers = rng.uniform(0.0, grid, size=n).astype(np.float32)
        if motion == "diagonal":
            row_phases = row_centers
            if ep:
                final_rows = rng.uniform(0.0, grid, size=n).astype(np.float32)
                row_phases = (final_rows - direction * speed * (L - 1)) % grid
    row_sigma = config.nuisance_sigma * 2.0 if motion == "translate" else config.nuisance_sigma
    sequences = np.zeros((n, L, grid, grid), dtype=np.float32)
    trail = np.zeros((n, grid, grid), dtype=np.float32)
    for t in range(L):
        if motion == "diagonal":
            col_center = (phases + direction * speed * t) % grid
            row_center = (row_phases + direction * speed * t) % grid
        elif motion == "rotate":
            angle = angle0 + direction * omega * t
            col_center = (center + radius * np.cos(angle)) % grid
            row_center = (center + radius * np.sin(angle)) % grid
        else:
            col_center = (phases + direction * speed * t) % grid
            row_center = row_centers
        col_dist = circular_distance(cols, col_center[:, None, None], grid)
        row_dist = circular_distance(rows, row_center[:, None, None], grid)
        pulse = np.exp(-0.5 * ((col_dist / config.nuisance_sigma) ** 2 + (row_dist / row_sigma) ** 2)).astype(np.float32)
        trail = config.nuisance_trail_decay * trail + pulse
        sequences[:, t] = pulse if ep and t == L - 1 else trail
    max_per_sample = sequences.reshape(n, -1).max(axis=1).clip(min=1e-8)
    return (sequences / max_per_sample[:, None, None, None]).astype(np.float32)


def circular_distance(a: np.ndarray, b: np.ndarray, period: int) -> np.ndarray:
    raw = np.abs(a - b)
    return np.minimum(raw, period - raw)


def split_metadata(config: GeneratorConfig, y: np.ndarray, nuisance_direction: np.ndarray, cf_direction: np.ndarray) -> dict:
    return {
        "benchmark_variant": config.benchmark_variant,
        "observation_layout": config.observation_layout,
        "nuisance_trail_decay": config.nuisance_trail_decay,
        "class_balance": {str(c): float(np.mean(y == c)) for c in sorted(np.unique(y).tolist())},
        "counterfactual_changed_fraction": float(np.mean(nuisance_direction != cf_direction)),
    }


def read_frames(path: str, max_frames: int = 2000) -> np.ndarray:
    import cv2
    cap = cv2.VideoCapture(path)
    frames = []
    while len(frames) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    cap.release()
    return np.stack(frames) if frames else np.zeros((0, 1, 1), np.uint8)


def extract_crops(frames: np.ndarray, grid: int, length: int, t_stride: int, short_side: int, per_clip: int, min_motion: float, rng: np.random.Generator) -> np.ndarray:
    import cv2
    if len(frames) < length * t_stride + 1:
        return np.zeros((0, length, grid, grid), np.uint8)
    h, w = frames.shape[1:]
    scale = short_side / min(h, w)
    frames = np.stack([
        cv2.resize(f, (max(grid, int(w * scale)), max(grid, int(h * scale))), interpolation=cv2.INTER_AREA)
        for f in frames
    ])
    H, W = frames.shape[1:]
    crops = []
    attempts = 0
    while len(crops) < per_clip and attempts < per_clip * 20:
        attempts += 1
        t0 = int(rng.integers(0, len(frames) - length * t_stride))
        r0 = int(rng.integers(0, H - grid + 1))
        c0 = int(rng.integers(0, W - grid + 1))
        clip = frames[t0 : t0 + length * t_stride : t_stride, r0 : r0 + grid, c0 : c0 + grid].astype(np.float32)
        if np.abs(np.diff(clip, axis=0)).mean() < min_motion:
            continue
        lo, hi = clip.min(), clip.max()
        if hi - lo < 8:
            continue
        clip = (clip - lo) / (hi - lo)
        crops.append((clip * 255).astype(np.uint8))
    return np.stack(crops) if crops else np.zeros((0, length, grid, grid), np.uint8)


def build_real_video_cache(src: Path, out: Path, grid: int = 16, length: int = 8, t_stride: int = 3, short_side: int = 48, per_clip: int = 3000, min_motion: float = 4.0, seed: int = 1234) -> np.ndarray:
    rng = np.random.default_rng(seed)
    all_crops, meta = [], []
    for path in sorted(src.glob("clip*.webm")) + sorted(src.glob("clip*.mp4")):
        frames = read_frames(str(path))
        crops = extract_crops(frames, grid, length, t_stride, short_side, per_clip, min_motion, rng)
        meta.append({"clip": path.name, "frames": int(len(frames)), "crops": int(len(crops))})
        if len(crops):
            all_crops.append(crops)
    crops = np.concatenate(all_crops) if all_crops else np.zeros((0,), np.uint8)
    rng.shuffle(crops)
    np.savez_compressed(out, crops=crops)
    out.with_suffix(".json").write_text(json.dumps({"params": {"src": str(src), "out": str(out), "grid": grid, "length": length}, "clips": meta}, indent=2), encoding="utf-8")
    return crops


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="data/real_video")
    p.add_argument("--out", default="data/real_video/cache_g16_L8_s5.npz")
    p.add_argument("--grid", type=int, default=16)
    p.add_argument("--length", type=int, default=8)
    p.add_argument("--t-stride", type=int, default=3)
    p.add_argument("--short-side", type=int, default=48)
    p.add_argument("--per-clip", type=int, default=3000)
    p.add_argument("--min-motion", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=1234)
    a = p.parse_args()
    crops = build_real_video_cache(Path(a.src), Path(a.out), a.grid, a.length, a.t_stride, a.short_side, a.per_clip, a.min_motion, a.seed)
    print(a.out, crops.shape)


if __name__ == "__main__":
    main()
