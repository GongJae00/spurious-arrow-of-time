from dataclasses import replace

import numpy as np

from src.data import GeneratorConfig, Split, SPLITS, generate_splits

# Table 3 constructions. FL-Trail is γ=0.78; OE-Simple is γ=0.
# OE-Strict / MF-Set / sinusoid replace the nuisance channel after generation.
# Overlay RNG is seed * salt + 1009 * split index.
# Table 9 FordA/HAR and Table A11 graph are the loaders below make().

PAPER = dict(
    grid_size=16,
    length=8,
    diffusion_alpha=0.22,
    diffusion_start_step=0,
    diffusion_steps_between_frames=4,
    core_noise_std=0.006,
    observation_noise_std=0.04,
    core_scale=1.0,
    nuisance_scale=1.2,
    nuisance_sigma=1.15,
    nuisance_speed=2.0,
    nuisance_trail_decay=0.78,
    nuisance_correlation=0.97,
    observation_layout="two_channel",
    benchmark_variant="endpoint_matched",
)

SIZES = dict(n_train=8192, n_val_iid=2048, n_iid_test=4096, n_ood_test=4096)

MF_CORE = dict(diffusion_start_step=8, diffusion_steps_between_frames=2, core_noise_std=0.045, core_noise_growth_power=0.0, observation_noise_std=0.01, nuisance_trail_decay=0.0)

# OE-Strict closed path, mod 16: first and last frames coincide.
OE_STRICT_CUM = np.array([0, 1, 2, 4, 7, 10, 13, 0])


def paper_config(seed: int = 0, **over) -> GeneratorConfig:
    fields = {**PAPER, **SIZES, "seed": seed, **over}
    return GeneratorConfig(**{k: fields[k] for k in GeneratorConfig.__dataclass_fields__ if k in fields})


def pulse(pos: np.ndarray, grid: int = 16, sigma: float = 1.15, scale: float = 1.2) -> np.ndarray:
    rows = np.arange(grid)[None, None, :, None]
    columns = np.arange(grid)[None, None, None, :]
    blob = np.exp(-0.5 * (((rows - grid / 2.0) ** 2) + ((columns - pos[:, :, None, None]) ** 2)) / (sigma ** 2))
    return (scale * blob / blob.max()).astype(np.float32)


def oe_strict_nuisance(direction: np.ndarray, rng: np.random.Generator, grid: int = 16, length: int = 8) -> np.ndarray:
    sample_count = len(direction)
    initial_position = rng.integers(0, grid, size=sample_count)
    positions = (initial_position[:, None] + OE_STRICT_CUM[None, :]) % grid
    positions = np.where(direction[:, None] < 0, positions[:, ::-1], positions)
    return pulse(positions, grid)


def mf_set_nuisance(direction: np.ndarray, rng: np.random.Generator, grid: int = 16, length: int = 8) -> np.ndarray:
    sample_count = len(direction)
    homogeneous = direction > 0
    shared_parity = rng.integers(0, 2, size=sample_count)
    homogeneous_columns = 2 * rng.integers(0, grid // 2, size=(sample_count, length)) + shared_parity[:, None]
    mixed_columns = rng.integers(0, grid, size=(sample_count, length))
    parity = mixed_columns % 2
    same_parity = (parity == parity[:, :1]).all(1)
    while same_parity.any():
        resampled_columns = rng.integers(0, grid, size=(int(same_parity.sum()), length))
        mixed_columns[same_parity] = resampled_columns
        parity = mixed_columns % 2
        same_parity = (parity == parity[:, :1]).all(1)
    columns = np.where(homogeneous[:, None], homogeneous_columns, mixed_columns)
    return pulse(columns, grid)


def sinusoid_nuisance(direction: np.ndarray, rng: np.random.Generator, grid: int = 16, length: int = 8) -> np.ndarray:
    sample_count = len(direction)
    initial_frequency = rng.integers(0, 8, size=sample_count)
    time = np.arange(length)
    frequency_bin = (initial_frequency[:, None] + direction[:, None] * time[None, :]) % 8 + 1
    columns = np.arange(grid)[None, None, None, :]
    wave = np.sin(2 * np.pi * frequency_bin[:, :, None, None] * columns / grid)
    frame = np.repeat(wave, grid, axis=2)
    return (0.6 * frame).astype(np.float32)


OVERLAY = {
    "oe_strict": (oe_strict_nuisance, 101),
    "mf_set": (mf_set_nuisance, 131),
    "sinusoid": (sinusoid_nuisance, 211),
}


def overlay_nuisance(splits: dict[str, Split], overlay: str, seed: int, corr_train: float = 0.97) -> dict[str, Split]:
    nuisance_generator, salt = OVERLAY[overlay]
    overlaid = {}
    for split_index, name in enumerate(SPLITS):
        alignment_probability = 1 - corr_train if name == "ood_test" else corr_train
        rng = np.random.default_rng(seed * salt + 1009 * split_index)
        split = splits[name]
        direction = np.where(rng.random(len(split.y)) < alignment_probability, split.y, 1 - split.y) * 2 - 1
        core = np.asarray(split.core_only)[:, :, None]
        nuisance = nuisance_generator(direction, rng)[:, :, None]
        mixed = np.concatenate([core, nuisance], 2).astype(np.float32)
        overlaid[name] = replace(
            split,
            nuisance_only=nuisance[:, :, 0],
            mixed=mixed,
            counterfactual=mixed,
            nuisance_direction=direction.astype(np.int64),
            counterfactual_direction=direction.astype(np.int64),
        )
    return overlaid


CONSTRUCTIONS = {
    # Table 3. OE-Strict, FL-Trail, MF-Set, MF-Core, OE-Core.
    "oe_strict": dict(nuisance_trail_decay=0.0, overlay="oe_strict"),
    "fl_trail": dict(nuisance_trail_decay=0.78),
    "mf_set": dict(nuisance_trail_decay=0.0, overlay="mf_set"),
    "mf_core": dict(**MF_CORE),
    "oe_core": dict(nuisance_trail_decay=0.0, core_process="directional_pulse"),
    "oe_simple": dict(nuisance_trail_decay=0.0),
    "oe_core_equalized": dict(nuisance_trail_decay=0.0, core_process="directional_pulse", core_direction_flip_prob=0.03),
    "sinusoid": dict(nuisance_trail_decay=0.0, overlay="sinusoid"),
}


def make(benchmark: str, seed: int, sizes: dict | None = None, extra: dict | None = None, corr_train: float = 0.97) -> dict[str, Split]:
    extra = extra or {}
    sizes = sizes or SIZES
    named = {**CONSTRUCTIONS[benchmark]}
    overlay = named.pop("overlay", None)
    if "overlay" in extra:
        overlay = extra["overlay"]
    fields = {k: v for k, v in extra.items() if k != "overlay"}
    config = paper_config(seed, **{**sizes, **named, **fields})
    splits = generate_splits(config)
    if overlay:
        return overlay_nuisance(splits, overlay, seed, corr_train=corr_train)
    return splits


def graph_setup(name: str = "karate"):
    # Table A11. Karate or Les Misérables. Degree order is the nuisance axis.
    import networkx as nx
    if name == "karate":
        graph = nx.karate_club_graph()
        club = nx.get_node_attributes(graph, "club")
        faction = np.array([0 if club[i] == "Mr. Hi" else 1 for i in graph.nodes()])
    if name == "lesmis":
        graph = nx.convert_node_labels_to_integers(nx.les_miserables_graph())
        fiedler = nx.fiedler_vector(graph, weight=None, seed=0)
        faction = (np.asarray(fiedler) > 0).astype(np.int64)
    n_nodes = graph.number_of_nodes()
    adjacency = nx.to_numpy_array(graph, weight=None)
    transition = adjacency / np.clip(adjacency.sum(1, keepdims=True), 1, None)
    order = np.argsort([-graph.degree(i) for i in range(n_nodes)])
    return transition.astype(np.float32), faction, order, n_nodes


def graph_split(transition, faction, order, n_nodes, n, split_seed, mode, length=8, diffusion_alpha=0.22, steps_between=4, diffusion_start=0, core_noise_std=0.006, observation_noise=0.04, nuisance_sigma=1.15, nuisance_speed=2.0, alignment=0.97):
    # Table A11. Graph diffusion core, directional nuisance on the node order.
    rng = np.random.default_rng(split_seed)
    y = np.zeros(n, dtype=np.int64)
    y[n // 2 :] = 1
    rng.shuffle(y)
    core = np.zeros((n, length, n_nodes), dtype=np.float32)
    nodes_by_faction = [np.where(faction == faction_id)[0] for faction_id in (0, 1)]
    source = np.array([rng.choice(nodes_by_faction[label]) for label in y])
    state = np.zeros((n, n_nodes), dtype=np.float32)
    state[np.arange(n), source] = 1.0
    for _ in range(diffusion_start):
        state = (1 - diffusion_alpha) * state + diffusion_alpha * (state @ transition.T)
    frame_index = 0
    for step in range(length * steps_between):
        if step % steps_between == 0:
            core[:, frame_index] = state
            frame_index += 1
        state = (1 - diffusion_alpha) * state + diffusion_alpha * (state @ transition.T)
    core += rng.normal(0, core_noise_std, core.shape).astype(np.float32)
    randomized = mode.startswith("ns_")
    base_mode = mode.replace("ns_", "")
    if randomized:
        alignment_probability = 0.5
    elif base_mode == "ood":
        alignment_probability = 1 - alignment
    else:
        alignment_probability = alignment
    aligned = rng.random(n) < alignment_probability
    base = np.where(y == 1, 1, -1)
    direction = np.where(aligned, base, -base).astype(np.int64)
    inverse_order = np.empty(n_nodes, dtype=np.int64)
    inverse_order[order] = np.arange(n_nodes)
    coordinates = inverse_order.astype(np.float32)
    final_position = rng.uniform(0, n_nodes, size=n).astype(np.float32)
    phase = (final_position - direction * nuisance_speed * (length - 1)) % n_nodes
    nuisance = np.zeros((n, length, n_nodes), dtype=np.float32)
    for time in range(length):
        column = (phase + direction * nuisance_speed * time) % n_nodes
        distance = np.abs(coordinates[None, :] - column[:, None])
        distance = np.minimum(distance, n_nodes - distance)
        nuisance[:, time] = np.exp(-0.5 * (distance / nuisance_sigma) ** 2)
    mixed = np.stack([core, 1.2 * nuisance], axis=2)[:, :, :, None, :]
    mixed = mixed + rng.normal(0, observation_noise, mixed.shape).astype(np.float32)
    return dict(
        mixed=mixed.astype(np.float32),
        core=core[:, :, None, None, :].astype(np.float32),
        nuisance=nuisance[:, :, None, None, :].astype(np.float32),
        y=y,
        direction=direction,
    )


def load_forda():
    # Table 9. Official FordA train/test, L=10 segments of 50.
    rows = [np.loadtxt(path) for path in ["data/ucr/FordA_TRAIN.tsv", "data/ucr/FordA_TEST.tsv"]]
    n_train = len(rows[0])
    table = np.concatenate(rows)
    y = (table[:, 0] > 0).astype(np.int64)
    series = table[:, 1:].astype(np.float32)
    series = (series - series.mean(1, keepdims=True)) / (series.std(1, keepdims=True) + 1e-8)
    return series.reshape(len(series), 10, 50), y, n_train


def load_har(mode="har"):
    # Table 9. HAR-coarse (dynamic vs static) or HAR-fine (walk vs walk-up).
    base = "data/ucr/har/UCI HAR Dataset"
    series_by_split, activities_by_split = [], []
    for split in ["train", "test"]:
        series_by_split.append(np.loadtxt(f"{base}/{split}/Inertial Signals/body_acc_x_{split}.txt"))
        activities_by_split.append(np.loadtxt(f"{base}/{split}/y_{split}.txt"))
    series = np.concatenate(series_by_split).astype(np.float32)
    activity = np.concatenate(activities_by_split)
    is_train = np.arange(len(activity)) < len(activities_by_split[0])
    if mode == "har2":
        keep = activity <= 2
        series, activity, is_train = series[keep], activity[keep], is_train[keep]
        y = (activity == 1).astype(np.int64)
    else:
        y = (activity <= 3).astype(np.int64)
    series = (series - series.mean(1, keepdims=True)) / (series.std(1, keepdims=True) + 1e-8)
    sample_times = np.linspace(0, 1, series.shape[1])
    resampled_times = np.linspace(0, 1, 10 * 50)
    series = np.stack([np.interp(resampled_times, sample_times, window) for window in series]).astype(np.float32)
    return series.reshape(len(series), 10, 50), y, int(is_train.sum())


def overlay_order_pulse(core, y, rng, corr, n, length=10, width=50):
    # Table 9. Order-encoded pulse on a real core; visited-position multiset is direction-independent.
    indices = rng.choice(len(y), size=n, replace=False)
    core_rows, labels = core[indices], y[indices]
    direction = np.where(rng.random(n) < corr, labels, 1 - labels)
    initial_position = rng.integers(0, width, size=n)
    time = np.arange(length)
    positions = (initial_position[:, None] + (2 * direction[:, None] - 1) * 5 * time[None, :]) % width
    grid = np.arange(width)[None, None, :]
    nuisance = np.exp(-0.5 * ((grid - positions[:, :, None]) ** 2) / (3.0 ** 2)).astype(np.float32) * 1.5
    return core_rows, nuisance, labels, direction.astype(np.int64)
