from dataclasses import replace

import numpy as np

from src.data import GeneratorConfig, Split, SPLITS, generate_splits

# Table 3 constructions. Trail-FL is γ=0.78; Simple OE is γ=0.
# OE-Strict / Set-MF / sinusoid replace the nuisance channel after generation.
# Overlay RNG is seed * salt + 1009 * split index.

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
    cols = np.arange(grid)[None, None, None, :]
    blob = np.exp(-0.5 * (((rows - grid / 2.0) ** 2) + ((cols - pos[:, :, None, None]) ** 2)) / (sigma ** 2))
    return (scale * blob / blob.max()).astype(np.float32)


def oe_strict_nuisance(direction: np.ndarray, rng: np.random.Generator, grid: int = 16, length: int = 8) -> np.ndarray:
    n = len(direction)
    p0 = rng.integers(0, grid, size=n)
    pos = (p0[:, None] + OE_STRICT_CUM[None, :]) % grid
    pos = np.where(direction[:, None] < 0, pos[:, ::-1], pos)
    return pulse(pos, grid)


def set_mf_nuisance(direction: np.ndarray, rng: np.random.Generator, grid: int = 16, length: int = 8) -> np.ndarray:
    n = len(direction)
    homog = direction > 0
    b = rng.integers(0, 2, size=n)
    cols_h = 2 * rng.integers(0, grid // 2, size=(n, length)) + b[:, None]
    cols_m = rng.integers(0, grid, size=(n, length))
    par = cols_m % 2
    same = (par == par[:, :1]).all(1)
    while same.any():
        redo = rng.integers(0, grid, size=(int(same.sum()), length))
        cols_m[same] = redo
        par = cols_m % 2
        same = (par == par[:, :1]).all(1)
    cols = np.where(homog[:, None], cols_h, cols_m)
    return pulse(cols, grid)


def sinusoid_nuisance(direction: np.ndarray, rng: np.random.Generator, grid: int = 16, length: int = 8) -> np.ndarray:
    n = len(direction)
    f0 = rng.integers(0, 8, size=n)
    t = np.arange(length)
    fbin = (f0[:, None] + direction[:, None] * t[None, :]) % 8 + 1
    x = np.arange(grid)[None, None, None, :]
    wave = np.sin(2 * np.pi * fbin[:, :, None, None] * x / grid)
    frame = np.repeat(wave, grid, axis=2)
    return (0.6 * frame).astype(np.float32)


OVERLAY = {
    "oe_strict": (oe_strict_nuisance, 101),
    "set_mf": (set_mf_nuisance, 131),
    "sinusoid": (sinusoid_nuisance, 211),
}


def overlay_nuisance(splits: dict[str, Split], overlay: str, seed: int, corr_train: float = 0.97) -> dict[str, Split]:
    fn, salt = OVERLAY[overlay]
    out = {}
    for i, name in enumerate(SPLITS):
        corr = 1 - corr_train if name == "ood_test" else corr_train
        rng = np.random.default_rng(seed * salt + 1009 * i)
        sp = splits[name]
        d = np.where(rng.random(len(sp.y)) < corr, sp.y, 1 - sp.y) * 2 - 1
        core = np.asarray(sp.core_only)[:, :, None]
        nu = fn(d, rng)[:, :, None]
        mixed = np.concatenate([core, nu], 2).astype(np.float32)
        out[name] = replace(
            sp,
            nuisance_only=nu[:, :, 0],
            mixed=mixed,
            counterfactual=mixed,
            nuisance_direction=d.astype(np.int64),
            counterfactual_direction=d.astype(np.int64),
        )
    return out


CONSTRUCTIONS = {
    # Table 3. OE-Strict, Trail-FL, Set-MF, MF-Core, OE-Core.
    "oe_strict": dict(nuisance_trail_decay=0.0, overlay="oe_strict"),
    "trail_fl": dict(nuisance_trail_decay=0.78),
    "set_mf": dict(nuisance_trail_decay=0.0, overlay="set_mf"),
    "mf_core": dict(**MF_CORE),
    "oe_core": dict(nuisance_trail_decay=0.0, core_process="directional_pulse"),
    "simple_oe": dict(nuisance_trail_decay=0.0),
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
    data = {k: v for k, v in extra.items() if k != "overlay"}
    cfg = paper_config(seed, **{**sizes, **named, **data})
    splits = generate_splits(cfg)
    if overlay:
        return overlay_nuisance(splits, overlay, seed, corr_train=corr_train)
    return splits


def graph_setup(name: str = "karate"):
    # Table A11.
    import networkx as nx
    if name == "karate":
        G = nx.karate_club_graph()
        club = nx.get_node_attributes(G, "club")
        faction = np.array([0 if club[i] == "Mr. Hi" else 1 for i in G.nodes()])
    if name == "lesmis":
        G = nx.convert_node_labels_to_integers(nx.les_miserables_graph())
        fiedler = nx.fiedler_vector(G, weight=None, seed=0)
        faction = (np.asarray(fiedler) > 0).astype(np.int64)
    n = G.number_of_nodes()
    A = nx.to_numpy_array(G, weight=None)
    P = A / np.clip(A.sum(1, keepdims=True), 1, None)
    order = np.argsort([-G.degree(i) for i in range(n)])
    return P.astype(np.float32), faction, order, n


def graph_split(P, faction, order, n_nodes, n, split_seed, mode, length=8, alpha=0.22, steps_between=4, diff_start=0, core_noise=0.006, obs_noise=0.04, nu_sigma=1.15, nu_speed=2.0, corr=0.97):
    # Table A11. Graph diffusion core, directional nuisance on the node order.
    rng = np.random.default_rng(split_seed)
    y = np.zeros(n, dtype=np.int64)
    y[n // 2 :] = 1
    rng.shuffle(y)
    core = np.zeros((n, length, n_nodes), dtype=np.float32)
    nodes_by_f = [np.where(faction == f)[0] for f in (0, 1)]
    src = np.array([rng.choice(nodes_by_f[label]) for label in y])
    state = np.zeros((n, n_nodes), dtype=np.float32)
    state[np.arange(n), src] = 1.0
    for _ in range(diff_start):
        state = (1 - alpha) * state + alpha * (state @ P.T)
    k = 0
    for step in range(length * steps_between):
        if step % steps_between == 0:
            core[:, k] = state
            k += 1
        state = (1 - alpha) * state + alpha * (state @ P.T)
    core += rng.normal(0, core_noise, core.shape).astype(np.float32)
    randomized = mode.startswith("ns_")
    base_mode = mode.replace("ns_", "")
    if randomized:
        p_align = 0.5
    elif base_mode == "ood":
        p_align = 1 - corr
    else:
        p_align = corr
    aligned = rng.random(n) < p_align
    base = np.where(y == 1, 1, -1)
    d = np.where(aligned, base, -base).astype(np.int64)
    inv_order = np.empty(n_nodes, dtype=np.int64)
    inv_order[order] = np.arange(n_nodes)
    coord = inv_order.astype(np.float32)
    final = rng.uniform(0, n_nodes, size=n).astype(np.float32)
    phase = (final - d * nu_speed * (length - 1)) % n_nodes
    nus = np.zeros((n, length, n_nodes), dtype=np.float32)
    for t in range(length):
        c = (phase + d * nu_speed * t) % n_nodes
        dist = np.abs(coord[None, :] - c[:, None])
        dist = np.minimum(dist, n_nodes - dist)
        nus[:, t] = np.exp(-0.5 * (dist / nu_sigma) ** 2)
    obs = np.stack([core, 1.2 * nus], axis=2)[:, :, :, None, :]
    obs = obs + rng.normal(0, obs_noise, obs.shape).astype(np.float32)
    return dict(
        mixed=obs.astype(np.float32),
        core=core[:, :, None, None, :].astype(np.float32),
        nuis=nus[:, :, None, None, :].astype(np.float32),
        y=y,
        d=d,
    )


def load_forda():
    # Table 9. Official FordA train/test, L=10 segments of 50.
    rows = [np.loadtxt(f) for f in ["data/ucr/FordA_TRAIN.tsv", "data/ucr/FordA_TEST.tsv"]]
    ntr = len(rows[0])
    a = np.concatenate(rows)
    y = (a[:, 0] > 0).astype(np.int64)
    x = a[:, 1:].astype(np.float32)
    x = (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-8)
    return x.reshape(len(x), 10, 50), y, ntr


def load_har(mode="har"):
    # Table 9. HAR-coarse (dynamic vs static) or HAR-fine (walk vs walk-up).
    base = "data/ucr/har/UCI HAR Dataset"
    xs, ys = [], []
    for split in ["train", "test"]:
        xs.append(np.loadtxt(f"{base}/{split}/Inertial Signals/body_acc_x_{split}.txt"))
        ys.append(np.loadtxt(f"{base}/{split}/y_{split}.txt"))
    x = np.concatenate(xs).astype(np.float32)
    yy = np.concatenate(ys)
    is_tr = np.arange(len(yy)) < len(ys[0])
    if mode == "har2":
        keep = yy <= 2
        x, yy, is_tr = x[keep], yy[keep], is_tr[keep]
        y = (yy == 1).astype(np.int64)
    else:
        y = (yy <= 3).astype(np.int64)
    x = (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-8)
    t_old = np.linspace(0, 1, x.shape[1])
    t_new = np.linspace(0, 1, 10 * 50)
    x = np.stack([np.interp(t_new, t_old, r) for r in x]).astype(np.float32)
    return x.reshape(len(x), 10, 50), y, int(is_tr.sum())


def overlay_order_pulse(xc, y, rng, corr, n, length=10, width=50):
    # Table 9. Order-encoded pulse on a real core; visited-position multiset is direction-independent.
    idx = rng.choice(len(y), size=n, replace=False)
    xcs, ys = xc[idx], y[idx]
    d = np.where(rng.random(n) < corr, ys, 1 - ys)
    p0 = rng.integers(0, width, size=n)
    t = np.arange(length)
    pos = (p0[:, None] + (2 * d[:, None] - 1) * 5 * t[None, :]) % width
    grid = np.arange(width)[None, None, :]
    nu = np.exp(-0.5 * ((grid - pos[:, :, None]) ** 2) / (3.0 ** 2)).astype(np.float32) * 1.5
    return xcs, nu, ys, d.astype(np.int64)
