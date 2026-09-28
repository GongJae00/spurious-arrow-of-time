import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.benchmark import OE_STRICT_OFFSETS, make, oe_strict_nuisance, overlay_order_pulse, mf_set_nuisance
from src.data import GeneratorConfig, generate_splits

# Table 3: OE-Strict closed path, MF-Set parity, endpoint matching, make().


def _probe_accuracy(x_train, y_train, x_test, y_test) -> float:
    probe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, solver="liblinear", random_state=0))
    probe.fit(x_train, y_train)
    return float(probe.score(x_test, y_test))


def _circular_column_slope(x: np.ndarray) -> np.ndarray:
    mass = np.clip(x, 0.0, None)
    width = x.shape[-1]
    columns = np.arange(width, dtype=np.float64)
    angle = 2.0 * np.pi * columns / width
    column_mass = mass.sum(axis=2)
    total = np.clip(column_mass.sum(axis=2), 1e-8, None)
    mean_cos = (column_mass * np.cos(angle)).sum(axis=2) / total
    mean_sin = (column_mass * np.sin(angle)).sum(axis=2) / total
    circular_mean = np.arctan2(mean_sin, mean_cos) * width / (2.0 * np.pi)
    unwrapped = np.unwrap(circular_mean * 2.0 * np.pi / width, axis=1)
    t = np.arange(x.shape[1], dtype=np.float64)
    t = t - t.mean()
    return ((unwrapped * t).sum(axis=1) / np.sum(t ** 2))[:, None]


def test_oe_strict_closed_path():
    assert list(OE_STRICT_OFFSETS) == [0, 1, 2, 4, 7, 10, 13, 0]
    direction = np.ones(32, dtype=np.int64)
    nuisance = oe_strict_nuisance(direction, np.random.default_rng(0))
    np.testing.assert_allclose(nuisance[:, 0], nuisance[:, -1])
    reverse_direction = -np.ones(32, dtype=np.int64)
    reverse_nuisance = oe_strict_nuisance(reverse_direction, np.random.default_rng(0))
    np.testing.assert_allclose(reverse_nuisance[:, 0], reverse_nuisance[:, -1])


def test_mf_set_parity_homogeneity():
    rng = np.random.default_rng(1)
    direction = np.array([1, 1, -1, -1] * 16)
    nuisance = mf_set_nuisance(direction, rng)
    columns = nuisance.argmax(axis=3)[:, :, 8]
    homogeneous = (columns % 2 == columns[:, :1] % 2).all(1)
    assert homogeneous[direction > 0].all()
    assert (~homogeneous[direction < 0]).all()


def test_endpoint_matched_controls_final_nuisance_leakage():
    config = GeneratorConfig(grid_size=12, length=6, n_train=512, n_val_iid=32, n_iid_test=512, n_ood_test=512, seed=123, diffusion_start_step=5, diffusion_steps_between_frames=3, benchmark_variant="endpoint_matched")
    splits = generate_splits(config)
    train = splits["train"]
    iid = splits["iid_test"]
    final_accuracy = _probe_accuracy(train.nuisance_only[:, -1].reshape(len(train.y), -1), train.y, iid.nuisance_only[:, -1].reshape(len(iid.y), -1), iid.y)
    motion_accuracy = _probe_accuracy(_circular_column_slope(train.nuisance_only), train.y, _circular_column_slope(iid.nuisance_only), iid.y)
    assert final_accuracy <= 0.65
    assert motion_accuracy >= 0.85
    assert train.metadata["benchmark_variant"] == "endpoint_matched"


def test_overlay_order_pulse_shape():
    rng = np.random.default_rng(0)
    core = rng.normal(size=(20, 10, 50)).astype(np.float32)
    y = np.array([0, 1] * 10)
    selected_core, nuisance, selected_labels, direction = overlay_order_pulse(core, y, rng, 0.97, 8)
    assert selected_core.shape == (8, 10, 50)
    assert nuisance.shape == (8, 10, 50)
    assert selected_labels.shape == (8,)
    assert set(np.unique(direction)).issubset({0, 1})


def test_make_fl_trail_accepts_overlapping_fields():
    splits = make("fl_trail", seed=0, sizes=dict(n_train=8, n_val_iid=4, n_iid_test=4, n_ood_test=4), extra={"nuisance_trail_decay": 0.78, "n_train": 8})
    assert splits["train"].mixed.shape[0] == 8
    assert splits["train"].metadata["nuisance_trail_decay"] == 0.78


def test_make_oe_strict_two_channel():
    splits = make("oe_strict", seed=0, sizes=dict(n_train=32, n_val_iid=8, n_iid_test=8, n_ood_test=8))
    mixed = splits["train"].mixed
    assert mixed.shape[2] == 2
    np.testing.assert_allclose(mixed[:, 0, 1], mixed[:, -1, 1])
