import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.benchmark import OE_STRICT_CUM, make, oe_strict_nuisance, overlay_order_pulse, set_mf_nuisance
from src.data import GeneratorConfig, generate_splits


def _probe_accuracy(x_train, y_train, x_test, y_test) -> float:
    probe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, solver="liblinear", random_state=0))
    probe.fit(x_train, y_train)
    return float(probe.score(x_test, y_test))


def _circular_column_slope(x: np.ndarray) -> np.ndarray:
    mass = np.clip(x, 0.0, None)
    width = x.shape[-1]
    cols = np.arange(width, dtype=np.float64)
    angle = 2.0 * np.pi * cols / width
    col_mass = mass.sum(axis=2)
    total = np.clip(col_mass.sum(axis=2), 1e-8, None)
    mean_cos = (col_mass * np.cos(angle)).sum(axis=2) / total
    mean_sin = (col_mass * np.sin(angle)).sum(axis=2) / total
    com = np.arctan2(mean_sin, mean_cos) * width / (2.0 * np.pi)
    unwrapped = np.unwrap(com * 2.0 * np.pi / width, axis=1)
    t = np.arange(x.shape[1], dtype=np.float64)
    t = t - t.mean()
    return ((unwrapped * t).sum(axis=1) / np.sum(t ** 2))[:, None]


def test_oe_strict_closed_path():
    assert list(OE_STRICT_CUM) == [0, 1, 2, 4, 7, 10, 13, 0]
    d = np.ones(32, dtype=np.int64)
    nu = oe_strict_nuisance(d, np.random.default_rng(0))
    np.testing.assert_allclose(nu[:, 0], nu[:, -1])
    dneg = -np.ones(32, dtype=np.int64)
    nu_b = oe_strict_nuisance(dneg, np.random.default_rng(0))
    np.testing.assert_allclose(nu_b[:, 0], nu_b[:, -1])


def test_set_mf_parity_homogeneity():
    rng = np.random.default_rng(1)
    d = np.array([1, 1, -1, -1] * 16)
    nu = set_mf_nuisance(d, rng)
    cols = nu.argmax(axis=3)[:, :, 8]
    homog = (cols % 2 == cols[:, :1] % 2).all(1)
    assert homog[d > 0].all()
    assert (~homog[d < 0]).all()


def test_endpoint_matched_controls_final_nuisance_leakage():
    cfg = GeneratorConfig(grid_size=12, length=6, n_train=512, n_val_iid=32, n_iid_test=512, n_ood_test=512, seed=123, diffusion_start_step=5, diffusion_steps_between_frames=3, benchmark_variant="endpoint_matched")
    splits = generate_splits(cfg)
    train = splits["train"]
    iid = splits["iid_test"]
    final_acc = _probe_accuracy(train.nuisance_only[:, -1].reshape(len(train.y), -1), train.y, iid.nuisance_only[:, -1].reshape(len(iid.y), -1), iid.y)
    motion_acc = _probe_accuracy(_circular_column_slope(train.nuisance_only), train.y, _circular_column_slope(iid.nuisance_only), iid.y)
    assert final_acc <= 0.65
    assert motion_acc >= 0.85
    assert train.metadata["benchmark_variant"] == "endpoint_matched"


def test_overlay_order_pulse_shape():
    rng = np.random.default_rng(0)
    xc = rng.normal(size=(20, 10, 50)).astype(np.float32)
    y = np.array([0, 1] * 10)
    xcs, nu, ys, d = overlay_order_pulse(xc, y, rng, 0.97, 8)
    assert xcs.shape == (8, 10, 50)
    assert nu.shape == (8, 10, 50)
    assert ys.shape == (8,)
    assert set(np.unique(d)).issubset({0, 1})


def test_make_trail_fl_accepts_overlapping_fields():
    splits = make("trail_fl", seed=0, sizes=dict(n_train=8, n_val_iid=4, n_iid_test=4, n_ood_test=4), extra={"nuisance_trail_decay": 0.78, "n_train": 8})
    assert splits["train"].mixed.shape[0] == 8
    assert splits["train"].metadata["nuisance_trail_decay"] == 0.78


def test_make_oe_strict_two_channel():
    splits = make("oe_strict", seed=0, sizes=dict(n_train=32, n_val_iid=8, n_iid_test=8, n_ood_test=8))
    mixed = splits["train"].mixed
    assert mixed.shape[2] == 2
    np.testing.assert_allclose(mixed[:, 0, 1], mixed[:, -1, 1])
