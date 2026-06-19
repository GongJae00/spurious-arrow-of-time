import numpy as np

from src.data.irreversible_source_inference import (
    IrreversibleSourceConfig,
    generate_irreversible_source_splits,
)


def small_config() -> IrreversibleSourceConfig:
    return IrreversibleSourceConfig(
        grid_size=12,
        length=6,
        n_train=64,
        n_val_iid=32,
        n_iid_test=32,
        n_ood_test=32,
        seed=123,
        diffusion_start_step=5,
        diffusion_steps_between_frames=3,
    )


def test_split_shapes_and_fields() -> None:
    splits = generate_irreversible_source_splits(small_config())
    train = splits["train"]
    assert train.core_only.shape == (64, 6, 12, 12)
    assert train.nuisance_only.shape == train.core_only.shape
    assert train.nuisance_counterfactual.shape == train.core_only.shape
    assert train.mixed.shape == train.core_only.shape
    assert train.counterfactual.shape == train.core_only.shape
    assert train.y.shape == (64,)
    assert train.source_index.shape == (64,)
    assert train.source_center.shape == (64, 2)
    assert set(np.unique(train.source_orientation)).issubset({0, 1})
    assert set(np.unique(train.nuisance_direction)).issubset({-1, 1})


def test_reproducible_generation() -> None:
    cfg = small_config()
    a = generate_irreversible_source_splits(cfg)["train"]
    b = generate_irreversible_source_splits(cfg)["train"]
    np.testing.assert_allclose(a.core_only, b.core_only)
    np.testing.assert_allclose(a.nuisance_only, b.nuisance_only)
    np.testing.assert_array_equal(a.y, b.y)


def test_nuisance_direction_reverses_ood() -> None:
    cfg = small_config()
    splits = generate_irreversible_source_splits(cfg)
    train_corr = np.corrcoef(splits["train"].y, splits["train"].nuisance_direction)[0, 1]
    ood_corr = np.corrcoef(splits["ood_test"].y, splits["ood_test"].nuisance_direction)[0, 1]
    assert train_corr > 0.7
    assert ood_corr < -0.7


def test_randomized_ood_correlation_near_zero() -> None:
    cfg = small_config()
    cfg = cfg.__class__(**{**cfg.__dict__, "ood_mode": "randomized", "n_ood_test": 512})
    splits = generate_irreversible_source_splits(cfg)
    ood_corr = np.corrcoef(splits["ood_test"].y, splits["ood_test"].nuisance_direction)[0, 1]
    assert abs(ood_corr) < 0.2


def test_partial_shift_correlation_target() -> None:
    cfg = small_config()
    cfg = cfg.__class__(
        **{
            **cfg.__dict__,
            "ood_mode": "partial_shift",
            "partial_shift_target_correlation": -0.25,
            "n_ood_test": 512,
        }
    )
    splits = generate_irreversible_source_splits(cfg)
    ood_corr = np.corrcoef(splits["ood_test"].y, splits["ood_test"].nuisance_direction)[0, 1]
    assert abs(ood_corr - (-0.25)) < 0.2


def test_no_spurious_correlation_train_mode() -> None:
    cfg = small_config()
    cfg = cfg.__class__(
        **{
            **cfg.__dict__,
            "train_nuisance_mode": "randomized",
            "ood_mode": "randomized",
            "n_train": 512,
            "n_iid_test": 512,
            "n_ood_test": 512,
        }
    )
    splits = generate_irreversible_source_splits(cfg)
    for split_name in ["train", "iid_test", "ood_test"]:
        split = splits[split_name]
        corr = np.corrcoef(split.y, split.nuisance_direction)[0, 1]
        assert abs(corr) < 0.2


def test_counterfactual_changes_nuisance_only() -> None:
    split = generate_irreversible_source_splits(small_config())["train"]
    assert np.mean(split.nuisance_direction != split.counterfactual_direction) > 0.95
    assert np.mean(np.abs(split.mixed - split.counterfactual)) > 0.01
    assert split.metadata["class_balance"]["0"] == 0.5
    assert split.metadata["class_balance"]["1"] == 0.5


def test_disable_nuisance_replaces_mixed_with_core_only() -> None:
    cfg = small_config()
    cfg = cfg.__class__(**{**cfg.__dict__, "disable_nuisance": True})
    split = generate_irreversible_source_splits(cfg)["train"]
    np.testing.assert_allclose(split.mixed, split.core_only)
    np.testing.assert_allclose(split.counterfactual, split.core_only)
