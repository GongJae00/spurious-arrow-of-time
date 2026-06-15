import numpy as np
import matplotlib.image as mpimg

from src.data.ink_advection_diffusion import (
    generate_ink_advection_diffusion_bench,
    generate_ink_advection_diffusion_splits,
)
from src.eval.ink_advection_diffusion_diagnostics import _save_visuals, diagnose_splits
from src.train.common import generate_splits_from_config


def test_ink_advection_diffusion_splits_shapes_and_threshold_reuse():
    splits = generate_ink_advection_diffusion_splits(
        n_train=128,
        n_val_iid=64,
        n_iid_test=64,
        n_ood_test=64,
        length=8,
        grid_size=16,
        seed=11,
    )

    assert set(splits) == {"train", "val_iid", "iid_test", "ood_test"}
    threshold = splits["train"]["metadata"]["core"]["label_threshold"]
    for split_name, split in splits.items():
        assert split["x"].shape[1:] == (8, 16 * 16)
        assert split["x_cf"].shape == split["x"].shape
        assert split["y"].shape == (split["x"].shape[0],)
        assert np.array_equal(split["y"], split["y_cf"])
        assert split["metadata"]["benchmark_name"] == "ink_advection_diffusion"
        assert split["metadata"]["benchmark_version"] == "ink_advection_diffusion"
        assert split["metadata"]["n_transitions"] == 7
        assert split["metadata"]["core"]["label_threshold"] == threshold
        if split_name == "train":
            assert split["metadata"]["core"]["label_threshold_source"] == "local"
        else:
            assert split["metadata"]["core"]["label_threshold_source"] == "train"


def test_ink_advection_diffusion_physical_sanity():
    split = generate_ink_advection_diffusion_bench(
        n_sequences=96,
        length=10,
        grid_size=18,
        split="train",
        spurious_mode="correlated",
        seed=12,
        return_fields=True,
    )
    metadata = split["metadata"]
    physics = metadata["physics"]
    ink = metadata["ink_advection_diffusion"]

    assert physics["core_mass_relative_error_max"] < 1.0e-3
    assert physics["spurious_mass_relative_error_max"] < 1.0e-3
    assert physics["core_min_concentration"] >= -1.0e-7
    assert physics["spurious_min_concentration"] >= -1.0e-7
    assert ink["core_spread_final_mean"] > ink["core_spread_initial_mean"]
    assert ink["core_entropy_final_mean"] > ink["core_entropy_initial_mean"]
    assert ink["signal_to_noise_std_ratio"] > 1.0
    assert split["core_field"].shape == (96, 10, 18, 18)
    assert split["spurious_field"].shape == (96, 10, 18, 18)


def test_ink_advection_diffusion_spurious_dynamic_correlation_reverses_ood():
    splits = generate_ink_advection_diffusion_splits(
        n_train=256,
        n_val_iid=128,
        n_iid_test=128,
        n_ood_test=128,
        length=8,
        grid_size=16,
        seed=13,
    )

    train_corr = splits["train"]["metadata"]["spurious"]["corr_y_spurious_dynamic_stat"]
    iid_corr = splits["iid_test"]["metadata"]["spurious"]["corr_y_spurious_dynamic_stat"]
    ood_corr = splits["ood_test"]["metadata"]["spurious"]["corr_y_spurious_dynamic_stat"]
    assert train_corr > 0.95
    assert iid_corr > 0.95
    assert ood_corr < -0.95


def test_ink_advection_diffusion_counterfactual_preserves_core_and_breaks_shortcut():
    split = generate_ink_advection_diffusion_bench(
        n_sequences=128,
        length=8,
        grid_size=16,
        split="train",
        spurious_mode="correlated",
        spurious_cf_mode="randomized",
        seed=14,
        return_fields=True,
    )

    assert np.array_equal(split["y"], split["y_cf"])
    assert split["metadata"]["counterfactual"]["preserves_y"] is True
    assert split["metadata"]["counterfactual"]["preserves_core_stat"] is True
    assert split["metadata"]["counterfactual"]["preserves_core_field"] is True
    assert split["metadata"]["counterfactual"]["changes_spurious_stat"] is True
    assert abs(split["metadata"]["counterfactual"]["corr_y_spurious_cf_dynamic_stat"]) < 0.25
    assert np.mean(np.abs(split["x"] - split["x_cf"])) > 0.001
    assert np.mean(np.abs(split["spurious_field"] - split["spurious_cf_field"])) > 0.001


def test_ink_advection_diffusion_data_diagnostics_pass_on_smoke_splits():
    splits = generate_ink_advection_diffusion_splits(
        n_train=128,
        n_val_iid=64,
        n_iid_test=64,
        n_ood_test=64,
        length=8,
        grid_size=16,
        seed=15,
    )

    report = diagnose_splits(splits)
    assert report["pass"] is True
    assert report["quality_gates"]["mass_conservation"] is True
    assert report["quality_gates"]["dynamic_spurious_corr_ood_reversed"] is True
    assert report["quality_gates"]["spurious_rule_breaks_ood"] is True
    assert report["quality_gates"]["counterfactual_preserves_core_and_label"] is True
    assert "inverse_ambiguity_claim_ready" in report["inverse_ambiguity_diagnostics"]


def test_ink_advection_diffusion_integrates_with_training_split_loader():
    cfg = {
        "seed": 16,
        "benchmark_name": "ink_advection_diffusion",
        "splits": {
            "train": {"n_sequences": 32, "spurious_mode": "correlated"},
            "val_iid": {"n_sequences": 16, "spurious_mode": "correlated"},
            "iid_test": {"n_sequences": 16, "spurious_mode": "correlated"},
            "ood_test": {"n_sequences": 16, "spurious_mode": "reversed"},
        },
        "data": {
            "length": 8,
            "grid_size": 16,
        },
    }

    splits = generate_splits_from_config(cfg)
    assert splits["train"]["metadata"]["benchmark_name"] == "ink_advection_diffusion"
    assert splits["train"]["x"].shape[0] == 32
    assert splits["val_iid"]["x"].shape[0] == 16
    assert splits["ood_test"]["metadata"]["spurious"]["spurious_mode"] == "reversed"


def test_ink_advection_diffusion_visuals_are_written(tmp_path):
    splits = generate_ink_advection_diffusion_splits(
        n_train=16,
        n_val_iid=8,
        n_iid_test=8,
        n_ood_test=16,
        length=6,
        grid_size=12,
        seed=17,
        return_fields=True,
    )

    _save_visuals(splits, tmp_path)

    for name in ("train_y1_diagnostic_sheet", "ood_test_y1_diagnostic_sheet"):
        png = tmp_path / f"{name}.png"
        pdf = tmp_path / f"{name}.pdf"
        assert png.exists()
        assert png.stat().st_size > 0
        image = mpimg.imread(png)
        assert image.shape[0] >= 1000
        assert image.shape[1] >= 1000
        assert pdf.exists()
        assert pdf.stat().st_size > 0
    assert not list(tmp_path.glob("*contact_sheet*"))
