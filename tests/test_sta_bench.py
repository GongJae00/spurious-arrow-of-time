import numpy as np
import pytest

from src.data.biased_ring import ring_entropy_production
from src.data.sta_bench import generate_sta_bench, generate_sta_splits


def test_generate_sta_splits_shapes_and_shared_mixing_matrix():
    splits = generate_sta_splits(
        n_train=128,
        n_val_iid=64,
        n_iid_test=64,
        n_ood_test=64,
        length=16,
        n_core_states=8,
        n_spur_states=8,
        obs_dim=12,
        seed=3,
    )
    assert set(splits) == {"train", "val_iid", "iid_test", "ood_test"}
    train_hash = splits["train"]["metadata"]["observation"]["mixing_matrix_hash"]
    for split, data in splits.items():
        assert data["x"].shape[1:] == (16, 12)
        assert data["c"].shape == (data["x"].shape[0], 16)
        assert data["s"].shape == (data["x"].shape[0], 16)
        assert data["x_cf"].shape == data["x"].shape
        assert data["metadata"]["observation"]["mixing_matrix_hash"] == train_hash
        assert data["metadata"]["n_transitions"] == 15


def test_train_threshold_is_reused_for_eval_splits():
    splits = generate_sta_splits(
        n_train=128,
        n_val_iid=64,
        n_iid_test=64,
        n_ood_test=64,
        length=16,
        seed=4,
    )
    threshold = splits["train"]["metadata"]["core"]["label_threshold"]
    assert splits["train"]["metadata"]["core"]["label_threshold_source"] == "local_calibration"
    for split in ("val_iid", "iid_test", "ood_test"):
        assert splits[split]["metadata"]["core"]["label_threshold"] == threshold
        assert splits[split]["metadata"]["core"]["label_threshold_source"] == "train"
        assert splits[split]["metadata"]["core"]["label_balance_fallback_used"] is False


def test_eval_split_does_not_use_label_balance_fallback():
    data = generate_sta_bench(
        n_sequences=64,
        length=16,
        n_core_states=8,
        n_spur_states=8,
        p_core=0.35,
        q_core=0.25,
        p_spur=0.45,
        q_spur=0.15,
        obs_dim=10,
        noise_std=0.1,
        split="val_iid",
        spurious_mode="correlated",
        seed=41,
        label_threshold=999.0,
    )
    expected = (data["core_score"] > 999.0).astype(np.int64)
    assert np.array_equal(data["y"], expected)
    assert data["metadata"]["core"]["label_balance_fallback_used"] is False
    assert data["metadata"]["core"]["class_balance"]["n1"] == 0


def test_eval_split_requires_train_threshold_unless_explicit_diagnostic():
    with pytest.raises(ValueError, match="label_threshold is required"):
        generate_sta_bench(
            n_sequences=32,
            length=12,
            n_core_states=8,
            n_spur_states=8,
            p_core=0.35,
            q_core=0.25,
            p_spur=0.45,
            q_spur=0.15,
            obs_dim=10,
            noise_std=0.1,
            split="iid_test",
            spurious_mode="correlated",
            seed=40,
        )

    data = generate_sta_bench(
        n_sequences=32,
        length=12,
        n_core_states=8,
        n_spur_states=8,
        p_core=0.35,
        q_core=0.25,
        p_spur=0.45,
        q_spur=0.15,
        obs_dim=10,
        noise_std=0.1,
        split="iid_test",
        spurious_mode="correlated",
        seed=40,
        allow_local_eval_calibration=True,
    )
    assert data["metadata"]["core"]["label_threshold_source"] == "local_calibration_diagnostic"


def test_label_balance_fallback_is_train_local_only():
    data = generate_sta_bench(
        n_sequences=32,
        length=8,
        n_core_states=8,
        n_spur_states=8,
        p_core=0.35,
        q_core=0.25,
        p_spur=0.45,
        q_spur=0.15,
        obs_dim=10,
        noise_std=0.1,
        split="train",
        spurious_mode="correlated",
        seed=42,
        label_mode="final_state_set",
    )
    assert data["metadata"]["core"]["label_threshold_source"] == "local_calibration"


def test_counterfactual_keeps_core_and_label_but_changes_nuisance():
    data = generate_sta_bench(
        n_sequences=128,
        length=16,
        n_core_states=8,
        n_spur_states=8,
        p_core=0.35,
        q_core=0.25,
        p_spur=0.45,
        q_spur=0.15,
        obs_dim=10,
        noise_std=0.1,
        split="train",
        spurious_mode="correlated",
        seed=5,
    )
    assert np.array_equal(data["c"], data["c_cf"])
    assert np.array_equal(data["y"], data["y_cf"])
    assert data["y"].shape == (128,)
    assert not np.array_equal(data["s"], data["s_cf"])
    assert np.mean(data["x"] != data["x_cf"]) > 0
    assert data["metadata"]["counterfactual"]["spurious_cf_mode"] == "randomized"


def test_main_trap_dynamic_correlation_and_ood_reversal():
    splits = generate_sta_splits(
        n_train=512,
        n_val_iid=256,
        n_iid_test=256,
        n_ood_test=256,
        length=24,
        p_spur=0.45,
        q_spur=0.15,
        seed=6,
        spurious_correlation_type="drift_direction",
        ood_spurious_mode="reversed",
    )
    train_corr = splits["train"]["metadata"]["spurious"]["corr_y_spurious_dynamic_stat"]
    iid_corr = splits["iid_test"]["metadata"]["spurious"]["corr_y_spurious_dynamic_stat"]
    ood_corr = splits["ood_test"]["metadata"]["spurious"]["corr_y_spurious_dynamic_stat"]
    assert train_corr > 0.5
    assert iid_corr > 0.5
    assert ood_corr < -0.5
    signed_by_y = splits["train"]["metadata"]["spurious"]["mean_signed_drift_by_y"]
    assert signed_by_y["1"] > 0.0
    assert signed_by_y["0"] < 0.0
    summary = splits["train"]["metadata"]["spurious"]["signed_drift_summary"]
    assert summary["max"] > 0.0
    assert summary["min"] < 0.0


def test_spurious_label_correlation_strength_controls_trap_strength():
    strong = generate_sta_splits(
        n_train=1024,
        n_val_iid=512,
        n_iid_test=512,
        n_ood_test=512,
        length=24,
        seed=60,
        spurious_label_correlation_strength=1.0,
    )
    weak = generate_sta_splits(
        n_train=1024,
        n_val_iid=512,
        n_iid_test=512,
        n_ood_test=512,
        length=24,
        seed=60,
        spurious_label_correlation_strength=0.5,
    )
    strong_corr = abs(
        strong["train"]["metadata"]["spurious"]["corr_y_spurious_dynamic_stat"]
    )
    weak_corr = abs(weak["train"]["metadata"]["spurious"]["corr_y_spurious_dynamic_stat"])
    assert strong_corr > weak_corr
    assert strong["train"]["metadata"]["spurious"]["orientation_match_rate"] == 1.0
    weak_match = weak["train"]["metadata"]["spurious"]["orientation_match_rate"]
    assert 0.65 <= weak_match <= 0.85
    assert (
        weak["train"]["metadata"]["spurious"]["spurious_label_correlation_strength"]
        == 0.5
    )


def test_invalid_spurious_label_correlation_strength_raises():
    with pytest.raises(ValueError, match="spurious_label_correlation_strength"):
        generate_sta_bench(
            n_sequences=32,
            length=12,
            n_core_states=8,
            n_spur_states=8,
            p_core=0.35,
            q_core=0.25,
            p_spur=0.45,
            q_spur=0.15,
            obs_dim=10,
            noise_std=0.1,
            split="train",
            spurious_mode="correlated",
            seed=61,
            spurious_label_correlation_strength=1.2,
        )


def test_label_noise_and_observation_dropout_are_logged_and_change_task():
    clean = generate_sta_bench(
        n_sequences=256,
        length=16,
        n_core_states=8,
        n_spur_states=8,
        p_core=0.35,
        q_core=0.25,
        p_spur=0.45,
        q_spur=0.15,
        obs_dim=10,
        noise_std=0.1,
        split="train",
        spurious_mode="correlated",
        seed=62,
        label_noise=0.0,
    )
    noisy = generate_sta_bench(
        n_sequences=256,
        length=16,
        n_core_states=8,
        n_spur_states=8,
        p_core=0.35,
        q_core=0.25,
        p_spur=0.45,
        q_spur=0.15,
        obs_dim=10,
        noise_std=0.1,
        split="train",
        spurious_mode="correlated",
        seed=62,
        label_noise=0.2,
        core_observation_dropout=0.25,
    )
    assert noisy["metadata"]["core"]["label_noise"] == 0.2
    assert noisy["metadata"]["observation"]["core_observation_dropout"] == 0.25
    assert not np.array_equal(clean["y"], noisy["y"])
    assert not np.array_equal(clean["x"], noisy["x"])


def test_ood_shift_variants_control_spurious_correlation_direction():
    attenuated = generate_sta_splits(
        n_train=1024,
        n_val_iid=512,
        n_iid_test=512,
        n_ood_test=512,
        length=24,
        seed=63,
        spurious_label_correlation_strength=0.8,
        ood_shift_type="attenuated",
        ood_spurious_label_correlation_strength=0.1,
    )
    randomized = generate_sta_splits(
        n_train=1024,
        n_val_iid=512,
        n_iid_test=512,
        n_ood_test=512,
        length=24,
        seed=63,
        spurious_label_correlation_strength=0.8,
        ood_shift_type="randomized",
    )
    mixed = generate_sta_splits(
        n_train=1024,
        n_val_iid=512,
        n_iid_test=512,
        n_ood_test=512,
        length=24,
        seed=63,
        spurious_label_correlation_strength=0.8,
        ood_shift_type="mixed",
        ood_spurious_label_correlation_strength=0.4,
    )
    assert attenuated["ood_test"]["metadata"]["spurious"]["spurious_mode"] == "correlated"
    assert abs(attenuated["ood_test"]["metadata"]["spurious"]["corr_y_spurious_dynamic_stat"]) < 0.2
    assert randomized["ood_test"]["metadata"]["spurious"]["spurious_mode"] == "randomized"
    assert abs(randomized["ood_test"]["metadata"]["spurious"]["corr_y_spurious_dynamic_stat"]) < 0.2
    assert mixed["ood_test"]["metadata"]["spurious"]["spurious_mode"] == "reversed"
    assert mixed["ood_test"]["metadata"]["spurious"]["corr_y_spurious_dynamic_stat"] < 0.0


def test_spurious_ep_can_be_configured_stronger_than_core_ep():
    data = generate_sta_bench(
        n_sequences=64,
        length=16,
        n_core_states=8,
        n_spur_states=8,
        p_core=0.35,
        q_core=0.25,
        p_spur=0.45,
        q_spur=0.15,
        obs_dim=10,
        noise_std=0.1,
        split="train",
        spurious_mode="correlated",
        seed=7,
    )
    core_ep = data["metadata"]["core"]["analytic_ep"]
    spur_ep = data["metadata"]["spurious"]["analytic_ep"]
    assert spur_ep > core_ep
    assert core_ep == ring_entropy_production(0.35, 0.25)
    assert spur_ep == ring_entropy_production(0.45, 0.15)


def test_split_spurious_modes_can_remove_spurious_correlation():
    splits = generate_sta_splits(
        n_train=256,
        n_val_iid=128,
        n_iid_test=128,
        n_ood_test=128,
        length=20,
        seed=8,
        split_spurious_modes={
            "train": "randomized",
            "val_iid": "randomized",
            "iid_test": "randomized",
            "ood_test": "randomized",
        },
    )
    for split in splits.values():
        corr = split["metadata"]["spurious"]["corr_y_spurious_dynamic_stat"]
        assert abs(corr) < 0.25


def test_randomized_labels_are_logged_and_break_core_label_rule():
    regular = generate_sta_bench(
        n_sequences=128,
        length=16,
        n_core_states=8,
        n_spur_states=8,
        p_core=0.35,
        q_core=0.25,
        p_spur=0.45,
        q_spur=0.15,
        obs_dim=10,
        noise_std=0.1,
        split="train",
        spurious_mode="correlated",
        seed=9,
        randomize_labels=False,
    )
    randomized = generate_sta_bench(
        n_sequences=128,
        length=16,
        n_core_states=8,
        n_spur_states=8,
        p_core=0.35,
        q_core=0.25,
        p_spur=0.45,
        q_spur=0.15,
        obs_dim=10,
        noise_std=0.1,
        split="train",
        spurious_mode="correlated",
        seed=9,
        randomize_labels=True,
    )
    assert randomized["metadata"]["core"]["label_randomized"] is True
    assert regular["metadata"]["core"]["label_randomized"] is False
    assert np.array_equal(np.sort(regular["y"]), np.sort(randomized["y"]))
    assert not np.array_equal(regular["y"], randomized["y"])


def test_no_counterfactual_change_control_returns_identical_pair():
    data = generate_sta_bench(
        n_sequences=64,
        length=16,
        n_core_states=8,
        n_spur_states=8,
        p_core=0.35,
        q_core=0.25,
        p_spur=0.45,
        q_spur=0.15,
        obs_dim=10,
        noise_std=0.1,
        split="train",
        spurious_mode="correlated",
        seed=10,
        counterfactual_no_change=True,
    )
    assert np.array_equal(data["s"], data["s_cf"])
    assert np.array_equal(data["c"], data["c_cf"])
    assert np.array_equal(data["y"], data["y_cf"])
    assert np.array_equal(data["x"], data["x_cf"])
    assert data["metadata"]["counterfactual"]["no_change"] is True


def test_static_spurious_control_uses_initial_sector_not_dynamic_direction():
    data = generate_sta_bench(
        n_sequences=512,
        length=20,
        n_core_states=8,
        n_spur_states=8,
        p_core=0.35,
        q_core=0.25,
        p_spur=0.45,
        q_spur=0.15,
        obs_dim=10,
        noise_std=0.1,
        split="train",
        spurious_mode="correlated",
        seed=11,
        spurious_correlation_type="initial_sector_static_control",
    )
    y = data["y"]
    initial_upper = data["s"][:, 0] >= 4
    assert np.mean(initial_upper[y == 1]) > 0.95
    assert np.mean(initial_upper[y == 0]) < 0.05
    dynamic_corr = data["metadata"]["spurious"]["corr_y_spurious_dynamic_stat"]
    assert abs(dynamic_corr) < 0.25
    spurious = data["metadata"]["spurious"]
    assert spurious["configured_initial_state_mode"] == "uniform_stationary"
    assert spurious["effective_initial_state_mode"] == "sector_conditioned"


def test_independent_same_marginal_preserves_empirical_marginal():
    data = generate_sta_bench(
        n_sequences=256,
        length=20,
        n_core_states=8,
        n_spur_states=8,
        p_core=0.35,
        q_core=0.25,
        p_spur=0.45,
        q_spur=0.15,
        obs_dim=10,
        noise_std=0.1,
        split="train",
        spurious_mode="correlated",
        seed=12,
        spurious_cf_mode="independent_same_marginal",
    )
    original = sorted(map(tuple, data["s"]))
    cf = sorted(map(tuple, data["s_cf"]))
    assert cf == original
    meta = data["metadata"]["counterfactual"]
    assert meta["cf_distribution_source"] == "empirical_permutation"
    assert meta["preserves_empirical_spurious_marginal"] is True
    original_corr = abs(data["metadata"]["spurious"]["corr_y_spurious_dynamic_stat"])
    cf_corr = abs(meta["corr_y_spurious_cf_dynamic_stat"])
    assert cf_corr < original_corr


def test_trajectory_arrow_statistic_logs_configured_statistic():
    data = generate_sta_bench(
        n_sequences=128,
        length=16,
        n_core_states=8,
        n_spur_states=8,
        p_core=0.35,
        q_core=0.25,
        p_spur=0.45,
        q_spur=0.15,
        obs_dim=10,
        noise_std=0.1,
        split="train",
        spurious_mode="correlated",
        seed=13,
        spurious_correlation_type="trajectory_arrow_statistic",
        trajectory_arrow_statistic="realized_forward_fraction",
    )
    spurious = data["metadata"]["spurious"]
    assert spurious["spurious_correlation_type"] == "trajectory_arrow_statistic"
    assert spurious["trajectory_arrow_statistic_config"] == "realized_forward_fraction"
    assert spurious["spurious_dynamic_stat_name"] == "realized_forward_fraction"


def test_unknown_trajectory_arrow_statistic_raises():
    with pytest.raises(ValueError, match="trajectory_arrow_statistic"):
        generate_sta_bench(
            n_sequences=32,
            length=12,
            n_core_states=8,
            n_spur_states=8,
            p_core=0.35,
            q_core=0.25,
            p_spur=0.45,
            q_spur=0.15,
            obs_dim=10,
            noise_std=0.1,
            split="train",
            spurious_mode="correlated",
            seed=14,
            spurious_correlation_type="trajectory_arrow_statistic",
            trajectory_arrow_statistic="not_a_statistic",
        )
