import numpy as np

from src.data.counterfactual import counterfactual_mode_to_spurious_mode, reuse_or_resample_noise


def test_counterfactual_mode_mapping():
    assert counterfactual_mode_to_spurious_mode("correlated", "randomized") == "randomized"
    assert counterfactual_mode_to_spurious_mode("correlated", "reversed") == "reversed"
    assert (
        counterfactual_mode_to_spurious_mode("correlated", "independent_same_marginal")
        == "independent_same_marginal"
    )
    assert (
        counterfactual_mode_to_spurious_mode("correlated", "resample_same_mode")
        == "correlated"
    )


def test_noise_reuse_or_resample():
    noise = np.ones((2, 3, 4), dtype=np.float32)
    reused = reuse_or_resample_noise(noise, reuse_noise=True, seed=0, noise_std=0.1)
    resampled = reuse_or_resample_noise(noise, reuse_noise=False, seed=0, noise_std=0.1)
    assert np.array_equal(reused, noise)
    assert not np.array_equal(resampled, noise)
    assert resampled.shape == noise.shape
