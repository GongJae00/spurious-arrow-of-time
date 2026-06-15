"""STA-Bench synthetic sequence generator.

STA-Bench keeps the core process label-generating and makes the nuisance process
spuriously correlated through dynamic trajectory properties, not static initial
state shortcuts in the main setting.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

from src.data.biased_ring import (
    net_clockwise_displacement,
    path_step_deltas,
    ring_transition_metadata,
    sample_biased_ring,
)
from src.data.counterfactual import counterfactual_mode_to_spurious_mode, reuse_or_resample_noise
from src.data.splits import SPLIT_ORDER, default_spurious_mode_for_split, validate_split


SPURIOUS_MODES = {"correlated", "reversed", "randomized"}
OOD_SHIFT_TYPES = {"reversed", "randomized", "attenuated", "mixed"}
SPURIOUS_CORRELATION_TYPES = {
    "drift_direction",
    "net_displacement",
    "trajectory_arrow_statistic",
    "initial_sector_static_control",
}
LABEL_MODES = {
    "core_net_displacement_median_threshold",
    "mean_sine_median_threshold",
    "final_state_set",
}
TRAJECTORY_ARROW_STATISTICS = {
    "net_clockwise_displacement",
    "realized_forward_fraction",
    "signed_step_sum",
}


@dataclass(frozen=True)
class STABenchSeeds:
    dataset_seed: int
    core_seed: int
    spurious_seed: int
    spurious_cf_seed: int
    noise_seed: int
    mixing_seed: int

    @classmethod
    def from_base(cls, seed: int, split_index: int = 0) -> "STABenchSeeds":
        base = int(seed) + 10_000 * split_index
        return cls(
            dataset_seed=base,
            core_seed=base + 101,
            spurious_seed=base + 202,
            spurious_cf_seed=base + 303,
            noise_seed=base + 404,
            mixing_seed=int(seed) + 505,
        )

    def as_dict(self) -> dict[str, int]:
        return {
            "dataset_seed": self.dataset_seed,
            "core_seed": self.core_seed,
            "spurious_seed": self.spurious_seed,
            "spurious_cf_seed": self.spurious_cf_seed,
            "noise_seed": self.noise_seed,
            "mixing_seed": self.mixing_seed,
        }


def _validate_common(
    n_sequences: int,
    length: int,
    n_core_states: int,
    n_spur_states: int,
    split: str,
    spurious_mode: str,
    spurious_correlation_type: str,
    label_mode: str,
    trajectory_arrow_statistic: str,
) -> None:
    if n_sequences <= 0:
        raise ValueError("n_sequences must be positive")
    if length < 2:
        raise ValueError("length L must be at least 2")
    if n_core_states < 3 or n_spur_states < 3:
        raise ValueError("ring state counts must be at least 3")
    validate_split(split)
    if spurious_mode not in SPURIOUS_MODES:
        raise ValueError(f"spurious_mode must be one of {sorted(SPURIOUS_MODES)}")
    if spurious_correlation_type not in SPURIOUS_CORRELATION_TYPES:
        raise ValueError(
            "spurious_correlation_type must be one of "
            f"{sorted(SPURIOUS_CORRELATION_TYPES)}"
        )
    if label_mode not in LABEL_MODES:
        raise ValueError(f"label_mode must be one of {sorted(LABEL_MODES)}")
    if trajectory_arrow_statistic not in TRAJECTORY_ARROW_STATISTICS:
        raise ValueError(
            "trajectory_arrow_statistic must be one of "
            f"{sorted(TRAJECTORY_ARROW_STATISTICS)}"
        )


def _validate_correlation_strength(value: float) -> float:
    if not np.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError("spurious_label_correlation_strength must be in [0, 1]")
    return float(value)


def _validate_probability(name: str, value: float) -> float:
    if not np.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return float(value)


def make_mixing_matrix(
    obs_dim: int,
    n_core_states: int,
    n_spur_states: int,
    seed: int,
    normalize_columns: bool = True,
) -> np.ndarray:
    if obs_dim <= 0:
        raise ValueError("obs_dim must be positive")
    rng = np.random.default_rng(seed)
    matrix = rng.normal(0.0, 1.0, size=(obs_dim, n_core_states + n_spur_states)).astype(
        np.float32
    )
    if normalize_columns:
        norms = np.linalg.norm(matrix, axis=0, keepdims=True)
        matrix = matrix / np.maximum(norms, 1e-12)
    return matrix.astype(np.float32)


def mixing_matrix_hash(matrix: np.ndarray) -> str:
    arr = np.ascontiguousarray(matrix.astype(np.float32))
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


def _one_hot(states: np.ndarray, n_states: int) -> np.ndarray:
    return np.eye(n_states, dtype=np.float32)[states]


def _core_scores(states: np.ndarray, n_states: int, label_mode: str) -> np.ndarray:
    if label_mode == "core_net_displacement_median_threshold":
        return net_clockwise_displacement(states, n_states).astype(np.float32)
    if label_mode == "mean_sine_median_threshold":
        angles = 2.0 * np.pi * states / float(n_states)
        return np.sin(angles).mean(axis=1).astype(np.float32)
    if label_mode == "final_state_set":
        return (states[:, -1] < (n_states // 2)).astype(np.float32)
    raise ValueError(f"unknown label_mode {label_mode!r}")


def _labels_from_scores(
    scores: np.ndarray,
    threshold: float | None,
    *,
    allow_balance_fallback: bool,
    label_noise: float = 0.0,
    seed: int | None = None,
) -> tuple[np.ndarray, float, bool]:
    label_noise = _validate_probability("label_noise", label_noise)
    if threshold is None:
        threshold = float(np.median(scores))
    y = (scores > threshold).astype(np.int64)
    fallback_used = False
    if y.min() == y.max():
        if not allow_balance_fallback:
            return y, float(threshold), fallback_used
        # Deterministic fallback for pathological ties; still depends only on core scores/order.
        order = np.argsort(scores, kind="mergesort")
        y = np.zeros_like(y)
        y[order[len(order) // 2 :]] = 1
        fallback_used = True
    if label_noise > 0.0:
        if seed is None:
            raise ValueError("seed is required when label_noise > 0")
        rng = np.random.default_rng(seed)
        flips = rng.random(y.shape[0]) < label_noise
        y = np.where(flips, 1 - y, y).astype(np.int64)
    return y, float(threshold), fallback_used


def _class_balance(y: np.ndarray) -> dict[str, float]:
    return {
        "n": int(y.shape[0]),
        "n0": int((y == 0).sum()),
        "n1": int((y == 1).sum()),
        "p1": float((y == 1).mean()),
    }


def _sample_spurious_by_label(
    y: np.ndarray,
    length: int,
    n_states: int,
    p_forward: float,
    p_backward: float,
    seed: int,
    mode: str,
    correlation_type: str,
    initial_state_mode: str,
    correlation_strength: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample nuisance chains and return states, signed drift, and orientation signal.

    `correlation_strength` is a continuous knob:
      1.0 means the label fully determines the spurious orientation.
      0.0 means the spurious orientation is randomized independent of the label.
    For `reversed`, the label-conditioned orientation is flipped before applying
    the same strength.
    """

    if mode not in SPURIOUS_MODES:
        raise ValueError(f"unknown spurious mode {mode!r}")
    correlation_strength = _validate_correlation_strength(correlation_strength)
    rng = np.random.default_rng(seed)

    def label_conditioned_bool(true_when_y1: bool) -> np.ndarray:
        desired = (y == 1) if true_when_y1 else (y == 0)
        if mode == "randomized":
            return rng.choice(np.array([False, True]), size=y.shape[0])
        keep_desired = rng.random(y.shape[0]) < ((1.0 + correlation_strength) / 2.0)
        return np.where(keep_desired, desired, ~desired)

    if correlation_type == "initial_sector_static_control":
        lower_hi = max(1, n_states // 2)
        upper_lo = lower_hi
        if upper_lo >= n_states:
            upper_lo = 0
        if mode == "correlated":
            use_upper = label_conditioned_bool(true_when_y1=True)
        elif mode == "reversed":
            use_upper = label_conditioned_bool(true_when_y1=False)
        else:
            use_upper = rng.choice(np.array([False, True]), size=y.shape[0])
        initial_states = np.empty(y.shape[0], dtype=np.int64)
        lower_count = int((~use_upper).sum())
        upper_count = int(use_upper.sum())
        initial_states[~use_upper] = rng.integers(0, lower_hi, size=lower_count)
        initial_states[use_upper] = rng.integers(upper_lo, n_states, size=upper_count)
        states = sample_biased_ring(
            n_sequences=y.shape[0],
            length=length,
            n_states=n_states,
            p_forward=p_forward,
            p_backward=p_backward,
            seed=seed + 17,
            initial_state_mode="provided",
            initial_states=initial_states,
        )
        signed_drift = np.full(y.shape[0], p_forward - p_backward, dtype=np.float32)
        orientation_signal = np.where(use_upper, 1, -1).astype(np.int64)
        return states, signed_drift, orientation_signal

    if mode == "correlated":
        positive = label_conditioned_bool(true_when_y1=True)
        orientation = np.where(positive, 1, -1)
    elif mode == "reversed":
        positive = label_conditioned_bool(true_when_y1=False)
        orientation = np.where(positive, 1, -1)
    else:
        orientation = rng.choice(np.array([-1, 1], dtype=np.int64), size=y.shape[0])

    states = np.empty((y.shape[0], length), dtype=np.int64)
    signed_drift = np.empty(y.shape[0], dtype=np.float32)
    for idx, sign in enumerate((-1, 1)):
        mask = orientation == sign
        if not np.any(mask):
            continue
        if sign == 1:
            p, q = p_forward, p_backward
        else:
            p, q = p_backward, p_forward
        sampled = sample_biased_ring(
            n_sequences=int(mask.sum()),
            length=length,
            n_states=n_states,
            p_forward=p,
            p_backward=q,
            seed=int(seed + 17 * (idx + 1)),
            initial_state_mode=initial_state_mode,
        )
        states[mask] = sampled
        signed_drift[mask] = p - q
    return states, signed_drift, orientation


def _observe(
    c: np.ndarray,
    s: np.ndarray,
    mixing_matrix: np.ndarray,
    noise: np.ndarray,
    n_core_states: int,
    n_spur_states: int,
    core_scale: float,
    spur_scale: float,
    core_dropout_prob: float = 0.0,
    spur_dropout_prob: float = 0.0,
    core_mask: np.ndarray | None = None,
    spur_mask: np.ndarray | None = None,
) -> np.ndarray:
    core_dropout_prob = _validate_probability("core_dropout_prob", core_dropout_prob)
    spur_dropout_prob = _validate_probability("spur_dropout_prob", spur_dropout_prob)
    core = core_scale * _one_hot(c, n_core_states)
    spur = spur_scale * _one_hot(s, n_spur_states)
    if core_dropout_prob > 0.0:
        if core_mask is None:
            raise ValueError("core_mask is required when core_dropout_prob > 0")
        core = core * core_mask
    if spur_dropout_prob > 0.0:
        if spur_mask is None:
            raise ValueError("spur_mask is required when spur_dropout_prob > 0")
        spur = spur * spur_mask
    h = np.concatenate([core, spur], axis=-1)
    x = np.einsum("oi,nli->nlo", mixing_matrix, h, optimize=True) + noise
    return x.astype(np.float32)


def _observation_mask(
    n_sequences: int,
    length: int,
    drop_prob: float,
    seed: int,
) -> np.ndarray | None:
    drop_prob = _validate_probability("observation_dropout_prob", drop_prob)
    if drop_prob <= 0.0:
        return None
    rng = np.random.default_rng(seed)
    return (rng.random((n_sequences, length, 1)) >= drop_prob).astype(np.float32)


def _safe_corr(y: np.ndarray, values: np.ndarray) -> float:
    y_f = y.astype(np.float64)
    v_f = values.astype(np.float64)
    if np.std(y_f) < 1e-12 or np.std(v_f) < 1e-12:
        return float("nan")
    return float(np.corrcoef(y_f, v_f)[0, 1])


def _safe_auc(y: np.ndarray, values: np.ndarray) -> float:
    if np.unique(y).shape[0] < 2 or np.unique(values).shape[0] < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y, values))
    except ValueError:
        return float("nan")


def _mean_by_y(y: np.ndarray, values: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    for cls in (0, 1):
        mask = y == cls
        out[str(cls)] = float(values[mask].mean()) if np.any(mask) else float("nan")
    return out


def _numeric_summary(values: np.ndarray) -> dict[str, float]:
    values = values.astype(np.float64)
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def _orientation_diagnostics(
    y: np.ndarray,
    orientation: np.ndarray,
    mode: str,
) -> dict[str, Any]:
    if mode == "correlated":
        desired = np.where(y == 1, 1, -1)
    elif mode == "reversed":
        desired = np.where(y == 1, -1, 1)
    else:
        desired = np.zeros_like(orientation)
    if mode not in {"correlated", "reversed"}:
        match_rate = float("nan")
    else:
        match_rate = float((orientation == desired).mean())
    return {
        "orientation_signal_summary": _numeric_summary(orientation.astype(np.float32)),
        "orientation_match_rate": match_rate,
        "mean_orientation_signal_by_y": _mean_by_y(y, orientation.astype(np.float32)),
    }


def _trajectory_statistic(
    states: np.ndarray,
    n_states: int,
    statistic: str,
) -> np.ndarray:
    deltas = net_clockwise_displacement(states, n_states).astype(np.float32)
    if statistic in {"net_clockwise_displacement", "signed_step_sum"}:
        return deltas
    if statistic == "realized_forward_fraction":
        steps = path_step_deltas(states, n_states)
        return (steps == 1).mean(axis=1).astype(np.float32)
    raise ValueError(f"unknown trajectory_arrow_statistic {statistic!r}")


def _dynamic_stats(
    y: np.ndarray,
    core_states: np.ndarray,
    spur_states: np.ndarray,
    core_n_states: int,
    spur_n_states: int,
    spurious_signed_drift: np.ndarray,
    spurious_correlation_type: str,
    trajectory_arrow_statistic: str,
) -> dict[str, Any]:
    core_stat = net_clockwise_displacement(core_states, core_n_states).astype(np.float32)
    spur_net = net_clockwise_displacement(spur_states, spur_n_states).astype(np.float32)
    if spurious_correlation_type == "drift_direction":
        spur_stat = spurious_signed_drift.astype(np.float32)
        stat_name = "signed_drift"
    elif spurious_correlation_type == "net_displacement":
        spur_stat = spur_net
        stat_name = "net_clockwise_displacement"
    elif spurious_correlation_type == "trajectory_arrow_statistic":
        spur_stat = _trajectory_statistic(spur_states, spur_n_states, trajectory_arrow_statistic)
        stat_name = trajectory_arrow_statistic
    else:
        spur_stat = spur_net
        stat_name = "static_control_net_displacement"
    return {
        "spurious_dynamic_stat": spur_stat,
        "core_dynamic_stat": core_stat,
        "spurious_dynamic_stat_name": stat_name,
        "corr_y_spurious_dynamic_stat": _safe_corr(y, spur_stat),
        "corr_y_core_dynamic_stat": _safe_corr(y, core_stat),
        "auc_y_from_spurious_dynamic_stat": _safe_auc(y, spur_stat),
        "auc_y_from_core_dynamic_stat": _safe_auc(y, core_stat),
        "mean_spurious_dynamic_stat_by_y": _mean_by_y(y, spur_stat),
        "mean_core_dynamic_stat_by_y": _mean_by_y(y, core_stat),
    }


def _permute_spurious_same_marginal(
    y: np.ndarray,
    s: np.ndarray,
    signed_drift: np.ndarray,
    orientation: np.ndarray,
    n_states: int,
    seed: int,
    spurious_correlation_type: str,
    trajectory_arrow_statistic: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Preserve empirical nuisance trajectories while reducing label association."""

    rng = np.random.default_rng(seed)
    if s.shape[0] == 1:
        perm = np.array([0], dtype=np.int64)
    else:
        original_stats = _dynamic_stats(
            y,
            s,
            s,
            n_states,
            n_states,
            signed_drift,
            spurious_correlation_type,
            trajectory_arrow_statistic,
        )
        original_corr = original_stats["corr_y_spurious_dynamic_stat"]
        original_abs_corr = abs(original_corr) if np.isfinite(original_corr) else float("inf")
        best_perm = np.arange(s.shape[0])
        best_abs_corr = original_abs_corr
        for _ in range(100):
            candidate = rng.permutation(s.shape[0])
            if np.array_equal(candidate, np.arange(s.shape[0])):
                candidate = np.roll(candidate, 1)
            candidate_stats = _dynamic_stats(
                y,
                s,
                s[candidate],
                n_states,
                n_states,
                signed_drift[candidate],
                spurious_correlation_type,
                trajectory_arrow_statistic,
            )
            corr = candidate_stats["corr_y_spurious_dynamic_stat"]
            abs_corr = abs(corr) if np.isfinite(corr) else 0.0
            if abs_corr <= best_abs_corr:
                best_abs_corr = abs_corr
                best_perm = candidate
            if best_abs_corr <= 0.05:
                break
        perm = best_perm
    return (
        s[perm].copy(),
        signed_drift[perm].copy(),
        orientation[perm].copy(),
        {
            "cf_distribution_source": "empirical_permutation",
            "preserves_empirical_spurious_marginal": True,
            "same_marginal_permutation_changed_order": bool(
                not np.array_equal(perm, np.arange(s.shape[0]))
            ),
        },
    )


def generate_sta_bench(
    n_sequences: int,
    length: int,
    n_core_states: int,
    n_spur_states: int,
    p_core: float,
    q_core: float,
    p_spur: float,
    q_spur: float,
    obs_dim: int,
    noise_std: float,
    split: str,
    spurious_mode: str,
    seed: int,
    *,
    label_mode: str = "core_net_displacement_median_threshold",
    label_threshold: float | None = None,
    spurious_correlation_type: str = "drift_direction",
    spurious_label_correlation_strength: float = 1.0,
    trajectory_arrow_statistic: str = "net_clockwise_displacement",
    mixing_matrix: np.ndarray | None = None,
    mixing_seed: int | None = None,
    normalize_mixing_columns: bool = True,
    core_scale: float = 1.0,
    spur_scale: float = 1.0,
    core_observation_dropout: float = 0.0,
    spur_observation_dropout: float = 0.0,
    label_noise: float = 0.0,
    initial_state_mode_core: str = "uniform_stationary",
    initial_state_mode_spurious: str = "uniform_stationary",
    spurious_cf_mode: str = "randomized",
    reuse_noise: bool = True,
    randomize_labels: bool = False,
    counterfactual_no_change: bool = False,
    allow_local_eval_calibration: bool = False,
) -> dict[str, Any]:
    """Generate one STA-Bench split.

    For final experiments prefer `generate_sta_splits`, which calibrates the
    core label threshold on train only and reuses one mixing matrix across all
    splits.
    """

    _validate_common(
        n_sequences,
        length,
        n_core_states,
        n_spur_states,
        split,
        spurious_mode,
        spurious_correlation_type,
        label_mode,
        trajectory_arrow_statistic,
    )
    spurious_label_correlation_strength = _validate_correlation_strength(
        spurious_label_correlation_strength
    )
    core_observation_dropout = _validate_probability(
        "core_observation_dropout", core_observation_dropout
    )
    spur_observation_dropout = _validate_probability(
        "spur_observation_dropout", spur_observation_dropout
    )
    label_noise = _validate_probability("label_noise", label_noise)
    seeds = STABenchSeeds.from_base(seed, SPLIT_ORDER.index(split))
    if mixing_seed is not None:
        seeds = STABenchSeeds(
            dataset_seed=seeds.dataset_seed,
            core_seed=seeds.core_seed,
            spurious_seed=seeds.spurious_seed,
            spurious_cf_seed=seeds.spurious_cf_seed,
            noise_seed=seeds.noise_seed,
            mixing_seed=int(mixing_seed),
        )

    if split != "train" and label_threshold is None and not allow_local_eval_calibration:
        raise ValueError(
            "label_threshold is required for non-train splits. Use generate_sta_splits "
            "so evaluation splits reuse the train threshold, or set "
            "allow_local_eval_calibration=True only for explicit diagnostics."
        )

    c = sample_biased_ring(
        n_sequences,
        length,
        n_core_states,
        p_core,
        q_core,
        seeds.core_seed,
        initial_state_mode=initial_state_mode_core,
    )
    core_scores = _core_scores(c, n_core_states, label_mode)
    if label_threshold is None:
        label_threshold_source = (
            "local_calibration" if split == "train" else "local_calibration_diagnostic"
        )
    else:
        label_threshold_source = "train" if split != "train" else "explicit"
    y, threshold, label_fallback_used = _labels_from_scores(
        core_scores,
        label_threshold,
        allow_balance_fallback=label_threshold is None,
        label_noise=label_noise,
        seed=seeds.dataset_seed + 707,
    )
    original_class_balance = _class_balance(y)
    if randomize_labels:
        rng_labels = np.random.default_rng(seeds.dataset_seed + 909)
        y = rng_labels.permutation(y)

    s, spur_signed_drift, spur_orientation = _sample_spurious_by_label(
        y,
        length,
        n_spur_states,
        p_spur,
        q_spur,
        seeds.spurious_seed,
        spurious_mode,
        spurious_correlation_type,
        initial_state_mode_spurious,
        spurious_label_correlation_strength,
    )

    if counterfactual_no_change:
        cf_split_mode = spurious_mode
        s_cf = s.copy()
        spur_cf_signed_drift = spur_signed_drift.copy()
        spur_cf_orientation = spur_orientation.copy()
        cf_extra = {
            "cf_distribution_source": "no_change",
            "preserves_empirical_spurious_marginal": True,
        }
    else:
        cf_split_mode = counterfactual_mode_to_spurious_mode(spurious_mode, spurious_cf_mode)
        if cf_split_mode == "independent_same_marginal":
            s_cf, spur_cf_signed_drift, spur_cf_orientation, cf_extra = (
                _permute_spurious_same_marginal(
                    y,
                    s,
                    spur_signed_drift,
                    spur_orientation,
                    n_spur_states,
                    seeds.spurious_cf_seed,
                    spurious_correlation_type,
                    trajectory_arrow_statistic,
                )
            )
        else:
            s_cf, spur_cf_signed_drift, spur_cf_orientation = _sample_spurious_by_label(
                y,
                length,
                n_spur_states,
                p_spur,
                q_spur,
                seeds.spurious_cf_seed,
                cf_split_mode,
                spurious_correlation_type,
                initial_state_mode_spurious,
                spurious_label_correlation_strength,
            )
            cf_extra = {
                "cf_distribution_source": f"sample_{cf_split_mode}",
                "preserves_empirical_spurious_marginal": False,
            }

    if mixing_matrix is None:
        mixing_matrix = make_mixing_matrix(
            obs_dim,
            n_core_states,
            n_spur_states,
            seeds.mixing_seed,
            normalize_columns=normalize_mixing_columns,
        )
    else:
        mixing_matrix = np.asarray(mixing_matrix, dtype=np.float32)
        expected_shape = (obs_dim, n_core_states + n_spur_states)
        if mixing_matrix.shape != expected_shape:
            raise ValueError(f"mixing_matrix must have shape {expected_shape}")

    rng_noise = np.random.default_rng(seeds.noise_seed)
    noise = rng_noise.normal(0.0, noise_std, size=(n_sequences, length, obs_dim)).astype(
        np.float32
    )
    if counterfactual_no_change:
        noise_cf = noise.copy()
    else:
        noise_cf = reuse_or_resample_noise(noise, reuse_noise, seeds.noise_seed + 1, noise_std)
    core_mask = _observation_mask(
        n_sequences,
        length,
        core_observation_dropout,
        seeds.noise_seed + 701,
    )
    spur_mask = _observation_mask(
        n_sequences,
        length,
        spur_observation_dropout,
        seeds.noise_seed + 702,
    )

    x = _observe(
        c,
        s,
        mixing_matrix,
        noise,
        n_core_states,
        n_spur_states,
        core_scale,
        spur_scale,
        core_dropout_prob=core_observation_dropout,
        spur_dropout_prob=spur_observation_dropout,
        core_mask=core_mask,
        spur_mask=spur_mask,
    )
    x_cf = _observe(
        c,
        s_cf,
        mixing_matrix,
        noise_cf,
        n_core_states,
        n_spur_states,
        core_scale,
        spur_scale,
        core_dropout_prob=core_observation_dropout,
        spur_dropout_prob=spur_observation_dropout,
        core_mask=core_mask,
        spur_mask=spur_mask,
    )

    dyn = _dynamic_stats(
        y,
        c,
        s,
        n_core_states,
        n_spur_states,
        spur_signed_drift,
        spurious_correlation_type,
        trajectory_arrow_statistic,
    )
    dyn_cf = _dynamic_stats(
        y,
        c,
        s_cf,
        n_core_states,
        n_spur_states,
        spur_cf_signed_drift,
        spurious_correlation_type,
        trajectory_arrow_statistic,
    )

    core_meta = ring_transition_metadata(p_core, q_core)
    spur_meta = ring_transition_metadata(p_spur, q_spur)
    orientation_diag = _orientation_diagnostics(y, spur_orientation, spurious_mode)
    orientation_cf_diag = _orientation_diagnostics(y, spur_cf_orientation, cf_split_mode)
    effective_spurious_initial_state_mode = (
        "sector_conditioned"
        if spurious_correlation_type == "initial_sector_static_control"
        else initial_state_mode_spurious
    )
    metadata: dict[str, Any] = {
        "benchmark_name": "sta_bench",
        "benchmark_version": "sta_bench_v1",
        "split": split,
        "length_L": int(length),
        "n_transitions": int(length - 1),
        "n_core_states": int(n_core_states),
        "n_spur_states": int(n_spur_states),
        "core": {
            **core_meta,
            "initial_state_mode": initial_state_mode_core,
            "configured_initial_state_mode": initial_state_mode_core,
            "effective_initial_state_mode": initial_state_mode_core,
            "label_mode": label_mode,
            "label_threshold": float(threshold),
            "label_threshold_source": label_threshold_source,
            "label_balance_fallback_used": bool(label_fallback_used),
            "label_noise": float(label_noise),
            "label_randomized": bool(randomize_labels),
            "original_class_balance": original_class_balance,
            "class_balance": _class_balance(y),
        },
        "spurious": {
            **spur_meta,
            "initial_state_mode": effective_spurious_initial_state_mode,
            "configured_initial_state_mode": initial_state_mode_spurious,
            "effective_initial_state_mode": effective_spurious_initial_state_mode,
            "base_p_forward": float(p_spur),
            "base_p_backward": float(q_spur),
            "base_signed_drift": float(spur_meta["signed_drift"]),
            "spurious_label_correlation_strength": float(spurious_label_correlation_strength),
            "expected_orientation_match_probability": (
                float((1.0 + spurious_label_correlation_strength) / 2.0)
                if spurious_mode != "randomized"
                else 0.5
            ),
            **orientation_diag,
            "signed_drift_summary": _numeric_summary(spur_signed_drift),
            "mean_signed_drift_by_y": _mean_by_y(y, spur_signed_drift),
            "spurious_mode": spurious_mode,
            "spurious_correlation_type": spurious_correlation_type,
            "trajectory_arrow_statistic_config": trajectory_arrow_statistic,
            "spurious_dynamic_stat_name": dyn["spurious_dynamic_stat_name"],
            "corr_y_spurious_dynamic_stat": dyn["corr_y_spurious_dynamic_stat"],
            "corr_y_core_dynamic_stat": dyn["corr_y_core_dynamic_stat"],
            "auc_y_from_spurious_dynamic_stat": dyn["auc_y_from_spurious_dynamic_stat"],
            "auc_y_from_core_dynamic_stat": dyn["auc_y_from_core_dynamic_stat"],
            "mean_spurious_dynamic_stat_by_y": dyn["mean_spurious_dynamic_stat_by_y"],
            "mean_core_dynamic_stat_by_y": dyn["mean_core_dynamic_stat_by_y"],
        },
        "observation": {
            "obs_dim": int(obs_dim),
            "core_scale": float(core_scale),
            "spur_scale": float(spur_scale),
            "noise_std": float(noise_std),
            "core_observation_dropout": float(core_observation_dropout),
            "spur_observation_dropout": float(spur_observation_dropout),
            "mixing_seed": int(seeds.mixing_seed),
            "mixing_matrix_hash": mixing_matrix_hash(mixing_matrix),
            "normalize_mixing_columns": bool(normalize_mixing_columns),
        },
        "counterfactual": {
            "spurious_cf_mode": spurious_cf_mode,
            "resolved_spurious_cf_mode": cf_split_mode,
            "reuse_noise": bool(reuse_noise),
            "no_change": bool(counterfactual_no_change),
            **cf_extra,
            "orientation_match_rate": orientation_cf_diag["orientation_match_rate"],
            "mean_orientation_signal_by_y": orientation_cf_diag["mean_orientation_signal_by_y"],
            "corr_y_spurious_cf_dynamic_stat": dyn_cf["corr_y_spurious_dynamic_stat"],
            "auc_y_from_spurious_cf_dynamic_stat": dyn_cf[
                "auc_y_from_spurious_dynamic_stat"
            ],
        },
        "seeds": seeds.as_dict(),
    }

    return {
        "x": x,
        "y": y,
        "c": c,
        "s": s,
        "c_cf": c.copy(),
        "x_cf": x_cf,
        "y_cf": y.copy(),
        "s_cf": s_cf,
        "mixing_matrix": mixing_matrix.copy(),
        "noise": noise,
        "noise_cf": noise_cf,
        "metadata": metadata,
        "core_score": core_scores,
        "spurious_dynamic_stat": dyn["spurious_dynamic_stat"],
        "core_dynamic_stat": dyn["core_dynamic_stat"],
        "spurious_orientation_signal": spur_orientation,
        "spurious_cf_orientation_signal": spur_cf_orientation,
    }


def generate_sta_splits(
    *,
    n_train: int = 10_000,
    n_val_iid: int = 2_000,
    n_iid_test: int = 5_000,
    n_ood_test: int = 5_000,
    length: int = 32,
    n_core_states: int = 8,
    n_spur_states: int = 8,
    p_core: float = 0.35,
    q_core: float = 0.25,
    p_spur: float = 0.45,
    q_spur: float = 0.15,
    obs_dim: int = 16,
    noise_std: float = 0.1,
    seed: int = 0,
    label_mode: str = "core_net_displacement_median_threshold",
    spurious_correlation_type: str = "drift_direction",
    spurious_label_correlation_strength: float = 1.0,
    trajectory_arrow_statistic: str = "net_clockwise_displacement",
    ood_spurious_mode: str = "reversed",
    ood_shift_type: str = "reversed",
    ood_spurious_label_correlation_strength: float | None = None,
    normalize_mixing_columns: bool = True,
    core_scale: float = 1.0,
    spur_scale: float = 1.0,
    core_observation_dropout: float = 0.0,
    spur_observation_dropout: float = 0.0,
    label_noise: float = 0.0,
    spurious_cf_mode: str = "randomized",
    reuse_noise: bool = True,
    split_spurious_modes: dict[str, str] | None = None,
    split_spurious_correlation_strengths: dict[str, float] | None = None,
    initial_state_mode_core: str = "uniform_stationary",
    initial_state_mode_spurious: str = "uniform_stationary",
    randomize_labels: bool = False,
    counterfactual_no_change: bool = False,
) -> dict[str, dict[str, Any]]:
    """Generate train/val_iid/iid_test/ood_test with shared A and train threshold."""

    spurious_label_correlation_strength = _validate_correlation_strength(
        spurious_label_correlation_strength
    )
    if ood_shift_type not in OOD_SHIFT_TYPES:
        raise ValueError(f"ood_shift_type must be one of {sorted(OOD_SHIFT_TYPES)}")
    if ood_spurious_label_correlation_strength is not None:
        ood_spurious_label_correlation_strength = _validate_correlation_strength(
            ood_spurious_label_correlation_strength
        )

    mixing_seed = STABenchSeeds.from_base(seed).mixing_seed
    mixing = make_mixing_matrix(
        obs_dim,
        n_core_states,
        n_spur_states,
        mixing_seed,
        normalize_columns=normalize_mixing_columns,
    )

    split_sizes = {
        "train": n_train,
        "val_iid": n_val_iid,
        "iid_test": n_iid_test,
        "ood_test": n_ood_test,
    }
    split_modes = {
        "train": default_spurious_mode_for_split("train"),
        "val_iid": default_spurious_mode_for_split("val_iid"),
        "iid_test": default_spurious_mode_for_split("iid_test"),
        "ood_test": ood_spurious_mode,
    }
    split_strengths = {
        split: spurious_label_correlation_strength
        for split in SPLIT_ORDER
    }
    if ood_shift_type == "randomized":
        split_modes["ood_test"] = "randomized"
        split_strengths["ood_test"] = 0.0
    elif ood_shift_type == "attenuated":
        split_modes["ood_test"] = "correlated"
        split_strengths["ood_test"] = (
            0.2
            if ood_spurious_label_correlation_strength is None
            else ood_spurious_label_correlation_strength
        )
    elif ood_shift_type == "mixed":
        split_modes["ood_test"] = "reversed"
        split_strengths["ood_test"] = (
            0.5
            if ood_spurious_label_correlation_strength is None
            else ood_spurious_label_correlation_strength
        )
    else:
        split_modes["ood_test"] = ood_spurious_mode
        if ood_spurious_label_correlation_strength is not None:
            split_strengths["ood_test"] = ood_spurious_label_correlation_strength
    if split_spurious_modes is not None:
        split_modes.update(split_spurious_modes)
    if split_spurious_correlation_strengths is not None:
        split_strengths.update(
            {
                split: _validate_correlation_strength(value)
                for split, value in split_spurious_correlation_strengths.items()
            }
        )
    outputs: dict[str, dict[str, Any]] = {}
    train_threshold: float | None = None
    for idx, split in enumerate(SPLIT_ORDER):
        data = generate_sta_bench(
            n_sequences=split_sizes[split],
            length=length,
            n_core_states=n_core_states,
            n_spur_states=n_spur_states,
            p_core=p_core,
            q_core=q_core,
            p_spur=p_spur,
            q_spur=q_spur,
            obs_dim=obs_dim,
            noise_std=noise_std,
            split=split,
            spurious_mode=split_modes[split],
            seed=seed,
            label_mode=label_mode,
            label_threshold=train_threshold,
            spurious_correlation_type=spurious_correlation_type,
            spurious_label_correlation_strength=split_strengths[split],
            trajectory_arrow_statistic=trajectory_arrow_statistic,
            mixing_matrix=mixing,
            mixing_seed=mixing_seed,
            normalize_mixing_columns=normalize_mixing_columns,
            core_scale=core_scale,
            spur_scale=spur_scale,
            core_observation_dropout=core_observation_dropout,
            spur_observation_dropout=spur_observation_dropout,
            label_noise=label_noise,
            initial_state_mode_core=initial_state_mode_core,
            initial_state_mode_spurious=initial_state_mode_spurious,
            spurious_cf_mode=spurious_cf_mode,
            reuse_noise=reuse_noise,
            randomize_labels=randomize_labels,
            counterfactual_no_change=counterfactual_no_change,
        )
        if idx == 0:
            train_threshold = float(data["metadata"]["core"]["label_threshold"])
        data["metadata"]["spurious"]["ood_shift_type"] = (
            ood_shift_type if split == "ood_test" else "train_like"
        )
        outputs[split] = data

    return outputs
