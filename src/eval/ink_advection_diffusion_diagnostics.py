"""Data-only diagnostics for Ink Advection-Diffusion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.data.ink_advection_diffusion import generate_ink_advection_diffusion_splits
from src.train.common import load_config, save_json


def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((np.asarray(y_true) == np.asarray(y_pred)).mean())


def _threshold_from_train(values: np.ndarray, y: np.ndarray) -> tuple[float, int]:
    values = np.asarray(values, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    if np.unique(y).shape[0] < 2:
        return float(np.median(values)), 1
    mean0 = float(values[y == 0].mean())
    mean1 = float(values[y == 1].mean())
    return 0.5 * (mean0 + mean1), 1 if mean1 >= mean0 else -1


def _predict(values: np.ndarray, threshold: float, direction: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if direction >= 0:
        return (values > threshold).astype(np.int64)
    return (values < threshold).astype(np.int64)


def _split_summary(split: dict[str, Any]) -> dict[str, Any]:
    meta = split["metadata"]
    physics = meta["physics"]
    core = meta["core"]
    spurious = meta["spurious"]
    observation = meta["observation"]
    return {
        "n_sequences": int(split["x"].shape[0]),
        "length_L": int(meta["length_L"]),
        "n_transitions": int(meta["n_transitions"]),
        "obs_shape": meta["obs_shape"],
        "class_balance": core["class_balance"],
        "label_threshold": core["label_threshold"],
        "label_threshold_source": core["label_threshold_source"],
        "core_entropy_initial_mean": core["entropy_initial_mean"],
        "core_entropy_final_mean": core["entropy_final_mean"],
        "core_spread_initial_mean": core["spread_initial_mean"],
        "core_spread_final_mean": core["spread_final_mean"],
        "core_source_recovery_final_mass_near_source": core[
            "source_recovery_final_mass_near_source"
        ],
        "core_mass_near_source_observed_start": core[
            "mass_near_source_observed_start"
        ],
        "core_mass_near_source_observed_final": core[
            "mass_near_source_observed_final"
        ],
        "core_mass_near_source_final_fraction": core[
            "mass_near_source_final_fraction"
        ],
        "core_peak_contrast_observed_start": core["peak_contrast_observed_start"],
        "core_peak_contrast_observed_final": core["peak_contrast_observed_final"],
        "core_peak_contrast_final_fraction": core["peak_contrast_final_fraction"],
        "core_peak_error_observed_final": core["peak_error_observed_final"],
        "core_center_error_observed_final": core["center_error_observed_final"],
        "corr_y_core_dynamic_stat": core["corr_y_core_dynamic_stat"],
        "corr_y_spurious_dynamic_stat": spurious["corr_y_spurious_dynamic_stat"],
        "auc_y_from_core_dynamic_stat": core["auc_y_from_core_dynamic_stat"],
        "auc_y_from_spurious_dynamic_stat": spurious[
            "auc_y_from_spurious_dynamic_stat"
        ],
        "corr_y_spurious_cf_dynamic_stat": meta["counterfactual"][
            "corr_y_spurious_cf_dynamic_stat"
        ],
        "cf_preserves_y": meta["counterfactual"]["preserves_y"],
        "cf_preserves_core_field": meta["counterfactual"]["preserves_core_field"],
        "cf_changes_spurious_flow": meta["counterfactual"]["changes_spurious_flow"],
        "core_mass_relative_error_mean": physics["core_mass_relative_error_mean"],
        "core_mass_relative_error_max": physics["core_mass_relative_error_max"],
        "core_min_concentration": physics["core_min_concentration"],
        "spurious_mass_relative_error_mean": physics[
            "spurious_mass_relative_error_mean"
        ],
        "spurious_mass_relative_error_max": physics[
            "spurious_mass_relative_error_max"
        ],
        "spurious_min_concentration": physics["spurious_min_concentration"],
        "signal_to_noise_std_ratio": observation["signal_to_noise_std_ratio"],
    }


def diagnose_splits(
    splits: dict[str, dict[str, Any]],
    *,
    min_signal_to_noise: float = 1.0,
    core_oracle_floor: float = 0.95,
    spurious_iid_floor: float = 0.90,
    max_source_mass_final_fraction: float = 0.85,
    max_peak_contrast_final_fraction: float = 0.85,
) -> dict[str, Any]:
    train = splits["train"]
    core_threshold, core_direction = _threshold_from_train(
        train["core_dynamic_stat"],
        train["y"],
    )
    spur_threshold, spur_direction = _threshold_from_train(
        train["spurious_dynamic_stat"],
        train["y"],
    )

    split_metrics: dict[str, Any] = {}
    for name, split in splits.items():
        y = split["y"]
        split_metrics[name] = {
            **_split_summary(split),
            "core_stat_oracle_accuracy": _accuracy(
                y,
                _predict(split["core_dynamic_stat"], core_threshold, core_direction),
            ),
            "spurious_train_rule_oracle_accuracy": _accuracy(
                y,
                _predict(
                    split["spurious_dynamic_stat"],
                    spur_threshold,
                    spur_direction,
                ),
            ),
        }

    thresholds = {
        name: split["metadata"]["core"]["label_threshold"]
        for name, split in splits.items()
    }
    train_corr = split_metrics["train"]["corr_y_spurious_dynamic_stat"]
    iid_corr = split_metrics["iid_test"]["corr_y_spurious_dynamic_stat"]
    ood_corr = split_metrics["ood_test"]["corr_y_spurious_dynamic_stat"]
    train_peak_error_final = split_metrics["train"]["core_peak_error_observed_final"]
    train_center_error_final = split_metrics["train"]["core_center_error_observed_final"]

    mass_errors = [
        split_metrics[name]["core_mass_relative_error_max"] for name in splits
    ] + [split_metrics[name]["spurious_mass_relative_error_max"] for name in splits]
    min_concentrations = [
        split_metrics[name]["core_min_concentration"] for name in splits
    ] + [split_metrics[name]["spurious_min_concentration"] for name in splits]
    signal_to_noise = [
        split_metrics[name]["signal_to_noise_std_ratio"] for name in splits
    ]

    result = {
        "schema_version": 1,
        "benchmark_name": "ink_advection_diffusion",
        "splits": split_metrics,
        "oracle_rules": {
            "core_stat_threshold": core_threshold,
            "core_stat_direction": core_direction,
            "spurious_train_rule_threshold": spur_threshold,
            "spurious_train_rule_direction": spur_direction,
        },
        "quality_gates": {
            "train_threshold_reused": bool(len(set(thresholds.values())) == 1),
            "mass_conservation": bool(max(mass_errors) <= 1e-3),
            "nonnegative_concentration": bool(min(min_concentrations) >= -1e-6),
            "spread_increase": bool(
                split_metrics["train"]["core_spread_final_mean"]
                > split_metrics["train"]["core_spread_initial_mean"]
            ),
            "entropy_increase": bool(
                split_metrics["train"]["core_entropy_final_mean"]
                > split_metrics["train"]["core_entropy_initial_mean"]
            ),
            "visible_signal": bool(min(signal_to_noise) >= min_signal_to_noise),
            "source_mass_information_decay": bool(
                split_metrics["train"]["core_mass_near_source_final_fraction"]
                <= max_source_mass_final_fraction
            ),
            "source_peak_contrast_decay": bool(
                split_metrics["train"]["core_peak_contrast_final_fraction"]
                <= max_peak_contrast_final_fraction
            ),
            "core_oracle_high_iid": bool(
                split_metrics["iid_test"]["core_stat_oracle_accuracy"] >= core_oracle_floor
            ),
            "core_oracle_high_ood": bool(
                split_metrics["ood_test"]["core_stat_oracle_accuracy"] >= core_oracle_floor
            ),
            "spurious_rule_high_iid": bool(
                split_metrics["iid_test"]["spurious_train_rule_oracle_accuracy"]
                >= spurious_iid_floor
            ),
            "spurious_rule_breaks_ood": bool(
                split_metrics["ood_test"]["spurious_train_rule_oracle_accuracy"] <= 0.1
                or train_corr * ood_corr < 0
            ),
            "dynamic_spurious_corr_train": bool(np.isfinite(train_corr) and abs(train_corr) > 0.5),
            "dynamic_spurious_corr_iid": bool(np.isfinite(iid_corr) and abs(iid_corr) > 0.5),
            "dynamic_spurious_corr_ood_reversed": bool(
                np.isfinite(ood_corr) and train_corr * ood_corr < 0
            ),
            "counterfactual_preserves_core_and_label": bool(
                all(
                    split["metadata"]["counterfactual"]["preserves_y"]
                    and split["metadata"]["counterfactual"]["preserves_core_field"]
                    for split in splits.values()
                )
            ),
            "counterfactual_changes_spurious_flow": bool(
                all(
                    split["metadata"]["counterfactual"]["changes_spurious_flow"]
                    for split in splits.values()
                )
            ),
        },
        "inverse_ambiguity_diagnostics": {
            "source_peak_remains_identifiable_final": bool(train_peak_error_final <= 0.5),
            "source_center_remains_identifiable_final": bool(
                train_center_error_final <= 0.5
            ),
            "inverse_ambiguity_claim_ready": bool(
                train_peak_error_final > 0.5 and train_center_error_final > 0.5
            ),
            "interpretation": (
                "Mass and peak contrast decay measure diffusion-like information "
                "spreading. Peak/center source errors measure whether the exact "
                "source remains trivially localizable. Do not claim a strong "
                "many-to-one inverse problem when peak or center remains "
                "identifiable."
            ),
        },
    }
    result["pass"] = bool(all(result["quality_gates"].values()))
    return result


def generate_splits_from_config(
    config: dict[str, Any],
    *,
    return_fields: bool = False,
) -> dict[str, dict[str, Any]]:
    data = config.get("data", {})
    splits = config.get("splits", {})
    observation = config.get("observation", {})
    cf = config.get("counterfactual", {})
    split_modes = {
        name: cfg.get("spurious_mode")
        for name, cfg in splits.items()
        if isinstance(cfg, dict) and cfg.get("spurious_mode") is not None
    }
    return generate_ink_advection_diffusion_splits(
        n_train=int(splits.get("train", {}).get("n_sequences", 10_000)),
        n_val_iid=int(splits.get("val_iid", {}).get("n_sequences", 2_000)),
        n_iid_test=int(splits.get("iid_test", {}).get("n_sequences", 5_000)),
        n_ood_test=int(splits.get("ood_test", {}).get("n_sequences", 5_000)),
        length=int(data.get("length", 16)),
        grid_size=int(data.get("grid_size", 32)),
        seed=int(config.get("seed", 0)),
        core_diffusion=float(data.get("core_diffusion", 0.16)),
        spurious_diffusion=float(data.get("spurious_diffusion", 0.12)),
        core_flow_x=float(data.get("core_flow_x", 0.0)),
        core_flow_y=float(data.get("core_flow_y", 0.0)),
        spurious_flow_scale=float(data.get("spurious_flow_scale", 0.8)),
        source_blur_sigma=float(data.get("source_blur_sigma", 1.0)),
        pre_observation_steps=int(data.get("pre_observation_steps", 0)),
        dt=float(data.get("dt", 0.35)),
        dx=float(data.get("dx", 1.0)),
        observation_noise_std=float(
            observation.get("noise_std", data.get("observation_noise_std", 0.003))
        ),
        core_scale=float(observation.get("core_scale", 1.0)),
        spur_scale=float(observation.get("spur_scale", 0.9)),
        label_mode=str(data.get("label_mode", "core_source_x_median_threshold")),
        spurious_label_correlation_strength=float(
            data.get("spurious_label_correlation_strength", 1.0)
        ),
        spurious_cf_mode=str(cf.get("spurious_cf_mode", "randomized")),
        reuse_noise=bool(cf.get("reuse_noise", True)),
        split_spurious_modes=split_modes,
        return_fields=return_fields,
    )


def diagnose_config(config: dict[str, Any]) -> dict[str, Any]:
    gates = config.get("diagnostics", {})
    return diagnose_splits(
        generate_splits_from_config(config),
        min_signal_to_noise=float(gates.get("min_signal_to_noise", 1.0)),
        core_oracle_floor=float(gates.get("core_oracle_floor", 0.95)),
        spurious_iid_floor=float(gates.get("spurious_iid_floor", 0.90)),
        max_source_mass_final_fraction=float(
            gates.get("max_source_mass_final_fraction", 0.85)
        ),
        max_peak_contrast_final_fraction=float(
            gates.get("max_peak_contrast_final_fraction", 0.85)
        ),
    )


def _save_visuals(splits: dict[str, dict[str, Any]], output_dir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    frames = [0, 5, 10, 15]

    def robust_limits(arrays: list[np.ndarray]) -> tuple[float, float]:
        flat = np.concatenate([np.asarray(arr).reshape(-1) for arr in arrays])
        lo = float(np.percentile(flat, 0.5))
        hi = float(np.percentile(flat, 99.7))
        if hi <= lo:
            hi = lo + 1.0
        return lo, hi

    def annotate(ax, source: np.ndarray | None, flow: float | None, color: str) -> None:
        if source is not None:
            ax.scatter(
                [float(source[0])],
                [float(source[1])],
                marker="x",
                s=28,
                linewidths=1.2,
                color=color,
                zorder=4,
            )
        if flow is not None:
            x0, x1 = (0.18, 0.36) if flow > 0 else (0.36, 0.18)
            ax.annotate(
                "",
                xy=(x1, 0.12),
                xytext=(x0, 0.12),
                xycoords="axes fraction",
                arrowprops={"arrowstyle": "-|>", "lw": 1.2, "color": color},
            )

    for split_name, label in (("train", 1), ("ood_test", 1)):
        split = splits[split_name]
        idxs = np.flatnonzero(split["y"] == label)
        if idxs.size == 0:
            continue
        idx = int(idxs[np.argmax(split["core_score"][idxs])])
        obs = split["x"][idx].reshape(split["metadata"]["obs_shape"])
        obs_cf = split["x_cf"][idx].reshape(split["metadata"]["obs_shape"])
        core = split["core_field"][idx]
        spur = split["spurious_field"][idx]
        delta = np.abs(obs - obs_cf)
        source = split["core_source"][idx]
        flow = float(split["spurious_dynamic_stat"][idx])
        cf_flow = float(split["spurious_cf_dynamic_stat"][idx])
        vmin, vmax = robust_limits([obs, obs_cf, core, spur])
        dmin, dmax = robust_limits([delta])
        frame_ids = [min(frame, obs.shape[0] - 1) for frame in frames]
        rows = [
            ("observed x", obs),
            ("core field", core),
            ("spurious field", spur),
            ("counterfactual x_cf", obs_cf),
            ("|x - x_cf|", delta),
        ]
        fig, axes = plt.subplots(
            len(rows),
            len(frame_ids),
            figsize=(7.0, 6.2),
            constrained_layout=True,
        )
        for r, (name, arr) in enumerate(rows):
            for c, frame in enumerate(frame_ids):
                ax = axes[r, c]
                if name == "|x - x_cf|":
                    ax.imshow(
                        arr[frame],
                        cmap="viridis",
                        vmin=dmin,
                        vmax=dmax,
                        interpolation="nearest",
                    )
                else:
                    ax.imshow(
                        arr[frame],
                        cmap="magma",
                        vmin=vmin,
                        vmax=vmax,
                        interpolation="nearest",
                    )
                if r == 0:
                    ax.set_title(f"t={frame}")
                if c == 0:
                    ax.set_ylabel(name, rotation=0, ha="right", va="center", labelpad=42)
                if name in {"observed x", "core field", "counterfactual x_cf"}:
                    annotate(ax, source, None, "white")
                if name == "spurious field":
                    annotate(ax, None, flow, "white")
                if name == "counterfactual x_cf":
                    annotate(ax, None, cf_flow, "cyan")
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
        fig.suptitle(
            (
                f"{split_name}, y={label}, source x={float(source[0]):.1f}, "
                f"flow={flow:+.2f}, cf flow={cf_flow:+.2f}"
            ),
            fontsize=10,
        )
        path = output_dir / f"{split_name}_y{label}_diagnostic_sheet.png"
        fig.savefig(path, dpi=350, bbox_inches="tight")
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run data-only InkAdvectionDiffusion diagnostics.",
    )
    parser.add_argument("--config", default="configs/ink_advection_diffusion_default.yaml")
    parser.add_argument("--output", default=None)
    parser.add_argument("--visual-output-dir", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.seed is not None:
        config["seed"] = int(args.seed)
    if args.smoke:
        config["splits"] = {
            "train": {"n_sequences": 128, "spurious_mode": "correlated"},
            "val_iid": {"n_sequences": 64, "spurious_mode": "correlated"},
            "iid_test": {"n_sequences": 64, "spurious_mode": "correlated"},
            "ood_test": {"n_sequences": 64, "spurious_mode": "reversed"},
        }
        config.setdefault("data", {})["grid_size"] = min(
            int(config.get("data", {}).get("grid_size", 32)),
            24,
        )
        config.setdefault("data", {})["length"] = min(
            int(config.get("data", {}).get("length", 16)),
            12,
        )
    splits = generate_splits_from_config(config, return_fields=bool(args.visual_output_dir))
    result = diagnose_splits(
        splits,
        min_signal_to_noise=float(config.get("diagnostics", {}).get("min_signal_to_noise", 1.0)),
        core_oracle_floor=float(config.get("diagnostics", {}).get("core_oracle_floor", 0.95)),
        spurious_iid_floor=float(config.get("diagnostics", {}).get("spurious_iid_floor", 0.90)),
        max_source_mass_final_fraction=float(
            config.get("diagnostics", {}).get("max_source_mass_final_fraction", 0.85)
        ),
        max_peak_contrast_final_fraction=float(
            config.get("diagnostics", {}).get("max_peak_contrast_final_fraction", 0.85)
        ),
    )
    if args.visual_output_dir:
        _save_visuals(splits, Path(args.visual_output_dir))
    if args.output:
        save_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
