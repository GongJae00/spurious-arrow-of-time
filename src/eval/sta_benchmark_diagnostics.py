"""Data-only diagnostics for STA-Bench benchmark quality.

This module checks the benchmark before model training: class balance, dynamic
shortcut strength, OOD reversal, EP ratios, threshold reuse, and simple oracle
rules from core/spurious trajectory statistics.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np

from src.train.common import apply_cli_overrides, generate_splits_from_config, load_config, save_json


def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((np.asarray(y_true) == np.asarray(y_pred)).mean())


def _threshold_from_train(values: np.ndarray, y: np.ndarray) -> tuple[float, int]:
    values = np.asarray(values, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    if np.unique(y).shape[0] < 2:
        return float(np.median(values)), 1
    mean0 = float(values[y == 0].mean())
    mean1 = float(values[y == 1].mean())
    threshold = 0.5 * (mean0 + mean1)
    direction = 1 if mean1 >= mean0 else -1
    return float(threshold), int(direction)


def _predict_from_threshold(values: np.ndarray, threshold: float, direction: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if direction >= 0:
        return (values > threshold).astype(np.int64)
    return (values < threshold).astype(np.int64)


def _split_summary(split: dict[str, Any]) -> dict[str, Any]:
    meta = split["metadata"]
    return {
        "n_sequences": int(split["x"].shape[0]),
        "length_L": int(meta["length_L"]),
        "n_transitions": int(meta["n_transitions"]),
        "class_balance": meta["core"]["class_balance"],
        "label_threshold": meta["core"]["label_threshold"],
        "label_threshold_source": meta["core"]["label_threshold_source"],
        "label_noise": meta["core"].get("label_noise", 0.0),
        "core_analytic_ep": meta["core"]["analytic_ep"],
        "spurious_analytic_ep": meta["spurious"]["analytic_ep"],
        "ep_ratio_spurious_over_core": (
            float(meta["spurious"]["analytic_ep"]) / float(meta["core"]["analytic_ep"])
            if float(meta["core"]["analytic_ep"]) > 0
            else float("inf")
        ),
        "spurious_label_correlation_strength": meta["spurious"].get(
            "spurious_label_correlation_strength"
        ),
        "spurious_mode": meta["spurious"].get("spurious_mode"),
        "ood_shift_type": meta["spurious"].get("ood_shift_type"),
        "orientation_match_rate": meta["spurious"].get("orientation_match_rate"),
        "corr_y_core_dynamic_stat": meta["spurious"]["corr_y_core_dynamic_stat"],
        "corr_y_spurious_dynamic_stat": meta["spurious"]["corr_y_spurious_dynamic_stat"],
        "auc_y_from_core_dynamic_stat": meta["spurious"]["auc_y_from_core_dynamic_stat"],
        "auc_y_from_spurious_dynamic_stat": meta["spurious"][
            "auc_y_from_spurious_dynamic_stat"
        ],
        "mean_core_dynamic_stat_by_y": meta["spurious"]["mean_core_dynamic_stat_by_y"],
        "mean_spurious_dynamic_stat_by_y": meta["spurious"][
            "mean_spurious_dynamic_stat_by_y"
        ],
        "counterfactual_corr_y_spurious_dynamic_stat": meta["counterfactual"][
            "corr_y_spurious_cf_dynamic_stat"
        ],
        "core_observation_dropout": meta["observation"].get("core_observation_dropout", 0.0),
        "spur_observation_dropout": meta["observation"].get("spur_observation_dropout", 0.0),
    }


def diagnose_splits(splits: dict[str, dict[str, Any]]) -> dict[str, Any]:
    train = splits["train"]
    core_threshold, core_direction = _threshold_from_train(
        train["core_dynamic_stat"], train["y"]
    )
    spur_threshold, spur_direction = _threshold_from_train(
        train["spurious_dynamic_stat"], train["y"]
    )
    split_metrics: dict[str, Any] = {}
    for name, split in splits.items():
        y = split["y"]
        core_pred = _predict_from_threshold(split["core_dynamic_stat"], core_threshold, core_direction)
        spur_pred = _predict_from_threshold(
            split["spurious_dynamic_stat"], spur_threshold, spur_direction
        )
        split_metrics[name] = {
            **_split_summary(split),
            "core_stat_oracle_accuracy": _accuracy(y, core_pred),
            "spurious_train_rule_oracle_accuracy": _accuracy(y, spur_pred),
        }

    hashes = {
        name: split["metadata"]["observation"]["mixing_matrix_hash"]
        for name, split in splits.items()
    }
    thresholds = {
        name: split["metadata"]["core"]["label_threshold"]
        for name, split in splits.items()
    }
    train_corr = split_metrics["train"]["corr_y_spurious_dynamic_stat"]
    ood_corr = split_metrics["ood_test"]["corr_y_spurious_dynamic_stat"]
    label_noise = float(split_metrics["train"].get("label_noise", 0.0) or 0.0)
    spurious_strength = float(
        split_metrics["iid_test"].get("spurious_label_correlation_strength", 1.0) or 0.0
    )
    iid_n = max(1, int(split_metrics["iid_test"]["n_sequences"]))
    finite_sample_tolerance = float(min(0.15, 2.5 / np.sqrt(iid_n)))
    core_oracle_floor = float(max(0.65, 0.98 - 1.2 * label_noise - finite_sample_tolerance))
    spurious_iid_floor = max(
        0.55,
        0.5 + 0.5 * spurious_strength - 0.08 - finite_sample_tolerance,
    )
    spurious_iid_floor = float(spurious_iid_floor)
    iid_spurious_accuracy = split_metrics["iid_test"]["spurious_train_rule_oracle_accuracy"]
    ood_spurious_accuracy = split_metrics["ood_test"]["spurious_train_rule_oracle_accuracy"]
    spurious_ood_min_drop = 0.15
    ood_breaks_or_degrades_spurious_rule = (
        ood_spurious_accuracy <= 0.35
        or train_corr * ood_corr < 0
        or ood_spurious_accuracy <= iid_spurious_accuracy - spurious_ood_min_drop
    )
    result = {
        "schema_version": 1,
        "benchmark_name": "sta_bench",
        "splits": split_metrics,
        "difficulty_adjusted_thresholds": {
            "core_oracle_floor": core_oracle_floor,
            "spurious_iid_floor": spurious_iid_floor,
            "spurious_ood_min_drop": spurious_ood_min_drop,
            "finite_sample_tolerance": finite_sample_tolerance,
            "label_noise": label_noise,
            "spurious_label_correlation_strength": spurious_strength,
        },
        "oracle_rules": {
            "core_stat_threshold": core_threshold,
            "core_stat_direction": core_direction,
            "spurious_train_rule_threshold": spur_threshold,
            "spurious_train_rule_direction": spur_direction,
        },
        "quality_gates": {
            "same_mixing_matrix": bool(len(set(hashes.values())) == 1),
            "train_threshold_reused": bool(len(set(thresholds.values())) == 1),
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
            "spurious_rule_breaks_ood": bool(ood_breaks_or_degrades_spurious_rule),
        },
    }
    result["pass"] = bool(all(result["quality_gates"].values()))
    return result


def diagnose_config(config: dict[str, Any]) -> dict[str, Any]:
    return diagnose_splits(generate_splits_from_config(config))


def diagnose_config_or_tiers(config: dict[str, Any]) -> dict[str, Any]:
    if config.get("experiment") not in {
        "benchmark_difficulty_sweep",
        "benchmark_challenge",
    } or not config.get("benchmark_tiers"):
        return diagnose_config(config)

    from src.experiments.run_experiment import expand_configs

    tier_results = {
        name: diagnose_config(tier_config)
        for name, tier_config in expand_configs(config)
    }
    return {
        "schema_version": 1,
        "kind": "benchmark_tier_diagnostics",
        "pass": bool(all(result["pass"] for result in tier_results.values())),
        "tiers": tier_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run data-only STA-Bench diagnostics.")
    parser.add_argument("--config", default="configs/sta_default.yaml")
    parser.add_argument("--output", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    config = apply_cli_overrides(
        config,
        argparse.Namespace(
            seed=args.seed,
            epochs=None,
            batch_size=None,
            smoke=args.smoke,
            device="cpu",
            output_dir="results",
            overwrite=False,
            resume=None,
            config=args.config,
        ),
    )
    result = diagnose_config_or_tiers(config)
    if args.output:
        save_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
