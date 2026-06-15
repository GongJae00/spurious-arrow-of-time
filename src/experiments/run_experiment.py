"""Run multi-seed STA-Bench experiments, sweeps, and negative controls."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import shutil
import traceback
from typing import Any

from src.data.biased_ring import ring_entropy_production, solve_biased_ring_for_ep
from src.eval.aggregate_results import aggregate_results
from src.eval.select_setpoint import select_val_iid_setpoints
from src.train.common import (
    _deep_update,
    apply_cli_overrides,
    load_config,
    make_run_dir,
    resolve_device,
    save_json,
    train_arrow_pretraining_method,
    train_supervised_method,
)
from src.utils.config import config_hash


SUPERVISED_METHODS = {"erm", "ib", "ep_min", "ep_max", "sib", "sid", "itm"}
ARROW_METHODS = {"ocp_style", "lens_like_arrow_classifier"}
ALL_METHODS = [
    "erm",
    "ib",
    "ep_min",
    "ep_max",
    "ocp_style",
    "lens_like_arrow_classifier",
    "sib",
    "sid",
    "itm",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _set_nested(config: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    cursor = config
    for key in path[:-1]:
        cursor = cursor.setdefault(key, {})
    cursor[path[-1]] = value


def clean_output_dir(output_dir: str | Path) -> None:
    """Remove an experiment output root when it is safely scoped under ./results."""

    path = Path(output_dir)
    resolved = path.resolve()
    allowed_root = (Path.cwd() / "results").resolve()
    if not resolved.is_relative_to(allowed_root):
        raise ValueError(
            f"--clean-output refuses to remove {resolved}; use a path under {allowed_root}"
        )
    if resolved == allowed_root:
        raise ValueError("--clean-output refuses to remove the entire results directory")
    if path.exists():
        shutil.rmtree(path)


def _run_one(
    method: str,
    config: dict[str, Any],
    output_dir: str | Path,
    experiment_name: str,
    device,
    overwrite: bool,
) -> dict[str, Any]:
    seed = int(config.get("seed", 0))
    run_dir: Path | None = None
    started_at = _utc_now()
    base_row: dict[str, Any] = {
        "method": method,
        "seed": seed,
        "run_dir": None,
        "status": "running",
        "started_at_utc": started_at,
        "finished_at_utc": None,
        "config_hash": config_hash(config),
        "sweep": config.get("sweep"),
        "sib_variant": config.get("sib_variant", config.get("method_variant")),
        "benchmark_tier": config.get("benchmark_tier"),
        "ablation": config.get("ablation"),
        "counterfactual_sweep": config.get("counterfactual_sweep"),
        "metrics": None,
        "error_type": None,
        "error_message": None,
        "traceback": None,
    }
    try:
        run_dir = make_run_dir(output_dir, experiment_name, method, config, seed, overwrite)
        base_row["run_dir"] = str(run_dir)
        if method in SUPERVISED_METHODS:
            metrics = train_supervised_method(method, config, run_dir, device)
        elif method in ARROW_METHODS:
            metrics = train_arrow_pretraining_method(method, config, run_dir, device)
        else:
            raise ValueError(f"unknown method {method!r}")
        return {
            **base_row,
            "status": "success",
            "finished_at_utc": _utc_now(),
            "metrics": metrics,
        }
    except Exception as exc:
        failure = {
            **base_row,
            "status": "failed",
            "finished_at_utc": _utc_now(),
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
        }
        if run_dir is None:
            fallback_run_dir = (
                Path(output_dir)
                / experiment_name
                / method
                / f"failed_before_run_dir_{config_hash(config)}_seed{seed}"
            )
            failure["run_dir"] = str(fallback_run_dir)
            run_dir = fallback_run_dir
        try:
            save_json(run_dir / "failure.json", failure)
        except Exception:
            pass
        return failure


def _expand_ep_ratio(config: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    ratios = config.get("ratios", [0.25, 0.5, 1.0, 2.0, 4.0, 8.0])
    move_rate = float(config.get("move_rate", 0.6))
    data_cfg = config.get("data", {})
    core_ep = ring_entropy_production(float(data_cfg["p_core"]), float(data_cfg["q_core"]))
    expanded = []
    for ratio in ratios:
        cfg = deepcopy(config)
        target = float(ratio) * core_ep
        solution = solve_biased_ring_for_ep(target, move_rate=move_rate)
        _set_nested(cfg, ("data", "p_spur"), solution.p_forward)
        _set_nested(cfg, ("data", "q_spur"), solution.p_backward)
        cfg["sweep"] = {
            "type": "ep_ratio",
            "requested_ratio": float(ratio),
            "actual_spur_ep": solution.actual_ep,
            "actual_ratio": solution.actual_ep / core_ep if core_ep > 0 else float("inf"),
            "solution": solution.__dict__,
        }
        expanded.append((f"ratio_{ratio}", cfg))
    return expanded


def _expand_setpoint(config: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    setpoint_cfg = config.get("setpoint", {})
    multipliers = setpoint_cfg.get(
        "multipliers", [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
    )
    data_cfg = config.get("data", {})
    core_ep = ring_entropy_production(float(data_cfg["p_core"]), float(data_cfg["q_core"]))
    expanded = []
    for multiplier in multipliers:
        cfg = deepcopy(config)
        target = float(multiplier) * core_ep
        cfg["setpoint"] = {
            **cfg.get("setpoint", {}),
            "mode": "fixed_grid",
            "fixed_target": target,
            "multiplier": float(multiplier),
        }
        cfg["sweep"] = {"type": "setpoint", "multiplier": float(multiplier), "target": target}
        expanded.append((f"setpoint_{multiplier}", cfg))
    if bool(setpoint_cfg.get("include_oracle_core_reference", False)):
        cfg = deepcopy(config)
        oracle_setpoint = {**cfg.get("setpoint", {})}
        for key in ("target", "fixed_target", "selected_target"):
            oracle_setpoint.pop(key, None)
        oracle_setpoint["mode"] = "oracle_core_reference"
        cfg["setpoint"] = {
            **oracle_setpoint,
        }
        cfg["sweep"] = {
            "type": "setpoint",
            "mode": "oracle_core_reference",
            "oracle_assisted": True,
            "target": None,
        }
        expanded.append(("oracle_core_reference", cfg))
    return expanded


def _expand_ablation(config: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    controls = config.get("controls", {})
    expanded = []
    for name, override in controls.items():
        cfg = _deep_update(config, override if isinstance(override, dict) else {})
        cfg["ablation"] = {"name": name}
        if name == "no_counterfactual_change":
            _set_nested(cfg, ("counterfactual", "spurious_cf_mode"), "resample_same_mode")
            _set_nested(cfg, ("counterfactual", "no_change"), True)
        expanded.append((name, cfg))
    return expanded


def _expand_counterfactual_sensitivity(config: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    sensitivity = config.get("counterfactual_sensitivity", {})
    modes = sensitivity.get(
        "modes",
        ["randomized", "reversed", "independent_same_marginal", "resample_same_mode"],
    )
    expanded = []
    for mode in modes:
        cfg = deepcopy(config)
        _set_nested(cfg, ("counterfactual", "spurious_cf_mode"), mode)
        _set_nested(cfg, ("counterfactual", "no_change"), False)
        cfg["counterfactual_sweep"] = {"spurious_cf_mode": mode, "no_change": False}
        expanded.append((f"cf_{mode}", cfg))
    if bool(sensitivity.get("include_no_change", True)):
        cfg = deepcopy(config)
        _set_nested(cfg, ("counterfactual", "spurious_cf_mode"), "resample_same_mode")
        _set_nested(cfg, ("counterfactual", "no_change"), True)
        cfg["counterfactual_sweep"] = {
            "spurious_cf_mode": "resample_same_mode",
            "no_change": True,
        }
        expanded.append(("cf_no_change", cfg))
    return expanded


def _expand_sib_ablation(config: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    ablation_cfg = config.get("sib_ablation", {})
    include = ablation_cfg.get(
        "include",
        [
            "full_sib",
            "no_cf_arrow_invariance",
            "no_setpoint",
            "no_dynamics_loss",
            "wrong_sigma_star",
            "randomized_cf_explicit",
            "resample_same_mode_cf",
        ],
    )
    data_cfg = config.get("data", {})
    core_ep = ring_entropy_production(float(data_cfg["p_core"]), float(data_cfg["q_core"]))
    wrong_sigma_multiplier = float(ablation_cfg.get("wrong_sigma_multiplier", 4.0))
    expanded = []
    for name in include:
        cfg = deepcopy(config)
        cfg["ablation"] = {"name": name, "type": "sib_internal"}
        if name == "full_sib":
            pass
        elif name == "no_cf_arrow_invariance":
            _set_nested(cfg, ("loss_weights", "eta_total"), 0.0)
            _set_nested(cfg, ("loss_weights", "eta_step"), 0.0)
        elif name == "no_setpoint":
            _set_nested(cfg, ("loss_weights", "rho"), 0.0)
        elif name == "no_dynamics_loss":
            _set_nested(cfg, ("loss_weights", "lambda_f"), 0.0)
            _set_nested(cfg, ("loss_weights", "lambda_r"), 0.0)
        elif name == "wrong_sigma_star":
            cfg["setpoint"] = {
                **cfg.get("setpoint", {}),
                "mode": "fixed_grid",
                "fixed_target": wrong_sigma_multiplier * core_ep,
                "wrong_sigma_multiplier": wrong_sigma_multiplier,
            }
        elif name == "randomized_cf_explicit":
            _set_nested(cfg, ("counterfactual", "spurious_cf_mode"), "randomized")
            _set_nested(cfg, ("counterfactual", "no_change"), False)
        elif name == "resample_same_mode_cf":
            _set_nested(cfg, ("counterfactual", "spurious_cf_mode"), "resample_same_mode")
            _set_nested(cfg, ("counterfactual", "no_change"), False)
        else:
            raise ValueError(f"unknown SIB ablation {name!r}")
        expanded.append((name, cfg))
    return expanded


def _expand_benchmark_tiers(config: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    tiers = config.get("benchmark_tiers", {})
    if not isinstance(tiers, dict) or not tiers:
        raise ValueError("benchmark tier experiments require a non-empty benchmark_tiers map")
    expanded = []
    for name, override in tiers.items():
        cfg = _deep_update(config, override if isinstance(override, dict) else {})
        cfg["benchmark_tier"] = {
            "name": str(name),
            "data": cfg.get("data", {}),
            "observation": cfg.get("observation", {}),
        }
        expanded.append((str(name), cfg))
    return expanded


def _default_closure_conditions() -> dict[str, dict[str, Any]]:
    """Return minimal closure conditions for spurious-arrow causality tests."""

    return {
        "correlated_reversed_ood": {
            "splits": {
                "train": {"spurious_mode": "correlated"},
                "val_iid": {"spurious_mode": "correlated"},
                "iid_test": {"spurious_mode": "correlated"},
                "ood_test": {"spurious_mode": "reversed"},
            },
            "closure_role": "main_spurious_arrow_trap",
        },
        "correlated_no_shift": {
            "splits": {
                "train": {"spurious_mode": "correlated"},
                "val_iid": {"spurious_mode": "correlated"},
                "iid_test": {"spurious_mode": "correlated"},
                "ood_test": {"spurious_mode": "correlated"},
            },
            "closure_role": "no_distribution_shift_control",
        },
        "randomized_no_shortcut": {
            "splits": {
                "train": {"spurious_mode": "randomized"},
                "val_iid": {"spurious_mode": "randomized"},
                "iid_test": {"spurious_mode": "randomized"},
                "ood_test": {"spurious_mode": "randomized"},
            },
            "closure_role": "no_spurious_arrow_control",
        },
    }


def _expand_closure_spurious_causality(config: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Expand the final closure experiment into a small set of causal controls.

    This is intentionally narrow: it tests whether the OOD failure is tied to a
    dynamic spurious-arrow shift rather than generic shortcut artifacts.
    """

    conditions = config.get("closure_conditions") or _default_closure_conditions()
    if not isinstance(conditions, dict) or not conditions:
        raise ValueError("closure_spurious_causality requires non-empty closure_conditions")
    expanded = []
    for name, override in conditions.items():
        if not isinstance(override, dict):
            raise ValueError(f"closure condition {name!r} must be a mapping")
        cfg = _deep_update(config, override)
        cfg["closure_condition"] = {
            "name": str(name),
            "role": str(override.get("closure_role", "diagnostic")),
            "splits": cfg.get("splits", {}),
            "benchmark_name": cfg.get("benchmark_name", cfg.get("data", {}).get("benchmark_name")),
        }
        cfg["sweep"] = {"type": "closure_spurious_causality", "condition": str(name)}
        expanded.append((str(name), cfg))
    return expanded


def expand_configs(config: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    experiment = config.get("experiment", "spurious_arrow_trap")
    if experiment == "closure_spurious_causality":
        return _expand_closure_spurious_causality(config)
    if experiment in {"benchmark_difficulty_sweep", "benchmark_challenge"}:
        return _expand_benchmark_tiers(config)
    if experiment == "ep_ratio_sweep":
        return _expand_ep_ratio(config)
    if experiment == "setpoint_sweep":
        return _expand_setpoint(config)
    if experiment == "ablation_negative_controls":
        return _expand_ablation(config)
    if experiment == "counterfactual_sensitivity":
        return _expand_counterfactual_sensitivity(config)
    if experiment == "sib_ablation":
        return _expand_sib_ablation(config)
    return [(str(experiment), config)]


def run_experiment(
    config: dict[str, Any],
    methods: list[str],
    seeds: list[int],
    output_dir: str | Path,
    device,
    overwrite: bool = False,
    allow_failures: bool = False,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {"runs": []}
    expanded = expand_configs(config)
    for suffix, cfg_base in expanded:
        for seed in seeds:
            cfg_seed = deepcopy(cfg_base)
            cfg_seed["seed"] = int(seed)
            experiment_name = str(cfg_seed.get("experiment", "sta_bench"))
            if len(expanded) > 1:
                experiment_name = f"{experiment_name}/{suffix}"
            for method in methods:
                row = _run_one(method, cfg_seed, output_dir, experiment_name, device, overwrite)
                if "status" not in row:
                    row = {
                        **row,
                        "status": "success",
                        "started_at_utc": row.get("started_at_utc"),
                        "finished_at_utc": row.get("finished_at_utc"),
                        "config_hash": row.get("config_hash"),
                        "error_type": None,
                        "error_message": None,
                        "traceback": None,
                    }
                manifest["runs"].append(row)
                save_json(Path(output_dir) / "manifest.json", manifest)
    save_json(Path(output_dir) / "manifest.json", manifest)
    aggregate_results(output_dir, Path(output_dir) / "aggregate.json")
    if config.get("experiment") == "setpoint_sweep" and bool(
        config.get("setpoint", {}).get("select_with_val_iid", True)
        ):
        select_val_iid_setpoints(output_dir, Path(output_dir) / "setpoint_selection.json")
    failed = [row for row in manifest["runs"] if row.get("status") == "failed"]
    if failed and not allow_failures:
        raise RuntimeError(f"{len(failed)} experiment run(s) failed; see manifest.json")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run STA-Bench experiments.")
    parser.add_argument("--config", default="configs/sta_default.yaml")
    parser.add_argument("--output-dir", default="results/experiment")
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--allow-failures",
        action="store_true",
        help="Record failed runs and continue without exiting nonzero.",
    )
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Remove the output directory before running. Refuses paths outside ./results.",
    )
    args = parser.parse_args()
    config = apply_cli_overrides(load_config(args.config), args)
    methods = args.methods or config.get("methods") or ALL_METHODS
    seeds = args.seeds or ([args.seed] if args.seed is not None else None)
    seeds = seeds or config.get("seeds") or [int(config.get("seed", 0))]
    device = resolve_device(args)
    if args.clean_output:
        clean_output_dir(args.output_dir)
    run_experiment(
        config,
        methods,
        seeds,
        args.output_dir,
        device,
        args.overwrite,
        allow_failures=args.allow_failures,
    )


if __name__ == "__main__":
    main()
