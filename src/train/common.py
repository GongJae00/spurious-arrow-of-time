"""Shared training and evaluation routines for active research methods."""

from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.data.ink_advection_diffusion import generate_ink_advection_diffusion_splits
from src.data.sta_bench import generate_sta_splits
from src.eval.metrics import ood_gap, reverse_sequence
from src.losses.arrow_score import (
    ArrowScoreOutput,
    calibrate_arrow_score,
    dynamics_nlls,
    latent_arrow_score,
)
from src.losses.ib_loss import gaussian_kl_to_standard_normal
from src.losses.itm_loss import itm_anti_collapse_loss, itm_loss
from src.losses.ocp_loss import binary_arrow_loss
from src.losses.sib_loss import sib_loss
from src.losses.sid_loss import sid_factor_diagnostics, sid_loss
from src.models.baselines import ArrowClassifier, ERMGRU, IBGRU
from src.models.classifiers import MLPClassifier
from src.models.encoders import pool_sequence
from src.models.itm import ITMModel
from src.models.sib import SIBModel
from src.models.sid import GradientReversal, SIDModel
from src.utils.checkpoint import load_checkpoint, save_checkpoint
from src.utils.config import config_hash, load_yaml
from src.utils.hardware import get_device, hardware_metadata
from src.utils.logging import JsonlLogger
from src.utils.seed import seed_everything

SupervisedMethod = Literal["erm", "ib", "ep_min", "ep_max", "sib", "sid", "itm"]
ArrowMethod = Literal["ocp_style", "lens_like_arrow_classifier"]
SIB_COLLAPSE_DETECTOR_KEYS = (
    "near_chance_task_detector",
    "constant_prediction_detector",
    "high_prediction_entropy_detector",
    "sigma_zero_collapse_detector",
    "loss_regularizer_dominance_detector",
    "dynamics_likelihood_explosion_detector",
    "latent_norm_collapse_detector",
)


class EncoderTaskClassifier(nn.Module):
    """Task classifier built from a pretrained sequence encoder."""

    def __init__(
        self,
        encoder: nn.Module,
        latent_dim: int,
        hidden_dim: int,
        pooling: str = "last",
        freeze_encoder: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.pooling = pooling
        self.freeze_encoder = freeze_encoder
        self.classifier = MLPClassifier(latent_dim, 2, hidden_dim)
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.freeze_encoder:
            self.encoder.eval()
            with torch.no_grad():
                z = self.encoder(x)
        else:
            z = self.encoder(x)
        logits = self.classifier(pool_sequence(z, self.pooling))
        return {"z": z, "logits": logits}


def build_arg_parser(method: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"Train {method} on STA-Bench or Ink Advection-Diffusion."
    )
    parser.add_argument("--config", default="configs/sta_default.yaml")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--smoke", action="store_true", help="Use tiny split sizes and epochs.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", default=None)
    return parser


def _deep_update(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    config = load_yaml(path)
    if "base_config" in config:
        base_path = Path(config["base_config"])
        if not base_path.is_absolute():
            relative_to_config = path.parent / base_path
            base_path = relative_to_config if relative_to_config.exists() else base_path
        base = load_config(base_path)
        config = _deep_update(base, {k: v for k, v in config.items() if k != "base_config"})
    config.setdefault("run", {})["config_path"] = str(path)
    return config


def apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    config = deepcopy(config)
    if args.seed is not None:
        config["seed"] = args.seed
    if args.epochs is not None:
        config.setdefault("training", {})["epochs"] = args.epochs
    if args.batch_size is not None:
        config.setdefault("training", {})["batch_size"] = args.batch_size
    if args.smoke:
        config.setdefault("splits", {})
        config["splits"] = {
            "train": {"n_sequences": 128, "spurious_mode": "correlated"},
            "val_iid": {"n_sequences": 64, "spurious_mode": "correlated"},
            "iid_test": {"n_sequences": 64, "spurious_mode": "correlated"},
            "ood_test": {"n_sequences": 64, "spurious_mode": "reversed"},
        }
        config.setdefault("data", {})["length"] = min(int(config["data"].get("length", 32)), 16)
        config.setdefault("training", {})["epochs"] = args.epochs or 2
        config["training"]["batch_size"] = args.batch_size or 32
    return config


def resolve_device(args: argparse.Namespace) -> torch.device:
    if args.device == "cpu":
        return torch.device("cpu")
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is not available")
        return torch.device("cuda")
    return get_device()


def make_run_dir(
    output_dir: str | Path,
    experiment: str,
    method: str,
    config: dict[str, Any],
    seed: int,
    overwrite: bool,
) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base_name = f"{timestamp}_{config_hash(config)}_seed{seed}"
    run_dir = Path(output_dir) / experiment / method / base_name
    if run_dir.exists() and not overwrite:
        for attempt in range(1, 10_000):
            candidate = Path(output_dir) / experiment / method / f"{base_name}_attempt{attempt}"
            if not candidate.exists():
                run_dir = candidate
                break
        else:
            raise FileExistsError(f"could not allocate unique run directory for {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=str)


def _git_output(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=Path(__file__).resolve().parents[2],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def git_metadata() -> dict[str, Any]:
    status = _git_output(["status", "--short"])
    return {
        "commit": _git_output(["rev-parse", "HEAD"]),
        "branch": _git_output(["rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": bool(status),
        "status_short": status,
    }


def _data_loader_settings(config: dict[str, Any], device: torch.device) -> dict[str, Any]:
    training = config.get("training", {})
    num_workers = int(training.get("num_workers", 0))
    return {
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": bool(training.get("persistent_workers", False)) and num_workers > 0,
        "worker_seed": int(config.get("seed", 0)),
    }


def _check_supported_config(config: dict[str, Any]) -> None:
    model_cfg = config.get("model", {})
    training_cfg = config.get("training", {})
    encoder = str(model_cfg.get("encoder", "gru"))
    if encoder != "gru":
        raise ValueError(f"unsupported model.encoder={encoder!r}; only 'gru' is implemented")
    optimizer = str(training_cfg.get("optimizer", "adamw")).lower()
    if optimizer != "adamw":
        raise ValueError(
            f"unsupported training.optimizer={optimizer!r}; only 'adamw' is implemented"
        )


def _is_static_spurious_control_config(config: dict[str, Any]) -> bool:
    ablation = config.get("ablation") or {}
    return (
        str(config.get("experiment", "")) == "static_spurious_control"
        or ablation.get("name") == "static_spurious_control"
    )


def _make_optimizer(parameters, training_cfg: dict[str, Any]) -> torch.optim.Optimizer:
    optimizer = str(training_cfg.get("optimizer", "adamw")).lower()
    if optimizer != "adamw":
        raise ValueError(
            f"unsupported training.optimizer={optimizer!r}; only 'adamw' is implemented"
        )
    return torch.optim.AdamW(
        parameters,
        lr=float(training_cfg.get("lr", 1e-3)),
        weight_decay=float(training_cfg.get("weight_decay", 1e-4)),
    )


def _early_stopping_metric_name(
    training_cfg: dict[str, Any],
    *,
    protocol: str | None = None,
) -> str:
    metric = str(training_cfg.get("early_stopping_metric", "val_iid_accuracy"))
    if protocol is not None and not metric.startswith(f"{protocol}_"):
        metric = f"{protocol}_{metric}"
    return metric


def run_metadata(
    method: str,
    config: dict[str, Any],
    run_dir: Path,
    device: torch.device,
    parameter_count_value: int,
    data_loader_config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "run_id": run_dir.name,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "method": method,
        "config_path": config.get("run", {}).get("config_path"),
        "config_hash": config_hash(config),
        "seed": int(config.get("seed", 0)),
        "device": str(device),
        "parameter_count": int(parameter_count_value),
        "git": git_metadata(),
        "hardware": hardware_metadata(device),
        "data_loader": data_loader_config,
        "deterministic": bool(config.get("training", {}).get("deterministic", False)),
    }


def generate_splits_from_config(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    _check_supported_config(config)
    data_cfg = config.get("data", {})
    split_cfg = config.get("splits", {})
    obs_cfg = config.get("observation", {})
    cf_cfg = config.get("counterfactual", {})
    benchmark_name = str(config.get("benchmark_name", data_cfg.get("benchmark_name", "sta_bench")))
    split_spurious_modes = {}
    for name, cfg in split_cfg.items():
        if isinstance(cfg, dict) and cfg.get("spurious_mode") is not None:
            split_spurious_modes[name] = cfg.get("spurious_mode")
    split_spurious_modes_arg = split_spurious_modes or None
    if benchmark_name == "ink_advection_diffusion":
        return generate_ink_advection_diffusion_splits(
            n_train=int(split_cfg.get("train", {}).get("n_sequences", 10_000)),
            n_val_iid=int(split_cfg.get("val_iid", {}).get("n_sequences", 2_000)),
            n_iid_test=int(split_cfg.get("iid_test", {}).get("n_sequences", 5_000)),
            n_ood_test=int(split_cfg.get("ood_test", {}).get("n_sequences", 5_000)),
            length=int(data_cfg.get("length", 16)),
            grid_size=int(data_cfg.get("grid_size", 32)),
            seed=int(config.get("seed", 0)),
            core_diffusion=float(data_cfg.get("core_diffusion", 0.16)),
            spurious_diffusion=float(data_cfg.get("spurious_diffusion", 0.12)),
            core_flow_x=float(data_cfg.get("core_flow_x", 0.0)),
            core_flow_y=float(data_cfg.get("core_flow_y", 0.0)),
            spurious_flow_scale=float(data_cfg.get("spurious_flow_scale", 0.8)),
            source_blur_sigma=float(data_cfg.get("source_blur_sigma", 1.0)),
            pre_observation_steps=int(data_cfg.get("pre_observation_steps", 0)),
            dt=float(data_cfg.get("dt", 0.35)),
            dx=float(data_cfg.get("dx", 1.0)),
            observation_noise_std=float(
                obs_cfg.get("noise_std", data_cfg.get("observation_noise_std", 0.003))
            ),
            core_scale=float(obs_cfg.get("core_scale", 1.0)),
            spur_scale=float(obs_cfg.get("spur_scale", 0.9)),
            label_mode=str(data_cfg.get("label_mode", "core_source_x_median_threshold")),
            spurious_label_correlation_strength=float(
                data_cfg.get("spurious_label_correlation_strength", 1.0)
            ),
            spurious_cf_mode=str(cf_cfg.get("spurious_cf_mode", "randomized")),
            reuse_noise=bool(cf_cfg.get("reuse_noise", True)),
            split_spurious_modes=split_spurious_modes_arg,
        )
    if benchmark_name != "sta_bench":
        raise ValueError(
            "benchmark_name must be one of 'sta_bench' or 'ink_advection_diffusion'"
        )
    init_cfg = data_cfg.get("initial_state_mode", {})
    initial_state_mode_core = (
        str(init_cfg.get("core", "uniform_stationary"))
        if isinstance(init_cfg, dict)
        else "uniform_stationary"
    )
    initial_state_mode_spurious = (
        str(init_cfg.get("spurious", "uniform_stationary"))
        if isinstance(init_cfg, dict)
        else "uniform_stationary"
    )
    spurious_correlation_type = str(
        data_cfg.get(
            "spurious_correlation_type",
            config.get("spurious_correlation_type", "drift_direction"),
        )
    )
    static_control = _is_static_spurious_control_config(config)
    if initial_state_mode_core != "uniform_stationary":
        raise ValueError(
            "core initial_state_mode must be uniform_stationary for configured experiments"
        )
    if initial_state_mode_spurious != "uniform_stationary" and not static_control:
        raise ValueError(
            "spurious non-stationary initialization is diagnostic-only; use the "
            "static_spurious_control ablation for sector/static shortcut controls"
        )
    if (
        static_control
        and initial_state_mode_spurious != "uniform_stationary"
        and spurious_correlation_type != "initial_sector_static_control"
    ):
        raise ValueError(
            "non-stationary spurious initialization in a static control must be paired with "
            "spurious_correlation_type=initial_sector_static_control"
        )
    if spurious_correlation_type == "initial_sector_static_control" and not static_control:
        raise ValueError(
            "initial_sector_static_control is diagnostic-only and must not be used as the "
            "main spurious_arrow_trap configuration"
        )
    ood_shift_type = str(data_cfg.get("ood_shift_type", "reversed"))
    split_spurious_modes = {}
    for name, cfg in split_cfg.items():
        if not isinstance(cfg, dict) or cfg.get("spurious_mode") is None:
            continue
        if name == "ood_test" and "ood_shift_type" in data_cfg:
            continue
        split_spurious_modes[name] = cfg.get("spurious_mode")
    if not split_spurious_modes:
        split_spurious_modes = None
    diagnostic = config.get("diagnostic")
    randomize_labels = diagnostic == "randomized_labels" or bool(
        data_cfg.get("randomize_labels", False)
    )
    return generate_sta_splits(
        n_train=int(split_cfg.get("train", {}).get("n_sequences", 10_000)),
        n_val_iid=int(split_cfg.get("val_iid", {}).get("n_sequences", 2_000)),
        n_iid_test=int(split_cfg.get("iid_test", {}).get("n_sequences", 5_000)),
        n_ood_test=int(split_cfg.get("ood_test", {}).get("n_sequences", 5_000)),
        length=int(data_cfg.get("length", 32)),
        n_core_states=int(data_cfg.get("n_core_states", 8)),
        n_spur_states=int(data_cfg.get("n_spur_states", 8)),
        p_core=float(data_cfg.get("p_core", 0.35)),
        q_core=float(data_cfg.get("q_core", 0.25)),
        p_spur=float(data_cfg.get("p_spur", 0.45)),
        q_spur=float(data_cfg.get("q_spur", 0.15)),
        obs_dim=int(data_cfg.get("obs_dim", 16)),
        noise_std=float(obs_cfg.get("noise_std", data_cfg.get("noise_std", 0.1))),
        seed=int(config.get("seed", 0)),
        label_mode=str(data_cfg.get("label_mode", "core_net_displacement_median_threshold")),
        spurious_correlation_type=spurious_correlation_type,
        spurious_label_correlation_strength=float(
            data_cfg.get("spurious_label_correlation_strength", 1.0)
        ),
        trajectory_arrow_statistic=str(
            data_cfg.get("trajectory_arrow_statistic", "net_clockwise_displacement")
        ),
        ood_spurious_mode=str(split_cfg.get("ood_test", {}).get("spurious_mode", "reversed")),
        ood_shift_type=ood_shift_type,
        ood_spurious_label_correlation_strength=(
            None
            if data_cfg.get("ood_spurious_label_correlation_strength") is None
            else float(data_cfg.get("ood_spurious_label_correlation_strength"))
        ),
        normalize_mixing_columns=bool(obs_cfg.get("normalize_mixing_columns", True)),
        core_scale=float(obs_cfg.get("core_scale", 1.0)),
        spur_scale=float(obs_cfg.get("spur_scale", 1.0)),
        core_observation_dropout=float(obs_cfg.get("core_observation_dropout", 0.0)),
        spur_observation_dropout=float(obs_cfg.get("spur_observation_dropout", 0.0)),
        label_noise=float(data_cfg.get("label_noise", 0.0)),
        spurious_cf_mode=str(cf_cfg.get("spurious_cf_mode", "randomized")),
        reuse_noise=bool(cf_cfg.get("reuse_noise", True)),
        split_spurious_modes=split_spurious_modes,
        initial_state_mode_core=initial_state_mode_core,
        initial_state_mode_spurious=initial_state_mode_spurious,
        randomize_labels=randomize_labels,
        counterfactual_no_change=bool(cf_cfg.get("no_change", False)),
    )


def _dataset_for_split(
    split: dict[str, Any], include_cf: bool = False, include_decomp_stats: bool = False
) -> TensorDataset:
    """Build TensorDataset for a split.

    include_cf: adds x_cf for SIB/SID counterfactual training.
    include_decomp_stats: adds core_dynamic_stat, spurious_dynamic_stat (normalized float)
        for full SID decomposition auxiliary losses (L_core_preservation, L_spur_*).
        These are only used for controlled synthetic benchmarks to enforce factor roles;
        the main task head and deployable path never receive the raw stats directly.
    """
    x = torch.as_tensor(split["x"], dtype=torch.float32)
    y = torch.as_tensor(split["y"], dtype=torch.long)
    tensors = [x, y]
    if include_cf:
        x_cf = torch.as_tensor(split["x_cf"], dtype=torch.float32)
        tensors.append(x_cf)
    if include_decomp_stats:
        c_stat = torch.as_tensor(split["core_dynamic_stat"], dtype=torch.float32)
        s_stat = torch.as_tensor(split["spurious_dynamic_stat"], dtype=torch.float32)
        tensors.extend([c_stat, s_stat])
    return TensorDataset(*tensors)


def make_loaders(
    splits: dict[str, dict[str, Any]],
    batch_size: int,
    include_cf: bool = False,
    include_decomp_stats: bool = False,
    device: torch.device | None = None,
    num_workers: int = 0,
    persistent_workers: bool = False,
    seed: int = 0,
) -> dict[str, DataLoader]:
    pin_memory = device is not None and device.type == "cuda"
    persistent_workers = bool(persistent_workers) and int(num_workers) > 0
    loaders: dict[str, DataLoader] = {}
    for idx, (name, data) in enumerate(splits.items()):
        generator = torch.Generator()
        generator.manual_seed(int(seed) + idx)

        def seed_worker(worker_id: int, base_seed: int = int(seed) + 10_000 * idx) -> None:
            worker_seed = (base_seed + worker_id) % (2**32)
            random.seed(worker_seed)
            np.random.seed(worker_seed)
            torch.manual_seed(worker_seed)

        loaders[name] = DataLoader(
            _dataset_for_split(data, include_cf=include_cf, include_decomp_stats=include_decomp_stats),
            batch_size=batch_size,
            shuffle=name == "train",
            num_workers=int(num_workers),
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            worker_init_fn=seed_worker if int(num_workers) > 0 else None,
            generator=generator,
        )
    return loaders


def _seeded_loader(
    dataset: TensorDataset,
    *,
    batch_size: int,
    shuffle: bool,
    device: torch.device | None,
    num_workers: int,
    persistent_workers: bool,
    seed: int,
) -> DataLoader:
    pin_memory = device is not None and device.type == "cuda"
    persistent_workers = bool(persistent_workers) and int(num_workers) > 0
    generator = torch.Generator()
    generator.manual_seed(int(seed))

    def seed_worker(worker_id: int) -> None:
        worker_seed = (int(seed) + worker_id) % (2**32)
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=int(num_workers),
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        worker_init_fn=seed_worker if int(num_workers) > 0 else None,
        generator=generator,
    )


def _model_common_kwargs(config: dict[str, Any], input_dim: int) -> dict[str, Any]:
    model_cfg = config.get("model", {})
    dyn_cfg = config.get("dynamics", {})
    return {
        "input_dim": input_dim,
        "hidden_dim": int(model_cfg.get("hidden_dim", 64)),
        "latent_dim": int(model_cfg.get("latent_dim", 16)),
        "pooling": str(model_cfg.get("pooling", "last")),
        "num_layers": int(model_cfg.get("num_layers", 1)),
        "dropout": float(model_cfg.get("dropout", 0.0)),
        "bidirectional": bool(model_cfg.get("bidirectional", False)),
        "min_logvar": float(dyn_cfg.get("min_logvar", -6.0)),
        "max_logvar": float(dyn_cfg.get("max_logvar", 2.0)),
        "fixed_variance": bool(dyn_cfg.get("fixed_variance", False)),
        "dynamics_hidden_dim": int(dyn_cfg.get("hidden_dim", model_cfg.get("hidden_dim", 64))),
    }


def build_supervised_model(method: SupervisedMethod, config: dict[str, Any], input_dim: int):
    kwargs = _model_common_kwargs(config, input_dim)
    if method == "erm":
        return ERMGRU(
            input_dim=input_dim,
            hidden_dim=kwargs["hidden_dim"],
            latent_dim=kwargs["latent_dim"],
            pooling=kwargs["pooling"],
            num_layers=kwargs["num_layers"],
            dropout=kwargs["dropout"],
            bidirectional=kwargs["bidirectional"],
        )
    if method == "ib":
        return IBGRU(
            input_dim=input_dim,
            hidden_dim=kwargs["hidden_dim"],
            latent_dim=kwargs["latent_dim"],
            pooling=kwargs["pooling"],
            min_logvar=kwargs["min_logvar"],
            max_logvar=kwargs["max_logvar"],
            num_layers=kwargs["num_layers"],
            dropout=kwargs["dropout"],
            bidirectional=kwargs["bidirectional"],
        )
    if method == "sid":
        sid_cfg = config.get("sid", {})
        latent_dim = int(config.get("model", {}).get("latent_dim", kwargs["latent_dim"]))
        return SIDModel(
            input_dim=input_dim,
            hidden_dim=kwargs["hidden_dim"],
            z_rev_dim=int(sid_cfg.get("z_rev_dim", latent_dim)),
            z_ir_task_dim=int(sid_cfg.get("z_ir_task_dim", latent_dim)),
            z_ir_spur_dim=int(sid_cfg.get("z_ir_spur_dim", latent_dim)),
            z_resid_dim=int(sid_cfg.get("z_resid_dim", 0)),
            pooling=kwargs["pooling"],
            num_layers=kwargs["num_layers"],
            dropout=kwargs["dropout"],
            bidirectional=kwargs["bidirectional"],
            min_logvar=kwargs["min_logvar"],
            max_logvar=kwargs["max_logvar"],
            fixed_variance=kwargs["fixed_variance"],
            dynamics_hidden_dim=kwargs["dynamics_hidden_dim"],
        )
    if method == "itm":
        itm_cfg = config.get("itm", {})
        latent_dim = int(config.get("model", {}).get("latent_dim", kwargs["latent_dim"]))
        return ITMModel(
            input_dim=input_dim,
            hidden_dim=kwargs["hidden_dim"],
            core_dim=int(itm_cfg.get("core_dim", latent_dim)),
            spur_dim=int(itm_cfg.get("spur_dim", latent_dim)),
            pooling=kwargs["pooling"],
            num_layers=kwargs["num_layers"],
            dropout=kwargs["dropout"],
            bidirectional=kwargs["bidirectional"],
            transition_hidden_dim=kwargs["dynamics_hidden_dim"],
        )
    return SIBModel(**kwargs)


def build_arrow_model(config: dict[str, Any], input_dim: int) -> ArrowClassifier:
    model_cfg = config.get("model", {})
    return ArrowClassifier(
        input_dim=input_dim,
        hidden_dim=int(model_cfg.get("hidden_dim", 64)),
        latent_dim=int(model_cfg.get("latent_dim", 16)),
        pooling=str(model_cfg.get("pooling", "last")),
        num_layers=int(model_cfg.get("num_layers", 1)),
        dropout=float(model_cfg.get("dropout", 0.0)),
        bidirectional=bool(model_cfg.get("bidirectional", False)),
    )


def parameter_count(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _tensor_summary(values: torch.Tensor, prefix: str) -> dict[str, float]:
    detached = values.detach().float().cpu()
    return {
        f"{prefix}_mean": float(detached.mean()),
        f"{prefix}_std": float(detached.std(unbiased=False)),
        f"{prefix}_min": float(detached.min()),
        f"{prefix}_max": float(detached.max()),
    }


def _dynamics_diagnostics(
    model: torch.nn.Module,
    z: torch.Tensor,
    score: ArrowScoreOutput,
    threshold: float,
    *,
    prefix: str = "",
) -> dict[str, float | bool]:
    """Return numerical diagnostics for latent arrow dynamics."""

    key = f"{prefix}_" if prefix else ""
    diagnostics: dict[str, float | bool] = {
        **_tensor_summary(score.sigma_total, f"{key}sigma_total"),
        **_tensor_summary(score.sigma_per_step, f"{key}sigma_per_step"),
        **_tensor_summary(score.sigma_steps, f"{key}sigma_steps"),
        **_tensor_summary(z.norm(dim=-1), f"{key}latent_norm"),
        f"{key}sigma_per_step_abs_max": float(
            score.sigma_per_step.abs().max().detach().cpu()
        ),
    }
    diagnostics[f"{key}sigma_threshold_exceeded"] = bool(
        diagnostics[f"{key}sigma_per_step_abs_max"] > threshold
    )
    if hasattr(model, "forward_dynamics") and hasattr(model, "reverse_dynamics"):
        z_t = z[:, :-1]
        z_next = z[:, 1:]
        diagnostics.update(model.forward_dynamics.logvar_summary(z_t, "forward"))
        diagnostics.update(model.reverse_dynamics.logvar_summary(z_next, "reverse"))
    return diagnostics


def _classification_diagnostics(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_correct = 0
    total = 0
    entropy_sum = 0.0
    pred_counts: torch.Tensor | None = None
    prob_sum: torch.Tensor | None = None
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            y = batch[1].to(device)
            logits = model(x)["logits"]
            probs = F.softmax(logits, dim=-1)
            pred = probs.argmax(dim=-1)
            n_classes = int(logits.shape[-1])
            if pred_counts is None:
                pred_counts = torch.zeros(n_classes, dtype=torch.float64)
                prob_sum = torch.zeros(n_classes, dtype=torch.float64)
            pred_counts += torch.bincount(pred.cpu(), minlength=n_classes).to(torch.float64)
            prob_sum += probs.detach().cpu().sum(dim=0).to(torch.float64)
            entropy_sum += float((-(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)).sum().cpu())
            total_correct += int((pred == y).sum().item())
            total += int(y.numel())
    denom = max(total, 1)
    pred_fracs = (pred_counts / denom).tolist() if pred_counts is not None else []
    mean_probs = (prob_sum / denom).tolist() if prob_sum is not None else []
    out = {
        "accuracy": total_correct / denom,
        "prediction_entropy": entropy_sum / denom,
        "max_pred_class_fraction": max(pred_fracs) if pred_fracs else 0.0,
    }
    for idx, value in enumerate(pred_fracs):
        out[f"pred_class_{idx}_fraction"] = float(value)
    for idx, value in enumerate(mean_probs):
        out[f"mean_prob_class_{idx}"] = float(value)
    return out


def _majority_baseline(split: dict[str, Any]) -> float:
    balance = split.get("metadata", {}).get("core", {}).get("class_balance", {})
    if "p1" in balance:
        p1 = float(balance["p1"])
        return max(p1, 1.0 - p1)
    y = np.asarray(split["y"])
    if y.size == 0:
        return 0.5
    return float(max(np.mean(y == 0), np.mean(y == 1)))


def _task_guard_threshold(
    task_guard_cfg: Mapping[str, Any],
    *,
    val_majority: float,
) -> float:
    threshold = val_majority + float(task_guard_cfg.get("min_margin_over_majority", 0.10))
    configured_min = task_guard_cfg.get("min_val_iid_accuracy")
    if isinstance(configured_min, (int, float)):
        threshold = float(configured_min)
    return threshold


def _task_guard_score(
    record: Mapping[str, Any],
    task_guard_cfg: Mapping[str, Any],
    detector_count: int,
) -> tuple[float, float, int]:
    metric = str(
        task_guard_cfg.get("selection_metric", "val_iid_accuracy_then_cf_stability")
    )
    val_iid = float(record.get("val_iid_accuracy", float("-inf")))
    cf_stability = float(record.get("val_iid_cf_prediction_consistency", 0.0))
    if metric == "val_iid_accuracy_then_cf_stability":
        return (val_iid, cf_stability, -int(detector_count))
    if metric == "cf_stability_then_val_iid":
        return (cf_stability, val_iid, -int(detector_count))
    raise ValueError(
        "task_guard.selection_metric must be one of "
        "val_iid_accuracy_then_cf_stability or cf_stability_then_val_iid"
    )


@torch.no_grad()
def _evaluate_supervised_selection(
    model: torch.nn.Module,
    loaders: Mapping[str, DataLoader],
    device: torch.device,
    *,
    include_cf: bool,
    arrow_calibration_config: Mapping[str, Any] | None = None,
) -> tuple[dict[str, float], dict[str, Any]]:
    selected_acc = {
        f"{name}_accuracy": evaluate_classifier(model, loader, device)
        for name, loader in loaders.items()
    }
    selected_acc["ood_gap"] = ood_gap(
        selected_acc["iid_test_accuracy"], selected_acc["ood_test_accuracy"]
    )
    selected_cf: dict[str, Any] = {}
    if include_cf:
        for name, loader in loaders.items():
            for metric, value in evaluate_counterfactual_sensitivity(
                model,
                loader,
                device,
                arrow_calibration_config=arrow_calibration_config,
            ).items():
                selected_cf[f"{name}_{metric}"] = value
    return selected_acc, selected_cf


def _scheduled_sib_weights(
    loss_weights: dict[str, Any],
    schedule_cfg: dict[str, Any],
    epoch: int,
) -> dict[str, Any]:
    effective = dict(loss_weights)
    task_warmup_epochs = int(schedule_cfg.get("task_warmup_epochs", 0))
    dynamics_warmup_epochs = int(schedule_cfg.get("dynamics_warmup_epochs", 0))
    warmup_epochs = task_warmup_epochs + dynamics_warmup_epochs
    ramp_epochs = int(schedule_cfg.get("regularizer_ramp_epochs", 0))
    if epoch < warmup_epochs:
        progress = 0.0
    elif ramp_epochs <= 0:
        progress = 1.0
    else:
        progress = min(max((epoch - warmup_epochs + 1) / float(ramp_epochs), 0.0), 1.0)
    for name in ("eta_total", "eta_step", "rho"):
        if name not in effective:
            continue
        start = float(schedule_cfg.get(f"{name}_start", 0.0))
        end = float(schedule_cfg.get(f"{name}_end", effective[name]))
        effective[name] = start + progress * (end - start)
    effective["schedule_progress"] = progress
    effective["schedule_warmup_active"] = epoch < warmup_epochs

    schedule_dynamics = bool(schedule_cfg.get("schedule_dynamics", False)) or any(
        key in schedule_cfg
        for key in ("lambda_f_start", "lambda_r_start", "lambda_f_end", "lambda_r_end")
    )
    if schedule_dynamics:
        if epoch < task_warmup_epochs:
            dynamics_progress = 0.0
        elif dynamics_warmup_epochs <= 0:
            dynamics_progress = 1.0
        else:
            dynamics_progress = min(
                max(
                    (epoch - task_warmup_epochs + 1)
                    / float(dynamics_warmup_epochs),
                    0.0,
                ),
                1.0,
            )
        for name in ("lambda_f", "lambda_r"):
            if name not in effective:
                continue
            start = float(schedule_cfg.get(f"{name}_start", 0.0))
            end = float(schedule_cfg.get(f"{name}_end", effective[name]))
            effective[name] = start + dynamics_progress * (end - start)
        effective["dynamics_schedule_progress"] = dynamics_progress
        effective["dynamics_schedule_warmup_active"] = dynamics_progress < 1.0
    else:
        effective["dynamics_schedule_progress"] = 1.0
        effective["dynamics_schedule_warmup_active"] = False
    return effective


def _scheduled_sid_weights(
    loss_weights: dict[str, Any],
    schedule_cfg: dict[str, Any],
    epoch: int,
) -> dict[str, Any]:
    """SID-specific schedule: task warmup, dynamics, cf ramp, adv after guard.

    Ramps the key regularizers (cf, sens, adv, preserve, capture, arrow_decomp, anti_collapse)
    from start values (usually 0) to their target after warmups + ramp_epochs.
    """
    effective = dict(loss_weights)
    task_warmup = int(schedule_cfg.get("task_warmup_epochs", 5))
    dyn_warmup = int(schedule_cfg.get("dynamics_warmup_epochs", 5))
    ramp = int(schedule_cfg.get("regularizer_ramp_epochs", 10))
    warmup_total = task_warmup + dyn_warmup

    if epoch < warmup_total:
        progress = 0.0
    elif ramp <= 0:
        progress = 1.0
    else:
        progress = min(max((epoch - warmup_total + 1) / float(ramp), 0.0), 1.0)

    # Ramp the main SID regularizers (cf invariance, sensitivity, decomp, anti, and the new role ones)
    for name in (
        "lambda_rev_cf",
        "lambda_task_ir_cf",
        "lambda_spur_sens",
        "lambda_spur_adv",
        "lambda_core_preserve",
        "lambda_spur_capture",
        "lambda_arrow_decomp",
        "lambda_anti_collapse",
    ):
        if name not in effective:
            continue
        start = float(schedule_cfg.get(f"{name}_start", 0.0))
        end = float(schedule_cfg.get(f"{name}_end", effective[name]))
        effective[name] = start + progress * (end - start)

    # Task and cf_task are usually always on, but we can gate adv etc. via the ramp
    effective["sid_schedule_progress"] = progress
    effective["sid_warmup_active"] = epoch < warmup_total
    effective["sid_adv_active"] = progress > 0.1  # rough "after guard"

    # Dynamics (arrow NLL on ir factors) ramp
    dyn_progress = 0.0
    if epoch >= task_warmup:
        if dyn_warmup <= 0:
            dyn_progress = 1.0
        else:
            dyn_progress = min(max((epoch - task_warmup + 1) / float(dyn_warmup), 0.0), 1.0)
    for name in ("lambda_task_forward", "lambda_task_reverse", "lambda_spur_forward", "lambda_spur_reverse"):
        if name not in effective:
            continue
        start = float(schedule_cfg.get(f"{name}_start", 0.0))
        end = float(schedule_cfg.get(f"{name}_end", effective.get(name, 1.0)))
        effective[name] = start + dyn_progress * (end - start)
    effective["sid_dynamics_progress"] = dyn_progress

    return effective


def _scheduled_itm_weights(
    loss_weights: dict[str, Any],
    schedule_cfg: dict[str, Any],
    epoch: int,
) -> dict[str, Any]:
    """Ramp ITM transition-mechanism role losses after task/transition warmup."""

    effective = dict(loss_weights)
    task_warmup = int(schedule_cfg.get("task_warmup_epochs", 5))
    transition_warmup = int(schedule_cfg.get("transition_warmup_epochs", 5))
    ramp = int(schedule_cfg.get("regularizer_ramp_epochs", 10))
    warmup_total = task_warmup + transition_warmup

    if epoch < warmup_total:
        role_progress = 0.0
    elif ramp <= 0:
        role_progress = 1.0
    else:
        role_progress = min(max((epoch - warmup_total + 1) / float(ramp), 0.0), 1.0)

    transition_progress = 0.0
    if epoch >= task_warmup:
        if transition_warmup <= 0:
            transition_progress = 1.0
        else:
            transition_progress = min(
                max((epoch - task_warmup + 1) / float(transition_warmup), 0.0),
                1.0,
            )

    for name in ("lambda_core_transition", "lambda_spur_transition"):
        if name not in effective:
            continue
        start = float(schedule_cfg.get(f"{name}_start", 0.0))
        end = float(schedule_cfg.get(f"{name}_end", effective[name]))
        effective[name] = start + transition_progress * (end - start)

    for name in (
        "lambda_core_mech_cf",
        "lambda_spur_mech_sens",
        "lambda_core_preserve",
        "lambda_spur_capture",
        "lambda_spur_adv",
        "lambda_anti_collapse",
    ):
        if name not in effective:
            continue
        start = float(schedule_cfg.get(f"{name}_start", 0.0))
        end = float(schedule_cfg.get(f"{name}_end", effective[name]))
        effective[name] = start + role_progress * (end - start)

    effective["itm_transition_schedule_progress"] = transition_progress
    effective["itm_schedule_progress"] = role_progress
    effective["itm_warmup_active"] = epoch < warmup_total
    return effective


def _sid_schedule_required_epochs(schedule_cfg: Mapping[str, Any]) -> int:
    """Number of epochs needed before SID role regularizers reach full weight."""

    task_warmup = max(int(schedule_cfg.get("task_warmup_epochs", 5)), 0)
    dyn_warmup = max(int(schedule_cfg.get("dynamics_warmup_epochs", 5)), 0)
    ramp = max(int(schedule_cfg.get("regularizer_ramp_epochs", 10)), 0)
    return max(task_warmup + dyn_warmup + ramp, 1)


def _itm_schedule_required_epochs(schedule_cfg: Mapping[str, Any]) -> int:
    """Number of epochs needed before ITM mechanism losses reach full weight."""

    task_warmup = max(int(schedule_cfg.get("task_warmup_epochs", 5)), 0)
    transition_warmup = max(int(schedule_cfg.get("transition_warmup_epochs", 5)), 0)
    ramp = max(int(schedule_cfg.get("regularizer_ramp_epochs", 10)), 0)
    return max(task_warmup + transition_warmup + ramp, 1)


def _resolve_epoch_floor(
    training_cfg: Mapping[str, Any],
    *,
    key: str,
    default: int,
    epochs: int,
) -> int:
    value = int(training_cfg.get(key, default))
    return min(max(value, 0), max(int(epochs), 0))


def _weighted_component_values(
    components: dict[str, float],
    weights: Mapping[str, Any],
) -> dict[str, float]:
    weighted = {
        "weighted_loss_task": float(components.get("task", 0.0)),
        "weighted_loss_task_cf": float(weights.get("lambda_cf_task", 1.0))
        * float(components.get("task_cf", 0.0)),
        "weighted_loss_forward_nll": float(weights.get("lambda_f", 1.0))
        * float(components.get("forward_nll", 0.0)),
        "weighted_loss_reverse_nll": float(weights.get("lambda_r", 1.0))
        * float(components.get("reverse_nll", 0.0)),
        "weighted_loss_cf_arrow_total": float(weights.get("eta_total", 1.0))
        * float(components.get("cf_arrow_total", 0.0)),
        "weighted_loss_cf_arrow_step": float(weights.get("eta_step", 0.1))
        * float(components.get("cf_arrow_step", 0.0)),
        "weighted_loss_setpoint": float(weights.get("rho", 1.0))
        * float(components.get("setpoint", 0.0)),
    }
    weighted["weighted_loss_dynamics"] = (
        weighted["weighted_loss_forward_nll"] + weighted["weighted_loss_reverse_nll"]
    )
    weighted["weighted_loss_regularizers"] = (
        weighted["weighted_loss_dynamics"]
        + weighted["weighted_loss_cf_arrow_total"]
        + weighted["weighted_loss_cf_arrow_step"]
        + weighted["weighted_loss_setpoint"]
    )
    task_scale = abs(weighted["weighted_loss_task"] + weighted["weighted_loss_task_cf"]) + 1e-8
    weighted["loss_component_ratio_regularizer_to_task_abs"] = (
        abs(weighted["weighted_loss_regularizers"]) / task_scale
    )
    weighted["loss_component_ratio_dynamics_to_task_abs"] = (
        abs(weighted["weighted_loss_dynamics"]) / task_scale
    )
    weighted["loss_component_ratio_cf_arrow_to_task_abs"] = (
        abs(weighted["weighted_loss_cf_arrow_total"] + weighted["weighted_loss_cf_arrow_step"])
        / task_scale
    )
    return weighted


def _sid_loss_scale_values(
    components: dict[str, float],
    weights: Mapping[str, Any],
) -> dict[str, float]:
    """Group weighted SID losses into task/dynamics/role fractions."""

    lambda_arrow = float(weights.get("lambda_arrow_decomp", 0.5))
    task = abs(float(components.get("weighted_task", 0.0))) + abs(
        float(components.get("weighted_cf_task", 0.0))
    )
    dynamics = abs(lambda_arrow * float(components.get("arrow_dynamics_nll", 0.0)))
    arrow_alignment = abs(lambda_arrow * float(components.get("arrow_alignment", 0.0)))
    role = sum(
        abs(float(components.get(name, 0.0)))
        for name in (
            "weighted_rev_cf_invariance",
            "weighted_task_ir_cf_invariance",
            "weighted_spur_ir_cf_sensitivity",
            "weighted_spur_adversary",
            "weighted_core_preservation",
            "weighted_spur_capture",
            "weighted_anti_collapse",
        )
    )
    total = task + dynamics + arrow_alignment + role
    denom = total + 1e-8
    return {
        "sid_weighted_task_loss_abs": task,
        "sid_weighted_dynamics_loss_abs": dynamics,
        "sid_weighted_role_loss_abs": role,
        "sid_weighted_arrow_alignment_loss_abs": arrow_alignment,
        "sid_weighted_loss_total_abs": total,
        "task_loss_fraction": task / denom,
        "dynamics_loss_fraction": dynamics / denom,
        "role_loss_fraction": role / denom,
        "arrow_alignment_loss_fraction": arrow_alignment / denom,
    }


def _module_grad_norm(module: torch.nn.Module) -> float:
    total = 0.0
    for parameter in module.parameters():
        if parameter.grad is None:
            continue
        value = float(parameter.grad.detach().float().norm(2).cpu())
        total += value * value
    return math.sqrt(total)


def _sid_gradient_norm_diagnostics(model: SIDModel) -> dict[str, float]:
    """Return grouped SID gradient norms after backward and before clipping."""

    return {
        "sid_grad_norm_encoder": _module_grad_norm(model.encoder),
        "sid_grad_norm_z_rev_proj": _module_grad_norm(model.z_rev_proj),
        "sid_grad_norm_z_ir_task_proj": _module_grad_norm(model.z_ir_task_proj),
        "sid_grad_norm_z_ir_spur_proj": _module_grad_norm(model.z_ir_spur_proj),
        "sid_grad_norm_classifier": _module_grad_norm(model.classifier),
        "sid_grad_norm_spur_adversary_head": _module_grad_norm(
            model.spur_adversary_head
        ),
        "sid_grad_norm_core_preservation_head": _module_grad_norm(
            model.core_preservation_head
        ),
        "sid_grad_norm_spur_capture_head": _module_grad_norm(model.spur_capture_head),
        "sid_grad_norm_task_dynamics": _module_grad_norm(model.task_forward_dynamics)
        + _module_grad_norm(model.task_reverse_dynamics),
        "sid_grad_norm_spur_dynamics": _module_grad_norm(model.spur_forward_dynamics)
        + _module_grad_norm(model.spur_reverse_dynamics),
    }


def _sib_specific_terms_active(weights: Mapping[str, Any]) -> bool:
    """Return whether SIB-specific training terms are active in the loss."""

    return any(
        abs(float(weights.get(name, default))) > 0.0
        for name, default in (
            ("lambda_cf_task", 1.0),
            ("lambda_f", 1.0),
            ("lambda_r", 1.0),
            ("eta_total", 1.0),
            ("eta_step", 0.1),
            ("rho", 1.0),
        )
    )


def _merge_epoch_diagnostics(
    totals: dict[str, float],
    bools: dict[str, bool],
    count: int,
) -> dict[str, float | bool]:
    merged = {key: value / max(count, 1) for key, value in totals.items()}
    merged.update(bools)
    return merged


def _accumulate_numeric(
    totals: dict[str, float],
    bools: dict[str, bool],
    values: dict[str, float | bool | str],
) -> None:
    for key, value in values.items():
        if isinstance(value, bool):
            bools[key] = bools.get(key, False) or value
        elif isinstance(value, (int, float)) and math.isfinite(float(value)):
            totals[key] = totals.get(key, 0.0) + float(value)


def _ep_regularizer_mode(method: str, loss_weights: dict[str, Any]) -> str:
    configured = loss_weights.get("ep_regularizer_mode", loss_weights.get("ep_regularizer"))
    if isinstance(configured, dict):
        configured = configured.get(method) or configured.get(f"default_for_{method}")
    if configured is None:
        configured = "abs_mean" if method in {"ep_min", "ep_max"} else "signed_mean"
    mode = str(configured)
    if mode not in {"signed_mean", "abs_mean", "squared_mean"}:
        raise ValueError(
            "EP regularizer mode must be one of signed_mean, abs_mean, squared_mean"
        )
    return mode


def _ep_regularizer_value(sigma_per_step: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "signed_mean":
        return sigma_per_step.mean()
    if mode == "abs_mean":
        return sigma_per_step.abs().mean()
    if mode == "squared_mean":
        return sigma_per_step.pow(2).mean()
    raise AssertionError(mode)


@torch.no_grad()
def evaluate_classifier(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    return _classification_diagnostics(model, loader, device)["accuracy"]


@torch.no_grad()
def evaluate_counterfactual_sensitivity(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    arrow_calibration_config: Mapping[str, Any] | None = None,
) -> dict[str, float | bool | None]:
    model.eval()
    total = 0
    pred_consistency_sum = 0.0
    cf_correct_sum = 0.0
    prob_drift_sum = 0.0
    delta_total_sum = 0.0
    delta_step_sum = 0.0
    calibrated_delta_total_sum = 0.0
    calibrated_delta_step_sum = 0.0
    calibration_cfg = arrow_calibration_config or {"mode": "none"}
    has_arrow_dynamics = hasattr(model, "forward_dynamics") and hasattr(
        model, "reverse_dynamics"
    )
    for batch in loader:
        x = batch[0].to(device)
        y = batch[1].to(device)
        x_cf = batch[2].to(device)
        out = model(x)
        out_cf = model(x_cf)
        probs = F.softmax(out["logits"], dim=-1)
        probs_cf = F.softmax(out_cf["logits"], dim=-1)
        batch_size = int(x.shape[0])
        pred_consistency_sum += float((probs.argmax(dim=-1) == probs_cf.argmax(dim=-1)).float().sum().cpu())
        cf_correct_sum += float((probs_cf.argmax(dim=-1) == y).float().sum().cpu())
        prob_drift_sum += float((probs - probs_cf).abs().sum(dim=-1).sum().cpu())
        if has_arrow_dynamics:
            score = latent_arrow_score(out["z"], model.forward_dynamics, model.reverse_dynamics)
            score_cf = latent_arrow_score(
                out_cf["z"], model.forward_dynamics, model.reverse_dynamics
            )
            delta_total_sum += float((score.sigma_total - score_cf.sigma_total).abs().sum().cpu())
            per_sample_step_delta = (score.sigma_steps - score_cf.sigma_steps).abs().mean(dim=1)
            delta_step_sum += float(per_sample_step_delta.sum().cpu())
            calibrated_score = calibrate_arrow_score(
                score,
                mode=str(calibration_cfg.get("mode", "none")),
                z=out["z"],
                forward_model=model.forward_dynamics,
                reverse_model=model.reverse_dynamics,
                reference_offset=float(
                    calibration_cfg.get(
                        "reference_offset", calibration_cfg.get("offset", 0.0)
                    )
                ),
            ).calibrated
            calibrated_score_cf = calibrate_arrow_score(
                score_cf,
                mode=str(calibration_cfg.get("mode", "none")),
                z=out_cf["z"],
                forward_model=model.forward_dynamics,
                reverse_model=model.reverse_dynamics,
                reference_offset=float(
                    calibration_cfg.get(
                        "reference_offset", calibration_cfg.get("offset", 0.0)
                    )
                ),
            ).calibrated
            calibrated_delta_total_sum += float(
                (calibrated_score.sigma_total - calibrated_score_cf.sigma_total)
                .abs()
                .sum()
                .cpu()
            )
            calibrated_per_sample_step_delta = (
                calibrated_score.sigma_steps - calibrated_score_cf.sigma_steps
            ).abs().mean(dim=1)
            calibrated_delta_step_sum += float(
                calibrated_per_sample_step_delta.sum().cpu()
            )
        total += batch_size
    denom = max(total, 1)
    return {
        "cf_arrow_metrics_available": has_arrow_dynamics,
        "cf_delta_arrow_total": delta_total_sum / denom if has_arrow_dynamics else None,
        "cf_delta_arrow_step": delta_step_sum / denom if has_arrow_dynamics else None,
        "cf_delta_arrow_total_calibrated": (
            calibrated_delta_total_sum / denom if has_arrow_dynamics else None
        ),
        "cf_delta_arrow_step_calibrated": (
            calibrated_delta_step_sum / denom if has_arrow_dynamics else None
        ),
        "cf_accuracy": cf_correct_sum / denom,
        "cf_prediction_consistency": pred_consistency_sum / denom,
        "cf_probability_l1_drift": prob_drift_sum / denom,
    }


def _resolve_sigma_target(config: dict[str, Any], splits: dict[str, dict[str, Any]]) -> float:
    resolved = _resolve_setpoint(config, splits, model=None, device=torch.device("cpu"))
    return float(resolved["sigma_target"])


def _one_hot_np(states, n_states: int):
    import numpy as np

    return np.eye(n_states, dtype=np.float32)[states]


def _core_only_observations(split: dict[str, Any]) -> torch.Tensor:
    import numpy as np

    metadata = split["metadata"]
    n_core_states = int(metadata["n_core_states"])
    n_spur_states = int(metadata["n_spur_states"])
    core_scale = float(metadata["observation"]["core_scale"])
    obs_dim = int(metadata["observation"]["obs_dim"])
    core = core_scale * _one_hot_np(split["c"], n_core_states)
    spur = np.zeros((*split["c"].shape, n_spur_states), dtype=np.float32)
    h = np.concatenate([core, spur], axis=-1)
    x = np.einsum("oi,nli->nlo", split["mixing_matrix"], h, optimize=True)
    return torch.as_tensor(x.reshape(split["x"].shape[0], split["x"].shape[1], obs_dim), dtype=torch.float32)


def _estimate_oracle_core_reference(
    model: torch.nn.Module,
    config: dict[str, Any],
    splits: dict[str, dict[str, Any]],
    device: torch.device,
) -> float:
    if not hasattr(model, "forward_dynamics") or not hasattr(model, "reverse_dynamics"):
        raise ValueError("oracle_core_reference requires a model with forward/reverse dynamics")
    reference = deepcopy(model).to(device)
    reference.train()
    setpoint_cfg = config.get("setpoint", {})
    training_cfg = config.get("training", {})
    epochs = int(setpoint_cfg.get("reference_epochs", 2))
    batch_size = int(setpoint_cfg.get("reference_batch_size", training_cfg.get("batch_size", 128)))
    x_core = _core_only_observations(splits["train"])
    loader = DataLoader(TensorDataset(x_core), batch_size=batch_size, shuffle=True)
    reference_training_cfg = {
        **training_cfg,
        "lr": float(setpoint_cfg.get("reference_lr", training_cfg.get("lr", 1e-3))),
    }
    opt = _make_optimizer(
        list(reference.encoder.parameters())
        + list(reference.forward_dynamics.parameters())
        + list(reference.reverse_dynamics.parameters()),
        reference_training_cfg,
    )
    grad_clip = float(training_cfg.get("grad_clip_norm", 1.0))
    for _ in range(epochs):
        for (x_batch,) in loader:
            x_batch = x_batch.to(device)
            opt.zero_grad(set_to_none=True)
            z = reference.encoder(x_batch)
            f_nll, r_nll = dynamics_nlls(z, reference.forward_dynamics, reference.reverse_dynamics)
            loss = f_nll + r_nll
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite oracle_core_reference dynamics loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(reference.parameters(), grad_clip)
            opt.step()
    reference.eval()
    sigma_values = []
    with torch.no_grad():
        for (x_batch,) in loader:
            z = reference.encoder(x_batch.to(device))
            score = latent_arrow_score(z, reference.forward_dynamics, reference.reverse_dynamics)
            sigma_values.append(score.sigma_per_step.detach().cpu())
    if not sigma_values:
        raise ValueError("oracle_core_reference could not compute reference sigma")
    return float(torch.cat(sigma_values).mean().item())


def _resolve_setpoint(
    config: dict[str, Any],
    splits: dict[str, dict[str, Any]],
    model: torch.nn.Module | None,
    device: torch.device,
) -> dict[str, Any]:
    setpoint = config.get("setpoint", {})
    training_cfg = config.get("training", {})
    mode = str(setpoint.get("mode", "fixed_grid"))
    if mode not in {
        "analytic_direct",
        "oracle_core_reference",
        "val_iid_sweep",
        "fixed_grid",
        "calibrated_val_reference",
    }:
        raise ValueError(f"unknown setpoint mode {mode!r}")
    if "target" in setpoint:
        target = float(setpoint["target"])
        source = "explicit_target"
    elif mode == "analytic_direct":
        target = float(splits["train"]["metadata"]["core"]["analytic_ep"])
        source = "train_core_analytic_ep"
    elif mode == "oracle_core_reference":
        if "fixed_target" in setpoint:
            target = float(setpoint["fixed_target"])
            source = "precomputed_oracle_core_reference"
        else:
            if model is None:
                raise ValueError("oracle_core_reference requires model when no fixed_target is set")
            target = _estimate_oracle_core_reference(model, config, splits, device)
            source = "estimated_oracle_core_reference"
    elif mode in {"val_iid_sweep", "calibrated_val_reference"}:
        if "selected_target" not in setpoint:
            raise ValueError(
                f"{mode} requires setpoint.selected_target from a val_iid-only selector"
            )
        target = float(setpoint["selected_target"])
        source = "val_iid_selected_target" if mode == "val_iid_sweep" else f"{mode}_selected_target"
    else:
        if "fixed_target" not in setpoint:
            raise ValueError("fixed_grid requires setpoint.fixed_target")
        target = float(setpoint["fixed_target"])
        source = "fixed_grid_target"
    resolved = {
        **setpoint,
        "mode": mode,
        "sigma_target": target,
        "target_source": source,
        "oracle_assisted": mode == "oracle_core_reference",
        "selection_split": (
            "val_iid" if mode in {"val_iid_sweep", "calibrated_val_reference"} else None
        ),
    }
    if mode == "oracle_core_reference" and source == "estimated_oracle_core_reference":
        resolved.update(
            {
                "reference_epochs": int(setpoint.get("reference_epochs", 2)),
                "reference_batch_size": int(
                    setpoint.get("reference_batch_size", training_cfg.get("batch_size", 128))
                ),
                "reference_lr": float(setpoint.get("reference_lr", training_cfg.get("lr", 1e-3))),
                "estimated_sigma_target": target,
            }
        )
    return resolved


def _calibrate_score_from_config(
    score: ArrowScoreOutput,
    config: dict[str, Any],
    *,
    z: torch.Tensor,
    model: torch.nn.Module,
) -> tuple[ArrowScoreOutput, dict[str, Any]]:
    calibration_cfg = config.get("arrow_calibration", {})
    mode = str(calibration_cfg.get("mode", "none"))
    offset = float(calibration_cfg.get("reference_offset", calibration_cfg.get("offset", 0.0)))
    calibrated = calibrate_arrow_score(
        score,
        mode=mode,
        z=z,
        forward_model=model.forward_dynamics,
        reverse_model=model.reverse_dynamics,
        reference_offset=offset,
    )
    return calibrated.calibrated, calibrated.metadata


def train_supervised_method(
    method: SupervisedMethod,
    config: dict[str, Any],
    run_dir: Path,
    device: torch.device,
    resume: str | None = None,
) -> dict[str, Any]:
    seed = int(config.get("seed", 0))
    training_cfg = config.get("training", {})
    _check_supported_config(config)
    seed_everything(seed, deterministic=bool(training_cfg.get("deterministic", False)))
    splits = generate_splits_from_config(config)
    input_dim = int(splits["train"]["x"].shape[-1])
    include_cf = method in {"sib", "sid", "itm"}
    # For full SID we include decomp stats (core/spurious dynamic) in batches so that
    # the auxiliary role-enforcement losses (adversary, core_preservation, spur_capture)
    # can be computed on controlled synthetic benchmarks. These aux terms only affect
    # the factor representations; the task head still only sees x-derived z_rev + z_ir_task.
    include_decomp_stats = method in {"sid", "itm"}
    loader_settings = _data_loader_settings(config, device)
    loaders = make_loaders(
        splits,
        batch_size=int(training_cfg.get("batch_size", 128)),
        include_cf=include_cf,
        include_decomp_stats=include_decomp_stats,
        device=device,
        num_workers=int(loader_settings["num_workers"]),
        persistent_workers=bool(loader_settings["persistent_workers"]),
        seed=int(loader_settings["worker_seed"]),
    )
    model = build_supervised_model(method, config, input_dim).to(device)
    if resume:
        payload = load_checkpoint(resume, map_location=device)
        model.load_state_dict(payload["model"])

    dynamics_cfg = config.get("dynamics", {})
    arrow_calibration_cfg = config.get("arrow_calibration", {"mode": "none"})
    loss_normalization_cfg = config.get("loss_normalization", {})
    sid_loss_normalization_cfg = config.get("sid_loss_normalization", {})
    sib_schedule_cfg = config.get("sib_schedule", {})
    sid_schedule_cfg = config.get("sid_schedule", {})
    task_guard_cfg = config.get("task_guard", {})
    opt = _make_optimizer(model.parameters(), training_cfg)
    logger = JsonlLogger(run_dir / "metrics.jsonl")
    if method == "sib" or config.get("setpoint") is not None:
        resolved_setpoint = _resolve_setpoint(config, splits, model=model, device=device)
    else:
        resolved_setpoint = {
            "mode": "not_applicable",
            "sigma_target": 0.0,
            "target_source": None,
            "oracle_assisted": False,
            "selection_split": None,
        }
    sigma_target = float(resolved_setpoint["sigma_target"])
    save_json(run_dir / "resolved_config.json", config)
    save_json(
        run_dir / "metadata.json",
        {
            **run_metadata(
                method,
                config,
                run_dir,
                device,
                parameter_count(model),
                loader_settings,
            ),
            "dataset_metadata": {k: v["metadata"] for k, v in splits.items()},
            "setpoint": resolved_setpoint,
            "dynamics_training": {
                "train_on_counterfactual": bool(
                    dynamics_cfg.get("train_on_counterfactual", False)
                ),
                "fixed_variance": bool(dynamics_cfg.get("fixed_variance", False)),
                "sigma_per_step_abs_threshold": float(
                    dynamics_cfg.get("sigma_per_step_abs_threshold", 1e4)
                ),
            },
            "sib_variant": config.get("sib_variant", config.get("method_variant")),
            "arrow_calibration": {
                "mode": str(arrow_calibration_cfg.get("mode", "none")),
                "uses_ood_test": False,
                "reference_offset": float(
                    arrow_calibration_cfg.get(
                        "reference_offset", arrow_calibration_cfg.get("offset", 0.0)
                    )
                ),
            },
            "loss_normalization": loss_normalization_cfg,
            "sid_loss_normalization": sid_loss_normalization_cfg,
            "sib_schedule": sib_schedule_cfg,
            "sid_schedule": sid_schedule_cfg,
            "task_guard": task_guard_cfg,
        },
    )

    best_val = -math.inf
    best_epoch = -1
    epochs = int(training_cfg.get("epochs", 50))
    patience = int(training_cfg.get("patience", epochs + 1))
    epochs_without_improvement = 0
    sid_required_epochs = _sid_schedule_required_epochs(sid_schedule_cfg) if method == "sid" else 0
    default_min_epochs = sid_required_epochs if method == "sid" else 0
    min_epochs_before_early_stopping = _resolve_epoch_floor(
        training_cfg,
        key="min_epochs_before_early_stopping",
        default=default_min_epochs,
        epochs=epochs,
    )
    min_epoch_for_checkpoint_selection = _resolve_epoch_floor(
        training_cfg,
        key="min_epoch_for_checkpoint_selection",
        default=min_epochs_before_early_stopping,
        epochs=epochs,
    )
    grad_clip = float(training_cfg.get("grad_clip_norm", 1.0))
    loss_weights = config.get("loss_weights", {})
    split_acc: dict[str, float] = {}
    early_metric = _early_stopping_metric_name(training_cfg)
    collapse_cfg = config.get("collapse_detection", {})
    val_majority = _majority_baseline(splits["val_iid"])
    majority_tolerance = float(collapse_cfg.get("majority_tolerance", 0.02))
    constant_pred_threshold = float(collapse_cfg.get("constant_prediction_threshold", 0.95))
    entropy_threshold = float(
        collapse_cfg.get("high_entropy_threshold", 0.95 * math.log(2.0))
    )
    sigma_zero_tolerance = float(collapse_cfg.get("sigma_zero_tolerance", 0.05))
    regularizer_dominance_multiple = float(
        collapse_cfg.get("regularizer_dominance_multiple", 10.0)
    )
    latent_norm_min = float(collapse_cfg.get("latent_norm_min", 1e-3))
    latent_norm_std_min = float(collapse_cfg.get("latent_norm_std_min", 1e-4))
    task_guard_enabled = bool(task_guard_cfg.get("enabled", False)) and method == "sib"
    task_guard_selection_split = str(task_guard_cfg.get("selection_split", "val_iid"))
    if task_guard_enabled and task_guard_selection_split != "val_iid":
        raise ValueError("task_guard.selection_split must be val_iid; OOD/test splits are forbidden")
    task_guard_min = _task_guard_threshold(task_guard_cfg, val_majority=val_majority)
    task_guard_best_score: tuple[float, float, int] | None = None
    task_guard_best_epoch = -1
    task_guard_n_eligible_epochs = 0
    task_guard_selected_checkpoint: str | None = None

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        component_totals: dict[str, float] = {}
        diagnostic_totals: dict[str, float] = {}
        diagnostic_bools: dict[str, bool] = {}
        diagnostic_literals: dict[str, str] = {}
        effective_loss_weights = dict(loss_weights)
        if method == "sib":
            effective_loss_weights = _scheduled_sib_weights(loss_weights, sib_schedule_cfg, epoch)
        elif method == "sid":
            sid_schedule_cfg = config.get("sid_schedule", {})
            effective_loss_weights = _scheduled_sid_weights(loss_weights, sid_schedule_cfg, epoch)
        elif method == "itm":
            itm_schedule_cfg = config.get("itm_schedule", {})
            effective_loss_weights = _scheduled_itm_weights(loss_weights, itm_schedule_cfg, epoch)
        for batch in loaders["train"]:
            opt.zero_grad(set_to_none=True)
            if include_cf and include_decomp_stats:
                # Controlled mechanism methods: x, y, x_cf, core_stat, spur_stat
                x, y, x_cf, core_stat, spur_stat = (t.to(device) for t in batch)
            elif include_cf:
                x, y, x_cf = (t.to(device) for t in batch)
                core_stat = spur_stat = None
            else:
                x, y = (t.to(device) for t in batch)
                core_stat = spur_stat = None
            out = model(x)
            task_loss = F.cross_entropy(out["logits"], y)
            diagnostics: dict[str, float | bool | str] = {}
            if method == "erm":
                loss = task_loss
                components = {"task": task_loss.detach()}
            elif method == "ib":
                kl = gaussian_kl_to_standard_normal(out["mu"], out["logvar"])
                beta = float(loss_weights.get("beta_latent", 0.0))
                loss = task_loss + beta * kl
                components = {"task": task_loss.detach(), "ib_kl": kl.detach()}
            elif method in {"ep_min", "ep_max"}:
                score = latent_arrow_score(out["z"], model.forward_dynamics, model.reverse_dynamics)
                f_nll, r_nll = dynamics_nlls(
                    out["z"], model.forward_dynamics, model.reverse_dynamics
                )
                eta = float(effective_loss_weights.get("eta_total", 1.0))
                sign = 1.0 if method == "ep_min" else -1.0
                ep_mode = _ep_regularizer_mode(method, effective_loss_weights)
                ep_regularizer = _ep_regularizer_value(score.sigma_per_step, ep_mode)
                loss = (
                    task_loss
                    + float(effective_loss_weights.get("lambda_f", 1.0)) * f_nll
                    + float(effective_loss_weights.get("lambda_r", 1.0)) * r_nll
                    + sign * eta * ep_regularizer
                )
                components = {
                    "task": task_loss.detach(),
                    "forward_nll": f_nll.detach(),
                    "reverse_nll": r_nll.detach(),
                    "sigma_per_step": score.sigma_per_step.mean().detach(),
                    "sigma_per_step_abs_mean": score.sigma_per_step.abs().mean().detach(),
                    "sigma_per_step_squared_mean": score.sigma_per_step.pow(2).mean().detach(),
                    "ep_regularizer": ep_regularizer.detach(),
                }
                diagnostics = {"ep_regularizer_mode": ep_mode}
                diagnostics.update(_dynamics_diagnostics(
                    model,
                    out["z"],
                    score,
                    float(dynamics_cfg.get("sigma_per_step_abs_threshold", 1e4)),
                ))
            elif method == "itm":
                out_cf = model(x_cf)
                core_stat_n = (
                    None
                    if core_stat is None
                    else (core_stat.float() - core_stat.float().mean())
                    / (core_stat.float().std(unbiased=False) + 1e-8)
                )
                spur_stat_n = (
                    None
                    if spur_stat is None
                    else (spur_stat.float() - spur_stat.float().mean())
                    / (spur_stat.float().std(unbiased=False) + 1e-8)
                )
                spur_adversary_loss = None
                if spur_stat_n is not None:
                    grl_task_rep = GradientReversal.apply(
                        out["task_rep"], float(config.get("itm", {}).get("grl_alpha", 1.0))
                    )
                    adv_pred = model.spur_adversary_head(grl_task_rep).squeeze(-1)
                    spur_adversary_loss = F.mse_loss(adv_pred, spur_stat_n)
                loss_out = itm_loss(
                    task_loss=task_loss,
                    task_loss_cf=F.cross_entropy(out_cf["logits"], y),
                    core_next_pred=out["core_next_pred"],
                    core_next_target=out["z_core"][:, 1:],
                    spur_next_pred=out["spur_next_pred"],
                    spur_next_target=out["z_spur"][:, 1:],
                    core_delta=out["core_delta"],
                    core_delta_cf=out_cf["core_delta"],
                    spur_delta=out["spur_delta"],
                    spur_delta_cf=out_cf["spur_delta"],
                    core_stat_pred=out["core_stat_pred"],
                    core_stat=core_stat,
                    spur_stat_pred=out["spurious_stat_pred"],
                    spur_stat=spur_stat,
                    spur_adversary_loss=spur_adversary_loss,
                    anti_collapse_loss=itm_anti_collapse_loss(out["z_core"], out["z_spur"]),
                    weights=effective_loss_weights,
                    spur_sensitivity_margin=float(
                        config.get("itm", {}).get("spur_sensitivity_margin", 0.1)
                    ),
                )
                loss = loss_out.total
                components = {k: v.detach() for k, v in loss_out.components.items()}
                components.update(
                    {
                        f"weighted_{key}": value.detach()
                        for key, value in loss_out.weighted_components.items()
                    }
                )
                with torch.no_grad():
                    core_cf_mse = F.mse_loss(out["core_delta"], out_cf["core_delta"])
                    spur_cf_mse = F.mse_loss(out["spur_delta"], out_cf["spur_delta"])
                    diagnostics = {
                        "itm_core_mech_cf_mse": float(core_cf_mse.detach().cpu()),
                        "itm_spur_mech_cf_mse": float(spur_cf_mse.detach().cpu()),
                        "itm_core_transition_fit": float(
                            components["core_transition_fit"].detach().cpu()
                        ),
                        "itm_spur_transition_fit": float(
                            components["spur_transition_fit"].detach().cpu()
                        ),
                        "itm_core_std_mean": float(
                            out["z_core"].float().reshape(-1, out["z_core"].shape[-1])
                            .std(dim=0, unbiased=False)
                            .mean()
                            .detach()
                            .cpu()
                        ),
                        "itm_spur_std_mean": float(
                            out["z_spur"].float().reshape(-1, out["z_spur"].shape[-1])
                            .std(dim=0, unbiased=False)
                            .mean()
                            .detach()
                            .cpu()
                        ),
                    }
                    if core_stat_n is not None and spur_stat_n is not None:
                        diagnostics["itm_core_preservation_mse"] = float(
                            components["core_preservation"].detach().cpu()
                        )
                        diagnostics["itm_spur_capture_mse"] = float(
                            components["spur_capture"].detach().cpu()
                        )
                        diagnostics["itm_spur_adversary_mse"] = (
                            0.0
                            if spur_adversary_loss is None
                            else float(spur_adversary_loss.detach().cpu())
                        )
                    diagnostics["itm_schedule_progress"] = float(
                        effective_loss_weights.get("itm_schedule_progress", 1.0)
                    )
                    diagnostics["itm_transition_schedule_progress"] = float(
                        effective_loss_weights.get(
                            "itm_transition_schedule_progress", 1.0
                        )
                    )
                    diagnostics["itm_warmup_active"] = bool(
                        effective_loss_weights.get("itm_warmup_active", False)
                    )
            elif method == "sid":
                out_cf = model(x_cf)
                task_f_nll, task_r_nll = dynamics_nlls(
                    out["factors"]["z_ir_task"],
                    model.task_forward_dynamics,
                    model.task_reverse_dynamics,
                )
                spur_f_nll, spur_r_nll = dynamics_nlls(
                    out["factors"]["z_ir_spur"],
                    model.spur_forward_dynamics,
                    model.spur_reverse_dynamics,
                )
                task_score = latent_arrow_score(
                    out["factors"]["z_ir_task"],
                    model.task_forward_dynamics,
                    model.task_reverse_dynamics,
                )
                spur_score = latent_arrow_score(
                    out["factors"]["z_ir_spur"],
                    model.spur_forward_dynamics,
                    model.spur_reverse_dynamics,
                )
                # Base arrow decomp = dynamics NLL fit on the two ir factors (per spec)
                base_arrow_nll = (
                    float(effective_loss_weights.get("lambda_task_forward", 1.0))
                    * task_f_nll
                    + float(effective_loss_weights.get("lambda_task_reverse", 1.0))
                    * task_r_nll
                    + float(effective_loss_weights.get("lambda_spur_forward", 1.0))
                    * spur_f_nll
                    + float(effective_loss_weights.get("lambda_spur_reverse", 1.0))
                    * spur_r_nll
                )

                # Actual alignment term for L_arrow_decomposition: make the *arrow evidence* (sigma)
                # in z_ir_task correlate with core stat, and in z_ir_spur with spurious stat.
                # This is optimized (not just logged).
                arrow_align_loss = torch.zeros((), device=device, dtype=torch.float32)
                if core_stat is not None and spur_stat is not None:
                    ts = task_score.sigma_per_step
                    ss = spur_score.sigma_per_step
                    if ts.dim() > 1:
                        ts = ts.mean(dim=1)
                    if ss.dim() > 1:
                        ss = ss.mean(dim=1)

                    def _align_term(a, b):
                        a = a.float().view(-1)
                        b = b.float().view(-1)
                        if a.numel() < 2 or a.std() < 1e-6 or b.std() < 1e-6:
                            return torch.zeros((), device=a.device, dtype=torch.float32)
                        a_n = (a - a.mean()) / (a.std() + 1e-8)
                        b_n = (b - b.mean()) / (b.std() + 1e-8)
                        corr = (a_n * b_n).mean()
                        return - corr   # maximize corr between arrow evidence and stat

                    arrow_align_loss = _align_term(ts, core_stat) + _align_term(ss, spur_stat)

                arrow_decomposition_loss = base_arrow_nll + arrow_align_loss

                # === Full SID decomposition auxiliary losses ===
                # These use the decomp stats (available only on synthetic controlled benchmarks).
                # They are *auxiliary regularizers* to enforce factor roles.
                # The task head (rev + ir_task) still only receives gradients from task + cf_task + invariance + adv (which penalizes spur info in task rep).
                core_preservation_loss = None
                spur_capture_loss = None
                spur_adversary_loss = None
                spur_adv_mse = None
                if core_stat is not None and spur_stat is not None:
                    # L_core_preservation: z_ir_task should predict core stat well
                    core_pred = model.core_preservation_pred(out["factors"])
                    core_preservation_loss = F.mse_loss(core_pred, core_stat)

                    # L_spur_capture: z_ir_spur should predict spurious stat well (encourage capture)
                    spur_cap_pred = model.spur_capture_pred(out["factors"])
                    spur_capture_loss = F.mse_loss(spur_cap_pred, spur_stat)

                    # L_spur_adversary (proper adversarial, per review feedback):
                    # - Use GradientReversal so the spur_adversary_head is trained to *minimize* its MSE
                    #   (i.e. become good at extracting spurious info from task_rep).
                    # - The task_rep (encoder side) receives reversed gradient, effectively maximizing
                    #   the head's MSE (fooling the adversary, removing spurious info from representation).
                    # This prevents the head itself from being incentivized to "break" (which caused
                    # previous explosion to 100k+ MSE).
                    task_rep = model.task_representation(out["factors"])
                    grl_task_rep = GradientReversal.apply(task_rep, 1.0)
                    adv_pred = model.spur_adversary_head(grl_task_rep)
                    # Standardize target for the adversary to keep scale stable and prevent explosion
                    spur_stat_n = (spur_stat - spur_stat.mean()) / (spur_stat.std() + 1e-8)
                    spur_adv_mse = F.mse_loss(adv_pred.squeeze(-1), spur_stat_n)
                    spur_adversary_loss = spur_adv_mse  # positive term; GRL handles the opposing objective for features

                loss_out = sid_loss(
                    task_loss=task_loss,
                    task_loss_cf=F.cross_entropy(out_cf["logits"], y),
                    factors=out["factors"],
                    factors_cf=out_cf["factors"],
                    weights=effective_loss_weights,
                    spur_sensitivity_margin=float(
                        config.get("sid", {}).get("spur_sensitivity_margin", 0.1)
                    ),
                    arrow_decomposition_loss=arrow_decomposition_loss,
                    core_preservation_loss=core_preservation_loss,
                    spur_capture_loss=spur_capture_loss,
                    spurious_adversary_loss=spur_adversary_loss,
                )
                loss = loss_out.total
                components = {k: v.detach() for k, v in loss_out.components.items()}
                components.update(
                    {
                        "task_forward_nll": task_f_nll.detach(),
                        "task_reverse_nll": task_r_nll.detach(),
                        "spur_forward_nll": spur_f_nll.detach(),
                        "spur_reverse_nll": spur_r_nll.detach(),
                        "task_sigma_per_step": task_score.sigma_per_step.mean().detach(),
                        "spur_sigma_per_step": spur_score.sigma_per_step.mean().detach(),
                        "arrow_dynamics_nll": base_arrow_nll.detach(),
                        "arrow_alignment": arrow_align_loss.detach(),
                    }
                )
                components.update(
                    {
                        f"weighted_{key}": value.detach()
                        for key, value in loss_out.weighted_components.items()
                    }
                )
                diagnostics = sid_factor_diagnostics(out["factors"], out_cf["factors"])
                diagnostics.update(
                    {
                        "sid_task_sigma_per_step_abs_max": float(
                            task_score.sigma_per_step.abs().max().detach().cpu()
                        ),
                        "sid_spur_sigma_per_step_abs_max": float(
                            spur_score.sigma_per_step.abs().max().detach().cpu()
                        ),
                    }
                )
                # Log schedule state for SID (required for audit and reproducibility)
                diagnostics["sid_schedule_progress"] = float(effective_loss_weights.get("sid_schedule_progress", 1.0))
                diagnostics["sid_warmup_active"] = bool(effective_loss_weights.get("sid_warmup_active", False))
                diagnostics["sid_adv_active"] = bool(effective_loss_weights.get("sid_adv_active", True))

                if core_stat is not None and spur_stat is not None:
                    # Log alignment diagnostics (for audit and paper)
                    with torch.no_grad():
                        ts = task_score.sigma_per_step  # already per-sample avg (shape [B])
                        ss = spur_score.sigma_per_step
                        # Pearson corr (per batch) between factor arrow and the dynamic stat
                        def _safe_corr(a, b):
                            a = a.float().view(-1)
                            b = b.float().view(-1)
                            if a.numel() < 2 or a.std() < 1e-6 or b.std() < 1e-6:
                                return 0.0
                            # manual pearson to avoid stack dim issues
                            a = a - a.mean()
                            b = b - b.mean()
                            denom = (a.std(unbiased=False) * b.std(unbiased=False)).clamp_min(1e-12)
                            return float((a * b).sum() / (a.numel() * denom))
                        diagnostics["sid_arrow_task_core_corr"] = _safe_corr(ts, core_stat)
                        diagnostics["sid_arrow_spur_spur_corr"] = _safe_corr(ss, spur_stat)
                        diagnostics["sid_core_preservation_mse"] = float(core_preservation_loss.detach().cpu()) if core_preservation_loss is not None else 0.0
                        diagnostics["sid_spur_capture_mse"] = float(spur_capture_loss.detach().cpu()) if spur_capture_loss is not None else 0.0
                        diagnostics["sid_spur_adversary_mse"] = float(spur_adv_mse.detach().cpu()) if spur_adv_mse is not None else 0.0
                        # Note: with GRL, spur_adversary_loss = +raw_mse (head trained to minimize it).
                        # sid_spur_adversary_mse (raw) high means the current task_rep fools the (improving) head well.
                        diagnostics["sid_arrow_align_loss"] = float(arrow_align_loss.detach().cpu()) if arrow_align_loss is not None else 0.0
            elif method == "sib":
                sib_terms_active = _sib_specific_terms_active(effective_loss_weights)
                if not sib_terms_active:
                    zero = task_loss.detach().new_zeros(())
                    loss = task_loss
                    components = {
                        "task": task_loss.detach(),
                        "task_cf": zero,
                        "forward_nll": zero,
                        "reverse_nll": zero,
                        "cf_arrow_total": zero,
                        "cf_arrow_step": zero,
                        "setpoint": zero,
                    }
                    diagnostics = {
                        "sib_task_only_training_fast_path": True,
                        "dynamics_train_on_counterfactual": False,
                    }
                else:
                    out_cf = model(x_cf)
                    raw_score = latent_arrow_score(
                        out["z"], model.forward_dynamics, model.reverse_dynamics
                    )
                    raw_score_cf = latent_arrow_score(
                        out_cf["z"], model.forward_dynamics, model.reverse_dynamics
                    )
                    score, calibration_metadata = _calibrate_score_from_config(
                        raw_score, config, z=out["z"], model=model
                    )
                    score_cf, cf_calibration_metadata = _calibrate_score_from_config(
                        raw_score_cf, config, z=out_cf["z"], model=model
                    )
                    f_nll, r_nll = dynamics_nlls(
                        out["z"], model.forward_dynamics, model.reverse_dynamics
                    )
                    train_cf_dynamics = bool(dynamics_cfg.get("train_on_counterfactual", False))
                    cf_dynamics_components: dict[str, torch.Tensor] = {}
                    if train_cf_dynamics:
                        f_cf_nll, r_cf_nll = dynamics_nlls(
                            out_cf["z"], model.forward_dynamics, model.reverse_dynamics
                        )
                        f_nll = 0.5 * (f_nll + f_cf_nll)
                        r_nll = 0.5 * (r_nll + r_cf_nll)
                        cf_dynamics_components = {
                            "forward_nll_cf": f_cf_nll.detach(),
                            "reverse_nll_cf": r_cf_nll.detach(),
                        }
                    loss_out = sib_loss(
                        task_loss=task_loss,
                        task_loss_cf=F.cross_entropy(out_cf["logits"], y),
                        forward_nll=f_nll,
                        reverse_nll=r_nll,
                        sigma_total=score.sigma_total,
                        sigma_total_cf=score_cf.sigma_total,
                        sigma_steps=score.sigma_steps,
                        sigma_steps_cf=score_cf.sigma_steps,
                        sigma_target=sigma_target,
                        weights=effective_loss_weights,
                        loss_normalization=loss_normalization_cfg,
                    )
                    loss = loss_out.total
                    components = {k: v.detach() for k, v in loss_out.components.items()}
                    components.update(cf_dynamics_components)
                    diagnostics = _dynamics_diagnostics(
                        model,
                        out["z"],
                        score,
                        float(dynamics_cfg.get("sigma_per_step_abs_threshold", 1e4)),
                        prefix="calibrated",
                    )
                    diagnostics.update(
                        _dynamics_diagnostics(
                            model,
                            out["z"],
                            raw_score,
                            float(dynamics_cfg.get("sigma_per_step_abs_threshold", 1e4)),
                            prefix="raw",
                        )
                    )
                    diagnostics.update(calibration_metadata)
                    diagnostics["sigma_per_step_mean"] = diagnostics[
                        "calibrated_sigma_per_step_mean"
                    ]
                    diagnostics["sigma_per_step_std"] = diagnostics[
                        "calibrated_sigma_per_step_std"
                    ]
                    diagnostics["sigma_total_mean"] = diagnostics[
                        "calibrated_sigma_total_mean"
                    ]
                    diagnostics["sigma_total_std"] = diagnostics[
                        "calibrated_sigma_total_std"
                    ]
                    diagnostics["sigma_total_min"] = diagnostics[
                        "calibrated_sigma_total_min"
                    ]
                    diagnostics["sigma_total_max"] = diagnostics[
                        "calibrated_sigma_total_max"
                    ]
                    diagnostics["sigma_steps_mean"] = diagnostics[
                        "calibrated_sigma_steps_mean"
                    ]
                    diagnostics["sigma_steps_std"] = diagnostics[
                        "calibrated_sigma_steps_std"
                    ]
                    diagnostics["sigma_steps_min"] = diagnostics[
                        "calibrated_sigma_steps_min"
                    ]
                    diagnostics["sigma_steps_max"] = diagnostics[
                        "calibrated_sigma_steps_max"
                    ]
                    diagnostics["latent_norm_mean"] = diagnostics[
                        "calibrated_latent_norm_mean"
                    ]
                    diagnostics["latent_norm_std"] = diagnostics[
                        "calibrated_latent_norm_std"
                    ]
                    diagnostics["sigma_per_step_abs_max"] = diagnostics[
                        "calibrated_sigma_per_step_abs_max"
                    ]
                    diagnostics["sigma_threshold_exceeded"] = diagnostics[
                        "calibrated_sigma_threshold_exceeded"
                    ]
                    diagnostics["arrow_calibration_cf_offset"] = float(
                        cf_calibration_metadata["arrow_calibration_offset"]
                    )
                    diagnostics["dynamics_train_on_counterfactual"] = train_cf_dynamics
                    diagnostics["sib_task_only_training_fast_path"] = False
            else:
                raise AssertionError(method)

            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss for method {method}")
            loss.backward()
            if (
                method == "sid"
                and isinstance(model, SIDModel)
                and bool(sid_loss_normalization_cfg.get("log_gradient_norms", False))
            ):
                diagnostics.update(_sid_gradient_norm_diagnostics(model))
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            epoch_loss += float(loss.detach().cpu())
            n_batches += 1
            for key, value in components.items():
                component_totals[key] = component_totals.get(key, 0.0) + float(value.detach().cpu())
            for key, value in diagnostics.items():
                if isinstance(value, str):
                    diagnostic_literals[key] = value
            _accumulate_numeric(diagnostic_totals, diagnostic_bools, diagnostics)

        split_eval: dict[str, float] = {}
        split_acc = {}
        for name, loader in loaders.items():
            stats = _classification_diagnostics(model, loader, device)
            split_acc[f"{name}_accuracy"] = stats["accuracy"]
            for stat_name, value in stats.items():
                split_eval[f"{name}_{stat_name}"] = value
        split_acc["ood_gap"] = ood_gap(
            split_acc["iid_test_accuracy"], split_acc["ood_test_accuracy"]
        )
        epoch_components = {
            key: value / max(n_batches, 1) for key, value in component_totals.items()
        }
        epoch_diagnostics = {
            **_merge_epoch_diagnostics(diagnostic_totals, diagnostic_bools, n_batches),
            **diagnostic_literals,
        }
        weighted_components = (
            _weighted_component_values(epoch_components, effective_loss_weights)
            if method == "sib"
            else {}
        )
        if method == "sid":
            weighted_components.update(
                _sid_loss_scale_values(epoch_components, effective_loss_weights)
            )
        if method == "sib":
            val_cf_metrics = {
                f"val_iid_{key}": value
                for key, value in evaluate_counterfactual_sensitivity(
                    model,
                    loaders["val_iid"],
                    device,
                    arrow_calibration_config=arrow_calibration_cfg,
                ).items()
            }
            weighted_components.update(val_cf_metrics)
            near_chance = split_acc["val_iid_accuracy"] <= val_majority + majority_tolerance
            regularizer_ratio = float(
                weighted_components.get("loss_component_ratio_regularizer_to_task_abs", 0.0)
            )
            sigma_mean = float(
                epoch_diagnostics.get(
                    "calibrated_sigma_per_step_mean",
                    epoch_diagnostics.get("sigma_per_step_mean", 0.0),
                )
            )
            sigma_std = float(
                epoch_diagnostics.get(
                    "calibrated_sigma_per_step_std",
                    epoch_diagnostics.get("sigma_per_step_std", 0.0),
                )
            )
            weighted_components.update(
                {
                    "val_iid_majority_baseline": val_majority,
                    "near_chance_task_detector": near_chance,
                    "constant_prediction_detector": bool(
                        split_eval.get("train_max_pred_class_fraction", 0.0)
                        >= constant_pred_threshold
                        or split_eval.get("val_iid_max_pred_class_fraction", 0.0)
                        >= constant_pred_threshold
                    ),
                    "high_prediction_entropy_detector": bool(
                        split_eval.get("val_iid_prediction_entropy", 0.0)
                        >= entropy_threshold
                    ),
                    "sigma_zero_collapse_detector": bool(
                        near_chance
                        and abs(sigma_mean) <= sigma_zero_tolerance
                        and sigma_std <= sigma_zero_tolerance
                    ),
                    "loss_regularizer_dominance_detector": bool(
                        regularizer_ratio >= regularizer_dominance_multiple
                    ),
                    "dynamics_likelihood_explosion_detector": bool(
                        epoch_diagnostics.get("raw_sigma_threshold_exceeded", False)
                        or epoch_diagnostics.get("calibrated_sigma_threshold_exceeded", False)
                        or float(epoch_diagnostics.get("forward_logvar_min", 0.0))
                        <= float(dynamics_cfg.get("min_logvar", -6.0)) + 1e-4
                        or float(epoch_diagnostics.get("reverse_logvar_min", 0.0))
                        <= float(dynamics_cfg.get("min_logvar", -6.0)) + 1e-4
                    ),
                    "latent_norm_collapse_detector": bool(
                        float(epoch_diagnostics.get("calibrated_latent_norm_mean", 1.0))
                        <= latent_norm_min
                        or float(epoch_diagnostics.get("calibrated_latent_norm_std", 1.0))
                        <= latent_norm_std_min
                    ),
                    "effective_eta_total": float(effective_loss_weights.get("eta_total", 0.0)),
                    "effective_eta_step": float(effective_loss_weights.get("eta_step", 0.0)),
                    "effective_rho": float(effective_loss_weights.get("rho", 0.0)),
                    "effective_lambda_f": float(
                        effective_loss_weights.get("lambda_f", 0.0)
                    ),
                    "effective_lambda_r": float(
                        effective_loss_weights.get("lambda_r", 0.0)
                    ),
                    "sib_schedule_progress": float(
                        effective_loss_weights.get("schedule_progress", 1.0)
                    ),
                    "sib_schedule_warmup_active": bool(
                        effective_loss_weights.get("schedule_warmup_active", False)
                    ),
                    "sib_dynamics_schedule_progress": float(
                        effective_loss_weights.get("dynamics_schedule_progress", 1.0)
                    ),
                    "sib_dynamics_schedule_warmup_active": bool(
                        effective_loss_weights.get(
                            "dynamics_schedule_warmup_active", False
                        )
                    ),
                }
            )
        record = {
            "epoch": epoch,
            "loss": epoch_loss / max(n_batches, 1),
            "checkpoint_selection_eligible": (epoch + 1) >= min_epoch_for_checkpoint_selection,
            "early_stopping_eligible": (epoch + 1) >= min_epochs_before_early_stopping,
            "min_epochs_before_early_stopping": min_epochs_before_early_stopping,
            "min_epoch_for_checkpoint_selection": min_epoch_for_checkpoint_selection,
            **split_acc,
            **split_eval,
            **{f"loss_{k}": float(v) for k, v in epoch_components.items()},
            **weighted_components,
            **epoch_diagnostics,
        }
        if method == "sib":
            detector_count = sum(
                1 for key in SIB_COLLAPSE_DETECTOR_KEYS if record.get(key) is True
            )
            task_guard_candidate_eligible = (
                task_guard_enabled
                and float(record.get("val_iid_accuracy", float("-inf"))) >= task_guard_min
            )
            task_guard_candidate_score = (
                _task_guard_score(record, task_guard_cfg, detector_count)
                if task_guard_enabled
                else None
            )
            record.update(
                {
                    "task_guard_enabled": task_guard_enabled,
                    "task_guard_min_val_iid_accuracy": (
                        task_guard_min if task_guard_enabled else None
                    ),
                    "task_guard_candidate_eligible": task_guard_candidate_eligible,
                    "task_guard_candidate_score": (
                        list(task_guard_candidate_score)
                        if task_guard_candidate_score is not None
                        else None
                    ),
                    "task_guard_collapse_detector_count": detector_count,
                }
            )
            if task_guard_candidate_eligible and task_guard_candidate_score is not None:
                task_guard_n_eligible_epochs += 1
                if (
                    task_guard_best_score is None
                    or task_guard_candidate_score > task_guard_best_score
                ):
                    task_guard_best_score = task_guard_candidate_score
                    task_guard_best_epoch = epoch
                    task_guard_selected_checkpoint = "task_guard_best.pt"
                    save_checkpoint(
                        run_dir / "task_guard_best.pt",
                        {
                            "model": model.state_dict(),
                            "epoch": epoch,
                            "task_guard_score": list(task_guard_best_score),
                            "task_guard_min_val_iid_accuracy": task_guard_min,
                        },
                    )
        logger.log(record)
        if early_metric not in split_acc:
            raise ValueError(
                f"early stopping metric {early_metric!r} is not available; "
                f"available={sorted(split_acc)}"
            )
        checkpoint_selection_eligible = (epoch + 1) >= min_epoch_for_checkpoint_selection
        early_stopping_eligible = (epoch + 1) >= min_epochs_before_early_stopping
        if not checkpoint_selection_eligible:
            epochs_without_improvement = 0
            continue
        if split_acc[early_metric] > best_val:
            best_val = split_acc[early_metric]
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(
                run_dir / "best.pt",
                {"model": model.state_dict(), "epoch": epoch, early_metric: best_val},
            )
        else:
            epochs_without_improvement += 1
            if early_stopping_eligible and epochs_without_improvement >= patience:
                break

    save_checkpoint(run_dir / "final.pt", {"model": model.state_dict(), "epoch": epoch})
    best_payload = load_checkpoint(run_dir / "best.pt", map_location=device)
    model.load_state_dict(best_payload["model"])
    unguarded_acc, unguarded_cf = _evaluate_supervised_selection(
        model,
        loaders,
        device,
        include_cf=include_cf,
        arrow_calibration_config=arrow_calibration_cfg,
    )
    selected_checkpoint = "best.pt"
    selected_epoch = best_epoch
    if task_guard_enabled and task_guard_selected_checkpoint is not None:
        guarded_payload = load_checkpoint(
            run_dir / task_guard_selected_checkpoint, map_location=device
        )
        model.load_state_dict(guarded_payload["model"])
        selected_checkpoint = task_guard_selected_checkpoint
        selected_epoch = int(guarded_payload.get("epoch", task_guard_best_epoch))
    selected_acc, selected_cf = _evaluate_supervised_selection(
        model,
        loaders,
        device,
        include_cf=include_cf,
        arrow_calibration_config=arrow_calibration_cfg,
    )
    task_guard_passed = (
        (not task_guard_enabled)
        or (
            selected_checkpoint == "task_guard_best.pt"
            and selected_acc["val_iid_accuracy"] >= task_guard_min
        )
    )
    selected_sid_schedule = (
        _scheduled_sid_weights(loss_weights, sid_schedule_cfg, selected_epoch)
        if method == "sid" and selected_epoch >= 0
        else {}
    )
    itm_schedule_cfg = config.get("itm_schedule", {})
    itm_required_epochs = _itm_schedule_required_epochs(itm_schedule_cfg)
    selected_itm_schedule = (
        _scheduled_itm_weights(loss_weights, itm_schedule_cfg, selected_epoch)
        if method == "itm" and selected_epoch >= 0
        else {}
    )
    final = {
        **selected_acc,
        **selected_cf,
        **{
            f"unguarded_{key}": value
            for key, value in {**unguarded_acc, **unguarded_cf}.items()
        },
        "best_val_iid_accuracy": best_val,
        "best_epoch": best_epoch,
        "epochs_completed": epoch + 1,
        "selected_checkpoint": selected_checkpoint,
        "selected_epoch": selected_epoch,
        "unguarded_selected_checkpoint": "best.pt",
        "unguarded_best_epoch": best_epoch,
        "min_epochs_before_early_stopping": min_epochs_before_early_stopping,
        "min_epoch_for_checkpoint_selection": min_epoch_for_checkpoint_selection,
        "checkpoint_selection_floor_satisfied": (
            selected_epoch >= 0
            and (selected_epoch + 1) >= min_epoch_for_checkpoint_selection
        ),
        "early_stopping_floor_satisfied": (
            int(epoch + 1) >= min_epochs_before_early_stopping
        ),
        "sid_schedule_required_epochs": sid_required_epochs if method == "sid" else None,
        "sid_selected_schedule_progress": (
            float(selected_sid_schedule.get("sid_schedule_progress", 1.0))
            if method == "sid"
            else None
        ),
        "sid_selected_dynamics_progress": (
            float(selected_sid_schedule.get("sid_dynamics_progress", 1.0))
            if method == "sid"
            else None
        ),
        "sid_schedule_floor_satisfied": (
            selected_epoch >= 0
            and (selected_epoch + 1) >= sid_required_epochs
            and float(selected_sid_schedule.get("sid_schedule_progress", 0.0)) >= 1.0
            if method == "sid"
            else None
        ),
        "itm_schedule_required_epochs": itm_required_epochs if method == "itm" else None,
        "itm_selected_schedule_progress": (
            float(selected_itm_schedule.get("itm_schedule_progress", 1.0))
            if method == "itm"
            else None
        ),
        "itm_selected_transition_schedule_progress": (
            float(selected_itm_schedule.get("itm_transition_schedule_progress", 1.0))
            if method == "itm"
            else None
        ),
        "itm_schedule_floor_satisfied": (
            selected_epoch >= 0
            and (selected_epoch + 1) >= itm_required_epochs
            and float(selected_itm_schedule.get("itm_schedule_progress", 0.0)) >= 1.0
            and float(
                selected_itm_schedule.get("itm_transition_schedule_progress", 0.0)
            )
            >= 1.0
            if method == "itm"
            else None
        ),
        "task_guard_enabled": task_guard_enabled,
        "task_guard_min_val_iid_accuracy": task_guard_min if task_guard_enabled else None,
        "task_guard_selection_split": (
            task_guard_selection_split if task_guard_enabled else None
        ),
        "task_guard_selection_metric": (
            str(
                task_guard_cfg.get(
                    "selection_metric", "val_iid_accuracy_then_cf_stability"
                )
            )
            if task_guard_enabled
            else None
        ),
        "task_guard_best_epoch": (
            task_guard_best_epoch if task_guard_enabled and task_guard_best_epoch >= 0 else None
        ),
        "task_guard_best_score": (
            list(task_guard_best_score)
            if task_guard_enabled and task_guard_best_score is not None
            else None
        ),
        "task_guard_n_eligible_epochs": (
            task_guard_n_eligible_epochs if task_guard_enabled else None
        ),
        "task_guard_passed": task_guard_passed,
        "task_guard_failed": task_guard_enabled and not task_guard_passed,
        "task_guard_no_eligible_checkpoint": (
            task_guard_enabled and task_guard_selected_checkpoint is None
        ),
    }
    save_json(run_dir / "final_metrics.json", final)
    return final


def _arrow_loader_from_split(
    split: dict[str, Any],
    batch_size: int,
    *,
    device: torch.device | None = None,
    num_workers: int = 0,
    persistent_workers: bool = False,
    seed: int = 0,
) -> DataLoader:
    x = torch.as_tensor(split["x"], dtype=torch.float32)
    x_rev = reverse_sequence(x, time_dim=1)
    labels = torch.cat(
        [torch.ones(x.shape[0], dtype=torch.long), torch.zeros(x.shape[0], dtype=torch.long)]
    )
    data = torch.cat([x, x_rev], dim=0)
    return _seeded_loader(
        TensorDataset(data, labels),
        batch_size=batch_size,
        shuffle=True,
        device=device,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        seed=seed,
    )


def _order_loader_from_split(
    split: dict[str, Any],
    batch_size: int,
    *,
    device: torch.device | None = None,
    num_workers: int = 0,
    persistent_workers: bool = False,
    seed: int = 0,
) -> DataLoader:
    x = torch.as_tensor(split["x"], dtype=torch.float32)
    midpoint = x.shape[1] // 2
    if midpoint <= 0 or midpoint >= x.shape[1]:
        raise ValueError("OCP-style order task requires at least two time points")
    x_swapped = torch.cat([x[:, midpoint:], x[:, :midpoint]], dim=1)
    labels = torch.cat(
        [torch.ones(x.shape[0], dtype=torch.long), torch.zeros(x.shape[0], dtype=torch.long)]
    )
    data = torch.cat([x, x_swapped], dim=0)
    return _seeded_loader(
        TensorDataset(data, labels),
        batch_size=batch_size,
        shuffle=True,
        device=device,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        seed=seed,
    )


def _pretraining_loader_for_method(
    method: ArrowMethod,
    split: dict[str, Any],
    batch_size: int,
    *,
    device: torch.device | None = None,
    num_workers: int = 0,
    persistent_workers: bool = False,
    seed: int = 0,
) -> tuple[DataLoader, str, str]:
    if method == "ocp_style":
        return (
            _order_loader_from_split(
                split,
                batch_size,
                device=device,
                num_workers=num_workers,
                persistent_workers=persistent_workers,
                seed=seed,
            ),
            "order_train_accuracy",
            "segment_order",
        )
    if method == "lens_like_arrow_classifier":
        return (
            _arrow_loader_from_split(
                split,
                batch_size,
                device=device,
                num_workers=num_workers,
                persistent_workers=persistent_workers,
                seed=seed,
            ),
            "arrow_train_accuracy",
            "forward_reverse",
        )
    raise AssertionError(method)


def _train_downstream_protocol(
    protocol: Literal["frozen_encoder", "fine_tuned_encoder"],
    pretrained_encoder: nn.Module,
    config: dict[str, Any],
    splits: dict[str, dict[str, Any]],
    device: torch.device,
    run_dir: Path,
) -> dict[str, Any]:
    """Train downstream task classifier using a pretrained encoder protocol."""

    model_cfg = config.get("model", {})
    training_cfg = config.get("training", {})
    pretraining_cfg = config.get("pretraining", {})
    _check_supported_config(config)
    latent_dim = int(model_cfg.get("latent_dim", 16))
    hidden_dim = int(model_cfg.get("hidden_dim", 64))
    pooling = str(model_cfg.get("pooling", "last"))
    downstream_epochs = int(pretraining_cfg.get("downstream_epochs", training_cfg.get("epochs", 50)))
    batch_size = int(training_cfg.get("batch_size", 128))
    loader_settings = _data_loader_settings(config, device)
    loaders = make_loaders(
        splits,
        batch_size=batch_size,
        include_cf=False,
        device=device,
        num_workers=int(loader_settings["num_workers"]),
        persistent_workers=bool(loader_settings["persistent_workers"]),
        seed=int(loader_settings["worker_seed"]),
    )

    task_model = EncoderTaskClassifier(
        encoder=deepcopy(pretrained_encoder),
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        pooling=pooling,
        freeze_encoder=protocol == "frozen_encoder",
    ).to(device)
    opt = _make_optimizer([p for p in task_model.parameters() if p.requires_grad], training_cfg)
    grad_clip = float(training_cfg.get("grad_clip_norm", 1.0))
    logger = JsonlLogger(run_dir / f"{protocol}_metrics.jsonl")
    best_val = -math.inf
    best_epoch = -1
    final_acc: dict[str, float] = {}
    patience = int(training_cfg.get("patience", downstream_epochs + 1))
    epochs_without_improvement = 0
    early_metric = _early_stopping_metric_name(training_cfg, protocol=protocol)
    for epoch in range(downstream_epochs):
        task_model.train()
        loss_sum = 0.0
        n_batches = 0
        for x, y in loaders["train"]:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = task_model(x)["logits"]
            loss = F.cross_entropy(logits, y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite downstream loss for {protocol}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(task_model.parameters(), grad_clip)
            opt.step()
            loss_sum += float(loss.detach().cpu())
            n_batches += 1

        split_acc = {
            f"{protocol}_{name}_accuracy": evaluate_classifier(task_model, loader, device)
            for name, loader in loaders.items()
        }
        split_acc[f"{protocol}_ood_gap"] = ood_gap(
            split_acc[f"{protocol}_iid_test_accuracy"],
            split_acc[f"{protocol}_ood_test_accuracy"],
        )
        record = {
            "epoch": epoch,
            "loss": loss_sum / max(n_batches, 1),
            **split_acc,
        }
        logger.log(record)
        final_acc = split_acc
        if early_metric not in split_acc:
            raise ValueError(
                f"early stopping metric {early_metric!r} is not available; "
                f"available={sorted(split_acc)}"
            )
        if split_acc[early_metric] > best_val:
            best_val = split_acc[early_metric]
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(
                run_dir / f"{protocol}_best.pt",
                {
                    "model": task_model.state_dict(),
                    "epoch": epoch,
                    early_metric: best_val,
                },
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break
    save_checkpoint(
        run_dir / f"{protocol}_final.pt",
        {"model": task_model.state_dict(), "epoch": epoch},
    )
    best_payload = load_checkpoint(run_dir / f"{protocol}_best.pt", map_location=device)
    task_model.load_state_dict(best_payload["model"])
    final_acc = {
        f"{protocol}_{name}_accuracy": evaluate_classifier(task_model, loader, device)
        for name, loader in loaders.items()
    }
    final_acc[f"{protocol}_ood_gap"] = ood_gap(
        final_acc[f"{protocol}_iid_test_accuracy"],
        final_acc[f"{protocol}_ood_test_accuracy"],
    )
    return {
        **final_acc,
        f"{protocol}_best_val_iid_accuracy": best_val,
        f"{protocol}_best_epoch": best_epoch,
        f"{protocol}_epochs_completed": epoch + 1,
        f"{protocol}_selected_checkpoint": f"{protocol}_best.pt",
    }


def train_arrow_pretraining_method(
    method: ArrowMethod,
    config: dict[str, Any],
    run_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    """Label-free train-split-only arrow pretraining plus downstream protocols."""

    seed = int(config.get("seed", 0))
    training_cfg = config.get("training", {})
    _check_supported_config(config)
    seed_everything(seed, deterministic=bool(training_cfg.get("deterministic", False)))
    splits = generate_splits_from_config(config)
    input_dim = int(splits["train"]["x"].shape[-1])
    batch_size = int(training_cfg.get("batch_size", 128))
    loader_settings = _data_loader_settings(config, device)
    train_loader, pretrain_metric_name, pretraining_objective = _pretraining_loader_for_method(
        method,
        splits["train"],
        batch_size,
        device=device,
        num_workers=int(loader_settings["num_workers"]),
        persistent_workers=bool(loader_settings["persistent_workers"]),
        seed=int(loader_settings["worker_seed"]),
    )
    model = build_arrow_model(config, input_dim).to(device)
    opt = _make_optimizer(model.parameters(), training_cfg)
    save_json(run_dir / "resolved_config.json", config)
    save_json(
        run_dir / "metadata.json",
        {
            **run_metadata(
                method,
                config,
                run_dir,
                device,
                parameter_count(model),
                loader_settings,
            ),
            "pretraining_split": "train",
            "pretraining_objective": pretraining_objective,
            "transductive": False,
            "downstream_protocols": ["frozen_encoder", "fine_tuned_encoder"],
            "dataset_metadata": {k: v["metadata"] for k, v in splits.items()},
        },
    )
    logger = JsonlLogger(run_dir / "metrics.jsonl")
    epochs = int(training_cfg.get("epochs", 50))
    grad_clip = float(training_cfg.get("grad_clip_norm", 1.0))
    final_pretrain_acc = 0.0
    for epoch in range(epochs):
        model.train()
        correct = 0
        total = 0
        loss_sum = 0.0
        for x, labels in train_loader:
            x = x.to(device)
            labels = labels.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)["logits"]
            loss = binary_arrow_loss(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite arrow loss for method {method}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            correct += int((logits.argmax(dim=-1) == labels).sum().item())
            total += int(labels.numel())
            loss_sum += float(loss.detach().cpu())
        final_pretrain_acc = correct / max(total, 1)
        logger.log(
            {
                "epoch": epoch,
                pretrain_metric_name: final_pretrain_acc,
                "pretraining_objective": pretraining_objective,
                "loss": loss_sum,
            }
        )
    save_checkpoint(run_dir / "final.pt", {"model": model.state_dict(), "epoch": epochs - 1})
    frozen = _train_downstream_protocol(
        "frozen_encoder", model.encoder, config, splits, device, run_dir
    )
    fine_tuned = _train_downstream_protocol(
        "fine_tuned_encoder", model.encoder, config, splits, device, run_dir
    )
    final = {
        pretrain_metric_name: final_pretrain_acc,
        "pretraining_split": "train",
        "pretraining_objective": pretraining_objective,
        "transductive": False,
        **frozen,
        **fine_tuned,
    }
    save_json(run_dir / "final_metrics.json", final)
    return final


def run_supervised_cli(method: SupervisedMethod) -> dict[str, Any]:
    parser = build_arg_parser(method)
    args = parser.parse_args()
    config = apply_cli_overrides(load_config(args.config), args)
    seed = int(config.get("seed", 0))
    device = resolve_device(args)
    experiment = str(config.get("experiment", config.get("benchmark_name", "sta_bench")))
    run_dir = make_run_dir(args.output_dir, experiment, method, config, seed, args.overwrite)
    return train_supervised_method(method, config, run_dir, device, resume=args.resume)


def run_arrow_cli(method: ArrowMethod) -> dict[str, Any]:
    parser = build_arg_parser(method)
    args = parser.parse_args()
    config = apply_cli_overrides(load_config(args.config), args)
    seed = int(config.get("seed", 0))
    device = resolve_device(args)
    experiment = str(config.get("experiment", config.get("benchmark_name", "sta_bench")))
    run_dir = make_run_dir(args.output_dir, experiment, method, config, seed, args.overwrite)
    return train_arrow_pretraining_method(method, config, run_dir, device)
