"""Run raw neural baselines for irreversible source inference."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import platform
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.data.irreversible_source_inference import (
    IrreversibleSourceConfig,
    IrreversibleSourceSplit,
    generate_irreversible_source_splits,
)
from src.eval.benchmark_diagnostics import run_diagnostics
from src.models.minimal_sequence import build_model, parameter_count


METHODS: dict[str, dict[str, Any]] = {
    "final_frame_mlp": {
        "model_type": "final_frame_mlp",
        "input_key": "mixed",
        "uses_counterfactual": False,
    },
    "sequence_erm": {
        "model_type": "sequence_cnn_gru",
        "input_key": "mixed",
        "uses_counterfactual": False,
    },
    "core_only_oracle": {
        "model_type": "sequence_cnn_gru",
        "input_key": "core_only",
        "uses_counterfactual": False,
    },
    "nuisance_only_oracle": {
        "model_type": "sequence_cnn_gru",
        "input_key": "nuisance_only",
        "uses_counterfactual": False,
    },
    "time_reversed_sequence": {
        "model_type": "sequence_cnn_gru",
        "input_key": "mixed_reversed",
        "uses_counterfactual": False,
    },
    "sequence_erm_lstm": {
        "model_type": "sequence_cnn_lstm",
        "input_key": "mixed",
        "uses_counterfactual": False,
    },
    "sequence_erm_tcn": {
        "model_type": "sequence_cnn_tcn",
        "input_key": "mixed",
        "uses_counterfactual": False,
    },
    "sequence_erm_transformer": {
        "model_type": "sequence_cnn_transformer",
        "input_key": "mixed",
        "uses_counterfactual": False,
    },
    "sequence_erm_temporal_pool": {
        "model_type": "sequence_cnn_temporal_pool",
        "input_key": "mixed",
        "uses_counterfactual": False,
    },
    "counterfactual_invariance": {
        "model_type": "sequence_cnn_gru",
        "input_key": "mixed",
        "uses_counterfactual": True,
    },
    "group_invariance_light": {
        "model_type": "sequence_cnn_gru",
        "input_key": "mixed",
        "uses_counterfactual": False,
        "uses_group_balancing": True,
    },
    "nuisance_channel_dropout": {
        "model_type": "sequence_cnn_gru",
        "input_key": "mixed",
        "uses_counterfactual": False,
        "channel_dropout_prob": 0.5,
    },
}


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def stable_hash(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:12]


def stable_run_seed(seed: int, scenario: str, method: str) -> int:
    raw = f"{seed}:{scenario}:{method}".encode("utf-8")
    return int(hashlib.sha256(raw).hexdigest()[:8], 16) % (2**31 - 1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def build_dataset_config(config: dict[str, Any], profile: dict[str, Any], seed: int) -> IrreversibleSourceConfig:
    data = deep_update(config.get("data", {}), profile.get("data", {}))
    data["seed"] = seed
    return IrreversibleSourceConfig(**data)


def split_tensor(split: IrreversibleSourceSplit, input_key: str) -> np.ndarray:
    if input_key == "mixed_reversed":
        return split.mixed[:, ::-1].copy()
    value = getattr(split, input_key)
    return np.asarray(value)


def infer_input_channels(x: np.ndarray) -> int:
    if x.ndim == 4:
        return 1
    if x.ndim == 5:
        return int(x.shape[2])
    raise ValueError(f"expected 4D or 5D input, got shape {x.shape}")


def normalize_with_train(
    train_x: np.ndarray,
    arrays: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    mean = float(train_x.mean())
    std = float(train_x.std())
    if std < 1e-6:
        std = 1.0
    return {name: ((value - mean) / std).astype(np.float32) for name, value in arrays.items()}


def make_loader(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
    x_cf: np.ndarray | None = None,
    group: np.ndarray | None = None,
) -> DataLoader:
    tensors = [torch.from_numpy(x).float(), torch.from_numpy(y).long()]
    if x_cf is not None:
        tensors.append(torch.from_numpy(x_cf).float())
    if group is not None:
        tensors.append(torch.from_numpy(group).long())
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        TensorDataset(*tensors),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
    )


def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    return float((logits.argmax(dim=1) == y).float().mean().item())


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    has_counterfactual: bool = False,
) -> dict[str, float]:
    model.eval()
    total = 0
    loss_sum = 0.0
    correct = 0
    cf_correct = 0
    cf_flip = 0
    cf_l1 = 0.0
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            y = batch[1].to(device)
            out = model(x)
            loss = F.cross_entropy(out.logits, y)
            batch_size = int(y.numel())
            total += batch_size
            loss_sum += float(loss.item()) * batch_size
            correct += int((out.logits.argmax(dim=1) == y).sum().item())
            if has_counterfactual and len(batch) == 3:
                x_cf = batch[2].to(device)
                out_cf = model(x_cf)
                pred = out.logits.argmax(dim=1)
                pred_cf = out_cf.logits.argmax(dim=1)
                cf_correct += int((pred_cf == y).sum().item())
                cf_flip += int((pred != pred_cf).sum().item())
                probs = F.softmax(out.logits, dim=1)
                probs_cf = F.softmax(out_cf.logits, dim=1)
                cf_l1 += float(torch.abs(probs - probs_cf).sum(dim=1).sum().item())
    metrics = {
        "loss": loss_sum / max(total, 1),
        "accuracy": correct / max(total, 1),
    }
    if has_counterfactual:
        metrics.update(
            {
                "accuracy_on_x_cf": cf_correct / max(total, 1),
                "cf_prediction_flip_rate": cf_flip / max(total, 1),
                "cf_probability_l1": cf_l1 / max(total, 1),
            }
        )
    return metrics


def train_one_method(
    *,
    method: str,
    splits: dict[str, IrreversibleSourceSplit],
    dataset_config: IrreversibleSourceConfig,
    training: dict[str, Any],
    model_config: dict[str, Any],
    out_dir: Path,
    seed: int,
    run_seed: int,
    device: torch.device,
) -> dict[str, Any]:
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}")
    method_spec = METHODS[method]
    input_key = method_spec["input_key"]
    uses_cf = bool(method_spec["uses_counterfactual"])
    uses_group_balancing = bool(method_spec.get("uses_group_balancing", False))
    channel_dropout_prob = float(method_spec.get("channel_dropout_prob", 0.0))

    raw = {name: split_tensor(split, input_key) for name, split in splits.items()}
    if uses_cf:
        raw_cf = {name: np.asarray(split.counterfactual) for name, split in splits.items()}
    else:
        raw_cf = {}
    normalized = normalize_with_train(raw["train"], raw)
    if uses_cf:
        normalized_cf = normalize_with_train(raw["train"], raw_cf)
    else:
        normalized_cf = {}

    batch_size = int(training.get("batch_size", 128))
    train_loader = make_loader(
        normalized["train"],
        splits["train"].y,
        batch_size=batch_size,
        shuffle=True,
        seed=run_seed,
        x_cf=normalized_cf.get("train"),
        group=(splits["train"].nuisance_direction > 0).astype(np.int64)
        if uses_group_balancing
        else None,
    )
    eval_loaders = {
        name: make_loader(
            x,
            splits[name].y,
            batch_size=batch_size,
            shuffle=False,
            seed=run_seed,
            x_cf=normalized_cf.get(name),
        )
        for name, x in normalized.items()
    }

    model = build_model(
        model_type=str(method_spec["model_type"]),
        grid_size=dataset_config.grid_size,
        hidden_dim=int(model_config.get("hidden_dim", 64)),
        num_layers=int(model_config.get("num_layers", 1)),
        dropout=float(model_config.get("dropout", 0.0)),
        input_channels=infer_input_channels(raw["train"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training.get("lr", 1e-3)),
        weight_decay=float(training.get("weight_decay", 1e-4)),
    )
    max_epochs = int(training.get("epochs", 30))
    patience = int(training.get("patience", 8))
    grad_clip = float(training.get("grad_clip_norm", 1.0))
    lambda_cf_task = float(training.get("lambda_cf_task", 1.0))
    lambda_pred = float(training.get("lambda_pred", 0.2))

    epoch_path = out_dir / "epoch_metrics.jsonl"
    best_state = None
    best_val = -math.inf
    best_epoch = 0
    stale_epochs = 0
    start = time.time()
    with epoch_path.open("a", encoding="utf-8") as epoch_file:
        for epoch in range(1, max_epochs + 1):
            model.train()
            epoch_loss = 0.0
            task_loss_sum = 0.0
            cf_task_loss_sum = 0.0
            consistency_loss_sum = 0.0
            seen = 0
            for batch in train_loader:
                x = batch[0].to(device)
                y = batch[1].to(device)
                if channel_dropout_prob > 0 and x.ndim == 5 and x.shape[2] >= 2:
                    # Randomly zero the nuisance channel (index 1) per sample so the
                    # model cannot always rely on the nuisance trajectory.
                    drop = (torch.rand(x.shape[0], device=device) < channel_dropout_prob)
                    if drop.any():
                        x = x.clone()
                        x[drop, :, 1] = 0.0
                optimizer.zero_grad(set_to_none=True)
                out = model(x)
                if uses_group_balancing:
                    group = batch[-1].to(device)
                    sample_loss = F.cross_entropy(out.logits, y, reduction="none")
                    group_losses = []
                    for group_id in torch.unique(group):
                        group_losses.append(sample_loss[group == group_id].mean())
                    task_loss = torch.stack(group_losses).mean()
                else:
                    task_loss = F.cross_entropy(out.logits, y)
                cf_task_loss = torch.zeros((), device=device)
                consistency_loss = torch.zeros((), device=device)
                if uses_cf:
                    x_cf = batch[2].to(device)
                    out_cf = model(x_cf)
                    cf_task_loss = F.cross_entropy(out_cf.logits, y)
                    probs = F.log_softmax(out_cf.logits, dim=1)
                    target_probs = F.softmax(out.logits.detach(), dim=1)
                    consistency_loss = F.kl_div(probs, target_probs, reduction="batchmean")
                loss = task_loss + lambda_cf_task * cf_task_loss + lambda_pred * consistency_loss
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite loss for {method} seed {seed}")
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                batch_size_seen = int(y.numel())
                seen += batch_size_seen
                epoch_loss += float(loss.item()) * batch_size_seen
                task_loss_sum += float(task_loss.item()) * batch_size_seen
                cf_task_loss_sum += float(cf_task_loss.item()) * batch_size_seen
                consistency_loss_sum += float(consistency_loss.item()) * batch_size_seen

            val_metrics = evaluate(model, eval_loaders["val_iid"], device, has_counterfactual=uses_cf)
            if uses_cf:
                validation_score = min(
                    val_metrics["accuracy"],
                    val_metrics["accuracy_on_x_cf"],
                )
            else:
                validation_score = val_metrics["accuracy"]
            record = {
                "seed": seed,
                "method": method,
                "epoch": epoch,
                "train_loss": epoch_loss / max(seen, 1),
                "task_loss": task_loss_sum / max(seen, 1),
                "cf_task_loss": cf_task_loss_sum / max(seen, 1),
                "consistency_loss": consistency_loss_sum / max(seen, 1),
                "val_iid_accuracy": val_metrics["accuracy"],
                "val_iid_selection_score": validation_score,
                "val_iid_loss": val_metrics["loss"],
            }
            if uses_cf:
                record["val_iid_accuracy_on_x_cf"] = val_metrics["accuracy_on_x_cf"]
            epoch_file.write(json.dumps(record) + "\n")
            if validation_score > best_val:
                best_val = validation_score
                best_epoch = epoch
                best_state = {
                    key: value.detach().cpu().clone() for key, value in model.state_dict().items()
                }
                stale_epochs = 0
            else:
                stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    split_metrics = {
        name: evaluate(model, loader, device, has_counterfactual=uses_cf)
        for name, loader in eval_loaders.items()
    }
    elapsed = time.time() - start
    result = {
        "seed": seed,
        "run_seed": run_seed,
        "method": method,
        "input_key": input_key,
        "uses_counterfactual": uses_cf,
        "best_epoch": best_epoch,
        "parameter_count": parameter_count(model),
        "training_time_seconds": elapsed,
        "val_iid_accuracy": split_metrics["val_iid"]["accuracy"],
        "iid_test_accuracy": split_metrics["iid_test"]["accuracy"],
        "ood_test_accuracy": split_metrics["ood_test"]["accuracy"],
        "ood_gap": split_metrics["iid_test"]["accuracy"] - split_metrics["ood_test"]["accuracy"],
        "split_metrics": split_metrics,
    }
    if uses_cf:
        result.update(
            {
                "cf_prediction_flip_rate": split_metrics["iid_test"]["cf_prediction_flip_rate"],
                "cf_probability_l1": split_metrics["iid_test"]["cf_probability_l1"],
                "cf_ood_prediction_flip_rate": split_metrics["ood_test"][
                    "cf_prediction_flip_rate"
                ],
                "cf_ood_probability_l1": split_metrics["ood_test"]["cf_probability_l1"],
            }
        )
    return result


def summarize_results(results: list[dict[str, Any]], primary_scenario: str) -> dict[str, Any]:
    primary_results = [
        result for result in results if str(result.get("scenario", primary_scenario)) == primary_scenario
    ]
    if not primary_results:
        primary_results = results
    by_method: dict[str, list[dict[str, Any]]] = {}
    for result in primary_results:
        by_method.setdefault(str(result["method"]), []).append(result)

    method_summary: dict[str, Any] = {}
    for method, rows in sorted(by_method.items()):
        method_summary[method] = {}
        for key in ["val_iid_accuracy", "iid_test_accuracy", "ood_test_accuracy", "ood_gap"]:
            values = np.array([float(row[key]) for row in rows], dtype=float)
            method_summary[method][key] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "n": int(len(values)),
                "values": values.round(6).tolist(),
            }
    if "sequence_erm" in by_method and "counterfactual_invariance" in by_method:
        erm = {int(row["seed"]): row for row in by_method["sequence_erm"]}
        cf = {int(row["seed"]): row for row in by_method["counterfactual_invariance"]}
        common = sorted(set(erm) & set(cf))
        reductions = [
            float(erm[seed]["ood_gap"] - cf[seed]["ood_gap"])
            for seed in common
        ]
        method_summary["main_gap_reduction"] = {
            "seeds": common,
            "values": [round(value, 6) for value in reductions],
            "mean": float(np.mean(reductions)) if reductions else None,
            "consistent_positive": bool(reductions and all(value > 0 for value in reductions)),
        }
    scenario_summary: dict[str, Any] = {}
    for result in results:
        scenario = str(result.get("scenario", primary_scenario))
        method = str(result["method"])
        scenario_summary.setdefault(scenario, {}).setdefault(method, []).append(result)
    compact_scenarios: dict[str, Any] = {}
    for scenario, methods in scenario_summary.items():
        compact_scenarios[scenario] = {}
        for method, rows in methods.items():
            compact_scenarios[scenario][method] = {}
            for key in ["iid_test_accuracy", "ood_test_accuracy", "ood_gap"]:
                values = np.array([float(row[key]) for row in rows], dtype=float)
                compact_scenarios[scenario][method][key] = {
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    "n": int(len(values)),
                }
    return {
        "primary_scenario": primary_scenario,
        "methods": method_summary,
        "scenarios": compact_scenarios,
    }


def markdown_summary(
    *,
    config_path: Path,
    profile_name: str,
    manifest: dict[str, Any],
    summary: dict[str, Any],
) -> str:
    lines = [
        "# Latest Main Result Summary",
        "",
        f"Profile: `{profile_name}`",
        f"Primary scenario: `{summary['primary_scenario']}`",
        f"Config: `{config_path}`",
        f"Command: `{manifest['command']}`",
        f"Device: `{manifest['device']}`",
        f"Runtime-limited: `{manifest['runtime_limited']}`",
        "",
        "## Method Table",
        "",
        "| Method | Val IID | IID Test | OOD Test | OOD Gap | Seeds |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method, stats in summary["methods"].items():
        if method == "main_gap_reduction":
            continue
        lines.append(
            "| {method} | {val:.3f} | {iid:.3f} | {ood:.3f} | {gap:.3f} | {n} |".format(
                method=method,
                val=stats["val_iid_accuracy"]["mean"],
                iid=stats["iid_test_accuracy"]["mean"],
                ood=stats["ood_test_accuracy"]["mean"],
                gap=stats["ood_gap"]["mean"],
                n=stats["ood_gap"]["n"],
            )
        )
    if "main_gap_reduction" in summary["methods"]:
        reduction = summary["methods"]["main_gap_reduction"]
        lines.extend(
            [
                "",
                "## Main Gap Reduction",
                "",
                f"Mean ERM-minus-counterfactual OOD-gap reduction: `{reduction['mean']}`",
                f"Consistent positive across common seeds: `{reduction['consistent_positive']}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Claim Status",
            "",
            claim_status_text(summary, manifest),
            "",
            "This summary separates neural model evidence from diagnostic feature probes.",
        ]
    )
    return "\n".join(lines) + "\n"


def claim_status_text(summary: dict[str, Any], manifest: dict[str, Any]) -> str:
    if manifest.get("runtime_limited") or manifest.get("profile") in {"smoke", "pilot"}:
        return (
            "Diagnostic only: this run is runtime-limited and cannot support a "
            "paper-level main claim."
        )
    methods = summary["methods"]
    if "sequence_erm" not in methods:
        return "Not supported: `sequence_erm` result is missing."
    erm_gap = methods["sequence_erm"]["ood_gap"]["mean"]
    if erm_gap < 0.20:
        return "Partially supported: neural ERM did not show the target OOD gap."
    if "counterfactual_invariance" not in methods:
        return "Partially supported: ERM gap exists, but counterfactual result is missing."
    cf_gap = methods["counterfactual_invariance"]["ood_gap"]["mean"]
    cf_ood = methods["counterfactual_invariance"]["ood_test_accuracy"]["mean"]
    cf_iid = methods["counterfactual_invariance"]["iid_test_accuracy"]["mean"]
    cf_ood_values = methods["counterfactual_invariance"]["ood_test_accuracy"].get("values", [])
    cf_seed_success = (
        sum(float(value) >= 0.80 for value in cf_ood_values) / len(cf_ood_values)
        if cf_ood_values
        else None
    )
    erm_ood = methods["sequence_erm"]["ood_test_accuracy"]["mean"]
    erm_iid = methods["sequence_erm"]["iid_test_accuracy"]["mean"]
    iid_preserved = cf_iid >= max(0.75, erm_iid - 0.15)
    stable = cf_seed_success is not None and cf_seed_success >= 0.80
    if cf_gap < erm_gap and cf_ood > erm_ood and iid_preserved and stable:
        return (
            "Supported in this controlled benchmark: ERM shows an OOD gap and "
            "counterfactual invariance reduces it with seed-level stability."
        )
    if cf_gap < erm_gap and cf_ood > erm_ood and iid_preserved and not stable:
        return (
            "Phenomenon supported: ERM shows an OOD gap. Counterfactual "
            "invariance improves mean OOD but is not seed-stable enough for a "
            "primary method-success claim."
        )
    if cf_gap < erm_gap and cf_ood > erm_ood and not iid_preserved:
        return (
            "Partially supported: ERM shows an OOD gap, but counterfactual "
            "training reduces the gap by sacrificing IID accuracy."
        )
    return "Partially supported: ERM shows an OOD gap, but counterfactual improvement is not established."


def run_experiment(
    config_path: Path,
    out_dir: Path,
    profile_name: str,
    write_docs_summary: bool = True,
) -> dict[str, Any]:
    config = load_yaml(config_path)
    profiles = config.get("profiles", {})
    if profile_name not in profiles:
        raise ValueError(f"profile {profile_name!r} is not defined")
    profile = profiles[profile_name]
    seeds = [int(seed) for seed in profile.get("seeds", [0])]
    methods = list(profile.get("methods", config.get("methods", [])))
    scenarios = profile.get("scenarios") or [{"name": "main_spurious_arrow", "data": {}}]
    primary_scenario = str(profile.get("primary_scenario", scenarios[0]["name"]))
    training = deep_update(config.get("training", {}), profile.get("training", {}))
    model_config = deep_update(config.get("model", {}), profile.get("model", {}))
    runtime_limited = bool(profile.get("runtime_limited", profile_name != "main"))

    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()
    all_results: list[dict[str, Any]] = []
    device_name = str(config.get("device", "auto"))
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    for scenario in scenarios:
        scenario_name = str(scenario["name"])
        scenario_profile = deep_update(profile, {"data": scenario.get("data", {})})
        scenario_methods = list(scenario.get("methods", methods))
        scenario_training = deep_update(training, scenario.get("training", {}))
        for seed in seeds:
            set_seed(seed)
            dataset_config = build_dataset_config(config, scenario_profile, seed)
            diagnostics = run_diagnostics(dataset_config)
            splits = generate_irreversible_source_splits(dataset_config)
            seed_dir = out_dir / scenario_name / f"seed_{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            with (seed_dir / "diagnostics.json").open("w", encoding="utf-8") as f:
                json.dump(diagnostics, f, indent=2)
            for method in scenario_methods:
                run_seed = stable_run_seed(seed, scenario_name, method)
                set_seed(run_seed)
                method_dir = seed_dir / method
                method_dir.mkdir(parents=True, exist_ok=True)
                result = train_one_method(
                    method=method,
                    splits=splits,
                    dataset_config=dataset_config,
                    training=scenario_training,
                    model_config=model_config,
                    out_dir=method_dir,
                    seed=seed,
                    run_seed=run_seed,
                    device=device,
                )
                result.update(
                    {
                        "profile": profile_name,
                        "scenario": scenario_name,
                        "config_hash": stable_hash(asdict(dataset_config)),
                        "dataset_config": asdict(dataset_config),
                        "benchmark_gate_passed": bool(diagnostics["gate"]["passed"]),
                    }
                )
                all_results.append(result)
                with metrics_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(result) + "\n")

    summary = summarize_results(all_results, primary_scenario=primary_scenario)
    manifest = {
        "command": " ".join(["python", "-m", "src.train.main_experiment", "--config", str(config_path), "--out", str(out_dir), "--profile", profile_name]),
        "profile": profile_name,
        "seeds": seeds,
        "methods": methods,
        "primary_scenario": primary_scenario,
        "scenarios": scenarios,
        "device": str(device),
        "runtime_limited": runtime_limited,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "config_path": str(config_path),
        "config_hash": stable_hash(config),
        "results_count": len(all_results),
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    summary_md = markdown_summary(
        config_path=config_path,
        profile_name=profile_name,
        manifest=manifest,
        summary=summary,
    )
    (out_dir / "summary.md").write_text(summary_md, encoding="utf-8")
    if write_docs_summary and not runtime_limited:
        docs_summary = Path("docs/latest_result_summary.md")
        docs_summary.write_text(summary_md, encoding="utf-8")
    return {"manifest": manifest, "summary": summary}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--profile", default="smoke")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_experiment(Path(args.config), Path(args.out), args.profile)
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
