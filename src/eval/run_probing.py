"""Run frozen-representation probes for trained STA-Bench models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.eval.metrics import reverse_sequence
from src.models.encoders import pool_sequence
from src.train.common import (
    build_arrow_model,
    build_supervised_model,
    generate_splits_from_config,
    save_json,
)
from src.utils.hardware import get_device


SUPERVISED_METHOD_NAMES = {"erm", "ib", "ep_min", "ep_max", "sib"}
ARROW_METHOD_NAMES = {"ocp_style", "lens_like_arrow_classifier"}


def _load_resolved_config(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "resolved_config.json"
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _build_model_for_probe(method: str, config: dict[str, Any], input_dim: int):
    if method in SUPERVISED_METHOD_NAMES:
        return build_supervised_model(method, config, input_dim)
    if method in ARROW_METHOD_NAMES:
        return build_arrow_model(config, input_dim)
    raise ValueError(f"unknown method {method!r}")


def load_probe_model(
    run_dir: str | Path,
    method: str,
    checkpoint: str = "best.pt",
    device: torch.device | None = None,
):
    run_dir = Path(run_dir)
    config = _load_resolved_config(run_dir)
    splits = generate_splits_from_config(config)
    input_dim = int(splits["train"]["x"].shape[-1])
    model = _build_model_for_probe(method, config, input_dim)
    device = device or torch.device("cpu")
    try:
        payload = torch.load(run_dir / checkpoint, map_location=device, weights_only=True)
    except TypeError:
        payload = torch.load(run_dir / checkpoint, map_location=device)
    model.load_state_dict(payload["model"])
    model.to(device)
    model.eval()
    return model, config, splits


@torch.inference_mode()
def encode_sequence(
    model: torch.nn.Module,
    x: np.ndarray | torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    x_t = torch.as_tensor(x, dtype=torch.float32, device=device)
    encoder = getattr(model, "encoder", None)
    if encoder is None:
        out = model(x_t)
        z = out.get("z_sequence", out["z"])
    else:
        z = encoder(x_t)
    return z.detach().cpu()


def _linear_probe_accuracy(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
) -> float:
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=500, C=1.0, solver="lbfgs", multi_class="auto"),
    )
    clf.fit(x_train, y_train)
    return float(clf.score(x_eval, y_eval))


def _pooled_features(z: torch.Tensor, pooling: str) -> np.ndarray:
    return pool_sequence(z, pooling).numpy()


def _flatten_time_features(z: torch.Tensor) -> np.ndarray:
    return z.reshape(-1, z.shape[-1]).numpy()


def _one_hot(states: np.ndarray, n_states: int) -> np.ndarray:
    return np.eye(n_states, dtype=np.float32)[states]


def _component_observations(split: dict[str, Any], component: str) -> np.ndarray:
    metadata = split["metadata"]
    n_core_states = int(metadata["n_core_states"])
    n_spur_states = int(metadata["n_spur_states"])
    mixing = split["mixing_matrix"]
    obs_dim = int(metadata["observation"]["obs_dim"])
    core_scale = float(metadata["observation"]["core_scale"])
    spur_scale = float(metadata["observation"]["spur_scale"])
    if component == "core":
        h_core = core_scale * _one_hot(split["c"], n_core_states)
        h_spur = np.zeros((*split["c"].shape, n_spur_states), dtype=np.float32)
    elif component == "spurious":
        h_core = np.zeros((*split["s"].shape, n_core_states), dtype=np.float32)
        h_spur = spur_scale * _one_hot(split["s"], n_spur_states)
    else:
        raise ValueError("component must be 'core' or 'spurious'")
    h = np.concatenate([h_core, h_spur], axis=-1)
    x = np.einsum("oi,nli->nlo", mixing, h, optimize=True)
    return x.astype(np.float32).reshape(split["x"].shape[0], split["x"].shape[1], obs_dim)


def _arrow_probe_features(
    model: torch.nn.Module,
    x: np.ndarray,
    pooling: str,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    x_t = torch.as_tensor(x, dtype=torch.float32)
    x_reverse = reverse_sequence(x_t, time_dim=1)
    labels = np.concatenate(
        [np.ones(x_t.shape[0], dtype=np.int64), np.zeros(x_t.shape[0], dtype=np.int64)]
    )
    z_forward = encode_sequence(model, x_t, device)
    z_reverse = encode_sequence(model, x_reverse, device)
    features = np.concatenate(
        [_pooled_features(z_forward, pooling), _pooled_features(z_reverse, pooling)], axis=0
    )
    return features, labels


def run_probes(
    run_dir: str | Path,
    method: str,
    checkpoint: str = "best.pt",
    probe_train_split: str = "val_iid",
    eval_splits: tuple[str, ...] = ("iid_test", "ood_test"),
    device: torch.device | None = None,
) -> dict[str, Any]:
    device = device or torch.device("cpu")
    model, config, splits = load_probe_model(run_dir, method, checkpoint, device)
    pooling = str(config.get("model", {}).get("pooling", "last"))
    train_split = splits[probe_train_split]

    z_train = encode_sequence(model, train_split["x"], device)
    train_pooled = _pooled_features(z_train, pooling)
    train_time = _flatten_time_features(z_train)
    results: dict[str, Any] = {
        "method": method,
        "checkpoint": checkpoint,
        "probe_train_split": probe_train_split,
        "eval_splits": list(eval_splits),
        "probe_capacity": "standardized_logistic_regression_C1_max_iter500",
    }

    for split_name in eval_splits:
        split = splits[split_name]
        z_eval = encode_sequence(model, split["x"], device)
        eval_pooled = _pooled_features(z_eval, pooling)
        eval_time = _flatten_time_features(z_eval)
        prefix = split_name
        results[f"{prefix}_label_probe_accuracy"] = _linear_probe_accuracy(
            train_pooled, train_split["y"], eval_pooled, split["y"]
        )
        results[f"{prefix}_core_state_probe_accuracy"] = _linear_probe_accuracy(
            train_time,
            train_split["c"].reshape(-1),
            eval_time,
            split["c"].reshape(-1),
        )
        results[f"{prefix}_spurious_state_probe_accuracy"] = _linear_probe_accuracy(
            train_time,
            train_split["s"].reshape(-1),
            eval_time,
            split["s"].reshape(-1),
        )

        for component, key in (("core", "core_arrow"), ("spurious", "spurious_arrow")):
            x_train_comp = _component_observations(train_split, component)
            x_eval_comp = _component_observations(split, component)
            arrow_train_x, arrow_train_y = _arrow_probe_features(
                model, x_train_comp, pooling, device
            )
            arrow_eval_x, arrow_eval_y = _arrow_probe_features(
                model, x_eval_comp, pooling, device
            )
            results[f"{prefix}_{key}_probe_accuracy"] = _linear_probe_accuracy(
                arrow_train_x, arrow_train_y, arrow_eval_x, arrow_eval_y
            )

    save_json(Path(run_dir) / "probe_metrics.json", results)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run frozen latent probes for a trained run.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument("--probe-train-split", default="val_iid")
    parser.add_argument("--eval-splits", nargs="+", default=["iid_test", "ood_test"])
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    device = get_device() if args.device == "auto" else torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is not available")
    run_probes(
        args.run_dir,
        args.method,
        checkpoint=args.checkpoint,
        probe_train_split=args.probe_train_split,
        eval_splits=tuple(args.eval_splits),
        device=device,
    )


if __name__ == "__main__":
    main()
