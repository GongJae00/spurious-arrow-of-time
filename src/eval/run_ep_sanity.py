"""Run direct state-level EP sanity checks for biased ring chains."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.data.biased_ring import (
    net_clockwise_displacement,
    ring_entropy_production,
    sample_biased_ring,
)
from src.data.sta_bench import make_mixing_matrix
from src.eval.ep_sanity import ep_ranking_correlations
from src.eval.metrics import reverse_sequence
from src.models.baselines import ArrowClassifier
from src.train.common import save_json


DEFAULT_PROCESSES = {
    "R0": (0.30, 0.30),
    "R1": (0.35, 0.25),
    "R2": (0.40, 0.20),
    "R3": (0.45, 0.15),
}


def direct_arrow_accuracy(states: np.ndarray, n_states: int) -> float:
    """Classify forward vs reversed using signed net displacement.

    This is a deliberately simple direct-state sanity proxy. It is not the
    learned latent score used by SIB.
    """

    forward_score = net_clockwise_displacement(states, n_states)
    reverse_score = net_clockwise_displacement(np.flip(states, axis=1), n_states)
    scores = np.concatenate([forward_score, reverse_score])
    labels = np.concatenate([np.ones(states.shape[0]), np.zeros(states.shape[0])])
    preds = (scores > 0).astype(float)
    ties = scores == 0
    # Ties carry no arrow evidence; assign half credit deterministically.
    correct = (preds == labels).astype(float)
    correct[ties] = 0.5
    return float(correct.mean())


def _one_hot(states: np.ndarray, n_states: int) -> np.ndarray:
    return np.eye(n_states, dtype=np.float32)[states]


def _mixed_ring_observations(
    states: np.ndarray,
    n_states: int,
    obs_dim: int,
    mixing_matrix: np.ndarray,
    noise_std: float,
    seed: int,
) -> np.ndarray:
    h = _one_hot(states, n_states)
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, noise_std, size=(states.shape[0], states.shape[1], obs_dim)).astype(
        np.float32
    )
    return (np.einsum("oi,nli->nlo", mixing_matrix, h, optimize=True) + noise).astype(
        np.float32
    )


def _decode_mixed_ring_states(x: np.ndarray, mixing_matrix: np.ndarray) -> np.ndarray:
    """Decode mixed one-hot ring observations with the known mixing matrix."""

    pinv = np.linalg.pinv(mixing_matrix.astype(np.float64))
    scores = np.einsum("nlo,ko->nlk", x.astype(np.float64), pinv, optimize=True)
    return scores.argmax(axis=-1).astype(np.int64)


def _fit_forward_transition_matrix(
    states: np.ndarray,
    n_states: int,
    smoothing: float = 1e-3,
) -> np.ndarray:
    counts = np.full((n_states, n_states), smoothing, dtype=np.float64)
    current = states[:, :-1].reshape(-1)
    nxt = states[:, 1:].reshape(-1)
    np.add.at(counts, (current, nxt), 1.0)
    return counts / counts.sum(axis=1, keepdims=True)


def _transition_log_ratio_per_step(
    states: np.ndarray,
    transition_matrix: np.ndarray,
) -> np.ndarray:
    current = states[:, :-1]
    nxt = states[:, 1:]
    sigma_steps = np.log(transition_matrix[current, nxt]) - np.log(
        transition_matrix[nxt, current]
    )
    return sigma_steps.sum(axis=1) / float(states.shape[1] - 1)


def _gate(
    *,
    spearman: float,
    r0_sigma: float,
    r0_arrow_accuracy: float | None = None,
    spearman_threshold: float = 0.8,
    r0_sigma_tolerance: float = 0.05,
    chance_tolerance: float = 0.15,
) -> dict[str, object]:
    spearman_ok = bool(np.isfinite(spearman) and spearman >= spearman_threshold)
    r0_sigma_ok = bool(abs(r0_sigma) <= r0_sigma_tolerance)
    if r0_arrow_accuracy is None:
        r0_arrow_ok = True
    else:
        r0_arrow_ok = bool(abs(r0_arrow_accuracy - 0.5) <= chance_tolerance)
    return {
        "pass": bool(spearman_ok and r0_sigma_ok and r0_arrow_ok),
        "spearman_ok": spearman_ok,
        "r0_sigma_ok": r0_sigma_ok,
        "r0_arrow_accuracy_ok": r0_arrow_ok,
        "thresholds": {
            "spearman_min": spearman_threshold,
            "abs_r0_sigma_max": r0_sigma_tolerance,
            "r0_arrow_accuracy_abs_from_chance_max": chance_tolerance,
        },
    }


def _with_centered_sigma_fields(
    rows: list[dict[str, object]],
    *,
    raw_key: str,
    calibrated_key: str,
) -> dict[str, float]:
    """Add a simple R0-centered diagnostic calibration to EP sanity rows."""

    if not rows:
        return {"offset": 0.0, "r0_raw": 0.0, "r0_calibrated": 0.0}
    offset = float(rows[0][raw_key])
    for row in rows:
        row[calibrated_key] = float(row[raw_key]) - offset
    return {
        "offset": offset,
        "r0_raw": float(rows[0][raw_key]),
        "r0_calibrated": float(rows[0][calibrated_key]),
    }


def _arrow_dataset(x: torch.Tensor) -> TensorDataset:
    x_reverse = reverse_sequence(x, time_dim=1)
    labels = torch.cat(
        [torch.ones(x.shape[0], dtype=torch.long), torch.zeros(x.shape[0], dtype=torch.long)]
    )
    return TensorDataset(torch.cat([x, x_reverse], dim=0), labels)


def _train_eval_latent_arrow_classifier(
    x_train: torch.Tensor,
    x_eval: torch.Tensor,
    *,
    hidden_dim: int,
    latent_dim: int,
    epochs: int,
    batch_size: int,
    seed: int,
) -> tuple[float, float]:
    torch.manual_seed(seed)
    train_loader = DataLoader(_arrow_dataset(x_train), batch_size=batch_size, shuffle=True)
    eval_dataset = _arrow_dataset(x_eval)
    model = ArrowClassifier(
        input_dim=int(x_train.shape[-1]),
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        pooling="last",
    )
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    for _ in range(epochs):
        model.train()
        for x_batch, y_batch in train_loader:
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x_batch)["logits"], y_batch)
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        x_eval_all, y_eval_all = eval_dataset.tensors
        logits = model(x_eval_all)["logits"]
        accuracy = float((logits.argmax(dim=-1) == y_eval_all).float().mean().item())
        margin = float((logits[:, 1] - logits[:, 0]).abs().mean().item())
    return accuracy, margin


def run_ep_sanity(
    output_dir: str | Path,
    n_sequences: int = 2048,
    length: int = 64,
    n_states: int = 8,
    seed: int = 0,
) -> dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx, (name, (p, q)) in enumerate(DEFAULT_PROCESSES.items()):
        states = sample_biased_ring(
            n_sequences=n_sequences,
            length=length,
            n_states=n_states,
            p_forward=p,
            p_backward=q,
            seed=seed + idx,
            initial_state_mode="uniform_stationary",
        )
        analytic_ep = ring_entropy_production(p, q)
        sigma_proxy = abs(float(net_clockwise_displacement(states, n_states).mean())) / float(
            length - 1
        )
        rows.append(
            {
                "process": name,
                "p_forward": p,
                "p_backward": q,
                "analytic_ep": analytic_ep,
                "direct_sigma_proxy": sigma_proxy,
                "arrow_classifier_accuracy": direct_arrow_accuracy(states, n_states),
            }
        )
    corr = ep_ranking_correlations(
        np.array([r["direct_sigma_proxy"] for r in rows]),
        np.array([r["analytic_ep"] for r in rows]),
    )
    calibration = _with_centered_sigma_fields(
        rows,
        raw_key="direct_sigma_proxy",
        calibrated_key="direct_sigma_proxy_r0_centered",
    )
    r0 = rows[0]
    result = {
        "rows": rows,
        "spearman": corr["spearman"],
        "pearson": corr["pearson"],
        "R0_arrow_classifier_accuracy": r0["arrow_classifier_accuracy"],
        "R0_sigma_per_step_mean": r0["direct_sigma_proxy"],
        "R0_sigma_per_step_std": 0.0,
        "R0_sigma_per_step_raw_mean": r0["direct_sigma_proxy"],
        "R0_sigma_per_step_raw_std": 0.0,
        "R0_sigma_per_step_calibrated_mean": r0["direct_sigma_proxy_r0_centered"],
        "R0_sigma_per_step_calibrated_std": 0.0,
        "sigma_calibration": {
            "mode": "r0_centering_diagnostic",
            "offset": calibration["offset"],
            "uses_ood_test": False,
        },
        "gate": _gate(
            spearman=corr["spearman"],
            r0_sigma=r0["direct_sigma_proxy"],
            r0_arrow_accuracy=r0["arrow_classifier_accuracy"],
        ),
        "notes": "Direct state-level sanity proxy; not learned latent entropy production.",
    }
    save_json(output_dir / "ep_sanity_direct.json", result)
    return result


def run_latent_ep_sanity(
    output_dir: str | Path,
    n_sequences: int = 2048,
    length: int = 48,
    n_states: int = 8,
    obs_dim: int = 16,
    noise_std: float = 0.02,
    epochs: int = 15,
    batch_size: int = 128,
    hidden_dim: int = 64,
    latent_dim: int = 16,
    seed: int = 0,
) -> dict[str, object]:
    """Run classifier-based latent sanity on mixed observations.

    This is Experiment 1B's recoverability gate: a learned encoder/classifier
    should recover forward-vs-reverse evidence from mixed observations and rank
    biased rings by analytic EP. It is not the SIB dynamics score.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mixing = make_mixing_matrix(obs_dim, n_states, 0, seed + 505, normalize_columns=True)
    rows = []
    for idx, (name, (p, q)) in enumerate(DEFAULT_PROCESSES.items()):
        states = sample_biased_ring(
            n_sequences=n_sequences,
            length=length,
            n_states=n_states,
            p_forward=p,
            p_backward=q,
            seed=seed + 100 * idx,
            initial_state_mode="uniform_stationary",
        )
        x = _mixed_ring_observations(
            states,
            n_states,
            obs_dim,
            mixing,
            noise_std,
            seed=seed + 200 * idx,
        )
        midpoint = n_sequences // 2
        accuracy, margin = _train_eval_latent_arrow_classifier(
            torch.as_tensor(x[:midpoint], dtype=torch.float32),
            torch.as_tensor(x[midpoint:], dtype=torch.float32),
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            epochs=epochs,
            batch_size=batch_size,
            seed=seed + 300 * idx,
        )
        rows.append(
            {
                "process": name,
                "p_forward": p,
                "p_backward": q,
                "analytic_ep": ring_entropy_production(p, q),
                "latent_arrow_classifier_accuracy": accuracy,
                "latent_arrow_classifier_margin": margin,
            }
        )

    corr = ep_ranking_correlations(
        np.array([r["latent_arrow_classifier_accuracy"] for r in rows]),
        np.array([r["analytic_ep"] for r in rows]),
    )
    calibration = _with_centered_sigma_fields(
        rows,
        raw_key="latent_arrow_classifier_margin",
        calibrated_key="latent_arrow_classifier_margin_r0_centered",
    )
    r0 = rows[0]
    calibrated_gate = _gate(
        spearman=corr["spearman"],
        r0_sigma=float(r0["latent_arrow_classifier_margin_r0_centered"]),
        r0_arrow_accuracy=r0["latent_arrow_classifier_accuracy"],
    )
    result = {
        "rows": rows,
        "spearman": corr["spearman"],
        "pearson": corr["pearson"],
        "R0_arrow_classifier_accuracy": r0["latent_arrow_classifier_accuracy"],
        "R0_arrow_classifier_margin": r0["latent_arrow_classifier_margin"],
        "R0_sigma_per_step_raw_mean": r0["latent_arrow_classifier_margin"],
        "R0_sigma_per_step_raw_std": 0.0,
        "R0_sigma_per_step_calibrated_mean": r0[
            "latent_arrow_classifier_margin_r0_centered"
        ],
        "R0_sigma_per_step_calibrated_std": 0.0,
        "sigma_calibration": {
            "mode": "r0_centering_diagnostic",
            "offset": calibration["offset"],
            "uses_ood_test": False,
        },
        "gate": _gate(
            spearman=corr["spearman"],
            r0_sigma=r0["latent_arrow_classifier_margin"],
            r0_arrow_accuracy=r0["latent_arrow_classifier_accuracy"],
        ),
        "calibrated_gate": calibrated_gate,
        "notes": (
            "Latent mixed-observation forward/reverse classifier sanity; "
            "not the SIB dynamics score and not physical entropy production."
        ),
    }
    save_json(output_dir / "ep_sanity_latent.json", result)
    return result


def run_decoded_transition_ep_sanity(
    output_dir: str | Path,
    n_sequences: int = 2048,
    length: int = 48,
    n_states: int = 8,
    obs_dim: int = 16,
    noise_std: float = 0.02,
    smoothing: float = 1e-3,
    seed: int = 0,
) -> dict[str, object]:
    """Run mixed-observation transition log-ratio sanity with oracle decoding.

    This estimates forward transition dynamics on decoded mixed observations and
    scores held-out transitions as log P(z_{t+1}|z_t) - log P(z_t|z_{t+1})
    under the same forward transition estimator. It verifies that the
    observation path can preserve transition-level arrow evidence, but it is
    oracle-assisted because the mixing matrix is known.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mixing = make_mixing_matrix(obs_dim, n_states, 0, seed + 505, normalize_columns=True)
    rows = []
    for idx, (name, (p, q)) in enumerate(DEFAULT_PROCESSES.items()):
        states = sample_biased_ring(
            n_sequences=n_sequences,
            length=length,
            n_states=n_states,
            p_forward=p,
            p_backward=q,
            seed=seed + 500 * idx,
            initial_state_mode="uniform_stationary",
        )
        x = _mixed_ring_observations(
            states,
            n_states,
            obs_dim,
            mixing,
            noise_std,
            seed=seed + 600 * idx,
        )
        decoded = _decode_mixed_ring_states(x, mixing)
        midpoint = n_sequences // 2
        transition_matrix = _fit_forward_transition_matrix(
            decoded[:midpoint],
            n_states,
            smoothing=smoothing,
        )
        sigma_per_step = _transition_log_ratio_per_step(
            decoded[midpoint:],
            transition_matrix,
        )
        rows.append(
            {
                "process": name,
                "p_forward": p,
                "p_backward": q,
                "analytic_ep": ring_entropy_production(p, q),
                "decoded_transition_sigma_per_step_mean": float(sigma_per_step.mean()),
                "decoded_transition_sigma_per_step_std": float(sigma_per_step.std()),
                "decode_accuracy": float((decoded == states).mean()),
            }
        )

    calibration = _with_centered_sigma_fields(
        rows,
        raw_key="decoded_transition_sigma_per_step_mean",
        calibrated_key="decoded_transition_sigma_per_step_r0_centered_mean",
    )
    corr = ep_ranking_correlations(
        np.abs(np.array([r["decoded_transition_sigma_per_step_mean"] for r in rows])),
        np.array([r["analytic_ep"] for r in rows]),
    )
    r0 = rows[0]
    calibrated_gate = _gate(
        spearman=corr["spearman"],
        r0_sigma=float(r0["decoded_transition_sigma_per_step_r0_centered_mean"]),
        r0_arrow_accuracy=None,
    )
    result = {
        "rows": rows,
        "spearman": corr["spearman"],
        "pearson": corr["pearson"],
        "R0_sigma_per_step_mean": r0["decoded_transition_sigma_per_step_mean"],
        "R0_sigma_per_step_std": r0["decoded_transition_sigma_per_step_std"],
        "R0_sigma_per_step_raw_mean": r0["decoded_transition_sigma_per_step_mean"],
        "R0_sigma_per_step_raw_std": r0["decoded_transition_sigma_per_step_std"],
        "R0_sigma_per_step_calibrated_mean": r0[
            "decoded_transition_sigma_per_step_r0_centered_mean"
        ],
        "R0_sigma_per_step_calibrated_std": r0["decoded_transition_sigma_per_step_std"],
        "R0_decode_accuracy": r0["decode_accuracy"],
        "sigma_calibration": {
            "mode": "r0_centering_diagnostic",
            "offset": calibration["offset"],
            "uses_ood_test": False,
        },
        "gate": _gate(
            spearman=corr["spearman"],
            r0_sigma=r0["decoded_transition_sigma_per_step_mean"],
            r0_arrow_accuracy=None,
        ),
        "calibrated_gate": calibrated_gate,
        "notes": (
            "Oracle-decoded mixed-observation transition log-ratio sanity. "
            "This is transition-level arrow evidence under the synthetic "
            "observation model; it is not physical entropy production and not "
            "a deployable learned estimator."
        ),
    }
    save_json(output_dir / "ep_sanity_decoded_transition.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run direct biased-ring EP sanity check.")
    parser.add_argument("--output-dir", default="results/ep_sanity")
    parser.add_argument("--n-sequences", type=int, default=2048)
    parser.add_argument("--length", type=int, default=64)
    parser.add_argument("--n-states", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--include-latent", action="store_true")
    parser.add_argument("--include-decoded-transition", action="store_true")
    parser.add_argument("--latent-epochs", type=int, default=15)
    args = parser.parse_args()
    direct = run_ep_sanity(args.output_dir, args.n_sequences, args.length, args.n_states, args.seed)
    summary: dict[str, object] = {"direct_state_level_1A": direct}
    if args.include_latent:
        latent = run_latent_ep_sanity(
            args.output_dir,
            n_sequences=max(args.n_sequences, 512),
            length=max(args.length, 24),
            n_states=args.n_states,
            epochs=args.latent_epochs,
            seed=args.seed,
        )
        summary["latent_classifier_mixed_observation_1B"] = latent
    if args.include_decoded_transition or args.include_latent:
        decoded = run_decoded_transition_ep_sanity(
            args.output_dir,
            n_sequences=max(args.n_sequences, 512),
            length=max(args.length, 24),
            n_states=args.n_states,
            seed=args.seed,
        )
        summary["oracle_decoded_transition_mixed_observation_1B"] = decoded
    if len(summary) > 1:
        save_json(Path(args.output_dir) / "ep_sanity_summary.json", summary)


if __name__ == "__main__":
    main()
