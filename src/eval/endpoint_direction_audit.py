"""Endpoint direction audit.

Trains a classifier on the FINAL frame only to predict the nuisance direction
(an auxiliary target, not the label). If endpoint matching works, this stays at
chance, showing the endpoint does not reveal the nuisance direction even when it
is the explicit target. The residue-visible variant (no endpoint matching) is
included as a positive control where the audit should detect leakage.

Usage:
  python -m src.eval.endpoint_direction_audit
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from torch import nn

from src.data.irreversible_source_inference import (
    IrreversibleSourceConfig,
    generate_split,
)


BASE = dict(
    grid_size=16, length=8, diffusion_alpha=0.22, diffusion_start_step=0,
    diffusion_steps_between_frames=4, core_noise_std=0.006, observation_noise_std=0.04,
    core_scale=1.0, nuisance_scale=1.2, nuisance_sigma=1.15, nuisance_speed=2.0,
    nuisance_trail_decay=0.78, nuisance_correlation=0.97, observation_layout="two_channel",
)


def final_frame_direction_accuracy(variant: str, seed: int, device: torch.device) -> float:
    cfg = IrreversibleSourceConfig(
        n_train=4096, n_val_iid=512, n_iid_test=2048, n_ood_test=512, seed=seed,
        benchmark_variant=variant, **BASE,
    )
    tr = generate_split(cfg, "train")
    te = generate_split(cfg, "iid_test")

    def prep(sp):
        x = np.asarray(sp.mixed)[:, -1]  # final frame only, shape [n, 2, H, W]
        x = x.reshape(len(x), -1).astype(np.float32)
        d = (np.asarray(sp.nuisance_direction) > 0).astype(np.int64)
        return torch.from_numpy(x), torch.from_numpy(d)

    xtr, dtr = prep(tr)
    xte, dte = prep(te)
    mean, std = xtr.mean(), xtr.std().clamp_min(1e-6)
    xtr, xte = (xtr - mean) / std, (xte - mean) / std
    xtr, dtr, xte, dte = xtr.to(device), dtr.to(device), xte.to(device), dte.to(device)

    torch.manual_seed(seed)
    net = nn.Sequential(
        nn.Linear(xtr.shape[1], 64), nn.ReLU(),
        nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 2),
    ).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(40):
        net.train()
        perm = torch.randperm(len(xtr), device=device)
        for i in range(0, len(xtr), 128):
            idx = perm[i:i + 128]
            opt.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(net(xtr[idx]), dtr[idx])
            loss.backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        acc = float((net(xte).argmax(1) == dte).float().mean().item())
    return acc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--out", default="results/extended/endpoint_direction_audit.json")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = {}
    for variant in ["endpoint_matched", "residue_visible"]:
        accs = [final_frame_direction_accuracy(variant, s, device) for s in range(args.seeds)]
        result[variant] = {"mean": float(np.mean(accs)), "std": float(np.std(accs, ddof=1)),
                           "values": [round(a, 4) for a in accs]}
        print(f"{variant:18s} final-frame direction acc: "
              f"{result[variant]['mean']:.3f} (+-{result[variant]['std']:.3f})")
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
