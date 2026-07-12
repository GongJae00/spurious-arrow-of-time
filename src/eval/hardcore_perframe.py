"""Per-frame core-label probe curve for the sequence-dependent core
configuration (all t=0..7, 5 seeds), plus reference sequence accuracies.

Usage:
  python -m src.eval.hardcore_perframe --seeds 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.eval.temporal_evidence_audit import (
    BASE,
    agg,
    to_tensor,
    train_mlp_probe,
)
from src.data.irreversible_source_inference import (
    IrreversibleSourceConfig,
    generate_split,
)

SIZES = dict(n_train=8192, n_val_iid=2048, n_iid_test=4096, n_ood_test=4096)
HARD = {"diffusion_start_step": 8, "diffusion_steps_between_frames": 2,
        "core_noise_std": 0.045, "core_noise_growth_power": 0.0,
        "observation_noise_std": 0.01, "nuisance_trail_decay": 0.0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--out", default="results/extended/hardcore_perframe.json")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    runs = []
    for s in range(args.seeds):
        cfg = IrreversibleSourceConfig(seed=s, **SIZES, **{**BASE, **HARD})
        tr = generate_split(cfg, "train")
        it = generate_split(cfg, "iid_test")
        ytr, yit = torch.from_numpy(tr.y), torch.from_numpy(it.y)
        accs = []
        L = tr.core_only.shape[1]
        for t in range(L):
            xtr = to_tensor(np.asarray(tr.core_only)[:, t].reshape(len(ytr), -1))
            xit = to_tensor(np.asarray(it.core_only)[:, t].reshape(len(yit), -1))
            accs.append(train_mlp_probe(xtr, ytr, [(xit, yit)], s * 100 + t, device)[0])
        runs.append(accs)
        print(f"seed {s}: {[round(a, 3) for a in accs]}", flush=True)

    result = {"per_frame": [agg([r[t] for r in runs]) for t in range(len(runs[0]))]}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print("means:", [round(a["mean"], 3) for a in result["per_frame"]])


if __name__ == "__main__":
    main()
