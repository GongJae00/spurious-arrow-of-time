"""Authoritative probe statistics for the FINAL sequence-dependent core
configuration (start=8, sb=2, core_noise=0.045 uniform, obs_noise=0.01):
temporal-mean probe and core-only ordered GRU, 5 seeds.

Usage:
  python -m src.eval.hardcore_probe_final --seeds 5
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
    train_sequence_model,
    eval_sequence,
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
    parser.add_argument("--out", default="results/extended/hardcore_probe_final.json")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    runs = []
    for s in range(args.seeds):
        cfg = IrreversibleSourceConfig(seed=s, **SIZES, **{**BASE, **HARD})
        sp = {k: generate_split(cfg, k) for k in ["train", "val_iid", "iid_test", "ood_test"]}
        tr, va, it, ot = (sp[k] for k in ["train", "val_iid", "iid_test", "ood_test"])
        ytr, yva, yit, yot = (torch.from_numpy(x.y) for x in (tr, va, it, ot))

        def mean_feat(x):
            m = np.asarray(x.core_only).mean(axis=1)
            return to_tensor(m.reshape(len(m), -1))

        mean_acc = train_mlp_probe(mean_feat(tr), ytr, [(mean_feat(it), yit)],
                                   s * 7 + 3, device)[0]

        mu = float(np.asarray(tr.core_only).mean())
        sd = float(np.asarray(tr.core_only).std()) or 1.0

        def seq(x):
            z = (np.asarray(x.core_only) - mu) / sd
            return to_tensor(z[:, :, None])

        m = train_sequence_model(seq(tr), ytr, seq(va), yva, s * 31 + 7, device,
                                 epochs=100, patience=30)
        gru_iid = eval_sequence(m, seq(it), yit, device)
        gru_ood = eval_sequence(m, seq(ot), yot, device)
        runs.append({"mean_probe": mean_acc, "gru_iid": gru_iid, "gru_ood": gru_ood})
        print(f"seed {s}: mean {mean_acc:.3f} gru {gru_iid:.3f}/{gru_ood:.3f}", flush=True)

    result = {k: agg([r[k] for r in runs]) for k in runs[0]}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    for k, v in result.items():
        print(f"{k}: {v['mean']:.3f} (+-{v['std']:.3f})")


if __name__ == "__main__":
    main()
