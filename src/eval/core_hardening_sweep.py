"""Parameter sweep for a sequence-dependent core variant.

Goal: find generator settings where the CLEAN core channel satisfies
  - best single-frame label probe     <= ~0.65
  - temporal-mean label probe          not much higher
  - full ordered-sequence GRU accuracy >= ~0.90 (IID and OOD)
so that recovering the label genuinely requires integrating multiple frames.

Knobs: diffusion_start_step (later start = more blurred first frame) and
core_noise_std (per-frame noise that single frames cannot average away).

Usage:
  python -m src.eval.core_hardening_sweep
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from src.eval.temporal_evidence_audit import (
    BASE,
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


def evaluate_config(start: int, noise: float, growth: float, seed: int, device,
                    steps_between: int = 4):
    cfg = IrreversibleSourceConfig(
        seed=seed, **SIZES,
        **{**BASE, "diffusion_start_step": start, "core_noise_std": noise,
           "core_noise_growth_power": growth, "nuisance_trail_decay": 0.0,
           "diffusion_steps_between_frames": steps_between},
    )
    splits = {s: generate_split(cfg, s) for s in ["train", "val_iid", "iid_test", "ood_test"]}
    tr, va, it, ot = (splits[k] for k in ["train", "val_iid", "iid_test", "ood_test"])
    ytr, yva = torch.from_numpy(tr.y), torch.from_numpy(va.y)
    yit, yot = torch.from_numpy(it.y), torch.from_numpy(ot.y)

    def core_frame(sp, t):
        x = np.asarray(sp.core_only)[:, t]
        return to_tensor(x.reshape(len(x), -1))

    frame_accs = []
    for t in [0, 3, 7]:
        acc = train_mlp_probe(core_frame(tr, t), ytr, [(core_frame(it, t), yit)],
                              seed * 100 + t, device)[0]
        frame_accs.append(acc)

    def core_mean(sp):
        x = np.asarray(sp.core_only).mean(axis=1)
        return to_tensor(x.reshape(len(x), -1))

    mean_acc = train_mlp_probe(core_mean(tr), ytr, [(core_mean(it), yit)],
                               seed * 100 + 55, device)[0]

    mu = float(np.asarray(tr.core_only).mean())
    sd = float(np.asarray(tr.core_only).std()) or 1.0

    def core_seq(sp):
        x = (np.asarray(sp.core_only) - mu) / sd  # train-stats standardization (main pipeline)
        return to_tensor(x[:, :, None])

    model = train_sequence_model(core_seq(tr), ytr, core_seq(va), yva, seed * 31 + 7, device,
                                 epochs=100, patience=30)
    gru_iid = eval_sequence(model, core_seq(it), yit, device)
    gru_ood = eval_sequence(model, core_seq(ot), yot, device)
    return {"frame_t0": frame_accs[0], "frame_t3": frame_accs[1], "frame_t7": frame_accs[2],
            "mean_probe": mean_acc, "gru_iid": gru_iid, "gru_ood": gru_ood}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results/extended/core_hardening_sweep.json")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    grid = list(itertools.product([6, 8, 10], [0.03, 0.05, 0.07]))
    results = {}
    for start, noise in grid:
        r = evaluate_config(start, noise, 0.0, 0, device, steps_between=2)
        key = f"start{start}_sb2_noise{noise}"
        results[key] = r
        print(f"{key:22s} t0 {r['frame_t0']:.3f} t3 {r['frame_t3']:.3f} t7 {r['frame_t7']:.3f} "
              f"mean {r['mean_probe']:.3f} gru {r['gru_iid']:.3f}/{r['gru_ood']:.3f}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
