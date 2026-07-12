"""Second-stage tuning for the hard-core variant.

Constraint set:
  - best single-frame core probe        <= ~0.65  (sequence-dependence)
  - core-only GRU                       >= ~0.90  (core accessible)
  - MIXED no-spurious ERM               >= ~0.85  (no-spurious gate holds)

The third constraint failed at core_noise_std=0.05 (mixed ERM stuck at chance),
so we scan slightly easier cores and check all three quantities per config.

Usage:
  python -m src.eval.hardcore_tune2
"""

from __future__ import annotations

import argparse
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


def norm_pair(tr_x, te_x):
    mu, sd = float(tr_x.mean()), float(tr_x.std()) or 1.0
    return (tr_x - mu) / sd, (te_x - mu) / sd


def check(start: int, sb: int, noise: float, seed: int, device):
    common = {**BASE, "diffusion_start_step": start,
              "diffusion_steps_between_frames": sb, "core_noise_std": noise,
              "core_noise_growth_power": 0.0, "nuisance_trail_decay": 0.0}
    # no-spurious mixture (randomized nuisance)
    cfg_ns = IrreversibleSourceConfig(seed=seed, **SIZES, **common,
                                      train_nuisance_mode="randomized",
                                      ood_mode="randomized")
    sp = {s: generate_split(cfg_ns, s) for s in ["train", "val_iid", "iid_test"]}
    tr, va, it = sp["train"], sp["val_iid"], sp["iid_test"]
    ytr, yva, yit = (torch.from_numpy(s.y) for s in (tr, va, it))

    # single-frame core probe (t=0, clean core channel)
    x0tr = to_tensor(np.asarray(tr.core_only)[:, 0].reshape(len(tr.y), -1))
    x0it = to_tensor(np.asarray(it.core_only)[:, 0].reshape(len(it.y), -1))
    t0 = train_mlp_probe(x0tr, ytr, [(x0it, yit)], seed * 100, device)[0]

    # core-only GRU (normalized, extended budget)
    ctr, cit = norm_pair(np.asarray(tr.core_only), np.asarray(it.core_only))
    cva = (np.asarray(va.core_only) - float(np.asarray(tr.core_only).mean())) / (
        float(np.asarray(tr.core_only).std()) or 1.0)
    m = train_sequence_model(to_tensor(ctr[:, :, None]), ytr, to_tensor(cva[:, :, None]),
                             yva, seed * 31 + 7, device, epochs=100, patience=30)
    core_gru = eval_sequence(m, to_tensor(cit[:, :, None]), yit, device)

    # mixed no-spurious ERM (normalized, extended budget)
    mtr, mit = norm_pair(np.asarray(tr.mixed), np.asarray(it.mixed))
    mva = (np.asarray(va.mixed) - float(np.asarray(tr.mixed).mean())) / (
        float(np.asarray(tr.mixed).std()) or 1.0)
    m2 = train_sequence_model(to_tensor(mtr), ytr, to_tensor(mva), yva,
                              seed * 31 + 8, device, epochs=100, patience=30)
    ns_erm = eval_sequence(m2, to_tensor(mit), yit, device)
    return {"t0": t0, "core_gru": core_gru, "nospur_erm": ns_erm}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results/extended/hardcore_tune2.json")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    candidates = [
        (8, 2, 0.040),
        (8, 2, 0.045),
        (10, 2, 0.035),
        (10, 2, 0.040),
        (12, 2, 0.030),
    ]
    results = {}
    for start, sb, noise in candidates:
        r = check(start, sb, noise, 0, device)
        key = f"start{start}_sb{sb}_noise{noise}"
        results[key] = r
        print(f"{key:24s} t0 {r['t0']:.3f} coreGRU {r['core_gru']:.3f} "
              f"nospurERM {r['nospur_erm']:.3f}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
