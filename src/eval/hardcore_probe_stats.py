"""Multi-seed probe statistics for the chosen hard-core configuration
(diffusion_start_step=8, steps_between=2, core_noise_std=0.05, growth=0).

Usage:
  python -m src.eval.hardcore_probe_stats --seeds 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.eval.core_hardening_sweep import evaluate_config
from src.eval.temporal_evidence_audit import agg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--out", default="results/extended/hardcore_probe_stats.json")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    runs = []
    for s in range(args.seeds):
        r = evaluate_config(8, 0.05, 0.0, s, device, steps_between=2)
        runs.append(r)
        print(f"seed {s}: " + " ".join(f"{k}={v:.3f}" for k, v in r.items()), flush=True)

    result = {k: agg([r[k] for r in runs]) for k in runs[0]}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    for k, v in result.items():
        print(f"{k}: {v['mean']:.3f} (+-{v['std']:.3f})")


if __name__ == "__main__":
    main()
