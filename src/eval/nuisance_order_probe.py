"""Order-sensitivity probe for the nuisance cue itself.

Trains a SequenceCNNGRU on the NUISANCE channel only (a guaranteed
direction-reader) and evaluates it under test-time frame shuffling and
frame-order reversal, in both benchmark variants:

- trail variant (nuisance_trail_decay=0.78): if OOD accuracy stays collapsed
  under shuffling/order reversal, the directional cue is frame-local
  (spatial trail asymmetry) and does not require temporal order.
- order-encoded variant (nuisance_trail_decay=0.0): the cue should vanish
  under shuffling (chance) and flip under order reversal.

Usage:
  python -m src.eval.nuisance_order_probe --seeds 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.eval.temporal_evidence_audit import (
    BASE,
    SPLIT_SIZES,
    agg,
    eval_sequence,
    to_tensor,
    train_sequence_model,
)
from src.data.irreversible_source_inference import (
    IrreversibleSourceConfig,
    generate_split,
)


def run_variant(trail_decay: float, seed: int, device):
    cfg = IrreversibleSourceConfig(
        seed=seed, **SPLIT_SIZES, **{**BASE, "nuisance_trail_decay": trail_decay},
    )
    splits = {s: generate_split(cfg, s) for s in ["train", "val_iid", "iid_test", "ood_test"]}
    tr, va, it, ot = (splits[k] for k in ["train", "val_iid", "iid_test", "ood_test"])

    def nu(sp):
        x = np.asarray(sp.nuisance_only)[:, :, None]  # add channel dim -> [n, L, 1, H, W]
        return to_tensor(x)

    xtr, xva, xit, xot = nu(tr), nu(va), nu(it), nu(ot)
    ytr, yva = torch.from_numpy(tr.y), torch.from_numpy(va.y)
    yit, yot = torch.from_numpy(it.y), torch.from_numpy(ot.y)

    model = train_sequence_model(xtr, ytr, xva, yva, seed * 13 + 3, device)
    return {
        "iid": eval_sequence(model, xit, yit, device),
        "ood": eval_sequence(model, xot, yot, device),
        "iid_shuffled": eval_sequence(model, xit, yit, device, "shuffled", seed),
        "ood_shuffled": eval_sequence(model, xot, yot, device, "shuffled", seed),
        "iid_reversed_order": eval_sequence(model, xit, yit, device, "reversed_order"),
        "ood_reversed_order": eval_sequence(model, xot, yot, device, "reversed_order"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--out", default="results/extended/nuisance_order_probe.json")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    result = {}
    for name, decay in [("trail", 0.78), ("order_encoded", 0.0)]:
        runs = []
        for s in range(args.seeds):
            runs.append(run_variant(decay, s, device))
            print(f"{name} seed {s}: " + " ".join(f"{k}={v:.3f}" for k, v in runs[-1].items()),
                  flush=True)
        result[name] = {k: agg([r[k] for r in runs]) for k in runs[0]}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    for name, d in result.items():
        print(f"=== {name} ===")
        for k, v in d.items():
            print(f"  {k}: {v['mean']:.3f} (+-{v['std']:.3f})")


if __name__ == "__main__":
    main()
