"""Fixed-data / multiple-initialization decomposition of basin selection.

For each of a few FIXED data seeds, train sequence ERM many times with
different model-initialization seeds (identical data splits, identical
recipe), under both the standard and the extended budget, in both variants.
Reports per-split basin fractions (core / nuisance-collapse / chance), which
separates dataset-realization effects from initialization/optimization
effects in the bimodal selection.

Usage:
  python -m src.eval.multi_init_experiment --data-seeds 3 --inits 15
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
    train_sequence_model,
    eval_sequence,
)
from src.data.irreversible_source_inference import (
    IrreversibleSourceConfig,
    generate_split,
)

SIZES = dict(n_train=8192, n_val_iid=2048, n_iid_test=4096, n_ood_test=4096)


def regime(iid: float, ood: float) -> str:
    if iid >= 0.8 and ood >= 0.8:
        return "core"
    if iid >= 0.8 and ood <= 0.2:
        return "collapse"
    return "chance"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-seeds", type=int, default=3)
    parser.add_argument("--inits", type=int, default=15)
    parser.add_argument("--out", default="results/extended/multi_init.json")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    budgets = {"standard": (40, 12), "extended": (100, 30)}
    variants = {"trail": 0.78, "order_encoded": 0.0}
    result = {}
    if Path(args.out).exists():
        result = json.loads(Path(args.out).read_text(encoding="utf-8"))
        print("resuming; completed cells:", sorted(result), flush=True)
    for vname, decay in variants.items():
        for ds in range(args.data_seeds):
            cfg = IrreversibleSourceConfig(
                seed=ds, **SIZES, **{**BASE, "nuisance_trail_decay": decay})
            sp = {s: generate_split(cfg, s)
                  for s in ["train", "val_iid", "iid_test", "ood_test"]}
            mu = float(np.asarray(sp["train"].mixed).mean())
            sd = float(np.asarray(sp["train"].mixed).std()) or 1.0

            def x(name):
                z = (np.asarray(sp[name].mixed) - mu) / sd
                return to_tensor(z)

            xtr, xva = x("train"), x("val_iid")
            xit, xot = x("iid_test"), x("ood_test")
            ytr = torch.from_numpy(sp["train"].y)
            yva = torch.from_numpy(sp["val_iid"].y)
            yit = torch.from_numpy(sp["iid_test"].y)
            yot = torch.from_numpy(sp["ood_test"].y)

            for bname, (ep, pat) in budgets.items():
                key = f"{vname}/data{ds}/{bname}"
                if key in result:
                    continue
                rows = []
                for init in range(args.inits):
                    m = train_sequence_model(
                        xtr, ytr, xva, yva, 90001 + 613 * init, device,
                        epochs=ep, patience=pat)
                    iid = eval_sequence(m, xit, yit, device)
                    ood = eval_sequence(m, xot, yot, device)
                    rows.append({"init": init, "iid": round(iid, 4),
                                 "ood": round(ood, 4),
                                 "regime": regime(iid, ood)})
                regs = [r["regime"] for r in rows]
                result[key] = {
                    "core": regs.count("core"),
                    "collapse": regs.count("collapse"),
                    "chance": regs.count("chance"),
                    "n": len(rows), "rows": rows,
                }
                print(f"{key}: core {regs.count('core')} "
                      f"collapse {regs.count('collapse')} "
                      f"chance {regs.count('chance')}", flush=True)
                Path(args.out).parent.mkdir(parents=True, exist_ok=True)
                with open(args.out, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
