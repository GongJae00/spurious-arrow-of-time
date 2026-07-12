"""Equalized-ceiling control on the order-encoded-core configuration.

In the original order-encoded-core control (occ_benchmark) the core's
predictive ceiling is 1.000 while the nuisance's is 0.97, so core selection
could reflect either equal-form accessibility or the predictiveness edge.
Here the core pulse direction disagrees with the label with probability
0.03 (core ceiling 0.97 = nuisance ceiling 0.97, identical order-encoded
cue form, identical pulse process), removing the predictiveness edge.

A core-following solution scores ~0.97 IID and ~0.97 OOD; a
nuisance-following solution scores ~0.97 IID and ~0.03 OOD.

Usage: python -m src.eval.occ_equalized --seeds 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import src.eval.robust_methods_bench as rmb
from src.eval.temporal_evidence_audit import (
    to_tensor, train_sequence_model, eval_sequence)
from src.data.irreversible_source_inference import (
    IrreversibleSourceConfig, generate_split)

EQ = {**rmb.BASE, "core_process": "directional_pulse",
      "core_direction_flip_prob": 0.03}


def run(seed, device):
    cfg = IrreversibleSourceConfig(seed=seed, **rmb.SIZES, **EQ)
    sp = {s: generate_split(cfg, s)
          for s in ["train", "val_iid", "iid_test", "ood_test"]}
    mu = float(np.asarray(sp["train"].mixed).mean())
    sd = float(np.asarray(sp["train"].mixed).std()) or 1.0

    def x(n):
        return to_tensor((np.asarray(sp[n].mixed) - mu) / sd)

    y = {n: torch.from_numpy(sp[n].y) for n in sp}
    m = train_sequence_model(x("train"), y["train"], x("val_iid"),
                             y["val_iid"], seed * 31 + 8, device,
                             epochs=100, patience=30)
    return (eval_sequence(m, x("iid_test"), y["iid_test"], device, "eq", seed),
            eval_sequence(m, x("ood_test"), y["ood_test"], device, "eq",
                          seed + 1))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--out", default="results/extended/occ_equalized.json")
    a = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outpath = Path(a.out)
    out = json.load(open(outpath, encoding="utf-8")) if outpath.exists() else {}
    for s in range(a.seeds):
        key = f"seed{s}"
        if key in out:
            continue
        iid, ood = run(s, device)
        out[key] = {"iid": round(iid, 4), "ood": round(ood, 4)}
        print(key, out[key], flush=True)
        outpath.parent.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(outpath, "w", encoding="utf-8"), indent=1)
    i = np.array([v["iid"] for v in out.values()])
    o = np.array([v["ood"] for v in out.values()])
    print(f"mean {i.mean():.3f}/{o.mean():.3f} | core {(o >= .8).sum()} "
          f"| nuis-coll {(o <= .2).sum()}", flush=True)


if __name__ == "__main__":
    main()
