"""Cue certification for the canonical strict order-encoded variant:
order interventions applied to the nuisance-only direction reader.

The backward sequence is the exact time reversal of the same closed path
(s_t^- = s_{L-1-t}^+), so the frame multiset is identical sample-wise; this
script certifies empirically that the nuisance-only reader behaves as an
order reader must: ordered accuracy high, full-sequence shuffle to chance,
full-sequence reversal flips predictions.

Usage: python -m src.eval.hardpair_certify --seeds 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.eval.temporal_evidence_audit import to_tensor
from src.eval.robust_methods_bench import accuracy
from src.eval.hardpair_oe import make, train_gru, L


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--out", default="results/extended/hardpair_certify.json")
    a = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outpath = Path(a.out)
    out = json.load(open(outpath, encoding="utf-8")) if outpath.exists() else {}
    for seed in range(a.seeds):
        key = f"seed{seed}"
        if key in out:
            continue
        sp = make(seed)
        iid, ood, m, mu, sd = train_gru(sp, seed + 5000, device, field=1,
                                        ret_model=True)
        x = to_tensor((sp["ood_test"][0][:, :, 1:2] - mu) / sd)
        y = torch.from_numpy(sp["ood_test"][1])
        r = {"ordered_iid": iid, "ordered_ood": ood}
        perm = torch.randperm(L)
        r["shuffled_ood"] = accuracy(m, x[:, perm], y, device)
        r["reversed_ood"] = accuracy(m, torch.flip(x, dims=[1]), y, device)
        out[key] = {k: round(v, 4) for k, v in r.items()}
        print(key, out[key], flush=True)
        outpath.parent.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(outpath, "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
