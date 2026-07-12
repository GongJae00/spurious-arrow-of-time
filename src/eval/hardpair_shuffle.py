"""Per-sample-independent random permutation on OE-Strict.

Because the frame multiset is identical sample-wise across directions, an
independent uniform permutation per sample makes the two classes'
shuffled-sequence distributions identical, so any classifier is at chance
in expectation. This distinguishes the destruction intervention
(per-sample shuffle) from a fixed temporal permutation, which re-encodes
the direction of an asymmetric closed path.

Usage: python -m src.eval.hardpair_shuffle --seeds 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.eval.temporal_evidence_audit import to_tensor
from src.eval.robust_methods_bench import accuracy
from src.eval.hardpair_oe import make, train_gru, L


def per_sample_shuffle(x, gen):
    idx = torch.argsort(torch.rand(x.shape[0], L, generator=gen), dim=1)
    view = idx.view(x.shape[0], L, *([1] * (x.dim() - 2)))
    return torch.gather(x, 1, view.expand_as(x))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--out", default="results/extended/hardpair_shuffle.json")
    a = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outpath = Path(a.out)
    out = json.load(open(outpath, encoding="utf-8")) if outpath.exists() else {}
    for seed in range(a.seeds):
        key = f"seed{seed}"
        if key in out:
            continue
        sp = make(seed)
        gen = torch.Generator().manual_seed(seed + 777)
        r = {}
        # nuisance-only reader
        _, _, m, mu, sd = train_gru(sp, seed + 5000, device, field=1,
                                    ret_model=True)
        xn = to_tensor((sp["ood_test"][0][:, :, 1:2] - mu) / sd)
        y = torch.from_numpy(sp["ood_test"][1])
        r["reader_persample_shuffle_ood"] = accuracy(
            m, per_sample_shuffle(xn, gen), y, device)
        del m
        # mixed ERM, nuisance channel only
        _, _, m, mu, sd = train_gru(sp, seed, device, ret_model=True)
        xm = to_tensor((sp["ood_test"][0] - mu) / sd)
        xs = xm.clone()
        xs[:, :, 1] = per_sample_shuffle(xm[:, :, 1], gen)
        r["mixed_persample_shuffle_nuis_ood"] = accuracy(m, xs, y, device)
        out[key] = {k: round(v, 4) for k, v in r.items()}
        print(key, out[key], flush=True)
        outpath.parent.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(outpath, "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
