"""Middle-class nuisance: order-invariant multi-frame direction cue.

Each nuisance frame contains one pulse at a random column. For d=+1 all
eight columns in a sequence share one random parity (all even or all odd);
for d=-1 the columns are drawn with mixed parity (at least one of each).
The per-frame marginal is uniform over all sixteen columns for both
directions, so no single frame carries the direction; the direction lives
in an unordered set statistic (parity homogeneity), so unordered-set and
temporal-summary probes decode it, and it survives frame shuffling and
order reversal. This instantiates the middle class of the cue-locality
spectrum: multi-frame but order-invariant.

Usage: python -m src.eval.midclass_oe --seeds 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.eval.temporal_evidence_audit import BASE, to_tensor
from src.eval.robust_methods_bench import SIZES, accuracy
from src.data.irreversible_source_inference import (
    IrreversibleSourceConfig, generate_split)
from src.eval.hardpair_oe import train_gru, probe, L, G


def nuisance_frames(d, rng):
    n = len(d)
    cols = np.empty((n, L), dtype=np.int64)
    homog = d > 0
    b = rng.integers(0, 2, size=n)
    cols_h = 2 * rng.integers(0, G // 2, size=(n, L)) + b[:, None]
    cols_m = rng.integers(0, G, size=(n, L))
    par = cols_m % 2
    same = (par == par[:, :1]).all(1)
    while same.any():
        redo = rng.integers(0, G, size=(int(same.sum()), L))
        cols_m[same] = redo
        par = cols_m % 2
        same = (par == par[:, :1]).all(1)
    cols = np.where(homog[:, None], cols_h, cols_m)
    rows = np.arange(G)[None, None, :, None]
    cgrid = np.arange(G)[None, None, None, :]
    blob = np.exp(-0.5 * (((rows - 8.0) ** 2) +
                          (cgrid - cols[:, :, None, None]) ** 2) / (1.15 ** 2))
    return (1.2 * blob / blob.max()).astype(np.float32)


def make(seed, corr_train=0.97):
    cfg = IrreversibleSourceConfig(seed=seed, **SIZES,
                                   **{**BASE, "nuisance_trail_decay": 0.0})
    sp = {s: generate_split(cfg, s)
          for s in ["train", "val_iid", "iid_test", "ood_test"]}
    out = {}
    for name, corr in [("train", corr_train), ("val_iid", corr_train),
                       ("iid_test", corr_train), ("ood_test", 1 - corr_train)]:
        rng = np.random.default_rng(seed * 131 + hash(name) % 1000)
        y = sp[name].y
        d = np.where(rng.random(len(y)) < corr, y, 1 - y) * 2 - 1
        core = np.asarray(sp[name].core_only)[:, :, None]
        nu = nuisance_frames(d, rng)[:, :, None]
        out[name] = (np.concatenate([core, nu], 2).astype(np.float32),
                     y, ((d > 0).astype(np.int64)))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--out", default="results/extended/midclass_oe.json")
    a = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outpath = Path(a.out)
    out = json.load(open(outpath, encoding="utf-8")) if outpath.exists() else {}
    for seed in range(a.seeds):
        key = f"seed{seed}"
        if key in out:
            continue
        sp = make(seed)
        r = {}
        iid, ood, m, mu, sd = train_gru(sp, seed, device, ret_model=True)
        r["erm"] = (iid, ood)
        xo = to_tensor((sp["ood_test"][0] - mu) / sd)
        yo = torch.from_numpy(sp["ood_test"][1])
        perm = torch.randperm(L)
        xs = xo.clone(); xs[:, :, 1] = xs[:, perm, 1]
        r["shuffle_nuis"] = accuracy(m, xs, yo, device)
        xr = xo.clone(); xr[:, :, 1] = torch.flip(xr[:, :, 1], dims=[1])
        r["reverse_nuis"] = accuracy(m, xr, yo, device)
        del m
        r["nuisance_only"] = train_gru(sp, seed + 5000, device, field=1)
        if seed < 5:
            nu_tr = torch.from_numpy(sp["train"][0][:, :, 1])
            nu_te = torch.from_numpy(sp["iid_test"][0][:, :, 1])
            dtr = torch.from_numpy(sp["train"][2])
            dte = torch.from_numpy(sp["iid_test"][2])
            best = 0.0
            for t in range(L):
                v = probe(nu_tr[:, t].reshape(len(nu_tr), -1), dtr,
                          nu_te[:, t].reshape(len(nu_te), -1), dte, device, 15)
                best = max(best, v)
            r["best_single_frame_dir"] = best
            srt_tr = torch.sort(nu_tr.reshape(len(nu_tr), L, -1), dim=1)[0]
            srt_te = torch.sort(nu_te.reshape(len(nu_te), L, -1), dim=1)[0]
            r["set_dir"] = probe(srt_tr.reshape(len(srt_tr), -1), dtr,
                                 srt_te.reshape(len(srt_te), -1), dte, device)
            mn_tr = nu_tr.mean(1).reshape(len(nu_tr), -1)
            mn_te = nu_te.mean(1).reshape(len(nu_te), -1)
            r["temporal_mean_dir"] = probe(mn_tr, dtr, mn_te, dte, device)
        out[key] = {k: (round(v, 4) if isinstance(v, float)
                        else [round(t, 4) for t in v]) for k, v in r.items()}
        print(key, out[key], flush=True)
        outpath.parent.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(outpath, "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
