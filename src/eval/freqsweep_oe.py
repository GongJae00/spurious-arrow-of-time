"""Independent generator family: frequency-sweep direction shortcut.

The nuisance frame is a spatial sinusoid whose frequency bin advances by
+1 (d=+1) or -1 (d=-1) per frame around the full cycle of eight bins, with
a uniform random starting bin. Both directions visit every bin exactly
once, so the frame multiset is identical across directions; a single frame
carries no direction, and the cue is carried by inter-frame frequency
order. This family shares no spatial-pulse structure with the main
generator and tests whether the audit assigns the correct locality class
on an independently designed carrier.

Usage: python -m src.eval.freqsweep_oe --seeds 10
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
    f0 = rng.integers(0, 8, size=n)
    t = np.arange(L)
    fbin = (f0[:, None] + d[:, None] * t[None, :]) % 8 + 1  # bins 1..8
    x = np.arange(G)[None, None, None, :]
    wave = np.sin(2 * np.pi * fbin[:, :, None, None] * x / G)
    frame = np.repeat(wave, G, axis=2)  # constant over rows
    return (0.6 * frame).astype(np.float32)


def make(seed, corr_train=0.97):
    cfg = IrreversibleSourceConfig(seed=seed, **SIZES,
                                   **{**BASE, "nuisance_trail_decay": 0.0})
    sp = {s: generate_split(cfg, s)
          for s in ["train", "val_iid", "iid_test", "ood_test"]}
    out = {}
    for name, corr in [("train", corr_train), ("val_iid", corr_train),
                       ("iid_test", corr_train), ("ood_test", 1 - corr_train)]:
        rng = np.random.default_rng(seed * 211 + hash(name) % 1000)
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
    p.add_argument("--out", default="results/extended/freqsweep_oe.json")
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
        xr = xo.clone(); xr[:, :, 1] = torch.flip(xr[:, :, 1], dims=[1])
        r["reverse_nuis"] = accuracy(m, xr, yo, device)
        xc = xo.clone(); xc[:, :, 0] = torch.flip(xc[:, :, 0], dims=[1])
        r["reverse_core"] = accuracy(m, xc, yo, device)
        gen = torch.Generator().manual_seed(seed + 99)
        idx = torch.argsort(torch.rand(xo.shape[0], L, generator=gen), dim=1)
        xs = xo.clone()
        view = idx.view(xo.shape[0], L, 1, 1, 1).expand_as(xo[:, :, 1:2])
        xs[:, :, 1:2] = torch.gather(xo[:, :, 1:2], 1, view)
        r["persample_shuffle_nuis"] = accuracy(m, xs, yo, device)
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
        out[key] = {k: (round(v, 4) if isinstance(v, float)
                        else [round(t, 4) for t in v]) for k, v in r.items()}
        print(key, out[key], flush=True)
        outpath.parent.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(outpath, "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
