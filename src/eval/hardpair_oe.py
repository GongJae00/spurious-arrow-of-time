"""Harder order-encoded nuisance: the (first, last) ordered frame pair is
direction-ambiguous by construction.

The nuisance pulse follows a non-uniform speed profile whose cumulative
offsets are C = [0, 1, 2, 4, 7, 10, 13, 16] mod 16 = [0, 1, 2, 4, 7, 10,
13, 0]: the pulse returns to its start position, so the first and last
frames are identical in distribution (and per sample) for both directions.
Forward (d=+1) plays the position list p0+C; backward (d=-1) plays it
reversed. The frame multiset is identical by reversal, single frames carry
no direction (p0 uniform), and the (first,last) ordered pair carries no
direction either; the direction lives in the interior ordering (the speed
profile ramps up forward and down backward).

Usage: python -m src.eval.hardpair_oe --seeds 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.eval.temporal_evidence_audit import BASE, to_tensor
from src.eval.robust_methods_bench import SIZES, accuracy, new_model
from src.data.irreversible_source_inference import (
    IrreversibleSourceConfig, generate_split)

L, G = 8, 16
CUM = np.array([0, 1, 2, 4, 7, 10, 13, 0])  # cumulative offsets mod 16


def nuisance_frames(d, rng):
    n = len(d)
    p0 = rng.integers(0, G, size=n)
    pos = (p0[:, None] + CUM[None, :]) % G
    pos = np.where(d[:, None] < 0, pos[:, ::-1], pos)
    rows = np.arange(G)[None, None, :, None]
    cols = np.arange(G)[None, None, None, :]
    blob = np.exp(-0.5 * (((rows - 8.0) ** 2) +
                          (cols - pos[:, :, None, None]) ** 2) / (1.15 ** 2))
    return (1.2 * blob / blob.max()).astype(np.float32)


def make(seed, corr_train=0.97):
    cfg = IrreversibleSourceConfig(seed=seed, **SIZES,
                                   **{**BASE, "nuisance_trail_decay": 0.0})
    sp = {s: generate_split(cfg, s)
          for s in ["train", "val_iid", "iid_test", "ood_test"]}
    out = {}
    for name, corr in [("train", corr_train), ("val_iid", corr_train),
                       ("iid_test", corr_train), ("ood_test", 1 - corr_train)]:
        rng = np.random.default_rng(seed * 101 + hash(name) % 1000)
        y = sp[name].y
        d = np.where(rng.random(len(y)) < corr, y, 1 - y) * 2 - 1
        core = np.asarray(sp[name].core_only)[:, :, None]
        nu = nuisance_frames(d, rng)[:, :, None]
        out[name] = (np.concatenate([core, nu], 2).astype(np.float32),
                     y, ((d > 0).astype(np.int64)))
    return out


def train_gru(sp, seed, device, field=None, ret_model=False, epochs=40, patience=12):
    mu = sp["train"][0].mean()
    sd = sp["train"][0].std() or 1.0

    def x(n):
        a = sp[n][0]
        if field is not None:
            a = a[:, :, field:field + 1]
        return to_tensor((a - mu) / sd)

    y = {n: torch.from_numpy(sp[n][1]) for n in sp}
    torch.manual_seed(seed)
    from src.models.minimal_sequence import build_model
    m = build_model("sequence_cnn_gru", grid_size=16, hidden_dim=64,
                    num_layers=1, dropout=0.0,
                    input_channels=1 if field is not None else 2).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    best, state, bad = -1, None, 0
    for _ in range(epochs):
        m.train()
        perm = torch.randperm(len(x("train")))
        for i in range(0, len(perm), 128):
            j = perm[i:i + 128]
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(m(x("train")[j].to(device)).logits,
                            y["train"][j].to(device)).backward()
            opt.step()
        m.eval()
        a = accuracy(m, x("val_iid"), y["val_iid"], device)
        if a > best:
            best, bad = a, 0
            state = {k: v.cpu().clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    m.load_state_dict(state)
    m.eval()
    res = (accuracy(m, x("iid_test"), y["iid_test"], device),
           accuracy(m, x("ood_test"), y["ood_test"], device))
    if ret_model:
        return res + (m, mu, sd)
    return res


def probe(xtr, ttr, xte, tte, device, epochs=30):
    md = nn.Sequential(nn.Linear(xtr.shape[1], 64), nn.ReLU(),
                       nn.Linear(64, 2)).to(device)
    opt = torch.optim.AdamW(md.parameters(), lr=1e-3)
    for _ in range(epochs):
        pm = torch.randperm(len(xtr))
        for i in range(0, len(xtr), 256):
            j = pm[i:i + 256]
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(md(xtr[j].to(device)), ttr[j].to(device)).backward()
            opt.step()
    md.eval()
    with torch.no_grad():
        return float((md(xte.to(device)).argmax(1).cpu() == tte).float().mean())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--out", default="results/extended/hardpair_oe30.json")
    p.add_argument("--nospurious", action="store_true")
    a = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outpath = Path(a.out)
    out = json.load(open(outpath, encoding="utf-8")) if outpath.exists() else {}
    for seed in range(a.seeds):
        key = f"seed{seed}"
        if key in out:
            continue
        if a.nospurious:
            sp = make(seed, corr_train=0.5)
            r = {"nospur": train_gru(sp, seed, device, epochs=100,
                                     patience=30)}
            out[key] = {k: [round(t, 4) for t in v] for k, v in r.items()}
            print(key, out[key], flush=True)
            outpath.parent.mkdir(parents=True, exist_ok=True)
            json.dump(out, open(outpath, "w", encoding="utf-8"), indent=1)
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
        xc = xo.clone(); xc[:, :, 0] = torch.flip(xc[:, :, 0], dims=[1])
        r["reverse_core"] = accuracy(m, xc, yo, device)
        del m
        r["nuisance_only"] = train_gru(sp, seed + 5000, device, field=1)
        if seed < 5:
            nu_tr = torch.from_numpy(sp["train"][0][:, :, 1])
            nu_te = torch.from_numpy(sp["iid_test"][0][:, :, 1])
            dtr = torch.from_numpy(sp["train"][2])
            dte = torch.from_numpy(sp["iid_test"][2])
            fl_tr = nu_tr[:, [0, L - 1]].reshape(len(nu_tr), -1)
            fl_te = nu_te[:, [0, L - 1]].reshape(len(nu_te), -1)
            r["firstlast_pair_dir"] = probe(fl_tr, dtr, fl_te, dte, device)
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
            aj_tr = nu_tr[:, [3, 4]].reshape(len(nu_tr), -1)
            aj_te = nu_te[:, [3, 4]].reshape(len(nu_te), -1)
            r["adjacent_pair_dir"] = probe(aj_tr, dtr, aj_te, dte, device)
        out[key] = {k: (round(v, 4) if isinstance(v, float)
                        else [round(t, 4) for t in v]) for k, v in r.items()}
        print(key, out[key], flush=True)
        outpath.parent.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(outpath, "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
