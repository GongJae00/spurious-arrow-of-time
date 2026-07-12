"""Semi-synthetic benchmark: real time-series core (UCR FordA, engine-noise
sensor data, binary labels) + synthetic order-encoded nuisance channel.

Each series (length 500) is cut into L=10 segments of W=50 samples (the
"frames").  The nuisance channel is a moving 1-D Gaussian bump whose start
position is uniform and whose step is +5 (d=+1) or -5 (d=-1) per segment,
wrap-around mod 50, so the visited position multiset {p0+5k mod 50} is
direction-independent: no single segment and no unordered segment set
carries the direction; only segment order does.  The direction is correlated
with the label at 0.97 in train/val/IID-test and 0.03 under OOD reversal.

Gate runs per seed: core-only, nuisance-only, mixed ERM (main), no-spurious
mixture, final-segment probes (label & direction), best single-segment
direction probe, unordered-set direction probe, and channel-selective order
interventions on the trained ERM model (shuffle / reverse only the nuisance
channel's segment order; reverse only the core channel's).

Usage: python -m src.eval.semisynthetic_ucr --seeds 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

L, W = 10, 50
CORR = 0.97


def load_forda():
    rows = []
    for f in ["data/ucr/FordA_TRAIN.tsv", "data/ucr/FordA_TEST.tsv"]:
        rows.append(np.loadtxt(f))
    a = np.concatenate(rows)
    y = (a[:, 0] > 0).astype(np.int64)
    x = a[:, 1:].astype(np.float32)
    x = (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-8)
    return x.reshape(len(x), L, W), y


def load_har(mode="har"):
    """UCI HAR body-acceleration windows (128 samples at 50 Hz), binarized
    into dynamic (walking variants, labels 1-3) vs. static (sitting,
    standing, lying, labels 4-6); windows are linearly resampled to length
    L*W so the same segmenting pipeline applies."""
    base = "data/ucr/har/UCI HAR Dataset"
    xs, ys = [], []
    for split in ["train", "test"]:
        x = np.loadtxt(f"{base}/{split}/Inertial Signals/body_acc_x_{split}.txt")
        yy = np.loadtxt(f"{base}/{split}/y_{split}.txt")
        xs.append(x)
        ys.append(yy)
    x = np.concatenate(xs).astype(np.float32)
    yy = np.concatenate(ys)
    if mode == "har2":
        # fine-grained dynamic pair: walking (1) vs walking upstairs (2)
        keep = yy <= 2
        x, yy = x[keep], yy[keep]
        y = (yy == 1).astype(np.int64)
    else:
        y = (yy <= 3).astype(np.int64)
    x = (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-8)
    t_old = np.linspace(0, 1, x.shape[1])
    t_new = np.linspace(0, 1, L * W)
    x = np.stack([np.interp(t_new, t_old, r) for r in x]).astype(np.float32)
    return x.reshape(len(x), L, W), y


def make_split(xc, y, rng, corr, n):
    idx = rng.choice(len(y), size=n, replace=False)
    xcs, ys = xc[idx], y[idx]
    d = np.where(rng.random(n) < corr, ys, 1 - ys)
    p0 = rng.integers(0, W, size=n)
    t = np.arange(L)
    pos = (p0[:, None] + (2 * d[:, None] - 1) * 5 * t[None, :]) % W
    grid = np.arange(W)[None, None, :]
    nu = np.exp(-0.5 * ((grid - pos[:, :, None]) ** 2) / (3.0 ** 2))
    nu = nu.astype(np.float32) * 1.5
    return xcs, nu, ys, d.astype(np.int64)


class SegGRU(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(ch * W, 64), nn.ReLU())
        self.gru = nn.GRU(64, 64, batch_first=True)
        self.head = nn.Linear(64, 2)

    def forward(self, x):  # [B, L, ch, W]
        b = x.shape[0]
        h = self.enc(x.reshape(b, L, -1))
        return self.head(self.gru(h)[0][:, -1])


def train(model, xtr, ytr, xva, yva, device, epochs=60):
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best, state, bad = -1, None, 0
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(xtr))
        for i in range(0, len(xtr), 128):
            j = perm[i:i + 128]
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(model(xtr[j].to(device)), ytr[j].to(device)).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            acc = (model(xva.to(device)).argmax(1).cpu() == yva).float().mean().item()
        if acc > best:
            best, bad = acc, 0
            state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 15:
                break
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def acc(model, x, y, device):
    return float((model(x.to(device)).argmax(1).cpu() == y).float().mean())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--dataset", default="forda", choices=["forda", "har", "har2"])
    p.add_argument("--out", default="results/extended/semisynthetic_ucr.json")
    a = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if a.dataset in ("har", "har2"):
        xc_all, y_all = load_har(a.dataset)
        n = len(y_all)
        cut = [int(n*0.62), int(n*0.70), int(n*0.85), n]
    else:
        xc_all, y_all = load_forda()
        cut = [3200, 3600, 4260, 4921]
    out = {}
    for seed in range(a.seeds):
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(y_all))
        pool = {}
        for name, s, e, corr in [("train", 0, cut[0], CORR),
                                 ("val", cut[0], cut[1], CORR),
                                 ("iid", cut[1], cut[2], CORR),
                                 ("ood", cut[2], cut[3], 1 - CORR)]:
            idx = order[s:e]
            pool[name] = make_split(xc_all[idx], y_all[idx],
                                    np.random.default_rng(seed * 7 + s),
                                    corr, len(idx))

        def T(name, mode):
            xcs, nu, ys, d = pool[name]
            core = torch.from_numpy(xcs)[:, :, None, :]
            nut = torch.from_numpy(nu)[:, :, None, :]
            if mode == "mixed":
                x = torch.cat([core, nut], 2)
            elif mode == "core":
                x = core
            else:
                x = nut
            return x, torch.from_numpy(ys), torch.from_numpy(d)

        r = {}
        torch.manual_seed(seed)
        m = train(SegGRU(1), *T("train", "core")[:2], *T("val", "core")[:2], device)
        r["core_only"] = [acc(m, *T("iid", "core")[:2], device),
                          acc(m, *T("ood", "core")[:2], device)]
        torch.manual_seed(seed)
        m = train(SegGRU(1), *T("train", "nu")[:2], *T("val", "nu")[:2], device)
        r["nuisance_only"] = [acc(m, *T("iid", "nu")[:2], device),
                              acc(m, *T("ood", "nu")[:2], device)]
        torch.manual_seed(seed)
        erm = train(SegGRU(2), *T("train", "mixed")[:2], *T("val", "mixed")[:2], device)
        xi, yi, _ = T("iid", "mixed")
        xo, yo, _ = T("ood", "mixed")
        r["erm"] = [acc(erm, xi, yi, device), acc(erm, xo, yo, device)]
        # channel-selective order interventions on the trained ERM model
        perm = torch.randperm(L)
        xs = xo.clone(); xs[:, :, 1] = xs[:, perm, 1]
        r["erm_shuffle_nuis"] = acc(erm, xs, yo, device)
        xr = xo.clone(); xr[:, :, 1] = torch.flip(xr[:, :, 1], dims=[1])
        r["erm_reverse_nuis"] = acc(erm, xr, yo, device)
        xrc = xo.clone(); xrc[:, :, 0] = torch.flip(xrc[:, :, 0], dims=[1])
        r["erm_reverse_core"] = acc(erm, xrc, yo, device)
        # no-spurious: independent nuisance in train and eval
        ns = {}
        for name in pool:
            xcs, nu, ys, d = pool[name]
            rng2 = np.random.default_rng(seed * 13 + hash(name) % 97)
            ns[name] = make_split(xcs, ys, rng2, 0.5, len(ys))
        pool_bak = {k: pool[k] for k in pool}
        pool.update(ns)
        torch.manual_seed(seed)
        m = train(SegGRU(2), *T("train", "mixed")[:2], *T("val", "mixed")[:2], device)
        r["no_spurious"] = [acc(m, *T("iid", "mixed")[:2], device),
                            acc(m, *T("ood", "mixed")[:2], device)]
        pool.update(pool_bak)
        # probes: single-segment / unordered-set / final-segment
        def probe(x, t, tgt, shuffle=False):
            xt = x[:, :, :] if t is None else x[:, t]
            model = nn.Sequential(nn.Linear(xt.reshape(len(xt), -1).shape[1], 64),
                                  nn.ReLU(), nn.Linear(64, 2)).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
            xtr = xt.reshape(len(xt), -1)
            for _ in range(30):
                pm = torch.randperm(len(xtr))
                for i in range(0, len(xtr), 256):
                    j = pm[i:i + 256]
                    opt.zero_grad(set_to_none=True)
                    F.cross_entropy(model(xtr[j].to(device)), tgt[j].to(device)).backward()
                    opt.step()
            model.eval()
            return model

        xn_tr, ytr_, dtr_ = T("train", "nu")
        xn_iid, yiid_, diid_ = T("iid", "nu")
        best_sf = 0.0
        for t in range(L):
            pm = probe(xn_tr, t, dtr_)
            with torch.no_grad():
                v = float((pm(xn_iid[:, t].reshape(len(xn_iid), -1).to(device))
                           .argmax(1).cpu() == diid_).float().mean())
            best_sf = max(best_sf, v)
        r["best_single_seg_dir"] = best_sf
        srt = torch.sort(xn_tr.reshape(len(xn_tr), L, -1), dim=1)[0]
        srti = torch.sort(xn_iid.reshape(len(xn_iid), L, -1), dim=1)[0]
        pm = probe(srt, None, dtr_)
        with torch.no_grad():
            r["unordered_set_dir"] = float(
                (pm(srti.reshape(len(srti), -1).to(device)).argmax(1).cpu()
                 == diid_).float().mean())
        xm_tr, ytr2, dtr2 = T("train", "mixed")
        xm_iid, yiid2, diid2 = T("iid", "mixed")
        pm = probe(xm_tr, L - 1, ytr2)
        with torch.no_grad():
            r["final_seg_label"] = float(
                (pm(xm_iid[:, L - 1].reshape(len(xm_iid), -1).to(device))
                 .argmax(1).cpu() == yiid2).float().mean())
        pm = probe(xm_tr, L - 1, dtr2)
        with torch.no_grad():
            r["final_seg_dir"] = float(
                (pm(xm_iid[:, L - 1].reshape(len(xm_iid), -1).to(device))
                 .argmax(1).cpu() == diid2).float().mean())

        out[f"seed{seed}"] = {k: (round(v, 4) if isinstance(v, float)
                                  else [round(t, 4) for t in v])
                              for k, v in r.items()}
        print(f"seed{seed}:", out[f"seed{seed}"], flush=True)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
