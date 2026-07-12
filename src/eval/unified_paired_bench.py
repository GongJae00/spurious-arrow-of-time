"""Unified paired comparison on the order-encoded variant.

One design for every major comparison: data seeds 0..S-1 (one dataset per
seed shared by all methods and architectures), model initialization stream
torch.manual_seed(seed) shared by every method with the same backbone, no
per-profile hashing.  Every method is trained once per seed and evaluated
under TWO model-selection protocols recorded from the same training run:

- sel_iid:   checkpoint with best IID-validation accuracy (corr 0.97)
- sel_shift: checkpoint with best accuracy on a held-out shifted validation
             environment (corr 0.50, nuisance-independent; never used for
             training and disjoint from the OOD test at corr 0.03)

Methods (GRU backbone): erm, groupdro_joint, irmv1, dann_dir, jtt,
frame_rand.  Architectures (ERM objective, same data/init pairing): lstm,
tcn, transformer, pooling.

Usage: python -m src.eval.unified_paired_bench --seeds 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import src.eval.robust_methods_bench as rmb
from src.eval.robust_methods_bench import (
    BASE, SIZES, EPOCHS, LR, WD, BS, GradReverse, make_data, tensors,
    accuracy)
from src.data.irreversible_source_inference import (
    IrreversibleSourceConfig, generate_split)
from src.models.minimal_sequence import build_model

ARCHS = {"lstm": "sequence_cnn_lstm", "tcn": "sequence_cnn_tcn",
         "transformer": "sequence_cnn_transformer",
         "pooling": "sequence_cnn_temporal_pool"}


def shifted_val(seed: int, mu: float, sd: float):
    cfg = IrreversibleSourceConfig(
        seed=seed, **SIZES, **{**BASE, "nuisance_correlation": 0.50})
    sp = generate_split(cfg, "val_iid")
    x = (np.asarray(sp.mixed) - mu) / sd
    return torch.from_numpy(x.astype(np.float32)), torch.from_numpy(sp.y)


def paired_model(arch_key: str, seed: int, device):
    torch.manual_seed(seed)
    name = "sequence_cnn_gru" if arch_key == "gru" else ARCHS[arch_key]
    return build_model(name, grid_size=16, hidden_dim=64, num_layers=1,
                       dropout=0.0, input_channels=2).to(device)


def frame_shuffle(xb: torch.Tensor) -> torch.Tensor:
    perm = torch.rand(xb.shape[0], xb.shape[1], device=xb.device).argsort(1)
    idx = perm[:, :, None, None, None].expand(-1, -1, *xb.shape[2:])
    return torch.gather(xb, 1, idx)


def train_dual(method: str, arch_key: str, seed: int, data, device):
    """Train once; return {sel: (iid, ood)} for both selection protocols."""
    splits = data
    mu = float(np.asarray(splits["train"].mixed).mean())
    sd = float(np.asarray(splits["train"].mixed).std()) or 1.0
    xtr, ytr, dtr = tensors(splits["train"], mu, sd)
    xva, yva, _ = tensors(splits["val_iid"], mu, sd)
    xit, yit, _ = tensors(splits["iid_test"], mu, sd)
    xot, yot, _ = tensors(splits["ood_test"], mu, sd)
    xsv, ysv = shifted_val(seed, mu, sd)

    model = paired_model(arch_key, seed, device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)

    adv = adv_opt = None
    if method == "dann_dir":
        torch.manual_seed(seed * 31 + 12)
        adv = nn.Sequential(nn.Linear(64, 64), nn.ReLU(),
                            nn.Linear(64, 2)).to(device)
        adv_opt = torch.optim.AdamW(adv.parameters(), lr=LR, weight_decay=WD)

    group = (ytr * 2 + dtr)
    gw = torch.ones(4, device=device) / 4

    sample_w = torch.ones(len(ytr))
    if method == "jtt":
        m1 = paired_model(arch_key, seed + 10007, device)
        o1 = torch.optim.AdamW(m1.parameters(), lr=LR, weight_decay=WD)
        for _ in range(5):
            perm = torch.randperm(len(xtr))
            for i in range(0, len(xtr), BS):
                idx = perm[i:i + BS]
                o1.zero_grad(set_to_none=True)
                F.cross_entropy(m1(xtr[idx].to(device)).logits,
                                ytr[idx].to(device)).backward()
                o1.step()
        m1.eval()
        with torch.no_grad():
            errs = []
            for i in range(0, len(xtr), 512):
                p = m1(xtr[i:i + 512].to(device)).logits.argmax(1).cpu()
                errs.append(p != ytr[i:i + 512])
            sample_w[torch.cat(errs)] = 5.0
        del m1

    # IRM second environment (corr 0.85), standardized jointly.
    xe = ye = None
    if method == "irmv1":
        env1 = make_data(seed + 7919, correlation=0.85, n_train=4096)
        allx = np.concatenate([np.asarray(splits["train"].mixed)[:4096],
                               np.asarray(env1["train"].mixed)])
        mu2, sd2 = float(allx.mean()), float(allx.std()) or 1.0
        xe, ye = [], []
        for sp in (splits["train"], env1["train"]):
            x = (np.asarray(sp.mixed)[:4096] - mu2) / sd2
            xe.append(torch.from_numpy(x.astype(np.float32)))
            ye.append(torch.from_numpy(sp.y[:4096]))
        xva, yva, _ = tensors(splits["val_iid"], mu2, sd2)
        xit, yit, _ = tensors(splits["iid_test"], mu2, sd2)
        xot, yot, _ = tensors(splits["ood_test"], mu2, sd2)
        xsv, ysv = shifted_val(seed, mu2, sd2)

    best = {"sel_iid": (-1.0, None), "sel_shift": (-1.0, None)}
    for epoch in range(EPOCHS):
        model.train()
        lam_adv = 2.0 / (1.0 + np.exp(-10 * epoch / EPOCHS)) - 1.0
        if method == "irmv1":
            lam = 1.0 if epoch < 5 else 1000.0
            perms = [torch.randperm(len(x)) for x in xe]
            for b in range(min(len(x) for x in xe) // BS):
                opt.zero_grad(set_to_none=True)
                total = 0.0
                for e in range(2):
                    idx = perms[e][b * BS:(b + 1) * BS]
                    xb, yb = xe[e][idx].to(device), ye[e][idx].to(device)
                    w = torch.ones(1, device=device, requires_grad=True)
                    risk = F.cross_entropy(model(xb).logits * w, yb)
                    g = torch.autograd.grad(risk, w, create_graph=True)[0]
                    total = total + risk + lam * (g ** 2).sum()
                loss = total / 2
                if lam > 1:
                    loss = loss / lam
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
        else:
            perm = torch.randperm(len(xtr))
            for i in range(0, len(xtr), BS):
                idx = perm[i:i + BS]
                xb, yb = xtr[idx].to(device), ytr[idx].to(device)
                if method == "frame_rand":
                    xb = frame_shuffle(xb)
                opt.zero_grad(set_to_none=True)
                out = model(xb)
                if method == "groupdro_joint":
                    gb = group[idx].to(device)
                    per = F.cross_entropy(out.logits, yb, reduction="none")
                    losses = torch.zeros(4, device=device)
                    for g in range(4):
                        m = gb == g
                        if m.any():
                            losses[g] = per[m].mean()
                    with torch.no_grad():
                        gw2 = gw * torch.exp(0.01 * losses)
                        gw.copy_(gw2 / gw2.sum())
                    loss = (gw * losses).sum()
                elif method == "dann_dir":
                    db = dtr[idx].to(device)
                    adv_opt.zero_grad(set_to_none=True)
                    rev = GradReverse.apply(out.representation, lam_adv)
                    loss = F.cross_entropy(out.logits, yb) + \
                        F.cross_entropy(adv(rev), db)
                elif method == "jtt":
                    wb = sample_w[idx].to(device)
                    per = F.cross_entropy(out.logits, yb, reduction="none")
                    loss = (wb * per).sum() / wb.sum()
                else:
                    loss = F.cross_entropy(out.logits, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                if adv_opt is not None:
                    adv_opt.step()
        model.eval()
        for sel, (xv, yv) in [("sel_iid", (xva, yva)),
                              ("sel_shift", (xsv, ysv))]:
            acc = accuracy(model, xv, yv, device)
            if acc > best[sel][0]:
                state = {k: v.detach().cpu().clone()
                         for k, v in model.state_dict().items()}
                best[sel] = (acc, state)

    out = {}
    for sel in best:
        model.load_state_dict(best[sel][1])
        model.eval()
        out[sel] = (accuracy(model, xit, yit, device),
                    accuracy(model, xot, yot, device))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--out", default="results/extended/unified_paired.json")
    a = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outpath = Path(a.out)
    out = json.load(open(outpath, encoding="utf-8")) if outpath.exists() else {}

    methods = ["erm", "groupdro_joint", "irmv1", "dann_dir", "jtt",
               "frame_rand"]
    jobs = [(m, "gru") for m in methods] + [("erm", k) for k in ARCHS]
    for seed in range(a.seeds):
        data = make_data(seed)
        for method, arch in jobs:
            key = f"{method}/{arch}/seed{seed}"
            if key in out:
                continue
            r = train_dual(method, arch, seed, data, device)
            out[key] = {sel: [round(v, 4) for v in r[sel]] for sel in r}
            print(key, out[key], flush=True)
            outpath.parent.mkdir(parents=True, exist_ok=True)
            json.dump(out, open(outpath, "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
