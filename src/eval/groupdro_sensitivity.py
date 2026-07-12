"""GroupDRO sensitivity on the order-encoded variant: group-balanced batch
sampling and step-size variations.

Usage:
  python -m src.eval.groupdro_sensitivity --seeds 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from src.eval.robust_methods_bench import (
    BS,
    EPOCHS,
    LR,
    PATIENCE,
    WD,
    accuracy,
    make_data,
    new_model,
    tensors,
)


def train_groupdro(seed: int, device, eta: float, balanced_sampler: bool):
    splits = make_data(seed)
    mu = float(np.asarray(splits["train"].mixed).mean())
    sd = float(np.asarray(splits["train"].mixed).std()) or 1.0
    xtr, ytr, dtr = tensors(splits["train"], mu, sd)
    xva, yva, _ = tensors(splits["val_iid"], mu, sd)
    xit, yit, _ = tensors(splits["iid_test"], mu, sd)
    xot, yot, _ = tensors(splits["ood_test"], mu, sd)
    group = (ytr * 2 + dtr)
    idx_by_g = [torch.where(group == g)[0] for g in range(4)]

    model = new_model(seed * 31 + 11, device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    gw = torch.ones(4, device=device) / 4
    g_cpu = torch.Generator().manual_seed(seed * 5 + 1)

    best_acc, best_state, bad = -1.0, None, 0
    n_batches = len(xtr) // BS
    for _ in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(xtr), generator=g_cpu)
        for b in range(n_batches):
            if balanced_sampler:
                per_g = BS // 4
                idx = torch.cat([
                    ig[torch.randint(len(ig), (per_g,), generator=g_cpu)]
                    for ig in idx_by_g
                ])
            else:
                idx = perm[b * BS:(b + 1) * BS]
            xb, yb, gb = xtr[idx].to(device), ytr[idx].to(device), group[idx].to(device)
            opt.zero_grad(set_to_none=True)
            per = F.cross_entropy(model(xb).logits, yb, reduction="none")
            losses = torch.zeros(4, device=device)
            for g in range(4):
                m = gb == g
                if m.any():
                    losses[g] = per[m].mean()
            with torch.no_grad():
                gw2 = gw * torch.exp(eta * losses)
                gw.copy_(gw2 / gw2.sum())
            loss = (gw * losses).sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        acc = accuracy(model, xva, yva, device)
        if acc > best_acc:
            best_acc, bad = acc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)
    model.eval()
    return accuracy(model, xit, yit, device), accuracy(model, xot, yot, device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--out", default="results/extended/groupdro_sensitivity.json")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    configs = {
        "eta0.01_balanced": dict(eta=0.01, balanced_sampler=True),
        "eta0.1_standard": dict(eta=0.1, balanced_sampler=False),
        "eta0.001_standard": dict(eta=0.001, balanced_sampler=False),
    }
    result = {}
    for name, kw in configs.items():
        rows = []
        for s in range(args.seeds):
            iid, ood = train_groupdro(s, device, **kw)
            rows.append({"seed": s, "iid": round(iid, 4), "ood": round(ood, 4)})
            print(f"{name} seed {s}: iid {iid:.3f} ood {ood:.3f}", flush=True)
        iids = np.array([r["iid"] for r in rows])
        oods = np.array([r["ood"] for r in rows])
        result[name] = {
            "iid_mean": float(iids.mean()), "ood_mean": float(oods.mean()),
            "ood_std": float(oods.std(ddof=1)),
            "core": int(((iids >= 0.8) & (oods >= 0.8)).sum()),
            "collapse": int(((iids >= 0.8) & (oods <= 0.2)).sum()),
            "rows": rows,
        }
        print(f"== {name}: iid {iids.mean():.3f} ood {oods.mean():.3f} "
              f"core {result[name]['core']} coll {result[name]['collapse']}", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
