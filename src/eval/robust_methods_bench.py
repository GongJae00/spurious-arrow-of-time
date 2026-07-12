"""Representative robustness baselines on the order-encoded variant.

Implements four established method families and evaluates them with the
benchmark protocol (main reversal, ten seeds, standard budget, IID-validation
selection):

- groupdro_joint: GroupDRO over the four joint groups (y, d_s) with online
  exponential group reweighting (privileged: uses generator group labels).
- irmv1: IRMv1 with two training environments that differ in the
  nuisance--label correlation (0.97 and 0.85), squared-gradient penalty.
- dann_dir: domain-adversarial removal of the nuisance direction from the
  representation via a gradient-reversal adversary (privileged: uses d_s).
- jtt: Just Train Twice; stage-1 ERM error set upweighted in stage 2
  (non-privileged).

Usage:
  python -m src.eval.robust_methods_bench --seeds 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.data.irreversible_source_inference import (
    IrreversibleSourceConfig,
    generate_split,
)
from src.models.minimal_sequence import build_model

BASE = dict(
    grid_size=16, length=8, diffusion_alpha=0.22, diffusion_start_step=0,
    diffusion_steps_between_frames=4, core_noise_std=0.006, observation_noise_std=0.04,
    core_scale=1.0, nuisance_scale=1.2, nuisance_sigma=1.15, nuisance_speed=2.0,
    nuisance_trail_decay=0.0, nuisance_correlation=0.97, observation_layout="two_channel",
    benchmark_variant="endpoint_matched",
)
SIZES = dict(n_train=8192, n_val_iid=2048, n_iid_test=4096, n_ood_test=4096)
EPOCHS, PATIENCE, LR, WD, BS = 40, 12, 1e-3, 1e-4, 128


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


def make_data(seed: int, correlation: float = 0.97, n_train: int | None = None):
    cfg = IrreversibleSourceConfig(
        seed=seed, **{**SIZES, **({"n_train": n_train} if n_train else {})},
        **{**BASE, "nuisance_correlation": correlation},
    )
    return {s: generate_split(cfg, s) for s in ["train", "val_iid", "iid_test", "ood_test"]}


def tensors(sp, mu, sd):
    x = (np.asarray(sp.mixed) - mu) / sd
    return (torch.from_numpy(x.astype(np.float32)),
            torch.from_numpy(sp.y),
            torch.from_numpy((sp.nuisance_direction > 0).astype(np.int64)))


@torch.no_grad()
def accuracy(model, x, y, device):
    preds = []
    for i in range(0, len(x), 512):
        preds.append(model(x[i:i + 512].to(device)).logits.argmax(1))
    return float((torch.cat(preds) == y.to(device)).float().mean().item())


def new_model(seed: int, device):
    torch.manual_seed(seed)
    return build_model("sequence_cnn_gru", grid_size=16, hidden_dim=64,
                       num_layers=1, dropout=0.0, input_channels=2).to(device)


def train_generic(method: str, seed: int, device):
    splits = make_data(seed)
    mu = float(np.asarray(splits["train"].mixed).mean())
    sd = float(np.asarray(splits["train"].mixed).std()) or 1.0
    xtr, ytr, dtr = tensors(splits["train"], mu, sd)
    xva, yva, _ = tensors(splits["val_iid"], mu, sd)
    xit, yit, _ = tensors(splits["iid_test"], mu, sd)
    xot, yot, _ = tensors(splits["ood_test"], mu, sd)

    model = new_model(seed * 31 + 11, device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)

    adv = None
    adv_opt = None
    if method == "dann_dir":
        torch.manual_seed(seed * 31 + 12)
        adv = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 2)).to(device)
        adv_opt = torch.optim.AdamW(adv.parameters(), lr=LR, weight_decay=WD)

    group = (ytr * 2 + dtr)  # joint (y, d) in {0,1,2,3}
    gw = torch.ones(4, device=device) / 4  # DRO weights

    sample_w = torch.ones(len(ytr))
    if method == "jtt":
        # stage 1: short ERM, collect error set
        m1 = new_model(seed * 31 + 13, device)
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
            err_mask = torch.cat(errs)
        sample_w[err_mask] = 5.0
        del m1

    best_acc, best_state, bad = -1.0, None, 0
    for epoch in range(EPOCHS):
        model.train()
        lam_adv = 2.0 / (1.0 + np.exp(-10 * epoch / EPOCHS)) - 1.0
        perm = torch.randperm(len(xtr))
        for i in range(0, len(xtr), BS):
            idx = perm[i:i + BS]
            xb, yb = xtr[idx].to(device), ytr[idx].to(device)
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
                adv_loss = F.cross_entropy(adv(rev), db)
                loss = F.cross_entropy(out.logits, yb) + adv_loss
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


def train_irm(seed: int, device):
    env0 = make_data(seed, correlation=0.97, n_train=4096)
    env1 = make_data(seed + 7919, correlation=0.85, n_train=4096)
    allx = np.concatenate([np.asarray(env0["train"].mixed),
                           np.asarray(env1["train"].mixed)])
    mu, sd = float(allx.mean()), float(allx.std()) or 1.0
    xe, ye = [], []
    for env in (env0, env1):
        x, y, _ = tensors(env["train"], mu, sd)
        xe.append(x)
        ye.append(y)
    xva, yva, _ = tensors(env0["val_iid"], mu, sd)
    xit, yit, _ = tensors(env0["iid_test"], mu, sd)
    xot, yot, _ = tensors(env0["ood_test"], mu, sd)

    model = new_model(seed * 31 + 14, device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    best_acc, best_state, bad = -1.0, None, 0
    for epoch in range(EPOCHS):
        model.train()
        lam = 1.0 if epoch < 5 else 1000.0
        perms = [torch.randperm(len(x)) for x in xe]
        n_batches = min(len(x) for x in xe) // BS
        for b in range(n_batches):
            opt.zero_grad(set_to_none=True)
            total = 0.0
            for e in range(2):
                idx = perms[e][b * BS:(b + 1) * BS]
                xb, yb = xe[e][idx].to(device), ye[e][idx].to(device)
                logits = model(xb).logits
                w = torch.ones(1, device=device, requires_grad=True)
                risk = F.cross_entropy(logits * w, yb)
                grad = torch.autograd.grad(risk, w, create_graph=True)[0]
                total = total + risk + lam * (grad ** 2).sum()
            loss = total / 2
            if lam > 1:
                loss = loss / lam
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
    parser.add_argument("--out", default="results/extended/robust_methods_oe.json")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    result = {}
    for method in ["groupdro_joint", "irmv1", "dann_dir", "jtt"]:
        rows = []
        for s in range(args.seeds):
            if method == "irmv1":
                iid, ood = train_irm(s, device)
            else:
                iid, ood = train_generic(method, s, device)
            rows.append({"seed": s, "iid": round(iid, 4), "ood": round(ood, 4)})
            print(f"{method} seed {s}: iid {iid:.3f} ood {ood:.3f}", flush=True)
        iids = np.array([r["iid"] for r in rows])
        oods = np.array([r["ood"] for r in rows])
        coll = int(((iids >= 0.8) & (oods <= 0.2)).sum())
        core = int(((iids >= 0.8) & (oods >= 0.8)).sum())
        result[method] = {
            "iid_mean": float(iids.mean()), "iid_std": float(iids.std(ddof=1)),
            "ood_mean": float(oods.mean()), "ood_std": float(oods.std(ddof=1)),
            "core": core, "collapse": coll, "chance": len(rows) - core - coll,
            "rows": rows,
        }
        print(f"== {method}: iid {iids.mean():.3f} ood {oods.mean():.3f} "
              f"core {core} coll {coll}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
