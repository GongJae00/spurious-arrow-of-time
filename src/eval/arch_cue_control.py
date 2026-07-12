"""Per-architecture single-cue controls on the order-encoded variant:
can each backbone learn the nuisance order cue (nuisance-only) and the core
(core-only) in isolation?  Plus a GroupDRO positive control at an easy
correlation (0.70), where minority joint groups are large.

Usage: python -m src.eval.arch_cue_control --seeds 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from src.eval.temporal_evidence_audit import BASE, to_tensor
from src.models.minimal_sequence import build_model
from src.eval.robust_methods_bench import make_data, tensors, accuracy
from src.data.irreversible_source_inference import (
    IrreversibleSourceConfig, generate_split)

ARCHS = {"lstm": "sequence_cnn_lstm", "tcn": "sequence_cnn_tcn",
         "transformer": "sequence_cnn_transformer",
         "pooling": "sequence_cnn_temporal_pool"}


def train_single_cue(arch, field, seed, device):
    cfg = IrreversibleSourceConfig(
        seed=seed, n_train=8192, n_val_iid=2048, n_iid_test=4096,
        n_ood_test=4096, **{**BASE, "nuisance_trail_decay": 0.0})
    sp = {s: generate_split(cfg, s)
          for s in ["train", "val_iid", "iid_test", "ood_test"]}
    mu = float(np.asarray(getattr(sp["train"], field)).mean())
    sd = float(np.asarray(getattr(sp["train"], field)).std()) or 1.0

    def x(n):
        return to_tensor(((np.asarray(getattr(sp[n], field)) - mu) / sd)[:, :, None])

    y = {n: torch.from_numpy(sp[n].y) for n in sp}
    torch.manual_seed(seed * 31 + 7)
    m = build_model(ARCHS[arch], grid_size=16, hidden_dim=64, num_layers=1,
                    dropout=0.0, input_channels=1).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    best, best_state, bad = -1.0, None, 0
    for _ in range(40):
        m.train()
        perm = torch.randperm(8192)
        for i in range(0, 8192, 128):
            idx = perm[i:i + 128]
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(m(x("train")[idx].to(device)).logits,
                            y["train"][idx].to(device)).backward()
            opt.step()
        m.eval()
        acc = accuracy(m, x("val_iid"), y["val_iid"], device)
        if acc > best:
            best, bad = acc, 0
            best_state = {k: v.cpu().clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= 12:
                break
    m.load_state_dict(best_state)
    m.eval()
    return (accuracy(m, x("iid_test"), y["iid_test"], device),
            accuracy(m, x("ood_test"), y["ood_test"], device))


def groupdro_easy(seed, device):
    import src.eval.robust_methods_bench as rmb
    splits = make_data(seed, correlation=0.70)
    mu = float(np.asarray(splits["train"].mixed).mean())
    sd = float(np.asarray(splits["train"].mixed).std()) or 1.0
    xtr, ytr, dtr = tensors(splits["train"], mu, sd)
    xva, yva, _ = tensors(splits["val_iid"], mu, sd)
    xit, yit, _ = tensors(splits["iid_test"], mu, sd)
    xot, yot, _ = tensors(splits["ood_test"], mu, sd)
    group = (ytr * 2 + dtr)
    m = rmb.new_model(seed * 31 + 11, device)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    gw = torch.ones(4, device=device) / 4
    best, best_state, bad = -1.0, None, 0
    for _ in range(40):
        m.train()
        perm = torch.randperm(len(xtr))
        for i in range(0, len(xtr), 128):
            idx = perm[i:i + 128]
            xb, yb, gb = xtr[idx].to(device), ytr[idx].to(device), group[idx].to(device)
            opt.zero_grad(set_to_none=True)
            per = F.cross_entropy(m(xb).logits, yb, reduction="none")
            losses = torch.zeros(4, device=device)
            for g in range(4):
                msk = gb == g
                if msk.any():
                    losses[g] = per[msk].mean()
            with torch.no_grad():
                gw2 = gw * torch.exp(0.01 * losses)
                gw.copy_(gw2 / gw2.sum())
            (gw * losses).sum().backward()
            opt.step()
        m.eval()
        acc = accuracy(m, xva, yva, device)
        if acc > best:
            best, bad = acc, 0
            best_state = {k: v.cpu().clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= 12:
                break
    m.load_state_dict(best_state)
    m.eval()
    return accuracy(m, xit, yit, device), accuracy(m, xot, yot, device)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--out", default="results/extended/arch_cue_control.json")
    a = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = {}
    for arch in ARCHS:
        for field in ["nuisance_only", "core_only"]:
            rows = [train_single_cue(arch, field, s, device)
                    for s in range(a.seeds)]
            i = np.array([r[0] for r in rows])
            o = np.array([r[1] for r in rows])
            out[f"{arch}/{field}"] = {
                "iid": float(i.mean()), "iid_std": float(i.std(ddof=1)),
                "ood": float(o.mean()), "ood_std": float(o.std(ddof=1))}
            print(f"{arch}/{field}: {i.mean():.3f}/{o.mean():.3f}", flush=True)
    rows = [groupdro_easy(s, device) for s in range(a.seeds)]
    i = np.array([r[0] for r in rows]); o = np.array([r[1] for r in rows])
    out["groupdro_corr0.70"] = {"iid": float(i.mean()), "ood": float(o.mean()),
                                "core": int(((i >= .8) & (o >= .8)).sum())}
    print(f"groupdro_corr0.70: {i.mean():.3f}/{o.mean():.3f}", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=2)


if __name__ == "__main__":
    main()
