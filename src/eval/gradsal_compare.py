"""Gradient-saliency baseline vs the cue-locality audit.

Post-hoc gradient attribution on trained mixed ERM models detects WHICH
channel carries the exploited cue, but produces near-identical output for a
frame-local cue (trail) and an order-encoded cue (simple OE): it cannot
assign cue locality. Reported per variant: share of input-gradient
magnitude on the nuisance channel, and the across-frame profile of that
saliency.

Usage: python -m src.eval.gradsal_compare --seeds 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from src.eval.temporal_evidence_audit import BASE, to_tensor
from src.eval.robust_methods_bench import SIZES
from src.models.minimal_sequence import build_model
from src.data.irreversible_source_inference import (
    IrreversibleSourceConfig, generate_split)


def train_mixed(variant, seed, device):
    decay = 0.78 if variant == "trail" else 0.0
    cfg = IrreversibleSourceConfig(seed=seed, **SIZES,
                                   **{**BASE, "nuisance_trail_decay": decay})
    sp = {s: generate_split(cfg, s)
          for s in ["train", "val_iid", "iid_test"]}
    mu = float(np.asarray(sp["train"].mixed).mean())
    sd = float(np.asarray(sp["train"].mixed).std()) or 1.0

    def x(n):
        return to_tensor((np.asarray(sp[n].mixed) - mu) / sd)

    y = {n: torch.from_numpy(sp[n].y) for n in sp}
    torch.manual_seed(seed)
    m = build_model("sequence_cnn_gru", grid_size=16, hidden_dim=64,
                    num_layers=1, dropout=0.0, input_channels=2).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    best, state, bad = -1, None, 0
    for _ in range(40):
        m.train()
        perm = torch.randperm(len(x("train")))
        for i in range(0, len(perm), 128):
            j = perm[i:i + 128]
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(m(x("train")[j].to(device)).logits,
                            y["train"][j].to(device)).backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            acc = float((m(x("val_iid")[:1024].to(device)).logits.argmax(1)
                         .cpu() == y["val_iid"][:1024]).float().mean())
        if acc > best:
            best, bad = acc, 0
            state = {k: v.cpu().clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= 12:
                break
    m.load_state_dict(state)
    m.train()  # cuDNN RNN backward needs train mode; dropout is 0
    xt = x("iid_test")[:512].to(device).requires_grad_(True)
    logits = m(xt).logits
    logits.gather(1, logits.argmax(1, keepdim=True)).sum().backward()
    g = xt.grad.abs()  # [B, L, 2, 16, 16]
    core_g = g[:, :, 0].mean(dim=(0, 2, 3))
    nuis_g = g[:, :, 1].mean(dim=(0, 2, 3))
    share = float(nuis_g.sum() / (core_g.sum() + nuis_g.sum()))
    prof = (nuis_g / nuis_g.sum()).detach().cpu().numpy().round(4).tolist()
    return share, prof


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--out", default="results/extended/gradsal_compare.json")
    a = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = {}
    for variant in ["trail", "oe"]:
        shares, profs = [], []
        for s in range(a.seeds):
            share, prof = train_mixed(variant, s, device)
            shares.append(share)
            profs.append(prof)
            print(variant, s, round(share, 3), prof, flush=True)
        out[variant] = {"nuis_share_mean": round(float(np.mean(shares)), 4),
                        "nuis_share_std": round(float(np.std(shares)), 4),
                        "frame_profile_mean":
                            np.mean(profs, axis=0).round(4).tolist()}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
