"""Post-hoc detector baselines vs the cue-locality audit.

Four standard detectors are applied to trained mixed ERM models on the
trail (frame-local) and simple order-encoded variants: input-gradient
saliency, input-times-gradient, per-frame temporal occlusion of the
nuisance channel, and channel permutation importance. Each reports the
share of evidence assigned to the nuisance channel. All four detect the
carrier channel in both variants, but their outputs do not identify
whether the cue is frame-local or order-encoded.

Usage: python -m src.eval.detector_compare --seeds 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from src.eval.gradsal_compare import train_mixed  # returns share/prof
from src.eval.temporal_evidence_audit import BASE, to_tensor
from src.eval.robust_methods_bench import SIZES, accuracy
from src.models.minimal_sequence import build_model
from src.data.irreversible_source_inference import (
    IrreversibleSourceConfig, generate_split)


def train_model(variant, seed, device):
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
    xt = x("iid_test")[:512]
    yt = y["iid_test"][:512]
    return m, xt, yt


def channel_share(vals_core, vals_nuis):
    return float(vals_nuis / (vals_core + vals_nuis))


def detectors(m, xt, yt, device):
    r = {}
    m.train()  # cuDNN RNN backward
    xg = xt.to(device).requires_grad_(True)
    logits = m(xg).logits
    logits.gather(1, logits.argmax(1, keepdim=True)).sum().backward()
    g = xg.grad.abs()
    r["saliency"] = channel_share(float(g[:, :, 0].mean()),
                                  float(g[:, :, 1].mean()))
    ig = (xg.grad * xg).abs()
    r["input_x_grad"] = channel_share(float(ig[:, :, 0].mean()),
                                      float(ig[:, :, 1].mean()))
    m.eval()
    with torch.no_grad():
        base = accuracy(m, xt, yt, device)
        drops = []
        for ch in [0, 1]:
            xo = xt.clone()
            xo[:, :, ch] = 0
            drops.append(max(base - accuracy(m, xo, yt, device), 0.0))
        r["occlusion"] = channel_share(drops[0] + 1e-6, drops[1] + 1e-6)
        drops = []
        gen = torch.Generator().manual_seed(0)
        perm = torch.randperm(len(xt), generator=gen)
        for ch in [0, 1]:
            xp = xt.clone()
            xp[:, :, ch] = xt[perm][:, :, ch]
            drops.append(max(base - accuracy(m, xp, yt, device), 0.0))
        r["perm_importance"] = channel_share(drops[0] + 1e-6, drops[1] + 1e-6)
    return r


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--out", default="results/extended/detector_compare.json")
    a = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = {}
    for variant in ["trail", "oe"]:
        acc = {}
        for s in range(a.seeds):
            m, xt, yt = train_model(variant, s, device)
            r = detectors(m, xt, yt, device)
            print(variant, s, {k: round(v, 3) for k, v in r.items()},
                  flush=True)
            for k, v in r.items():
                acc.setdefault(k, []).append(v)
        out[variant] = {k: [round(float(np.mean(v)), 4),
                            round(float(np.std(v)), 4)]
                        for k, v in acc.items()}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
