"""G4 robustness across reference learners.

The no-spurious recoverability gate (G4) is defined relative to a
pre-specified learner and budget. This script checks whether the G4
verdict on the canonical strict variant is stable across three reference
backbones (GRU, TCN, Transformer) at the certification budget.

Usage: python -m src.eval.g4_multilearner --seeds 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from src.eval.temporal_evidence_audit import to_tensor
from src.eval.robust_methods_bench import accuracy
from src.models.minimal_sequence import build_model
from src.eval.hardpair_oe import make

ARCHS = {"gru": "sequence_cnn_gru", "tcn": "sequence_cnn_tcn",
         "transformer": "sequence_cnn_transformer"}


def train_arch(sp, arch, seed, device, epochs=100, patience=30):
    mu = sp["train"][0].mean()
    sd = sp["train"][0].std() or 1.0

    def x(n):
        return to_tensor((sp[n][0] - mu) / sd)

    y = {n: torch.from_numpy(sp[n][1]) for n in sp}
    torch.manual_seed(seed)
    m = build_model(ARCHS[arch], grid_size=16, hidden_dim=64, num_layers=1,
                    dropout=0.0, input_channels=2).to(device)
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
    return (accuracy(m, x("iid_test"), y["iid_test"], device),
            accuracy(m, x("ood_test"), y["ood_test"], device))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--out", default="results/extended/g4_multilearner.json")
    a = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outpath = Path(a.out)
    out = json.load(open(outpath, encoding="utf-8")) if outpath.exists() else {}
    for arch in ARCHS:
        for seed in range(a.seeds):
            key = f"{arch}/seed{seed}"
            if key in out:
                continue
            sp = make(seed, corr_train=0.5)
            iid, ood = train_arch(sp, arch, seed, device)
            out[key] = [round(iid, 4), round(ood, 4)]
            print(key, out[key], flush=True)
            outpath.parent.mkdir(parents=True, exist_ok=True)
            json.dump(out, open(outpath, "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
