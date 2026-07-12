"""Direct measurement of cue accessibility: sample efficiency and
epochs-to-criterion for each single-cue reference task.

Cues: diffusion core (main), multi-frame core, order-encoded nuisance pulse.
For each cue and training-set size, train the reference CNN-GRU and record
(a) IID test accuracy and (b) the first epoch reaching 95% of the cue's
predictive ceiling on validation.

Usage: python -m src.eval.cue_accessibility --seeds 5
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
from src.data.irreversible_source_inference import (
    IrreversibleSourceConfig, generate_split)

HARD = {"diffusion_start_step": 8, "diffusion_steps_between_frames": 2,
        "core_noise_std": 0.045, "core_noise_growth_power": 0.0,
        "observation_noise_std": 0.01}
CUES = {
    "diffusion_core": (dict(), "core_only", 1.0),
    "multiframe_core": (HARD, "core_only", 0.961),
    "oe_nuisance": ({"nuisance_trail_decay": 0.0}, "nuisance_only", 0.97),
}
SIZES = [1024, 8192]


def run(cue, n_train, seed, device):
    over, field, ceiling = CUES[cue]
    cfg = IrreversibleSourceConfig(
        seed=seed, n_train=n_train, n_val_iid=2048, n_iid_test=4096,
        n_ood_test=256, **{**BASE, **over})
    sp = {s: generate_split(cfg, s) for s in ["train", "val_iid", "iid_test"]}

    def x(n):
        a = np.asarray(getattr(sp[n], field))
        mu = float(np.asarray(getattr(sp["train"], field)).mean())
        sd = float(np.asarray(getattr(sp["train"], field)).std()) or 1.0
        return to_tensor(((a - mu) / sd)[:, :, None])

    tgt = "y" if field == "core_only" else None
    def yv(n):
        s = sp[n]
        return torch.from_numpy(s.y if tgt else s.y)  # nuisance-only labels: y (correlated task)

    xtr, xva, xit = x("train"), x("val_iid"), x("iid_test")
    ytr, yva, yit = yv("train"), yv("val_iid"), yv("iid_test")
    torch.manual_seed(seed * 31 + 7)
    m = build_model("sequence_cnn_gru", grid_size=16, hidden_dim=64,
                    num_layers=1, dropout=0.0, input_channels=1).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = 0.95 * ceiling
    hit, best = None, -1.0
    best_state = None
    for ep in range(1, 101):
        m.train()
        perm = torch.randperm(len(xtr))
        for i in range(0, len(xtr), 128):
            idx = perm[i:i + 128]
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(m(xtr[idx].to(device)).logits,
                            ytr[idx].to(device)).backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            acc = float((m(xva.to(device)).logits.argmax(1)
                         == yva.to(device)).float().mean())
        if acc > best:
            best = acc
            best_state = {k: v.cpu().clone() for k, v in m.state_dict().items()}
        if hit is None and acc >= crit:
            hit = ep
        if hit is not None and ep >= hit + 5:
            break
    m.load_state_dict(best_state)
    m.eval()
    with torch.no_grad():
        iid = float((m(xit.to(device)).logits.argmax(1)
                     == yit.to(device)).float().mean())
    return iid, (hit if hit is not None else 101)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--out", default="results/extended/cue_accessibility.json")
    a = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = {}
    for cue in CUES:
        for n in SIZES:
            rows = [run(cue, n, s, device) for s in range(a.seeds)]
            accs = np.array([r[0] for r in rows])
            eps = np.array([r[1] for r in rows])
            out[f"{cue}/n{n}"] = {
                "iid_mean": float(accs.mean()), "iid_std": float(accs.std(ddof=1)),
                "epochs_to_95pct_ceiling": [int(e) for e in eps],
                "epochs_median": float(np.median(eps))}
            print(f"{cue}/n{n}: iid {accs.mean():.3f} epochs {sorted(eps.tolist())}",
                  flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=2)


if __name__ == "__main__":
    main()
