"""Strict order-cue audit addressing three referee questions.

1. Is the 0.549 best single-frame direction accuracy in the order-encoded
   variant residual nuisance leakage, or core-channel leakage through the
   label--direction correlation?  -> per-frame direction probes on the
   NUISANCE CHANNEL ONLY (and, as control, on the core channel only).
2. Can any permutation-invariant set representation decode the direction?
   -> DeepSets-style probe (shared frame encoder + mean pooling) trained on
   the unordered multiset of nuisance frames.
3. Does the mixed sequence-ERM collapse flip/vanish when ONLY the nuisance
   channel's frame order is intervened on (core order preserved)?
   -> channel-selective order interventions on trained mixed ERM models.

Usage:
  python -m src.eval.strict_order_audit --seeds 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.eval.temporal_evidence_audit import (
    BASE,
    SPLIT_SIZES,
    agg,
    to_tensor,
    train_mlp_probe,
    train_sequence_model,
)
from src.data.irreversible_source_inference import (
    IrreversibleSourceConfig,
    generate_split,
)


def make_splits(seed: int, trail_decay: float):
    cfg = IrreversibleSourceConfig(
        seed=seed, **SPLIT_SIZES, **{**BASE, "nuisance_trail_decay": trail_decay},
    )
    return {s: generate_split(cfg, s) for s in ["train", "val_iid", "iid_test", "ood_test"]}


def channel_probes(splits, seed: int, device):
    """Per-frame direction probes on nuisance channel only / core channel only."""
    tr, it = splits["train"], splits["iid_test"]
    dtr = torch.from_numpy((tr.nuisance_direction > 0).astype(np.int64))
    dit = torch.from_numpy((it.nuisance_direction > 0).astype(np.int64))
    L = tr.mixed.shape[1]
    out = {"dir_nuis_only": [], "dir_core_only": []}
    for t in range(L):
        for key, name in [("nuisance_only", "dir_nuis_only"), ("core_only", "dir_core_only")]:
            x = np.asarray(getattr(tr, key))[:, t]
            xtr = to_tensor(x.reshape(len(x), -1))
            x = np.asarray(getattr(it, key))[:, t]
            xit = to_tensor(x.reshape(len(x), -1))
            acc = train_mlp_probe(xtr, dtr, [(xit, dit)], seed * 100 + t, device)[0]
            out[name].append(acc)
    return out


class SetProbe(nn.Module):
    """DeepSets-style permutation-invariant direction classifier."""

    def __init__(self, frame_dim: int, hidden: int = 64) -> None:
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(frame_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(),
                                  nn.Linear(hidden, 2))

    def forward(self, x):  # x: [B, L, D]
        h = self.enc(x)
        pooled = torch.cat([h.mean(dim=1), h.max(dim=1).values], dim=1)
        return self.head(pooled)


def set_probe_direction(splits, seed: int, device, epochs: int = 40):
    """Train a permutation-invariant set probe on nuisance frames -> direction."""
    tr, it = splits["train"], splits["iid_test"]

    def prep(sp):
        x = np.asarray(sp.nuisance_only)
        return to_tensor(x.reshape(len(x), x.shape[1], -1))

    xtr, xit = prep(tr), prep(it)
    mean, std = xtr.mean(), xtr.std().clamp_min(1e-6)
    xtr, xit = (xtr - mean) / std, (xit - mean) / std
    dtr = torch.from_numpy((tr.nuisance_direction > 0).astype(np.int64)).to(device)
    dit = torch.from_numpy((it.nuisance_direction > 0).astype(np.int64)).to(device)
    xtr, xit = xtr.to(device), xit.to(device)
    torch.manual_seed(seed * 17 + 5)
    net = SetProbe(xtr.shape[-1]).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(len(xtr), device=device)
        for i in range(0, len(xtr), 128):
            idx = perm[i:i + 128]
            xb = xtr[idx]
            fp = torch.randperm(xb.shape[1], generator=g)  # fresh shuffle each batch
            xb = xb[:, fp]
            opt.zero_grad(set_to_none=True)
            nn.functional.cross_entropy(net(xb), dtr[idx]).backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        acc = float((net(xit).argmax(1) == dit).float().mean().item())
    return acc


@torch.no_grad()
def eval_channel_intervention(model, x, y, device, channel: int, mode: str, seed: int = 0):
    """Intervene on ONE channel's frame order; keep the other channel intact."""
    x = x.clone()
    if mode == "shuffled":
        g = torch.Generator().manual_seed(seed)
        for i in range(len(x)):
            fp = torch.randperm(x.shape[1], generator=g)
            x[i, :, channel] = x[i, fp, channel]
    elif mode == "reversed_order":
        x[:, :, channel] = x.flip(1)[:, :, channel]
    preds = []
    for i in range(0, len(x), 512):
        preds.append(model(x[i:i + 512].to(device)).logits.argmax(1))
    return float((torch.cat(preds) == y.to(device)).float().mean().item())


def mixed_erm_channel_tests(splits, seed: int, device):
    tr, va, it, ot = (splits[k] for k in ["train", "val_iid", "iid_test", "ood_test"])
    xtr, xva = to_tensor(tr.mixed), to_tensor(va.mixed)
    ytr, yva = torch.from_numpy(tr.y), torch.from_numpy(va.y)
    xit, xot = to_tensor(it.mixed), to_tensor(ot.mixed)
    yit, yot = torch.from_numpy(it.y), torch.from_numpy(ot.y)
    model = train_sequence_model(xtr, ytr, xva, yva, seed * 31 + 7, device)
    res = {}
    for split, x, y in [("iid", xit, yit), ("ood", xot, yot)]:
        res[f"{split}_original"] = eval_channel_intervention(model, x, y, device, 1, "none")
        res[f"{split}_nuis_shuffle"] = eval_channel_intervention(model, x, y, device, 1, "shuffled", seed)
        res[f"{split}_nuis_reverse"] = eval_channel_intervention(model, x, y, device, 1, "reversed_order")
        res[f"{split}_core_reverse"] = eval_channel_intervention(model, x, y, device, 0, "reversed_order")
    return res


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--out", default="results/extended/strict_order_audit.json")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    result = {}
    for name, decay in [("order_encoded", 0.0), ("trail", 0.78)]:
        ch_runs, set_runs, erm_runs = [], [], []
        for s in range(args.seeds):
            splits = make_splits(s, decay)
            ch = channel_probes(splits, s, device)
            ch_runs.append(ch)
            set_runs.append(set_probe_direction(splits, s, device))
            if name == "order_encoded":
                erm_runs.append(mixed_erm_channel_tests(splits, s, device))
            print(f"{name} seed {s}: nuis-only dir "
                  f"{[round(v, 3) for v in ch['dir_nuis_only']]} set {set_runs[-1]:.3f}",
                  flush=True)
        L = len(ch_runs[0]["dir_nuis_only"])
        result[name] = {
            "dir_nuis_only": [agg([r["dir_nuis_only"][t] for r in ch_runs]) for t in range(L)],
            "dir_core_only": [agg([r["dir_core_only"][t] for r in ch_runs]) for t in range(L)],
            "set_probe_direction": agg(set_runs),
        }
        if erm_runs:
            result[name]["mixed_erm_channel"] = {k: agg([r[k] for r in erm_runs]) for k in erm_runs[0]}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    for name, d in result.items():
        print(f"=== {name} ===")
        print("  nuis-only dir per-frame:", [round(a["mean"], 3) for a in d["dir_nuis_only"]])
        print("  core-only dir per-frame:", [round(a["mean"], 3) for a in d["dir_core_only"]])
        print("  set probe:", round(d["set_probe_direction"]["mean"], 3),
              "+-", round(d["set_probe_direction"]["std"], 3))
        if "mixed_erm_channel" in d:
            for k, v in d["mixed_erm_channel"].items():
                print(f"  {k}: {v['mean']:.3f} (+-{v['std']:.3f})")


if __name__ == "__main__":
    main()
