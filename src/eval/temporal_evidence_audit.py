"""Temporal-evidence audit for the main endpoint-matched setting.

Answers three referee questions with direct measurements:

1. Is the nuisance direction decodable from any SINGLE frame (t=0..L-1),
   not just the final frame?  (per-frame direction probes)
2. Does the sequence-ERM shortcut require temporal ORDER, or does it
   survive frame shuffling / order reversal at test time and even
   fully-shuffled training?  (order-sensitivity tests)
3. Is the CORE task solvable from a single early frame, i.e. is the
   core-only reference really doing sequence inverse inference?
   (per-frame core label probes)

All runs use the main configuration (endpoint_matched, correlation 0.97,
two-channel layout, 8192 train / 4096 IID test / 4096 OOD test).

Usage:
  python -m src.eval.temporal_evidence_audit --seeds 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.data.irreversible_source_inference import (
    IrreversibleSourceConfig,
    generate_split,
)
from src.models.minimal_sequence import build_model

BASE = dict(
    grid_size=16, length=8, diffusion_alpha=0.22, diffusion_start_step=0,
    diffusion_steps_between_frames=4, core_noise_std=0.006, observation_noise_std=0.04,
    core_scale=1.0, nuisance_scale=1.2, nuisance_sigma=1.15, nuisance_speed=2.0,
    nuisance_trail_decay=0.78, nuisance_correlation=0.97, observation_layout="two_channel",
    benchmark_variant="endpoint_matched",
)
SPLIT_SIZES = dict(n_train=8192, n_val_iid=2048, n_iid_test=4096, n_ood_test=4096)


def make_splits(seed: int):
    cfg = IrreversibleSourceConfig(seed=seed, **SPLIT_SIZES, **BASE)
    return {s: generate_split(cfg, s) for s in ["train", "val_iid", "iid_test", "ood_test"]}


def to_tensor(a: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(a.astype(np.float32)))


def train_mlp_probe(xtr, ytr, xte_list, seed: int, device, epochs: int = 40):
    """Small MLP probe (matches the endpoint audit probe). Returns accuracies."""
    mean, std = xtr.mean(), xtr.std().clamp_min(1e-6)
    xtr = ((xtr - mean) / std).to(device)
    ytr = ytr.to(device)
    torch.manual_seed(seed)
    net = nn.Sequential(
        nn.Linear(xtr.shape[1], 64), nn.ReLU(),
        nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 2),
    ).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(len(xtr), device=device)
        for i in range(0, len(xtr), 128):
            idx = perm[i:i + 128]
            opt.zero_grad(set_to_none=True)
            nn.functional.cross_entropy(net(xtr[idx]), ytr[idx]).backward()
            opt.step()
    net.eval()
    accs = []
    with torch.no_grad():
        for xte, yte in xte_list:
            xte = ((xte - mean) / std).to(device)
            accs.append(float((net(xte).argmax(1) == yte.to(device)).float().mean().item()))
    return accs


def per_frame_probes(splits, seed: int, device):
    """Per-frame label and direction probes on the mixed observation, plus core-only label probes."""
    L = splits["train"].mixed.shape[1]
    out = {"frame": list(range(L)),
           "label_iid": [], "label_ood": [],
           "dir_iid": [], "dir_ood": [],
           "core_label_iid": [], "core_label_ood": []}
    tr, it, ot = splits["train"], splits["iid_test"], splits["ood_test"]
    ytr, yit, yot = map(lambda s: torch.from_numpy(s.y), (tr, it, ot))
    dtr = torch.from_numpy((tr.nuisance_direction > 0).astype(np.int64))
    dit = torch.from_numpy((it.nuisance_direction > 0).astype(np.int64))
    dot = torch.from_numpy((ot.nuisance_direction > 0).astype(np.int64))
    for t in range(L):
        def frame(sp, key="mixed"):
            x = np.asarray(getattr(sp, key))[:, t]
            return to_tensor(x.reshape(len(x), -1))
        xtr, xit, xot = frame(tr), frame(it), frame(ot)
        li, lo = train_mlp_probe(xtr, ytr, [(xit, yit), (xot, yot)], seed * 100 + t, device)
        di, do = train_mlp_probe(xtr, dtr, [(xit, dit), (xot, dot)], seed * 100 + t + 50, device)
        out["label_iid"].append(li); out["label_ood"].append(lo)
        out["dir_iid"].append(di); out["dir_ood"].append(do)
        cxtr, cxit, cxot = frame(tr, "core_only"), frame(it, "core_only"), frame(ot, "core_only")
        ci, co = train_mlp_probe(cxtr, ytr, [(cxit, yit), (cxot, yot)], seed * 100 + t + 90, device)
        out["core_label_iid"].append(ci); out["core_label_ood"].append(co)
    return out


def summary_probes(splits, seed: int, device):
    """Order-invariant summary-statistic probes: temporal mean / std / first+last pair."""
    tr, it, ot = splits["train"], splits["iid_test"], splits["ood_test"]
    ytr, yit, yot = map(lambda s: torch.from_numpy(s.y), (tr, it, ot))
    dtr = torch.from_numpy((tr.nuisance_direction > 0).astype(np.int64))
    dit = torch.from_numpy((it.nuisance_direction > 0).astype(np.int64))
    dot = torch.from_numpy((ot.nuisance_direction > 0).astype(np.int64))
    feats = {
        "temporal_mean": lambda x: x.mean(axis=1),
        "temporal_std": lambda x: x.std(axis=1),
        "first_frame": lambda x: x[:, 0],
        "middle_frame": lambda x: x[:, x.shape[1] // 2],
        "first_last_pair": lambda x: np.concatenate([x[:, 0], x[:, -1]], axis=1),
    }
    out = {}
    for name, fn in feats.items():
        def make(sp):
            x = fn(np.asarray(sp.mixed))
            return to_tensor(x.reshape(len(x), -1))
        xtr, xit, xot = make(tr), make(it), make(ot)
        li, lo = train_mlp_probe(xtr, ytr, [(xit, yit), (xot, yot)], seed * 7 + 1, device)
        di, do = train_mlp_probe(xtr, dtr, [(xit, dit), (xot, dot)], seed * 7 + 2, device)
        out[name] = {"label_iid": li, "label_ood": lo, "dir_iid": di, "dir_ood": do}
    return out


def train_sequence_model(xtr, ytr, xval, yval, seed: int, device,
                         shuffle_frames: bool = False, epochs: int = 40, patience: int = 12):
    """Train SequenceCNNGRU with the main-recipe hyperparameters."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_model("sequence_cnn_gru", grid_size=xtr.shape[-1], hidden_dim=64,
                        num_layers=1, dropout=0.0, input_channels=xtr.shape[2]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_acc, best_state, bad = -1.0, None, 0
    xval_d, yval_d = xval.to(device), yval.to(device)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(xtr))
        for i in range(0, len(xtr), 128):
            idx = perm[i:i + 128]
            xb = xtr[idx].to(device)
            if shuffle_frames:
                fp = torch.randperm(xb.shape[1], device=device)
                xb = xb[:, fp]
            yb = ytr[idx].to(device)
            opt.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(xb).logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            xv = xval_d
            if shuffle_frames:
                fp = torch.randperm(xv.shape[1], device=device)
                xv = xv[:, fp]
            acc = float((model(xv).logits.argmax(1) == yval_d).float().mean().item())
        if acc > best_acc:
            best_acc, bad = acc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model


@torch.no_grad()
def eval_sequence(model, x, y, device, mode: str = "ordered", seed: int = 0):
    x = x.clone()
    if mode == "shuffled":
        g = torch.Generator().manual_seed(seed)
        for i in range(len(x)):  # independent permutation per sample
            x[i] = x[i][torch.randperm(x.shape[1], generator=g)]
    elif mode == "reversed_order":
        x = x.flip(1)
    accs = []
    yd = y.to(device)
    preds = []
    for i in range(0, len(x), 512):
        preds.append(model(x[i:i + 512].to(device)).logits.argmax(1))
    pred = torch.cat(preds)
    return float((pred == yd).float().mean().item())


def order_tests(splits, seed: int, device):
    tr, va, it, ot = splits["train"], splits["val_iid"], splits["iid_test"], splits["ood_test"]
    xtr, xva = to_tensor(tr.mixed), to_tensor(va.mixed)
    ytr, yva = torch.from_numpy(tr.y), torch.from_numpy(va.y)
    xit, xot = to_tensor(it.mixed), to_tensor(ot.mixed)
    yit, yot = torch.from_numpy(it.y), torch.from_numpy(ot.y)

    erm = train_sequence_model(xtr, ytr, xva, yva, seed * 31 + 7, device)
    res = {
        "erm_iid": eval_sequence(erm, xit, yit, device),
        "erm_ood": eval_sequence(erm, xot, yot, device),
        "erm_iid_shuffled": eval_sequence(erm, xit, yit, device, "shuffled", seed),
        "erm_ood_shuffled": eval_sequence(erm, xot, yot, device, "shuffled", seed),
        "erm_iid_reversed_order": eval_sequence(erm, xit, yit, device, "reversed_order"),
        "erm_ood_reversed_order": eval_sequence(erm, xot, yot, device, "reversed_order"),
    }
    sh = train_sequence_model(xtr, ytr, xva, yva, seed * 31 + 8, device, shuffle_frames=True)
    res.update({
        "shuftrain_iid": eval_sequence(sh, xit, yit, device, "shuffled", seed + 1),
        "shuftrain_ood": eval_sequence(sh, xot, yot, device, "shuffled", seed + 2),
        "shuftrain_iid_ordered": eval_sequence(sh, xit, yit, device),
        "shuftrain_ood_ordered": eval_sequence(sh, xot, yot, device),
    })
    return res


def agg(values):
    a = np.asarray(values, dtype=np.float64)
    return {"mean": float(a.mean()), "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
            "values": [round(float(v), 4) for v in a]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--trail-decay", type=float, default=None,
                        help="override nuisance_trail_decay (0.0 = order-encoded variant)")
    parser.add_argument("--out", default="results/extended/temporal_evidence_audit.json")
    args = parser.parse_args()
    if args.trail_decay is not None:
        BASE["nuisance_trail_decay"] = args.trail_decay
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    per_frame_runs, summary_runs, order_runs = [], [], []
    for s in range(args.seeds):
        splits = make_splits(s)
        per_frame_runs.append(per_frame_probes(splits, s, device))
        summary_runs.append(summary_probes(splits, s, device))
        order_runs.append(order_tests(splits, s, device))
        print(f"seed {s}: dir per-frame IID "
              f"{[round(v, 3) for v in per_frame_runs[-1]['dir_iid']]} | "
              f"erm_ood {order_runs[-1]['erm_ood']:.3f} "
              f"shuffled {order_runs[-1]['erm_ood_shuffled']:.3f} "
              f"shuftrain_ood {order_runs[-1]['shuftrain_ood']:.3f}", flush=True)

    L = len(per_frame_runs[0]["frame"])
    result = {"config": {**BASE, **SPLIT_SIZES, "seeds": args.seeds}}
    result["per_frame"] = {
        key: [agg([r[key][t] for r in per_frame_runs]) for t in range(L)]
        for key in ["label_iid", "label_ood", "dir_iid", "dir_ood",
                    "core_label_iid", "core_label_ood"]
    }
    result["summary_probes"] = {
        name: {k: agg([r[name][k] for r in summary_runs])
               for k in ["label_iid", "label_ood", "dir_iid", "dir_ood"]}
        for name in summary_runs[0]
    }
    result["order_tests"] = {k: agg([r[k] for r in order_runs]) for k in order_runs[0]}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("\n=== per-frame direction decodability (IID) ===")
    for t in range(L):
        print(f"  t={t}: {result['per_frame']['dir_iid'][t]['mean']:.3f}")
    print("=== per-frame label from mixed (IID/OOD) ===")
    for t in range(L):
        print(f"  t={t}: {result['per_frame']['label_iid'][t]['mean']:.3f} / "
              f"{result['per_frame']['label_ood'][t]['mean']:.3f}")
    print("=== per-frame core-only label (IID) ===")
    for t in range(L):
        print(f"  t={t}: {result['per_frame']['core_label_iid'][t]['mean']:.3f}")
    print("=== order tests ===")
    for k, v in result["order_tests"].items():
        print(f"  {k}: {v['mean']:.3f} (+-{v['std']:.3f})")
    print("=== summary probes ===")
    for name, d in result["summary_probes"].items():
        print(f"  {name}: label {d['label_iid']['mean']:.3f}/{d['label_ood']['mean']:.3f} "
              f"dir {d['dir_iid']['mean']:.3f}")


if __name__ == "__main__":
    main()
