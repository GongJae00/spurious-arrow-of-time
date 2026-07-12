"""Search for a real-video-nuisance configuration in which all six gates
pass AND sequence ERM collapses (a positive semi-synthetic shortcut case).

For each candidate configuration this screens the five accuracy gates
(core-only, nuisance-only, final-frame, no-spurious mixture, main mixture)
over a few seeds.  Order gates (shuffle/reversal) are run separately for the
selected configuration.

Usage: python -m src.eval.rv_positive_search --seeds 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.eval.temporal_evidence_audit import to_tensor
from src.models.minimal_sequence import build_model
from src.data.irreversible_source_inference import (
    IrreversibleSourceConfig, generate_split)

RV_BASE = dict(
    grid_size=16, length=8, diffusion_alpha=0.22, diffusion_start_step=0,
    diffusion_steps_between_frames=4, core_noise_std=0.006,
    observation_noise_std=0.04, core_scale=1.0, nuisance_scale=1.2,
    nuisance_sigma=1.15, nuisance_speed=2.0, nuisance_trail_decay=0.78,
    nuisance_correlation=0.97, observation_layout="two_channel",
    benchmark_variant="endpoint_matched", nuisance_motion="real_video",
    real_video_cache="data/real_video/cache_g16_L8_s5.npz",
)
SIZES = dict(n_train=8192, n_val_iid=2048, n_iid_test=2048, n_ood_test=2048)

CONFIGS = {
    "A_std_c05_n15": {"real_video_standardize": True, "core_scale": 0.5,
                      "nuisance_scale": 1.5},
    "B_std_c035_n15": {"real_video_standardize": True, "core_scale": 0.35,
                       "nuisance_scale": 1.5},
    "C_c05_n15": {"core_scale": 0.5, "nuisance_scale": 1.5},
    "D_std_c05_n15_on08": {"real_video_standardize": True, "core_scale": 0.5,
                           "nuisance_scale": 1.5,
                           "observation_noise_std": 0.08},
    "E_roll_std_c07_n15": {"real_video_standardize": True,
                           "real_video_endpoint_roll": True,
                           "core_scale": 0.7, "nuisance_scale": 1.5},
    "F_roll_std_c085_n15": {"real_video_standardize": True,
                            "real_video_endpoint_roll": True,
                            "core_scale": 0.85, "nuisance_scale": 1.5},
    "G_roll_std_c10_n15": {"real_video_standardize": True,
                           "real_video_endpoint_roll": True,
                           "core_scale": 1.0, "nuisance_scale": 1.5},
    "H_roll_std_c06_n15": {"real_video_standardize": True,
                           "real_video_endpoint_roll": True,
                           "core_scale": 0.6, "nuisance_scale": 1.5},
}


def make_splits(seed, over, no_spurious=False):
    extra = {}
    if no_spurious:
        extra = {"train_nuisance_mode": "randomized",
                 "ood_mode": "randomized"}
    cfg = IrreversibleSourceConfig(seed=seed, **SIZES,
                                   **{**RV_BASE, **over, **extra})
    return {s: generate_split(cfg, s)
            for s in ["train", "val_iid", "iid_test", "ood_test"]}


def get_x(sp, field, mu, sd):
    a = np.asarray(getattr(sp, field))
    if a.ndim == 4:
        a = a[:, :, None]
    return to_tensor((a - mu) / sd)


def train_gru(sp, field, seed, device, channels):
    mu = float(np.asarray(getattr(sp["train"], field)).mean())
    sd = float(np.asarray(getattr(sp["train"], field)).std()) or 1.0
    x = {n: get_x(sp[n], field, mu, sd) for n in sp}
    y = {n: torch.from_numpy(sp[n].y) for n in sp}
    torch.manual_seed(seed)
    m = build_model("sequence_cnn_gru", grid_size=16, hidden_dim=64,
                    num_layers=1, dropout=0.0,
                    input_channels=channels).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    best, best_state, bad = -1.0, None, 0
    n = len(x["train"])
    for _ in range(40):
        m.train()
        perm = torch.randperm(n)
        for i in range(0, n, 128):
            idx = perm[i:i + 128]
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(m(x["train"][idx].to(device)).logits,
                            y["train"][idx].to(device)).backward()
            opt.step()
        m.eval()
        accs = []
        with torch.no_grad():
            for i in range(0, len(x["val_iid"]), 512):
                accs.append(m(x["val_iid"][i:i + 512].to(device))
                            .logits.argmax(1).cpu())
        acc = float((torch.cat(accs) == y["val_iid"]).float().mean())
        if acc > best:
            best, bad = acc, 0
            best_state = {k: v.cpu().clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= 12:
                break
    m.load_state_dict(best_state)
    m.eval()

    def ev(name):
        preds = []
        with torch.no_grad():
            for i in range(0, len(x[name]), 512):
                preds.append(m(x[name][i:i + 512].to(device))
                             .logits.argmax(1).cpu())
        return float((torch.cat(preds) == y[name]).float().mean())

    return ev("iid_test"), ev("ood_test")


def final_frame_mlp(sp, seed, device):
    def last(name):
        a = np.asarray(getattr(sp[name], "mixed"))[:, -1]
        return a.reshape(len(a), -1)

    mu, sd = last("train").mean(), last("train").std() or 1.0
    x = {n: torch.from_numpy(((last(n) - mu) / sd).astype(np.float32))
         for n in sp}
    y = {n: torch.from_numpy(sp[n].y) for n in sp}
    torch.manual_seed(seed + 5)
    m = nn.Sequential(nn.Linear(x["train"].shape[1], 128), nn.ReLU(),
                      nn.Linear(128, 2)).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(20):
        perm = torch.randperm(len(x["train"]))
        for i in range(0, len(perm), 256):
            idx = perm[i:i + 256]
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(m(x["train"][idx].to(device)),
                            y["train"][idx].to(device)).backward()
            opt.step()
    m.eval()

    def ev(name):
        with torch.no_grad():
            p = m(x[name].to(device)).argmax(1).cpu()
        return float((p == y[name]).float().mean())

    return ev("iid_test"), ev("ood_test")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--configs", default="")
    p.add_argument("--out", default="results/extended/rv_positive_search.json")
    a = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outpath = Path(a.out)
    out = json.load(open(outpath, encoding="utf-8")) if outpath.exists() else {}
    keys = [k for k in a.configs.split(",") if k] or list(CONFIGS)
    for name in keys:
        over = CONFIGS[name]
        for seed in range(a.seeds):
            base = f"{name}/seed{seed}"
            if f"{base}/erm" in out:
                continue
            sp = make_splits(seed, over)
            r = {}
            r["erm"] = train_gru(sp, "mixed", seed * 31 + 11, device, 2)
            r["core_only"] = train_gru(sp, "core_only", seed * 31 + 12,
                                       device, 1)
            r["nuis_only"] = train_gru(sp, "nuisance_only", seed * 31 + 13,
                                       device, 1)
            r["final_frame"] = final_frame_mlp(sp, seed * 31 + 14, device)
            spn = make_splits(seed, over, no_spurious=True)
            r["no_spurious"] = train_gru(spn, "mixed", seed * 31 + 11,
                                         device, 2)
            for k, v in r.items():
                out[f"{base}/{k}"] = [round(v[0], 4), round(v[1], 4)]
                print(f"{base}/{k}: {v[0]:.3f}/{v[1]:.3f}", flush=True)
            outpath.parent.mkdir(parents=True, exist_ok=True)
            json.dump(out, open(outpath, "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
