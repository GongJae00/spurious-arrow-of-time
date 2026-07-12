"""Real-graph-topology source localization with a directional nuisance.

Core task: a diffusion source is planted on Zachary's karate-club graph (a real
social network with two known factions). The label is the faction of the source
node, and the model must recover it from the diffusion trace over the graph.

Nuisance: an independent pulse travels over a fixed node ordering in direction
d_s (forward/backward), endpoint-matched so the final frame does not reveal the
direction. During training d_s is aligned with the label with probability 0.97;
under OOD the alignment is reversed. The no-spurious control randomizes d_s.

This extends the benchmark from a synthetic grid to a real graph topology while
keeping the intervention structure fully controlled.

Usage:
  python -m src.eval.graph_source_experiment --seeds 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from src.models.minimal_sequence import build_model

L = 8
ALPHA = 0.6
STEPS_BETWEEN = 8
DIFF_START = 2  # diffusion steps before the first observed frame
CORE_NOISE = 0.01
OBS_NOISE = 0.03
NU_SIGMA = 1.6
NU_SPEED = 3
CORR = 0.97


def graph_setup(name: str = "karate"):
    if name == "karate":
        G = nx.karate_club_graph()
        club = nx.get_node_attributes(G, "club")
        faction = np.array([0 if club[i] == "Mr. Hi" else 1 for i in G.nodes()])
    elif name == "lesmis":
        G = nx.les_miserables_graph()
        G = nx.convert_node_labels_to_integers(G)
        # two communities via the Fiedler vector (spectral bipartition)
        fiedler = nx.fiedler_vector(G, weight=None, seed=0)
        faction = (np.asarray(fiedler) > 0).astype(np.int64)
    else:
        raise ValueError(name)
    n = G.number_of_nodes()
    A = nx.to_numpy_array(G, weight=None)
    deg = A.sum(1, keepdims=True)
    P = A / np.clip(deg, 1, None)  # row-normalized
    order = np.argsort([-G.degree(i) for i in range(n)])  # fixed node ordering
    return P.astype(np.float32), faction, order, n


def make_split(P, faction, order, n_nodes, n, split_seed, mode):
    """mode: 'train'/'iid' (correlated), 'ood' (reversed), or with prefix 'ns_' randomized."""
    rng = np.random.default_rng(split_seed)
    y = np.zeros(n, dtype=np.int64); y[n // 2:] = 1; rng.shuffle(y)
    # core: diffusion from a random source node in faction y
    core = np.zeros((n, L, n_nodes), dtype=np.float32)
    nodes_by_f = [np.where(faction == f)[0] for f in (0, 1)]
    src = np.array([rng.choice(nodes_by_f[label]) for label in y])
    state = np.zeros((n, n_nodes), dtype=np.float32)
    state[np.arange(n), src] = 1.0
    for _ in range(DIFF_START):
        state = (1 - ALPHA) * state + ALPHA * (state @ P.T)
    k = 0
    for step in range(L * STEPS_BETWEEN):
        if step % STEPS_BETWEEN == 0:
            core[:, k] = state; k += 1
        state = (1 - ALPHA) * state + ALPHA * (state @ P.T)
    core += rng.normal(0, CORE_NOISE, core.shape).astype(np.float32)
    # nuisance direction
    randomized = mode.startswith("ns_")
    base_mode = mode.replace("ns_", "")
    if randomized:
        p_align = 0.5
    elif base_mode == "ood":
        p_align = 1 - CORR
    else:
        p_align = CORR
    aligned = rng.random(n) < p_align
    base = np.where(y == 1, 1, -1)
    d = np.where(aligned, base, -base).astype(np.int64)
    # nuisance: pulse over node ordering, endpoint matched
    pos_idx = np.arange(n_nodes, dtype=np.float32)
    inv_order = np.empty(n_nodes, dtype=np.int64); inv_order[order] = np.arange(n_nodes)
    coord = inv_order.astype(np.float32)  # each node's position in the ordering
    final = rng.uniform(0, n_nodes, size=n).astype(np.float32)
    phase = (final - d * NU_SPEED * (L - 1)) % n_nodes
    nus = np.zeros((n, L, n_nodes), dtype=np.float32)
    for t in range(L):
        c = (phase + d * NU_SPEED * t) % n_nodes
        dist = np.abs(coord[None, :] - c[:, None])
        dist = np.minimum(dist, n_nodes - dist)
        nus[:, t] = np.exp(-0.5 * (dist / NU_SIGMA) ** 2)
    obs = np.stack([core, 1.2 * nus], axis=2)[:, :, :, None, :]  # [n, L, 2, 1, N]
    obs = obs + rng.normal(0, OBS_NOISE, obs.shape).astype(np.float32)
    return dict(mixed=obs.astype(np.float32),
                core=core[:, :, None, None, :].astype(np.float32),
                nuis=nus[:, :, None, None, :].astype(np.float32),
                y=y, d=d)


class FinalFrameMLPFlat(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(dim, 64), nn.ReLU(),
                                 nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 2))

    def forward(self, x):
        out = self.net(x[:, -1])
        class O: pass
        o = O(); o.logits = out
        return o


def train_eval(model, tr_x, tr_y, te, device, epochs=40, patience=12, seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    xtr = torch.from_numpy(tr_x).to(device); ytr = torch.from_numpy(tr_y).to(device)
    val_x, val_y = te["val"]
    xval = torch.from_numpy(val_x).to(device); yval = torch.from_numpy(val_y).to(device)
    best, best_state, stale = -1, None, 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(xtr), device=device)
        for i in range(0, len(xtr), 128):
            idx = perm[i:i + 128]
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(xtr[idx]).logits, ytr[idx])
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            acc = float((model(xval).logits.argmax(1) == yval).float().mean())
        if acc > best:
            best, best_state, stale = acc, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            stale += 1
            if stale >= patience: break
    model.load_state_dict(best_state)
    out = {}
    for name, (x, yy) in te.items():
        if name == "val": continue
        with torch.no_grad():
            xx = torch.from_numpy(x).to(device)
            out[name] = float((model(xx).logits.argmax(1) == torch.from_numpy(yy).to(device)).float().mean())
    return out


def run_seed(seed, device, graph="karate"):
    P, faction, order, N = graph_setup(graph)
    res = {}
    for scen, prefix in [("main", ""), ("no_spurious", "ns_")]:
        tr = make_split(P, faction, order, N, 4096, seed * 100 + 1, prefix + "train")
        va = make_split(P, faction, order, N, 1024, seed * 100 + 2, prefix + "train")
        ii = make_split(P, faction, order, N, 2048, seed * 100 + 3, prefix + "train")
        oo = make_split(P, faction, order, N, 2048, seed * 100 + 4, prefix + "ood")
        methods = {"sequence_erm": ("mixed", "seq"), "core_only": ("core", "seq"),
                   "nuisance_only": ("nuis", "seq"), "final_frame": ("mixed", "mlp")}
        if scen == "no_spurious":
            methods = {"sequence_erm": ("mixed", "seq")}
        for m, (key, kind) in methods.items():
            def norm(a, mu, sd): return ((a - mu) / sd).astype(np.float32)
            mu, sd = tr[key].mean(), max(tr[key].std(), 1e-6)
            te = {"val": (norm(va[key], mu, sd), va["y"]),
                  "iid": (norm(ii[key], mu, sd), ii["y"]),
                  "ood": (norm(oo[key], mu, sd), oo["y"])}
            ch = tr[key].shape[2]
            if kind == "seq":
                model = build_model("sequence_cnn_gru", grid_size=N, hidden_dim=64,
                                    input_channels=ch).to(device)
            else:
                model = FinalFrameMLPFlat(ch * 1 * N).to(device)
            r = train_eval(model, norm(tr[key], mu, sd), tr["y"], te, device, seed=seed)
            res[f"{scen}/{m}"] = r
    return res


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--graph", default="karate")
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    if not args.out:
        args.out = f"results/extended/graph_source/{args.graph}_summary.json"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    allres = {}
    for s in range(args.seeds):
        torch.manual_seed(s)
        allres[s] = run_seed(s, device, args.graph)
        print(f"seed {s}:", {k: {kk: round(vv, 3) for kk, vv in v.items()}
                             for k, v in allres[s].items()})
    # aggregate
    agg = {}
    keys = allres[0].keys()
    for k in keys:
        for split in ("iid", "ood"):
            vals = [allres[s][k][split] for s in allres]
            agg[f"{k}/{split}"] = {"mean": float(np.mean(vals)),
                                   "std": float(np.std(vals, ddof=1)),
                                   "values": [round(v, 4) for v in vals]}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(agg, open(args.out, "w"), indent=2)
    print("\n=== aggregate ===")
    for k, v in agg.items():
        print(f"{k:38s} {v['mean']:.3f} ({v['std']:.3f})")


if __name__ == "__main__":
    main()
