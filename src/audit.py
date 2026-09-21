import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.data import Split
from src.evaluate import as_tensor, accuracy, aggregate
from src.train import train_sequence, eval_sequence


def train_mlp_probe(xtr, ytr, xte_list, seed: int, device, epochs: int = 40):
    mean, std = xtr.mean(), xtr.std().clamp_min(1e-6)
    xtr = ((xtr - mean) / std).to(device)
    ytr = ytr.to(device)
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(xtr.shape[1], 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 2)).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(len(xtr), device=device)
        for i in range(0, len(xtr), 128):
            idx = perm[i : i + 128]
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(net(xtr[idx]), ytr[idx]).backward()
            opt.step()
    net.eval()
    accs = []
    with torch.no_grad():
        for xte, yte in xte_list:
            xte = ((xte - mean) / std).to(device)
            accs.append(float((net(xte).argmax(1) == yte.to(device)).float().mean().item()))
    return accs


def probe(xtr, ttr, xte, tte, device, epochs=30):
    md = nn.Sequential(nn.Linear(xtr.shape[1], 64), nn.ReLU(), nn.Linear(64, 2)).to(device)
    opt = torch.optim.AdamW(md.parameters(), lr=1e-3)
    for _ in range(epochs):
        pm = torch.randperm(len(xtr))
        for i in range(0, len(xtr), 256):
            j = pm[i : i + 256]
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(md(xtr[j].to(device)), ttr[j].to(device)).backward()
            opt.step()
    md.eval()
    with torch.no_grad():
        return float((md(xte.to(device)).argmax(1).cpu() == tte).float().mean())


def per_frame_probes(splits: dict[str, Split], seed: int, device, field: str = "mixed"):
    L = getattr(splits["train"], field).shape[1]
    out = {"frame": list(range(L)), "label_iid": [], "label_ood": [], "dir_iid": [], "dir_ood": [], "core_label_iid": [], "core_label_ood": []}
    tr, it, ot = splits["train"], splits["iid_test"], splits["ood_test"]
    ytr, yit, yot = map(lambda s: torch.from_numpy(s.y), (tr, it, ot))
    dtr = torch.from_numpy((tr.nuisance_direction > 0).astype(np.int64))
    dit = torch.from_numpy((it.nuisance_direction > 0).astype(np.int64))
    dot = torch.from_numpy((ot.nuisance_direction > 0).astype(np.int64))

    def frame(sp, t, key="mixed"):
        x = np.asarray(getattr(sp, key))[:, t]
        return as_tensor(x.reshape(len(x), -1))

    for t in range(L):
        xtr, xit, xot = frame(tr, t, field), frame(it, t, field), frame(ot, t, field)
        li, lo = train_mlp_probe(xtr, ytr, [(xit, yit), (xot, yot)], seed * 100 + t, device)
        di, do = train_mlp_probe(xtr, dtr, [(xit, dit), (xot, dot)], seed * 100 + t + 50, device)
        out["label_iid"].append(li)
        out["label_ood"].append(lo)
        out["dir_iid"].append(di)
        out["dir_ood"].append(do)
        cxtr, cxit, cxot = frame(tr, t, "core_only"), frame(it, t, "core_only"), frame(ot, t, "core_only")
        ci, co = train_mlp_probe(cxtr, ytr, [(cxit, yit), (cxot, yot)], seed * 100 + t + 90, device)
        out["core_label_iid"].append(ci)
        out["core_label_ood"].append(co)
    return out


def summary_probes(splits: dict[str, Split], seed: int, device):
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
        def make(sp, fn=fn):
            x = fn(np.asarray(sp.mixed))
            return as_tensor(x.reshape(len(x), -1))
        xtr, xit, xot = make(tr), make(it), make(ot)
        li, lo = train_mlp_probe(xtr, ytr, [(xit, yit), (xot, yot)], seed * 7 + 1, device)
        di, do = train_mlp_probe(xtr, dtr, [(xit, dit), (xot, dot)], seed * 7 + 2, device)
        out[name] = {"label_iid": li, "label_ood": lo, "dir_iid": di, "dir_ood": do}
    return out


def order_tests(splits: dict[str, Split], seed: int, device):
    tr, va, it, ot = splits["train"], splits["val_iid"], splits["iid_test"], splits["ood_test"]
    xtr, xva = as_tensor(tr.mixed), as_tensor(va.mixed)
    ytr, yva = torch.from_numpy(tr.y), torch.from_numpy(va.y)
    xit, xot = as_tensor(it.mixed), as_tensor(ot.mixed)
    yit, yot = torch.from_numpy(it.y), torch.from_numpy(ot.y)
    erm = train_sequence(xtr, ytr, xva, yva, seed * 31 + 7, device)
    res = {
        "erm_iid": eval_sequence(erm, xit, yit, device),
        "erm_ood": eval_sequence(erm, xot, yot, device),
        "erm_iid_shuffled": eval_sequence(erm, xit, yit, device, "shuffled", seed),
        "erm_ood_shuffled": eval_sequence(erm, xot, yot, device, "shuffled", seed),
        "erm_iid_reversed_order": eval_sequence(erm, xit, yit, device, "reversed_order"),
        "erm_ood_reversed_order": eval_sequence(erm, xot, yot, device, "reversed_order"),
    }
    sh = train_sequence(xtr, ytr, xva, yva, seed * 31 + 8, device, shuffle_frames=True)
    res.update({
        "shuftrain_iid": eval_sequence(sh, xit, yit, device, "shuffled", seed + 1),
        "shuftrain_ood": eval_sequence(sh, xot, yot, device, "shuffled", seed + 2),
        "shuftrain_iid_ordered": eval_sequence(sh, xit, yit, device),
        "shuftrain_ood_ordered": eval_sequence(sh, xot, yot, device),
    })
    return res


def channel_probes(splits: dict[str, Split], seed: int, device):
    tr, it = splits["train"], splits["iid_test"]
    dtr = torch.from_numpy((tr.nuisance_direction > 0).astype(np.int64))
    dit = torch.from_numpy((it.nuisance_direction > 0).astype(np.int64))
    L = tr.mixed.shape[1]
    out = {"dir_nuis_only": [], "dir_core_only": []}
    for t in range(L):
        for key, name in [("nuisance_only", "dir_nuis_only"), ("core_only", "dir_core_only")]:
            xtr = as_tensor(np.asarray(getattr(tr, key))[:, t].reshape(len(tr.y), -1))
            xit = as_tensor(np.asarray(getattr(it, key))[:, t].reshape(len(it.y), -1))
            out[name].append(train_mlp_probe(xtr, dtr, [(xit, dit)], seed * 100 + t, device)[0])
    return out


class SetProbe(nn.Module):
    def __init__(self, frame_dim: int, hidden: int = 64):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(frame_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, 2))

    def forward(self, x):
        h = self.enc(x)
        return self.head(torch.cat([h.mean(dim=1), h.max(dim=1).values], dim=1))


def set_probe_direction(splits: dict[str, Split], seed: int, device, epochs: int = 40):
    tr, it = splits["train"], splits["iid_test"]

    def prep(sp):
        x = np.asarray(sp.nuisance_only)
        return as_tensor(x.reshape(len(x), x.shape[1], -1))

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
            idx = perm[i : i + 128]
            xb = xtr[idx]
            xb = xb[:, torch.randperm(xb.shape[1], generator=g)]
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(net(xb), dtr[idx]).backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        return float((net(xit).argmax(1) == dit).float().mean().item())


@torch.no_grad()
def eval_channel_intervention(model, x, y, device, channel: int, mode: str, seed: int = 0):
    x = x.clone()
    if mode == "shuffled":
        g = torch.Generator().manual_seed(seed)
        for i in range(len(x)):
            fp = torch.randperm(x.shape[1], generator=g)
            x[i, :, channel] = x[i, fp, channel]
    elif mode == "reversed_order":
        x[:, :, channel] = x.flip(1)[:, :, channel]
    preds = [model(x[i : i + 512].to(device)).logits.argmax(1) for i in range(0, len(x), 512)]
    return float((torch.cat(preds) == y.to(device)).float().mean().item())


def mixed_erm_channel_tests(splits: dict[str, Split], seed: int, device):
    tr, va, it, ot = (splits[k] for k in ["train", "val_iid", "iid_test", "ood_test"])
    xtr, xva = as_tensor(tr.mixed), as_tensor(va.mixed)
    ytr, yva = torch.from_numpy(tr.y), torch.from_numpy(va.y)
    xit, xot = as_tensor(it.mixed), as_tensor(ot.mixed)
    yit, yot = torch.from_numpy(it.y), torch.from_numpy(ot.y)
    model = train_sequence(xtr, ytr, xva, yva, seed * 31 + 7, device)
    res = {}
    for split, x, y in [("iid", xit, yit), ("ood", xot, yot)]:
        res[f"{split}_original"] = eval_channel_intervention(model, x, y, device, 1, "none")
        res[f"{split}_nuis_shuffle"] = eval_channel_intervention(model, x, y, device, 1, "shuffled", seed)
        res[f"{split}_nuis_reverse"] = eval_channel_intervention(model, x, y, device, 1, "reversed_order")
        res[f"{split}_core_reverse"] = eval_channel_intervention(model, x, y, device, 0, "reversed_order")
    return res


def per_sample_shuffle(x, gen, length=None):
    L = x.shape[1] if length is None else length
    idx = torch.argsort(torch.rand(x.shape[0], L, generator=gen), dim=1)
    view = idx.view(x.shape[0], L, *([1] * (x.dim() - 2)))
    return torch.gather(x, 1, view.expand_as(x))


def final_frame_direction_accuracy(splits: dict[str, Split], seed: int, device) -> float:
    tr, te = splits["train"], splits["iid_test"]

    def prep(sp):
        x = np.asarray(sp.mixed)[:, -1].reshape(len(sp.y), -1).astype(np.float32)
        d = (np.asarray(sp.nuisance_direction) > 0).astype(np.int64)
        return torch.from_numpy(x), torch.from_numpy(d)

    xtr, dtr = prep(tr)
    xte, dte = prep(te)
    return train_mlp_probe(xtr, dtr, [(xte, dte)], seed, device)[0]


def nuisance_order(splits: dict[str, Split], seed: int, device):
    tr, va, it, ot = (splits[k] for k in ["train", "val_iid", "iid_test", "ood_test"])

    def nu(sp):
        x = np.asarray(sp.nuisance_only)
        if x.ndim == 4:
            x = x[:, :, None]
        return as_tensor(x)

    model = train_sequence(nu(tr), torch.from_numpy(tr.y), nu(va), torch.from_numpy(va.y), seed * 13 + 3, device)
    yit, yot = torch.from_numpy(it.y), torch.from_numpy(ot.y)
    xit, xot = nu(it), nu(ot)
    return {
        "iid": eval_sequence(model, xit, yit, device),
        "ood": eval_sequence(model, xot, yot, device),
        "iid_shuffled": eval_sequence(model, xit, yit, device, "shuffled", seed),
        "ood_shuffled": eval_sequence(model, xot, yot, device, "shuffled", seed),
        "iid_reversed_order": eval_sequence(model, xit, yit, device, "reversed_order"),
        "ood_reversed_order": eval_sequence(model, xot, yot, device, "reversed_order"),
    }


def g6_locality(single_frame: float, set_acc: float, ordered: float, shuffled: float, route_a: bool) -> str:
    if single_frame >= 0.8:
        return "frame-local"
    if set_acc >= 0.8:
        return "order-invariant multi-frame"
    if ordered >= 0.8 and (route_a or shuffled <= 0.6):
        return "order-encoded"
    return "inconclusive"


def run_audit(splits: dict[str, Split], seed: int, device, route_a: bool = False) -> dict:
    tr, va, it, ot = (splits[k] for k in ["train", "val_iid", "iid_test", "ood_test"])
    def seq(sp, key):
        a = np.asarray(getattr(sp, key))
        if a.ndim == 4:
            a = a[:, :, None]
        return as_tensor(a)
    core = train_sequence(seq(tr, "core_only"), torch.from_numpy(tr.y), seq(va, "core_only"), torch.from_numpy(va.y), seed * 31 + 1, device)
    g1 = eval_sequence(core, seq(it, "core_only"), torch.from_numpy(it.y), device)
    nuis = train_sequence(seq(tr, "nuisance_only"), torch.from_numpy(tr.y), seq(va, "nuisance_only"), torch.from_numpy(va.y), seed * 31 + 2, device)
    g2 = eval_sequence(nuis, seq(it, "nuisance_only"), torch.from_numpy(it.y), device)
    g3 = final_frame_direction_accuracy(splits, seed, device)
    frames = per_frame_probes(splits, seed, device)
    single = max(frames["dir_iid"])
    set_acc = set_probe_direction(splits, seed, device)
    order = order_tests(splits, seed, device)
    locality = g6_locality(single, set_acc, order["erm_iid"], order["erm_iid_shuffled"], route_a)
    return {
        "g1_core": g1,
        "g2_nuisance": g2,
        "g3_endpoint": g3,
        "g5_iid": order["erm_iid"],
        "g5_ood": order["erm_ood"],
        "g6_single_frame": single,
        "g6_set": set_acc,
        "g6_locality": locality,
        "per_frame": frames,
        "order": order,
    }
