import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.data import Split
from src.evaluate import as_tensor
from src.train import eval_sequence, train_sequence

# Algorithm 1 lives on Audit.gate1–gate6. Table 5 / Figure 4b sit after run().


def train_mlp_probe(xtr, ytr, xte_list, seed: int, device, epochs: int = 40):
    # Gates 3 and 6.
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
    # Table 5.
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


def per_sample_shuffle(x, gen):
    L = x.shape[1]
    idx = torch.argsort(torch.rand(x.shape[0], L, generator=gen), dim=1)
    view = idx.view(x.shape[0], L, *([1] * (x.dim() - 2)))
    return torch.gather(x, 1, view.expand_as(x))


class SetProbe(nn.Module):
    def __init__(self, frame_dim: int, hidden: int = 64):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(frame_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, 2))

    def forward(self, x):
        h = self.enc(x)
        return self.head(torch.cat([h.mean(dim=1), h.max(dim=1).values], dim=1))


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


class Audit:
    # Algorithm 1. Privileged access: core-only, nuisance-only, mixed, direction.

    def __init__(self, splits: dict[str, Split], seed: int, device, route_a: bool = False, nospur_splits: dict[str, Split] | None = None):
        self.splits = splits
        self.seed = seed
        self.device = device
        self.route_a = route_a
        self.nospur_splits = nospur_splits

    def seq(self, sp, key):
        a = np.asarray(getattr(sp, key))
        if a.ndim == 4:
            a = a[:, :, None]
        return as_tensor(a)

    def gate1(self) -> dict:
        # Gate 1. Core accessible: core-only sequence ERM on IID and OOD.
        tr, va, it, ot = self.splits["train"], self.splits["val_iid"], self.splits["iid_test"], self.splits["ood_test"]
        core = train_sequence(self.seq(tr, "core_only"), torch.from_numpy(tr.y), self.seq(va, "core_only"), torch.from_numpy(va.y), self.seed * 31 + 1, self.device)
        return {
            "iid": eval_sequence(core, self.seq(it, "core_only"), torch.from_numpy(it.y), self.device),
            "ood": eval_sequence(core, self.seq(ot, "core_only"), torch.from_numpy(ot.y), self.device),
        }

    def gate2(self) -> dict:
        # Gate 2. Nuisance predictive: nuisance-only sequence ERM on IID and OOD.
        tr, va, it, ot = self.splits["train"], self.splits["val_iid"], self.splits["iid_test"], self.splits["ood_test"]
        nuis = train_sequence(self.seq(tr, "nuisance_only"), torch.from_numpy(tr.y), self.seq(va, "nuisance_only"), torch.from_numpy(va.y), self.seed * 31 + 2, self.device)
        return {
            "iid": eval_sequence(nuis, self.seq(it, "nuisance_only"), torch.from_numpy(it.y), self.device),
            "ood": eval_sequence(nuis, self.seq(ot, "nuisance_only"), torch.from_numpy(ot.y), self.device),
        }

    def gate3(self) -> float:
        # Gate 3. Endpoint controlled: final-frame direction.
        tr, te = self.splits["train"], self.splits["iid_test"]

        def prep(sp):
            x = np.asarray(sp.mixed)[:, -1].reshape(len(sp.y), -1).astype(np.float32)
            d = (np.asarray(sp.nuisance_direction) > 0).astype(np.int64)
            return torch.from_numpy(x), torch.from_numpy(d)

        xtr, dtr = prep(tr)
        xte, dte = prep(te)
        return train_mlp_probe(xtr, dtr, [(xte, dte)], self.seed, self.device)[0]

    def gate4(self):
        # Gate 4. Core recoverable: no-spurious mixed ERM, certification budget 100/30.
        if self.nospur_splits is None:
            return None
        ntr, nva, nit, not_sp = (self.nospur_splits[k] for k in ["train", "val_iid", "iid_test", "ood_test"])
        rec = train_sequence(self.seq(ntr, "mixed"), torch.from_numpy(ntr.y), self.seq(nva, "mixed"), torch.from_numpy(nva.y), self.seed * 31 + 3, self.device, epochs=100, patience=30)
        return {
            "iid": eval_sequence(rec, self.seq(nit, "mixed"), torch.from_numpy(nit.y), self.device),
            "ood": eval_sequence(rec, self.seq(not_sp, "mixed"), torch.from_numpy(not_sp.y), self.device),
        }

    def gate5(self) -> dict:
        # Gate 5. Reversal attribution: mixed ERM; OOD reverses P(d_s | y).
        tr, va, it, ot = self.splits["train"], self.splits["val_iid"], self.splits["iid_test"], self.splits["ood_test"]
        xtr, xva = as_tensor(tr.mixed), as_tensor(va.mixed)
        ytr, yva = torch.from_numpy(tr.y), torch.from_numpy(va.y)
        xit, xot = as_tensor(it.mixed), as_tensor(ot.mixed)
        yit, yot = torch.from_numpy(it.y), torch.from_numpy(ot.y)
        self.mixed_erm = train_sequence(xtr, ytr, xva, yva, self.seed * 31 + 7, self.device)
        return {
            "erm_iid": eval_sequence(self.mixed_erm, xit, yit, self.device),
            "erm_ood": eval_sequence(self.mixed_erm, xot, yot, self.device),
        }

    def gate6(self, reversal) -> dict:
        # Gate 6. Cue locality. `reversal` is Gate 5.
        tr, va, it, ot = self.splits["train"], self.splits["val_iid"], self.splits["iid_test"], self.splits["ood_test"]
        seed, device = self.seed, self.device
        ytr, yit, yot = map(lambda s: torch.from_numpy(s.y), (tr, it, ot))
        dtr = torch.from_numpy((tr.nuisance_direction > 0).astype(np.int64))
        dit = torch.from_numpy((it.nuisance_direction > 0).astype(np.int64))
        dot = torch.from_numpy((ot.nuisance_direction > 0).astype(np.int64))

        def frame(sp, t, key="mixed"):
            x = np.asarray(getattr(sp, key))[:, t]
            return as_tensor(x.reshape(len(x), -1))

        # Single-frame classifier, each t, on mixed and on core.
        L = tr.mixed.shape[1]
        frames = {"frame": list(range(L)), "label_iid": [], "label_ood": [], "dir_iid": [], "dir_ood": [], "core_label_iid": [], "core_label_ood": []}
        for t in range(L):
            xtr, xit, xot = frame(tr, t), frame(it, t), frame(ot, t)
            li, lo = train_mlp_probe(xtr, ytr, [(xit, yit), (xot, yot)], seed * 100 + t, device)
            di, do = train_mlp_probe(xtr, dtr, [(xit, dit), (xot, dot)], seed * 100 + t + 50, device)
            frames["label_iid"].append(li)
            frames["label_ood"].append(lo)
            frames["dir_iid"].append(di)
            frames["dir_ood"].append(do)
            cxtr, cxit, cxot = frame(tr, t, "core_only"), frame(it, t, "core_only"), frame(ot, t, "core_only")
            ci, co = train_mlp_probe(cxtr, ytr, [(cxit, yit), (cxot, yot)], seed * 100 + t + 90, device)
            frames["core_label_iid"].append(ci)
            frames["core_label_ood"].append(co)

        # Order-invariant summaries: temporal mean, std, first, middle, first-last.
        feats = {
            "temporal_mean": lambda x: x.mean(axis=1),
            "temporal_std": lambda x: x.std(axis=1),
            "first_frame": lambda x: x[:, 0],
            "middle_frame": lambda x: x[:, x.shape[1] // 2],
            "first_last_pair": lambda x: np.concatenate([x[:, 0], x[:, -1]], axis=1),
        }
        summary = {}
        for name, fn in feats.items():
            def make(sp, fn=fn):
                x = fn(np.asarray(sp.mixed))
                return as_tensor(x.reshape(len(x), -1))
            xtr, xit, xot = make(tr), make(it), make(ot)
            li, lo = train_mlp_probe(xtr, ytr, [(xit, yit), (xot, yot)], seed * 7 + 1, device)
            di, do = train_mlp_probe(xtr, dtr, [(xit, dit), (xot, dot)], seed * 7 + 2, device)
            summary[name] = {"label_iid": li, "label_ood": lo, "dir_iid": di, "dir_ood": do}

        # Unordered-set probe on the nuisance channel.
        def prep(sp):
            x = np.asarray(sp.nuisance_only)
            return as_tensor(x.reshape(len(x), x.shape[1], -1))

        xtr, xit = prep(tr), prep(it)
        mean, std = xtr.mean(), xtr.std().clamp_min(1e-6)
        xtr, xit = ((xtr - mean) / std).to(device), ((xit - mean) / std).to(device)
        dtr_d = dtr.to(device)
        dit_d = dit.to(device)
        torch.manual_seed(seed * 17 + 5)
        net = SetProbe(xtr.shape[-1]).to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
        g = torch.Generator(device="cpu")
        g.manual_seed(seed)
        for _ in range(40):
            net.train()
            perm = torch.randperm(len(xtr), device=device)
            for i in range(0, len(xtr), 128):
                idx = perm[i : i + 128]
                xb = xtr[idx]
                xb = xb[:, torch.randperm(xb.shape[1], generator=g)]
                opt.zero_grad(set_to_none=True)
                F.cross_entropy(net(xb), dtr_d[idx]).backward()
                opt.step()
        net.eval()
        with torch.no_grad():
            set_acc = float((net(xit).argmax(1) == dit_d).float().mean().item())

        # Ordered readout: shuffle and reverse of the Gate 5 mixed ERM.
        xtr, xva = as_tensor(tr.mixed), as_tensor(va.mixed)
        yva = torch.from_numpy(va.y)
        xit, xot = as_tensor(it.mixed), as_tensor(ot.mixed)
        erm = self.mixed_erm
        order = {
            "erm_iid": reversal["erm_iid"],
            "erm_ood": reversal["erm_ood"],
            "erm_iid_shuffled": eval_sequence(erm, xit, yit, device, "shuffled", seed),
            "erm_ood_shuffled": eval_sequence(erm, xot, yot, device, "shuffled", seed),
            "erm_iid_reversed_order": eval_sequence(erm, xit, yit, device, "reversed_order"),
            "erm_ood_reversed_order": eval_sequence(erm, xot, yot, device, "reversed_order"),
        }

        sh = train_sequence(xtr, ytr, xva, yva, seed * 31 + 8, device, shuffle_frames=True)
        order["shuftrain_iid"] = eval_sequence(sh, xit, yit, device, "shuffled", seed + 1)
        order["shuftrain_ood"] = eval_sequence(sh, xot, yot, device, "shuffled", seed + 2)
        order["shuftrain_iid_ordered"] = eval_sequence(sh, xit, yit, device)
        order["shuftrain_ood_ordered"] = eval_sequence(sh, xot, yot, device)

        single = max(frames["dir_iid"])

        # Route A is the constructor flag. Route B is shuffle ≤ 0.6.
        if single >= 0.8:
            loc = "frame-local"
        elif set_acc >= 0.8:
            loc = "order-invariant multi-frame"
        elif order["erm_iid"] >= 0.8 and (self.route_a or order["erm_iid_shuffled"] <= 0.6):
            loc = "order-encoded"
        else:
            loc = "inconclusive"

        return {
            "per_frame": frames,
            "summary": summary,
            "set": set_acc,
            "locality": loc,
            "order": order,
        }

    def run(self) -> dict:
        # Phase 1 — admissibility of a shortcut reading (Gates 1–5)
        gate1 = self.gate1()
        gate2 = self.gate2()
        gate3 = self.gate3()
        gate4 = self.gate4()
        gate5 = self.gate5()
        # Phase 2 — locate the cue on the locality spectrum (Gate 6)
        gate6 = self.gate6(gate5)
        return {
            "gate1": gate1,
            "gate2": gate2,
            "gate3": gate3,
            "gate4": gate4,
            "gate5": gate5,
            "gate6": gate6,
        }

    def mixed_channel(self) -> dict:
        # Table 5. Mixed ERM, then shuffle/reverse one channel.
        tr, va, it, ot = (self.splits[k] for k in ["train", "val_iid", "iid_test", "ood_test"])
        xtr, xva = as_tensor(tr.mixed), as_tensor(va.mixed)
        ytr, yva = torch.from_numpy(tr.y), torch.from_numpy(va.y)
        xit, xot = as_tensor(it.mixed), as_tensor(ot.mixed)
        yit, yot = torch.from_numpy(it.y), torch.from_numpy(ot.y)
        device, seed = self.device, self.seed
        model = train_sequence(xtr, ytr, xva, yva, seed * 31 + 7, device)
        res = {}
        for split, x, y in [("iid", xit, yit), ("ood", xot, yot)]:
            res[f"{split}_original"] = eval_channel_intervention(model, x, y, device, 1, "none")
            res[f"{split}_nuis_shuffle"] = eval_channel_intervention(model, x, y, device, 1, "shuffled", seed)
            res[f"{split}_nuis_reverse"] = eval_channel_intervention(model, x, y, device, 1, "reversed_order")
            res[f"{split}_core_reverse"] = eval_channel_intervention(model, x, y, device, 0, "reversed_order")
        return res

    def nuisance_order(self) -> dict:
        # Figure 4b. Nuisance-only ERM under shuffle and reverse.
        tr, va, it, ot = (self.splits[k] for k in ["train", "val_iid", "iid_test", "ood_test"])

        def nu(sp):
            x = np.asarray(sp.nuisance_only)
            if x.ndim == 4:
                x = x[:, :, None]
            return as_tensor(x)

        device, seed = self.device, self.seed
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

    def channel_probes(self) -> dict:
        # Table 6. Per-frame direction on nuisance-only and core-only.
        tr, it = self.splits["train"], self.splits["iid_test"]
        dtr = torch.from_numpy((tr.nuisance_direction > 0).astype(np.int64))
        dit = torch.from_numpy((it.nuisance_direction > 0).astype(np.int64))
        L = tr.mixed.shape[1]
        out = {"dir_nuis_only": [], "dir_core_only": []}
        seed, device = self.seed, self.device
        for t in range(L):
            for key, name in [("nuisance_only", "dir_nuis_only"), ("core_only", "dir_core_only")]:
                xtr = as_tensor(np.asarray(getattr(tr, key))[:, t].reshape(len(tr.y), -1))
                xit = as_tensor(np.asarray(getattr(it, key))[:, t].reshape(len(it.y), -1))
                out[name].append(train_mlp_probe(xtr, dtr, [(xit, dit)], seed * 100 + t, device)[0])
        return out
