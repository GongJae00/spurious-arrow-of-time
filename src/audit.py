import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.data import Split
from src.evaluate import as_tensor
from src.train import eval_sequence, train_sequence

# Algorithm 1 lives on Audit.gate1–gate6. Table 5 / Figure 4b sit after run().


def train_mlp_probe(x_train, y_train, test_pairs, seed: int, device, epochs: int = 40):
    # Gates 3 and 6.
    mean, std = x_train.mean(), x_train.std().clamp_min(1e-6)
    x_train = ((x_train - mean) / std).to(device)
    y_train = y_train.to(device)
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(x_train.shape[1], 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 2)).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(len(x_train), device=device)
        for i in range(0, len(x_train), 128):
            idx = perm[i : i + 128]
            optimizer.zero_grad(set_to_none=True)
            F.cross_entropy(net(x_train[idx]), y_train[idx]).backward()
            optimizer.step()
    net.eval()
    accs = []
    with torch.no_grad():
        for x_test, y_test in test_pairs:
            x_test = ((x_test - mean) / std).to(device)
            accs.append(float((net(x_test).argmax(1) == y_test.to(device)).float().mean().item()))
    return accs


def probe(x_train, target_train, x_test, target_test, device, epochs=30):
    # Table 5.
    net = nn.Sequential(nn.Linear(x_train.shape[1], 64), nn.ReLU(), nn.Linear(64, 2)).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3)
    for _ in range(epochs):
        perm = torch.randperm(len(x_train))
        for i in range(0, len(x_train), 256):
            j = perm[i : i + 256]
            optimizer.zero_grad(set_to_none=True)
            F.cross_entropy(net(x_train[j].to(device)), target_train[j].to(device)).backward()
            optimizer.step()
    net.eval()
    with torch.no_grad():
        return float((net(x_test.to(device)).argmax(1).cpu() == target_test).float().mean())


def per_sample_shuffle(x, generator):
    # Table 5. Each sample is permuted on its own. The frame multiset stays.
    length = x.shape[1]
    idx = torch.argsort(torch.rand(x.shape[0], length, generator=generator), dim=1)
    view = idx.view(x.shape[0], length, *([1] * (x.dim() - 2)))
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
    # Table 5. Shuffle or reverse one channel of the mixed input.
    x = x.clone()
    if mode == "shuffled":
        generator = torch.Generator().manual_seed(seed)
        for i in range(len(x)):
            frame_perm = torch.randperm(x.shape[1], generator=generator)
            x[i, :, channel] = x[i, frame_perm, channel]
    elif mode == "reversed_order":
        x[:, :, channel] = x.flip(1)[:, :, channel]
    preds = [model(x[i : i + 512].to(device)).logits.argmax(1) for i in range(0, len(x), 512)]
    return float((torch.cat(preds) == y.to(device)).float().mean().item())


class Audit:
    # Algorithm 1. Privileged access: core-only, nuisance-only, mixed, direction.

    def __init__(self, splits: dict[str, Split], seed: int, device, construction_certified: bool = False, no_spurious_splits: dict[str, Split] | None = None):
        self.splits = splits
        self.seed = seed
        self.device = device
        self.construction_certified = construction_certified
        self.no_spurious_splits = no_spurious_splits

    def sequence(self, split, key):
        array = np.asarray(getattr(split, key))
        if array.ndim == 4:
            array = array[:, :, None]
        return as_tensor(array)

    def gate1(self) -> dict:
        # Gate 1. Core accessible: core-only sequence ERM on IID and OOD.
        train, val, iid, ood = (self.splits[name] for name in ("train", "val_iid", "iid_test", "ood_test"))
        core = train_sequence(self.sequence(train, "core_only"), torch.from_numpy(train.y), self.sequence(val, "core_only"), torch.from_numpy(val.y), self.seed * 31 + 1, self.device)
        return {
            "iid": eval_sequence(core, self.sequence(iid, "core_only"), torch.from_numpy(iid.y), self.device),
            "ood": eval_sequence(core, self.sequence(ood, "core_only"), torch.from_numpy(ood.y), self.device),
        }

    def gate2(self) -> dict:
        # Gate 2. Nuisance predictive: nuisance-only sequence ERM on IID and OOD.
        train, val, iid, ood = (self.splits[name] for name in ("train", "val_iid", "iid_test", "ood_test"))
        nuisance = train_sequence(self.sequence(train, "nuisance_only"), torch.from_numpy(train.y), self.sequence(val, "nuisance_only"), torch.from_numpy(val.y), self.seed * 31 + 2, self.device)
        return {
            "iid": eval_sequence(nuisance, self.sequence(iid, "nuisance_only"), torch.from_numpy(iid.y), self.device),
            "ood": eval_sequence(nuisance, self.sequence(ood, "nuisance_only"), torch.from_numpy(ood.y), self.device),
        }

    def gate3(self) -> float:
        # Gate 3. Endpoint controlled: final-frame direction.
        train, iid = self.splits["train"], self.splits["iid_test"]

        def final_frame(split):
            frame = np.asarray(split.mixed)[:, -1].reshape(len(split.y), -1).astype(np.float32)
            direction = (np.asarray(split.nuisance_direction) > 0).astype(np.int64)
            return torch.from_numpy(frame), torch.from_numpy(direction)

        x_train, d_train = final_frame(train)
        x_iid, d_iid = final_frame(iid)
        return train_mlp_probe(x_train, d_train, [(x_iid, d_iid)], self.seed, self.device)[0]

    def gate4(self):
        # Gate 4. Core recoverable: no-spurious mixed ERM, certification budget 100/30.
        if self.no_spurious_splits is None:
            return None
        train, val, iid, ood = (self.no_spurious_splits[name] for name in ("train", "val_iid", "iid_test", "ood_test"))
        recovered = train_sequence(self.sequence(train, "mixed"), torch.from_numpy(train.y), self.sequence(val, "mixed"), torch.from_numpy(val.y), self.seed * 31 + 3, self.device, epochs=100, patience=30)
        return {
            "iid": eval_sequence(recovered, self.sequence(iid, "mixed"), torch.from_numpy(iid.y), self.device),
            "ood": eval_sequence(recovered, self.sequence(ood, "mixed"), torch.from_numpy(ood.y), self.device),
        }

    def gate5(self) -> dict:
        # Gate 5. Reversal attribution: mixed ERM; OOD reverses P(d_s | y).
        train, val, iid, ood = (self.splits[name] for name in ("train", "val_iid", "iid_test", "ood_test"))
        x_train, x_val = as_tensor(train.mixed), as_tensor(val.mixed)
        y_train, y_val = torch.from_numpy(train.y), torch.from_numpy(val.y)
        x_iid, x_ood = as_tensor(iid.mixed), as_tensor(ood.mixed)
        y_iid, y_ood = torch.from_numpy(iid.y), torch.from_numpy(ood.y)
        self.mixed_erm = train_sequence(x_train, y_train, x_val, y_val, self.seed * 31 + 7, self.device)
        return {
            "erm_iid": eval_sequence(self.mixed_erm, x_iid, y_iid, self.device),
            "erm_ood": eval_sequence(self.mixed_erm, x_ood, y_ood, self.device),
        }

    def gate6(self, reversal) -> dict:
        # Gate 6. Cue locality. `reversal` is Gate 5.
        train, val, iid, ood = (self.splits[name] for name in ("train", "val_iid", "iid_test", "ood_test"))
        y_train, y_iid, y_ood = torch.from_numpy(train.y), torch.from_numpy(iid.y), torch.from_numpy(ood.y)
        d_train = torch.from_numpy((train.nuisance_direction > 0).astype(np.int64))
        d_iid = torch.from_numpy((iid.nuisance_direction > 0).astype(np.int64))
        d_ood = torch.from_numpy((ood.nuisance_direction > 0).astype(np.int64))

        def frame(split, t, key="mixed"):
            array = np.asarray(getattr(split, key))[:, t]
            return as_tensor(array.reshape(len(array), -1))

        # Single-frame classifier, each t, on mixed and on core.
        length = train.mixed.shape[1]
        frames = {"frame": list(range(length)), "label_iid": [], "label_ood": [], "dir_iid": [], "dir_ood": [], "core_label_iid": [], "core_label_ood": []}
        for t in range(length):
            mixed_train, mixed_iid, mixed_ood = frame(train, t), frame(iid, t), frame(ood, t)
            label_iid, label_ood = train_mlp_probe(mixed_train, y_train, [(mixed_iid, y_iid), (mixed_ood, y_ood)], self.seed * 100 + t, self.device)
            dir_iid, dir_ood = train_mlp_probe(mixed_train, d_train, [(mixed_iid, d_iid), (mixed_ood, d_ood)], self.seed * 100 + t + 50, self.device)
            frames["label_iid"].append(label_iid)
            frames["label_ood"].append(label_ood)
            frames["dir_iid"].append(dir_iid)
            frames["dir_ood"].append(dir_ood)
            core_train, core_iid, core_ood = frame(train, t, "core_only"), frame(iid, t, "core_only"), frame(ood, t, "core_only")
            core_label_iid, core_label_ood = train_mlp_probe(core_train, y_train, [(core_iid, y_iid), (core_ood, y_ood)], self.seed * 100 + t + 90, self.device)
            frames["core_label_iid"].append(core_label_iid)
            frames["core_label_ood"].append(core_label_ood)

        # Order-invariant summaries: temporal mean, std, first, middle, first-last.
        summaries = {
            "temporal_mean": lambda x: x.mean(axis=1),
            "temporal_std": lambda x: x.std(axis=1),
            "first_frame": lambda x: x[:, 0],
            "middle_frame": lambda x: x[:, x.shape[1] // 2],
            "first_last_pair": lambda x: np.concatenate([x[:, 0], x[:, -1]], axis=1),
        }
        summary = {}
        for name, summary_fn in summaries.items():
            def encode(split, summary_fn=summary_fn):
                array = summary_fn(np.asarray(split.mixed))
                return as_tensor(array.reshape(len(array), -1))
            mixed_train, mixed_iid, mixed_ood = encode(train), encode(iid), encode(ood)
            label_iid, label_ood = train_mlp_probe(mixed_train, y_train, [(mixed_iid, y_iid), (mixed_ood, y_ood)], self.seed * 7 + 1, self.device)
            dir_iid, dir_ood = train_mlp_probe(mixed_train, d_train, [(mixed_iid, d_iid), (mixed_ood, d_ood)], self.seed * 7 + 2, self.device)
            summary[name] = {"label_iid": label_iid, "label_ood": label_ood, "dir_iid": dir_iid, "dir_ood": dir_ood}

        # Unordered-set probe on the nuisance channel.
        def nuisance_frames(split):
            array = np.asarray(split.nuisance_only)
            return as_tensor(array.reshape(len(array), array.shape[1], -1))

        nuisance_train, nuisance_iid = nuisance_frames(train), nuisance_frames(iid)
        mean, std = nuisance_train.mean(), nuisance_train.std().clamp_min(1e-6)
        nuisance_train, nuisance_iid = ((nuisance_train - mean) / std).to(self.device), ((nuisance_iid - mean) / std).to(self.device)
        d_train_device, d_iid_device = d_train.to(self.device), d_iid.to(self.device)
        torch.manual_seed(self.seed * 17 + 5)
        net = SetProbe(nuisance_train.shape[-1]).to(self.device)
        optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed)
        for _ in range(40):
            net.train()
            perm = torch.randperm(len(nuisance_train), device=self.device)
            for i in range(0, len(nuisance_train), 128):
                idx = perm[i : i + 128]
                batch = nuisance_train[idx]
                batch = batch[:, torch.randperm(batch.shape[1], generator=generator)]
                optimizer.zero_grad(set_to_none=True)
                F.cross_entropy(net(batch), d_train_device[idx]).backward()
                optimizer.step()
        net.eval()
        with torch.no_grad():
            set_acc = float((net(nuisance_iid).argmax(1) == d_iid_device).float().mean().item())

        # Ordered readout: shuffle and reverse of the Gate 5 mixed ERM.
        mixed_train, mixed_val = as_tensor(train.mixed), as_tensor(val.mixed)
        y_val = torch.from_numpy(val.y)
        mixed_iid, mixed_ood = as_tensor(iid.mixed), as_tensor(ood.mixed)
        erm = self.mixed_erm
        order = {
            "erm_iid": reversal["erm_iid"],
            "erm_ood": reversal["erm_ood"],
            "erm_iid_shuffled": eval_sequence(erm, mixed_iid, y_iid, self.device, "shuffled", self.seed),
            "erm_ood_shuffled": eval_sequence(erm, mixed_ood, y_ood, self.device, "shuffled", self.seed),
            "erm_iid_reversed_order": eval_sequence(erm, mixed_iid, y_iid, self.device, "reversed_order"),
            "erm_ood_reversed_order": eval_sequence(erm, mixed_ood, y_ood, self.device, "reversed_order"),
        }

        shuffled = train_sequence(mixed_train, y_train, mixed_val, y_val, self.seed * 31 + 8, self.device, shuffle_frames=True)
        order["shuftrain_iid"] = eval_sequence(shuffled, mixed_iid, y_iid, self.device, "shuffled", self.seed + 1)
        order["shuftrain_ood"] = eval_sequence(shuffled, mixed_ood, y_ood, self.device, "shuffled", self.seed + 2)
        order["shuftrain_iid_ordered"] = eval_sequence(shuffled, mixed_iid, y_iid, self.device)
        order["shuftrain_ood_ordered"] = eval_sequence(shuffled, mixed_ood, y_ood, self.device)

        single = max(frames["dir_iid"])

        # Route A is construction-certified. Route B is shuffle ≤ 0.6.
        if single >= 0.8:
            locality = "frame-local"
        elif set_acc >= 0.8:
            locality = "order-invariant multi-frame"
        elif order["erm_iid"] >= 0.8 and (self.construction_certified or order["erm_iid_shuffled"] <= 0.6):
            locality = "order-encoded"
        else:
            locality = "inconclusive"

        return {
            "per_frame": frames,
            "summary": summary,
            "set": set_acc,
            "locality": locality,
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
        train, val, iid, ood = (self.splits[name] for name in ("train", "val_iid", "iid_test", "ood_test"))
        x_train, x_val = as_tensor(train.mixed), as_tensor(val.mixed)
        y_train, y_val = torch.from_numpy(train.y), torch.from_numpy(val.y)
        x_iid, x_ood = as_tensor(iid.mixed), as_tensor(ood.mixed)
        y_iid, y_ood = torch.from_numpy(iid.y), torch.from_numpy(ood.y)
        model = train_sequence(x_train, y_train, x_val, y_val, self.seed * 31 + 7, self.device)
        interventions = {}
        for split_name, x, y in [("iid", x_iid, y_iid), ("ood", x_ood, y_ood)]:
            interventions[f"{split_name}_original"] = eval_channel_intervention(model, x, y, self.device, 1, "none")
            interventions[f"{split_name}_nuis_shuffle"] = eval_channel_intervention(model, x, y, self.device, 1, "shuffled", self.seed)
            interventions[f"{split_name}_nuis_reverse"] = eval_channel_intervention(model, x, y, self.device, 1, "reversed_order")
            interventions[f"{split_name}_core_reverse"] = eval_channel_intervention(model, x, y, self.device, 0, "reversed_order")
        return interventions

    def nuisance_order(self) -> dict:
        # Figure 4b. Nuisance-only ERM under shuffle and reverse.
        train, val, iid, ood = (self.splits[name] for name in ("train", "val_iid", "iid_test", "ood_test"))

        def nuisance_of(split):
            array = np.asarray(split.nuisance_only)
            if array.ndim == 4:
                array = array[:, :, None]
            return as_tensor(array)

        model = train_sequence(nuisance_of(train), torch.from_numpy(train.y), nuisance_of(val), torch.from_numpy(val.y), self.seed * 13 + 3, self.device)
        y_iid, y_ood = torch.from_numpy(iid.y), torch.from_numpy(ood.y)
        x_iid, x_ood = nuisance_of(iid), nuisance_of(ood)
        return {
            "iid": eval_sequence(model, x_iid, y_iid, self.device),
            "ood": eval_sequence(model, x_ood, y_ood, self.device),
            "iid_shuffled": eval_sequence(model, x_iid, y_iid, self.device, "shuffled", self.seed),
            "ood_shuffled": eval_sequence(model, x_ood, y_ood, self.device, "shuffled", self.seed),
            "iid_reversed_order": eval_sequence(model, x_iid, y_iid, self.device, "reversed_order"),
            "ood_reversed_order": eval_sequence(model, x_ood, y_ood, self.device, "reversed_order"),
        }

    def channel_probes(self) -> dict:
        # Table 6. Per-frame direction on nuisance-only and core-only.
        train, iid = self.splits["train"], self.splits["iid_test"]
        d_train = torch.from_numpy((train.nuisance_direction > 0).astype(np.int64))
        d_iid = torch.from_numpy((iid.nuisance_direction > 0).astype(np.int64))
        length = train.mixed.shape[1]
        probes = {"dir_nuis_only": [], "dir_core_only": []}
        for t in range(length):
            for key, name in [("nuisance_only", "dir_nuis_only"), ("core_only", "dir_core_only")]:
                x_train = as_tensor(np.asarray(getattr(train, key))[:, t].reshape(len(train.y), -1))
                x_iid = as_tensor(np.asarray(getattr(iid, key))[:, t].reshape(len(iid.y), -1))
                probes[name].append(train_mlp_probe(x_train, d_train, [(x_iid, d_iid)], self.seed * 100 + t, self.device)[0])
        return probes
