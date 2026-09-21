import copy
import hashlib
import math
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.benchmark import SIZES, make, paper_config
from src.data import GeneratorConfig, Split, generate_split
from src.evaluate import accuracy, evaluate
from src.models import build_model


METHODS = {
    "final_frame_mlp": {"model_type": "final_frame_mlp", "input_key": "mixed", "uses_counterfactual": False, "uses_group_balancing": False, "channel_dropout_prob": 0.0},
    "sequence_erm": {"model_type": "sequence_cnn_gru", "input_key": "mixed", "uses_counterfactual": False, "uses_group_balancing": False, "channel_dropout_prob": 0.0},
    "core_only_oracle": {"model_type": "sequence_cnn_gru", "input_key": "core_only", "uses_counterfactual": False, "uses_group_balancing": False, "channel_dropout_prob": 0.0},
    "nuisance_only_oracle": {"model_type": "sequence_cnn_gru", "input_key": "nuisance_only", "uses_counterfactual": False, "uses_group_balancing": False, "channel_dropout_prob": 0.0},
    "sequence_erm_lstm": {"model_type": "sequence_cnn_lstm", "input_key": "mixed", "uses_counterfactual": False, "uses_group_balancing": False, "channel_dropout_prob": 0.0},
    "sequence_erm_tcn": {"model_type": "sequence_cnn_tcn", "input_key": "mixed", "uses_counterfactual": False, "uses_group_balancing": False, "channel_dropout_prob": 0.0},
    "sequence_erm_transformer": {"model_type": "sequence_cnn_transformer", "input_key": "mixed", "uses_counterfactual": False, "uses_group_balancing": False, "channel_dropout_prob": 0.0},
    "sequence_erm_temporal_pool": {"model_type": "sequence_cnn_temporal_pool", "input_key": "mixed", "uses_counterfactual": False, "uses_group_balancing": False, "channel_dropout_prob": 0.0},
    "counterfactual_invariance": {"model_type": "sequence_cnn_gru", "input_key": "mixed", "uses_counterfactual": True, "uses_group_balancing": False, "channel_dropout_prob": 0.0},
    "group_invariance_light": {"model_type": "sequence_cnn_gru", "input_key": "mixed", "uses_counterfactual": False, "uses_group_balancing": True, "channel_dropout_prob": 0.0},
    "nuisance_channel_dropout": {"model_type": "sequence_cnn_gru", "input_key": "mixed", "uses_counterfactual": False, "uses_group_balancing": False, "channel_dropout_prob": 0.5},
}

ARCHS = {
    "gru": "sequence_cnn_gru",
    "lstm": "sequence_cnn_lstm",
    "tcn": "sequence_cnn_tcn",
    "transformer": "sequence_cnn_transformer",
    "pooling": "sequence_cnn_temporal_pool",
}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def stable_run_seed(seed: int, scenario: str, method: str) -> int:
    raw = f"{seed}:{scenario}:{method}".encode("utf-8")
    return int(hashlib.sha256(raw).hexdigest()[:8], 16) % (2 ** 31 - 1)


def deep_update(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        cur = out[key] if key in out else None
        if type(value) is dict and type(cur) is dict:
            out[key] = deep_update(cur, value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def infer_input_channels(x: np.ndarray) -> int:
    if x.ndim == 4:
        return 1
    return int(x.shape[2])


def normalize_with_train(train_x: np.ndarray, arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    mean = float(train_x.mean())
    std = float(train_x.std())
    if std < 1e-6:
        std = 1.0
    return {name: ((value - mean) / std).astype(np.float32) for name, value in arrays.items()}


def make_loader(x, y, batch_size, shuffle, seed, x_cf=None, group=None) -> DataLoader:
    tensors = [torch.from_numpy(x).float(), torch.from_numpy(y).long()]
    if x_cf is not None:
        tensors.append(torch.from_numpy(x_cf).float())
    if group is not None:
        tensors.append(torch.from_numpy(group).long())
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=shuffle, generator=generator)


def field_array(split: Split, input_key: str) -> np.ndarray:
    a = np.asarray(getattr(split, input_key))
    if a.ndim == 4:
        a = a[:, :, None]
    return a


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


def frame_shuffle(xb: torch.Tensor) -> torch.Tensor:
    perm = torch.rand(xb.shape[0], xb.shape[1], device=xb.device).argsort(1)
    idx = perm.view(xb.shape[0], xb.shape[1], *([1] * (xb.dim() - 2))).expand_as(xb)
    return torch.gather(xb, 1, idx)


def train_one_method(method: str, splits: dict[str, Split], dataset_config: GeneratorConfig, training: dict, model_config: dict, seed: int, run_seed: int, device: torch.device) -> dict:
    spec = METHODS[method]
    input_key = spec["input_key"]
    uses_cf = spec["uses_counterfactual"]
    uses_group = spec["uses_group_balancing"]
    channel_dropout_prob = spec["channel_dropout_prob"]
    raw = {name: field_array(split, input_key) for name, split in splits.items()}
    raw_cf = {name: np.asarray(split.counterfactual) for name, split in splits.items()} if uses_cf else {}
    if uses_cf:
        for name, a in list(raw_cf.items()):
            if a.ndim == 4:
                raw_cf[name] = a[:, :, None]
    normalized = normalize_with_train(raw["train"], raw)
    normalized_cf = normalize_with_train(raw["train"], raw_cf) if uses_cf else {}
    batch_size = training["batch_size"]
    train_loader = make_loader(
        normalized["train"],
        splits["train"].y,
        batch_size=batch_size,
        shuffle=True,
        seed=run_seed,
        x_cf=normalized_cf["train"] if uses_cf else None,
        group=(splits["train"].nuisance_direction > 0).astype(np.int64) if uses_group else None,
    )
    eval_loaders = {
        name: make_loader(x, splits[name].y, batch_size=batch_size, shuffle=False, seed=run_seed, x_cf=normalized_cf[name] if uses_cf else None)
        for name, x in normalized.items()
    }
    model = build_model(
        model_type=spec["model_type"],
        grid_size=dataset_config.grid_size,
        hidden_dim=model_config["hidden_dim"],
        num_layers=model_config["num_layers"],
        dropout=model_config["dropout"],
        input_channels=infer_input_channels(raw["train"]),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training["lr"], weight_decay=training["weight_decay"])
    max_epochs = training["epochs"]
    patience = training["patience"]
    grad_clip = training["grad_clip_norm"]
    lambda_cf_task = training["lambda_cf_task"]
    lambda_pred = training["lambda_pred"]
    best_state = None
    best_val = -math.inf
    best_epoch = 0
    stale_epochs = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        for batch in train_loader:
            x = batch[0].to(device)
            y = batch[1].to(device)
            if channel_dropout_prob > 0 and x.ndim == 5 and x.shape[2] >= 2:
                drop = torch.rand(x.shape[0], device=device) < channel_dropout_prob
                if drop.any():
                    x = x.clone()
                    x[drop, :, 1] = 0.0
            optimizer.zero_grad(set_to_none=True)
            out = model(x)
            if uses_group:
                group = batch[-1].to(device)
                sample_loss = F.cross_entropy(out.logits, y, reduction="none")
                group_losses = [sample_loss[group == gid].mean() for gid in torch.unique(group)]
                task_loss = torch.stack(group_losses).mean()
            else:
                task_loss = F.cross_entropy(out.logits, y)
            cf_task_loss = torch.zeros((), device=device)
            consistency_loss = torch.zeros((), device=device)
            if uses_cf:
                out_cf = model(batch[2].to(device))
                cf_task_loss = F.cross_entropy(out_cf.logits, y)
                consistency_loss = F.kl_div(F.log_softmax(out_cf.logits, dim=1), F.softmax(out.logits.detach(), dim=1), reduction="batchmean")
            loss = task_loss + lambda_cf_task * cf_task_loss + lambda_pred * consistency_loss
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        val_metrics = evaluate(model, eval_loaders["val_iid"], device, has_counterfactual=uses_cf)
        validation_score = min(val_metrics["accuracy"], val_metrics["accuracy_on_x_cf"]) if uses_cf else val_metrics["accuracy"]
        if validation_score > best_val:
            best_val = validation_score
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    split_metrics = {name: evaluate(model, loader, device, has_counterfactual=uses_cf) for name, loader in eval_loaders.items()}
    return {
        "seed": seed,
        "run_seed": run_seed,
        "method": method,
        "input_key": input_key,
        "uses_counterfactual": uses_cf,
        "best_epoch": best_epoch,
        "val_iid_accuracy": split_metrics["val_iid"]["accuracy"],
        "iid_test_accuracy": split_metrics["iid_test"]["accuracy"],
        "ood_test_accuracy": split_metrics["ood_test"]["accuracy"],
        "ood_gap": split_metrics["iid_test"]["accuracy"] - split_metrics["ood_test"]["accuracy"],
        "split_metrics": split_metrics,
        "model": model,
    }


def train_sequence(xtr, ytr, xval, yval, seed, device, shuffle_frames=False, epochs=40, patience=12, grid_size=16, input_channels=None, model_type="sequence_cnn_gru"):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if input_channels is None:
        input_channels = 1 if xtr.ndim == 4 else int(xtr.shape[2])
    model = build_model(model_type, grid_size=grid_size, hidden_dim=64, num_layers=1, dropout=0.0, input_channels=input_channels).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_acc, best_state, bad = -1.0, None, 0
    xval_d, yval_d = xval.to(device), yval.to(device)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(xtr))
        for i in range(0, len(xtr), 128):
            idx = perm[i : i + 128]
            xb = xtr[idx].to(device)
            if shuffle_frames:
                xb = xb[:, torch.randperm(xb.shape[1], device=device)]
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(model(xb).logits, ytr[idx].to(device)).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            xv = xval_d[:, torch.randperm(xval_d.shape[1], device=device)] if shuffle_frames else xval_d
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
def eval_sequence(model, x, y, device, mode: str = "ordered", seed: int = 0) -> float:
    x = x.clone()
    if mode == "shuffled":
        g = torch.Generator().manual_seed(seed)
        for i in range(len(x)):
            x[i] = x[i][torch.randperm(x.shape[1], generator=g)]
    elif mode == "reversed_order":
        x = x.flip(1)
    preds = [model(x[i : i + 512].to(device)).logits.argmax(1) for i in range(0, len(x), 512)]
    return float((torch.cat(preds) == y.to(device)).float().mean().item())


def tensors(split: Split, mu: float, sd: float):
    x = (np.asarray(split.mixed) - mu) / sd
    return (
        torch.from_numpy(x.astype(np.float32)),
        torch.from_numpy(split.y),
        torch.from_numpy((split.nuisance_direction > 0).astype(np.int64)),
    )


def train_robust(method: str, splits: dict[str, Split], seed: int, device: torch.device, epochs=40, patience=12, lr=1e-3, wd=1e-4, bs=128, irm_lambda=1000.0, eta=0.01, balanced_sampler=False):
    mu = float(np.asarray(splits["train"].mixed).mean())
    sd = float(np.asarray(splits["train"].mixed).std()) or 1.0
    xtr, ytr, dtr = tensors(splits["train"], mu, sd)
    xva, yva, _ = tensors(splits["val_iid"], mu, sd)
    xit, yit, _ = tensors(splits["iid_test"], mu, sd)
    xot, yot, _ = tensors(splits["ood_test"], mu, sd)
    torch.manual_seed(seed * 31 + 11)
    model = build_model("sequence_cnn_gru", grid_size=splits["train"].core_only.shape[-1], hidden_dim=64, input_channels=infer_input_channels(np.asarray(splits["train"].mixed))).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    adv = adv_opt = None
    if method == "dann_dir":
        torch.manual_seed(seed * 31 + 12)
        adv = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 2)).to(device)
        adv_opt = torch.optim.AdamW(adv.parameters(), lr=lr, weight_decay=wd)
    group = ytr * 2 + dtr
    gw = torch.ones(4, device=device) / 4
    idx_by_g = [torch.where(group == g)[0] for g in range(4)]
    sample_w = torch.ones(len(ytr))
    if method == "jtt":
        torch.manual_seed(seed * 31 + 13)
        m1 = build_model("sequence_cnn_gru", grid_size=splits["train"].core_only.shape[-1], hidden_dim=64, input_channels=infer_input_channels(np.asarray(splits["train"].mixed))).to(device)
        o1 = torch.optim.AdamW(m1.parameters(), lr=lr, weight_decay=wd)
        for _ in range(5):
            perm = torch.randperm(len(xtr))
            for i in range(0, len(xtr), bs):
                idx = perm[i : i + bs]
                o1.zero_grad(set_to_none=True)
                F.cross_entropy(m1(xtr[idx].to(device)).logits, ytr[idx].to(device)).backward()
                o1.step()
        m1.eval()
        with torch.no_grad():
            errs = [m1(xtr[i : i + 512].to(device)).logits.argmax(1).cpu() != ytr[i : i + 512] for i in range(0, len(xtr), 512)]
        sample_w[torch.cat(errs)] = 5.0
        del m1
    g_cpu = torch.Generator().manual_seed(seed * 5 + 1)
    best_acc, best_state, bad = -1.0, None, 0
    n_batches = len(xtr) // bs
    for epoch in range(epochs):
        model.train()
        lam_adv = 2.0 / (1.0 + np.exp(-10 * epoch / epochs)) - 1.0
        perm = torch.randperm(len(xtr), generator=g_cpu)
        for b in range(n_batches):
            if method.startswith("groupdro") and balanced_sampler:
                per_g = bs // 4
                idx = torch.cat([ig[torch.randint(len(ig), (per_g,), generator=g_cpu)] for ig in idx_by_g])
            else:
                idx = perm[b * bs : (b + 1) * bs]
            xb, yb = xtr[idx].to(device), ytr[idx].to(device)
            if method == "frame_rand":
                xb = frame_shuffle(xb)
            opt.zero_grad(set_to_none=True)
            out = model(xb)
            if method.startswith("groupdro"):
                gb = group[idx].to(device)
                per = F.cross_entropy(out.logits, yb, reduction="none")
                losses = torch.zeros(4, device=device)
                for g in range(4):
                    m = gb == g
                    if m.any():
                        losses[g] = per[m].mean()
                with torch.no_grad():
                    gw2 = gw * torch.exp(eta * losses)
                    gw.copy_(gw2 / gw2.sum())
                loss = (gw * losses).sum()
            elif method == "dann_dir":
                adv_opt.zero_grad(set_to_none=True)
                loss = F.cross_entropy(out.logits, yb) + F.cross_entropy(adv(GradReverse.apply(out.representation, lam_adv)), dtr[idx].to(device))
            elif method == "jtt":
                wb = sample_w[idx].to(device)
                per = F.cross_entropy(out.logits, yb, reduction="none")
                loss = (wb * per).sum() / wb.sum()
            else:
                loss = F.cross_entropy(out.logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if adv_opt is not None:
                adv_opt.step()
        model.eval()
        acc = accuracy(model, xva, yva, device)
        if acc > best_acc:
            best_acc, bad = acc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return accuracy(model, xit, yit, device), accuracy(model, xot, yot, device), model


def shifted_val(seed: int, mu: float, sd: float):
    cfg = paper_config(seed, **SIZES, nuisance_correlation=0.50)
    sp = generate_split(cfg, "val_iid")
    x = (np.asarray(sp.mixed) - mu) / sd
    return torch.from_numpy(x.astype(np.float32)), torch.from_numpy(sp.y)


def train_dual(method: str, arch_key: str, seed: int, splits: dict[str, Split], device: torch.device, epochs=40, lr=1e-3, wd=1e-4, bs=128):
    mu = float(np.asarray(splits["train"].mixed).mean())
    sd = float(np.asarray(splits["train"].mixed).std()) or 1.0
    xtr, ytr, dtr = tensors(splits["train"], mu, sd)
    xva, yva, _ = tensors(splits["val_iid"], mu, sd)
    xit, yit, _ = tensors(splits["iid_test"], mu, sd)
    xot, yot, _ = tensors(splits["ood_test"], mu, sd)
    xsv, ysv = shifted_val(seed, mu, sd)
    torch.manual_seed(seed)
    model = build_model(ARCHS[arch_key], grid_size=16, hidden_dim=64, num_layers=1, dropout=0.0, input_channels=2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    adv = adv_opt = None
    if method == "dann_dir":
        torch.manual_seed(seed * 31 + 12)
        adv = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 2)).to(device)
        adv_opt = torch.optim.AdamW(adv.parameters(), lr=lr, weight_decay=wd)
    group = ytr * 2 + dtr
    gw = torch.ones(4, device=device) / 4
    sample_w = torch.ones(len(ytr))
    if method == "jtt":
        torch.manual_seed(seed + 10007)
        m1 = build_model(ARCHS[arch_key], grid_size=16, hidden_dim=64, input_channels=2).to(device)
        o1 = torch.optim.AdamW(m1.parameters(), lr=lr, weight_decay=wd)
        for _ in range(5):
            perm = torch.randperm(len(xtr))
            for i in range(0, len(xtr), bs):
                idx = perm[i : i + bs]
                o1.zero_grad(set_to_none=True)
                F.cross_entropy(m1(xtr[idx].to(device)).logits, ytr[idx].to(device)).backward()
                o1.step()
        m1.eval()
        with torch.no_grad():
            errs = [m1(xtr[i : i + 512].to(device)).logits.argmax(1).cpu() != ytr[i : i + 512] for i in range(0, len(xtr), 512)]
        sample_w[torch.cat(errs)] = 5.0
        del m1
    xe = ye = None
    if method == "irmv1":
        env1 = make("simple_oe", seed + 7919, extra={"nuisance_correlation": 0.85}, sizes={**SIZES, "n_train": 4096})
        allx = np.concatenate([np.asarray(splits["train"].mixed)[:4096], np.asarray(env1["train"].mixed)])
        mu2, sd2 = float(allx.mean()), float(allx.std()) or 1.0
        xe, ye = [], []
        for sp in (splits["train"], env1["train"]):
            x = (np.asarray(sp.mixed)[:4096] - mu2) / sd2
            xe.append(torch.from_numpy(x.astype(np.float32)))
            ye.append(torch.from_numpy(sp.y[:4096]))
        xva, yva, _ = tensors(splits["val_iid"], mu2, sd2)
        xit, yit, _ = tensors(splits["iid_test"], mu2, sd2)
        xot, yot, _ = tensors(splits["ood_test"], mu2, sd2)
        xsv, ysv = shifted_val(seed, mu2, sd2)
    best = {"sel_iid": (-1.0, None), "sel_shift": (-1.0, None)}
    for epoch in range(epochs):
        model.train()
        lam_adv = 2.0 / (1.0 + np.exp(-10 * epoch / epochs)) - 1.0
        if method == "irmv1":
            lam = 1.0 if epoch < 5 else 1000.0
            perms = [torch.randperm(len(x)) for x in xe]
            for b in range(min(len(x) for x in xe) // bs):
                opt.zero_grad(set_to_none=True)
                total = 0.0
                for e in range(2):
                    idx = perms[e][b * bs : (b + 1) * bs]
                    xb, yb = xe[e][idx].to(device), ye[e][idx].to(device)
                    w = torch.ones(1, device=device, requires_grad=True)
                    risk = F.cross_entropy(model(xb).logits * w, yb)
                    g = torch.autograd.grad(risk, w, create_graph=True)[0]
                    total = total + risk + lam * (g ** 2).sum()
                loss = total / 2
                if lam > 1:
                    loss = loss / lam
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
        else:
            perm = torch.randperm(len(xtr))
            for i in range(0, len(xtr), bs):
                idx = perm[i : i + bs]
                xb, yb = xtr[idx].to(device), ytr[idx].to(device)
                if method == "frame_rand":
                    xb = frame_shuffle(xb)
                opt.zero_grad(set_to_none=True)
                out = model(xb)
                if method == "groupdro_joint":
                    gb = group[idx].to(device)
                    per = F.cross_entropy(out.logits, yb, reduction="none")
                    losses = torch.zeros(4, device=device)
                    for g in range(4):
                        msk = gb == g
                        if msk.any():
                            losses[g] = per[msk].mean()
                    with torch.no_grad():
                        gw2 = gw * torch.exp(0.01 * losses)
                        gw.copy_(gw2 / gw2.sum())
                    loss = (gw * losses).sum()
                elif method == "dann_dir":
                    adv_opt.zero_grad(set_to_none=True)
                    loss = F.cross_entropy(out.logits, yb) + F.cross_entropy(adv(GradReverse.apply(out.representation, lam_adv)), dtr[idx].to(device))
                elif method == "jtt":
                    wb = sample_w[idx].to(device)
                    per = F.cross_entropy(out.logits, yb, reduction="none")
                    loss = (wb * per).sum() / wb.sum()
                else:
                    loss = F.cross_entropy(out.logits, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                if adv_opt is not None:
                    adv_opt.step()
        model.eval()
        for sel, (xv, yv) in [("sel_iid", (xva, yva)), ("sel_shift", (xsv, ysv))]:
            acc = accuracy(model, xv, yv, device)
            if acc > best[sel][0]:
                state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best[sel] = (acc, state)
    out = {}
    for sel in best:
        model.load_state_dict(best[sel][1])
        model.eval()
        out[sel] = (accuracy(model, xit, yit, device), accuracy(model, xot, yot, device))
    return out


def train_irm(splits0: dict[str, Split], splits1: dict[str, Split], seed: int, device: torch.device, epochs=40, patience=12, irm_lambda=1000.0, bs=128, lr=1e-3, wd=1e-4):
    allx = np.concatenate([np.asarray(splits0["train"].mixed), np.asarray(splits1["train"].mixed)])
    mu, sd = float(allx.mean()), float(allx.std()) or 1.0
    xe, ye = [], []
    for env in (splits0, splits1):
        x, y, _ = tensors(env["train"], mu, sd)
        xe.append(x)
        ye.append(y)
    xva, yva, _ = tensors(splits0["val_iid"], mu, sd)
    xit, yit, _ = tensors(splits0["iid_test"], mu, sd)
    xot, yot, _ = tensors(splits0["ood_test"], mu, sd)
    torch.manual_seed(seed * 31 + 14)
    model = build_model("sequence_cnn_gru", grid_size=splits0["train"].core_only.shape[-1], hidden_dim=64, input_channels=infer_input_channels(np.asarray(splits0["train"].mixed))).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    best_acc, best_state, bad = -1.0, None, 0
    for epoch in range(epochs):
        model.train()
        lam = 1.0 if epoch < 5 else irm_lambda
        perms = [torch.randperm(len(x)) for x in xe]
        n_batches = min(len(x) for x in xe) // bs
        for b in range(n_batches):
            opt.zero_grad(set_to_none=True)
            total = 0.0
            for e in range(2):
                idx = perms[e][b * bs : (b + 1) * bs]
                xb, yb = xe[e][idx].to(device), ye[e][idx].to(device)
                w = torch.ones(1, device=device, requires_grad=True)
                risk = F.cross_entropy(model(xb).logits * w, yb)
                grad = torch.autograd.grad(risk, w, create_graph=True)[0]
                total = total + risk + lam * (grad ** 2).sum()
            loss = total / 2
            if lam > 1:
                loss = loss / lam
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        acc = accuracy(model, xva, yva, device)
        if acc > best_acc:
            best_acc, bad = acc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return accuracy(model, xit, yit, device), accuracy(model, xot, yot, device), model
