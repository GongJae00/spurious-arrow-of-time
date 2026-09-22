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

# Reference learner, then Table 7, then Table A6: GroupDRO, DANN, JTT, IRM.

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


def train_sequence(x_train, y_train, x_val, y_val, seed, device, shuffle_frames=False, epochs=40, patience=12, grid_size=16, model_type="sequence_cnn_gru"):
    # Reference learner. CNN+GRU, standard budget 40/12 (Table A20).
    torch.manual_seed(seed)
    np.random.seed(seed)
    channels = 1 if x_train.ndim == 4 else int(x_train.shape[2])
    model = build_model(model_type, grid_size=grid_size, hidden_dim=64, num_layers=1, dropout=0.0, input_channels=channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_acc, best_state, bad = -1.0, None, 0
    x_val_device, y_val_device = x_val.to(device), y_val.to(device)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(x_train))
        for i in range(0, len(x_train), 128):
            idx = perm[i : i + 128]
            batch = x_train[idx].to(device)
            if shuffle_frames:
                batch = batch[:, torch.randperm(batch.shape[1], device=device)]
            optimizer.zero_grad(set_to_none=True)
            F.cross_entropy(model(batch).logits, y_train[idx].to(device)).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            x_eval = x_val_device[:, torch.randperm(x_val_device.shape[1], device=device)] if shuffle_frames else x_val_device
            acc = float((model(x_eval).logits.argmax(1) == y_val_device).float().mean().item())
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
    # ordered, or Gate 6 shuffled / reversed_order.
    x = x.clone()
    if mode == "shuffled":
        generator = torch.Generator().manual_seed(seed)
        for i in range(len(x)):
            x[i] = x[i][torch.randperm(x.shape[1], generator=generator)]
    elif mode == "reversed_order":
        x = x.flip(1)
    preds = [model(x[i : i + 512].to(device)).logits.argmax(1) for i in range(0, len(x), 512)]
    return float((torch.cat(preds) == y.to(device)).float().mean().item())


def train_one_method(method: str, splits: dict[str, Split], dataset_config: GeneratorConfig, training: dict, model_config: dict, seed: int, run_seed: int, device: torch.device) -> dict:
    # Table 7. CNN+GRU ERM and invariance methods.
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

            # nuisance-channel dropout
            if channel_dropout_prob > 0 and x.ndim == 5 and x.shape[2] >= 2:
                drop = torch.rand(x.shape[0], device=device) < channel_dropout_prob
                if drop.any():
                    x = x.clone()
                    x[drop, :, 1] = 0.0

            optimizer.zero_grad(set_to_none=True)
            out = model(x)

            # group balancing
            if uses_group:
                group = batch[-1].to(device)
                sample_loss = F.cross_entropy(out.logits, y, reduction="none")
                group_losses = [sample_loss[group == group_id].mean() for group_id in torch.unique(group)]
                task_loss = torch.stack(group_losses).mean()
            else:
                # ERM
                task_loss = F.cross_entropy(out.logits, y)

            cf_task_loss = torch.zeros((), device=device)
            consistency_loss = torch.zeros((), device=device)
            # counterfactual invariance
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


class GradReverse(torch.autograd.Function):
    # Table A6. DANN reverses the direction-adversary gradient.
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.scale * grad, None


def frame_shuffle(batch: torch.Tensor) -> torch.Tensor:
    # Table A6. One permutation of each sample's frames.
    perm = torch.rand(batch.shape[0], batch.shape[1], device=batch.device).argsort(1)
    idx = perm.view(batch.shape[0], batch.shape[1], *([1] * (batch.dim() - 2))).expand_as(batch)
    return torch.gather(batch, 1, idx)


def tensors(split: Split, mean: float, std: float):
    # Table A6. Train-normalized mixed input, label, and direction.
    x = (np.asarray(split.mixed) - mean) / std
    return (
        torch.from_numpy(x.astype(np.float32)),
        torch.from_numpy(split.y),
        torch.from_numpy((split.nuisance_direction > 0).astype(np.int64)),
    )


def train_robust(method: str, splits: dict[str, Split], seed: int, device: torch.device, epochs=40, patience=12, learning_rate=1e-3, weight_decay=1e-4, batch_size=128, group_step=0.01, balanced_sampler=False):
    # Table A6. GroupDRO / DANN / JTT / frame-rand. GroupDRO uses (y, direction) groups.
    mean = float(np.asarray(splits["train"].mixed).mean())
    std = float(np.asarray(splits["train"].mixed).std()) or 1.0
    x_train, y_train, d_train = tensors(splits["train"], mean, std)
    x_val, y_val, _ = tensors(splits["val_iid"], mean, std)
    x_iid, y_iid, _ = tensors(splits["iid_test"], mean, std)
    x_ood, y_ood, _ = tensors(splits["ood_test"], mean, std)
    torch.manual_seed(seed * 31 + 11)
    model = build_model("sequence_cnn_gru", grid_size=splits["train"].core_only.shape[-1], hidden_dim=64, input_channels=infer_input_channels(np.asarray(splits["train"].mixed))).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    adversary = adversary_optimizer = None
    if method == "dann_dir":
        torch.manual_seed(seed * 31 + 12)
        adversary = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 2)).to(device)
        adversary_optimizer = torch.optim.AdamW(adversary.parameters(), lr=learning_rate, weight_decay=weight_decay)
    group = y_train * 2 + d_train
    group_weight = torch.ones(4, device=device) / 4
    index_by_group = [torch.where(group == group_id)[0] for group_id in range(4)]
    jtt_weights = torch.ones(len(y_train))
    if method == "jtt":
        torch.manual_seed(seed * 31 + 13)
        stage1 = build_model("sequence_cnn_gru", grid_size=splits["train"].core_only.shape[-1], hidden_dim=64, input_channels=infer_input_channels(np.asarray(splits["train"].mixed))).to(device)
        stage1_optimizer = torch.optim.AdamW(stage1.parameters(), lr=learning_rate, weight_decay=weight_decay)
        for _ in range(5):
            perm = torch.randperm(len(x_train))
            for i in range(0, len(x_train), batch_size):
                idx = perm[i : i + batch_size]
                stage1_optimizer.zero_grad(set_to_none=True)
                F.cross_entropy(stage1(x_train[idx].to(device)).logits, y_train[idx].to(device)).backward()
                stage1_optimizer.step()
        stage1.eval()
        with torch.no_grad():
            errs = [stage1(x_train[i : i + 512].to(device)).logits.argmax(1).cpu() != y_train[i : i + 512] for i in range(0, len(x_train), 512)]
        jtt_weights[torch.cat(errs)] = 5.0
        del stage1
    generator = torch.Generator().manual_seed(seed * 5 + 1)
    best_acc, best_state, bad = -1.0, None, 0
    n_batches = len(x_train) // batch_size
    for epoch in range(epochs):
        model.train()
        adversary_scale = 2.0 / (1.0 + np.exp(-10 * epoch / epochs)) - 1.0
        perm = torch.randperm(len(x_train), generator=generator)
        for b in range(n_batches):
            if method.startswith("groupdro") and balanced_sampler:
                per_group = batch_size // 4
                idx = torch.cat([group_index[torch.randint(len(group_index), (per_group,), generator=generator)] for group_index in index_by_group])
            else:
                idx = perm[b * batch_size : (b + 1) * batch_size]
            batch_x, batch_y = x_train[idx].to(device), y_train[idx].to(device)
            if method == "frame_rand":
                batch_x = frame_shuffle(batch_x)
            optimizer.zero_grad(set_to_none=True)
            out = model(batch_x)
            if method.startswith("groupdro"):
                group_batch = group[idx].to(device)
                per_sample = F.cross_entropy(out.logits, batch_y, reduction="none")
                losses = torch.zeros(4, device=device)
                for group_id in range(4):
                    m = group_batch == group_id
                    if m.any():
                        losses[group_id] = per_sample[m].mean()
                with torch.no_grad():
                    updated_weight = group_weight * torch.exp(group_step * losses)
                    group_weight.copy_(updated_weight / updated_weight.sum())
                loss = (group_weight * losses).sum()
            elif method == "dann_dir":
                adversary_optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(out.logits, batch_y) + F.cross_entropy(adversary(GradReverse.apply(out.representation, adversary_scale)), d_train[idx].to(device))
            elif method == "jtt":
                sample_weight = jtt_weights[idx].to(device)
                per_sample = F.cross_entropy(out.logits, batch_y, reduction="none")
                loss = (sample_weight * per_sample).sum() / sample_weight.sum()
            else:
                loss = F.cross_entropy(out.logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if adversary_optimizer is not None:
                adversary_optimizer.step()
        model.eval()
        acc = accuracy(model, x_val, y_val, device)
        if acc > best_acc:
            best_acc, bad = acc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return accuracy(model, x_iid, y_iid, device), accuracy(model, x_ood, y_ood, device), model


def shifted_val(seed: int, mean: float, std: float):
    # Table A6. sel_shift: a second val draw at correlation 0.50.
    config = paper_config(seed, **SIZES, nuisance_correlation=0.50)
    split = generate_split(config, "val_iid")
    x = (np.asarray(split.mixed) - mean) / std
    return torch.from_numpy(x.astype(np.float32)), torch.from_numpy(split.y)


def train_dual(method: str, arch_key: str, seed: int, splits: dict[str, Split], device: torch.device, epochs=40, learning_rate=1e-3, weight_decay=1e-4, batch_size=128):
    # Table A6. One run, two selection rules (sel_iid, sel_shift).
    mean = float(np.asarray(splits["train"].mixed).mean())
    std = float(np.asarray(splits["train"].mixed).std()) or 1.0
    x_train, y_train, d_train = tensors(splits["train"], mean, std)
    x_val, y_val, _ = tensors(splits["val_iid"], mean, std)
    x_iid, y_iid, _ = tensors(splits["iid_test"], mean, std)
    x_ood, y_ood, _ = tensors(splits["ood_test"], mean, std)
    x_shift, y_shift = shifted_val(seed, mean, std)
    torch.manual_seed(seed)
    model = build_model(ARCHS[arch_key], grid_size=16, hidden_dim=64, num_layers=1, dropout=0.0, input_channels=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    adversary = adversary_optimizer = None
    if method == "dann_dir":
        torch.manual_seed(seed * 31 + 12)
        adversary = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 2)).to(device)
        adversary_optimizer = torch.optim.AdamW(adversary.parameters(), lr=learning_rate, weight_decay=weight_decay)
    group = y_train * 2 + d_train
    group_weight = torch.ones(4, device=device) / 4
    jtt_weights = torch.ones(len(y_train))
    if method == "jtt":
        torch.manual_seed(seed + 10007)
        stage1 = build_model(ARCHS[arch_key], grid_size=16, hidden_dim=64, input_channels=2).to(device)
        stage1_optimizer = torch.optim.AdamW(stage1.parameters(), lr=learning_rate, weight_decay=weight_decay)
        for _ in range(5):
            perm = torch.randperm(len(x_train))
            for i in range(0, len(x_train), batch_size):
                idx = perm[i : i + batch_size]
                stage1_optimizer.zero_grad(set_to_none=True)
                F.cross_entropy(stage1(x_train[idx].to(device)).logits, y_train[idx].to(device)).backward()
                stage1_optimizer.step()
        stage1.eval()
        with torch.no_grad():
            errs = [stage1(x_train[i : i + 512].to(device)).logits.argmax(1).cpu() != y_train[i : i + 512] for i in range(0, len(x_train), 512)]
        jtt_weights[torch.cat(errs)] = 5.0
        del stage1
    env_x = env_y = None
    if method == "irmv1":
        env1 = make("simple_oe", seed + 7919, extra={"nuisance_correlation": 0.85}, sizes={**SIZES, "n_train": 4096})
        both_train = np.concatenate([np.asarray(splits["train"].mixed)[:4096], np.asarray(env1["train"].mixed)])
        mean_env, std_env = float(both_train.mean()), float(both_train.std()) or 1.0
        env_x, env_y = [], []
        for split in (splits["train"], env1["train"]):
            x = (np.asarray(split.mixed)[:4096] - mean_env) / std_env
            env_x.append(torch.from_numpy(x.astype(np.float32)))
            env_y.append(torch.from_numpy(split.y[:4096]))
        x_val, y_val, _ = tensors(splits["val_iid"], mean_env, std_env)
        x_iid, y_iid, _ = tensors(splits["iid_test"], mean_env, std_env)
        x_ood, y_ood, _ = tensors(splits["ood_test"], mean_env, std_env)
        x_shift, y_shift = shifted_val(seed, mean_env, std_env)
    best = {"sel_iid": (-1.0, None), "sel_shift": (-1.0, None)}
    for epoch in range(epochs):
        model.train()
        adversary_scale = 2.0 / (1.0 + np.exp(-10 * epoch / epochs)) - 1.0
        if method == "irmv1":
            irm_penalty = 1.0 if epoch < 5 else 1000.0
            perms = [torch.randperm(len(x)) for x in env_x]
            for b in range(min(len(x) for x in env_x) // batch_size):
                optimizer.zero_grad(set_to_none=True)
                total = 0.0
                for e in range(2):
                    idx = perms[e][b * batch_size : (b + 1) * batch_size]
                    batch_x, batch_y = env_x[e][idx].to(device), env_y[e][idx].to(device)
                    w = torch.ones(1, device=device, requires_grad=True)
                    risk = F.cross_entropy(model(batch_x).logits * w, batch_y)
                    grad = torch.autograd.grad(risk, w, create_graph=True)[0]
                    total = total + risk + irm_penalty * (grad ** 2).sum()
                loss = total / 2
                if irm_penalty > 1:
                    loss = loss / irm_penalty
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        else:
            perm = torch.randperm(len(x_train))
            for i in range(0, len(x_train), batch_size):
                idx = perm[i : i + batch_size]
                batch_x, batch_y = x_train[idx].to(device), y_train[idx].to(device)
                if method == "frame_rand":
                    batch_x = frame_shuffle(batch_x)
                optimizer.zero_grad(set_to_none=True)
                out = model(batch_x)
                if method == "groupdro_joint":
                    group_batch = group[idx].to(device)
                    per_sample = F.cross_entropy(out.logits, batch_y, reduction="none")
                    losses = torch.zeros(4, device=device)
                    for group_id in range(4):
                        msk = group_batch == group_id
                        if msk.any():
                            losses[group_id] = per_sample[msk].mean()
                    with torch.no_grad():
                        updated_weight = group_weight * torch.exp(0.01 * losses)
                        group_weight.copy_(updated_weight / updated_weight.sum())
                    loss = (group_weight * losses).sum()
                elif method == "dann_dir":
                    adversary_optimizer.zero_grad(set_to_none=True)
                    loss = F.cross_entropy(out.logits, batch_y) + F.cross_entropy(adversary(GradReverse.apply(out.representation, adversary_scale)), d_train[idx].to(device))
                elif method == "jtt":
                    sample_weight = jtt_weights[idx].to(device)
                    per_sample = F.cross_entropy(out.logits, batch_y, reduction="none")
                    loss = (sample_weight * per_sample).sum() / sample_weight.sum()
                else:
                    loss = F.cross_entropy(out.logits, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if adversary_optimizer is not None:
                    adversary_optimizer.step()
        model.eval()
        for sel, (xv, yv) in [("sel_iid", (x_val, y_val)), ("sel_shift", (x_shift, y_shift))]:
            acc = accuracy(model, xv, yv, device)
            if acc > best[sel][0]:
                state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best[sel] = (acc, state)
    out = {}
    for sel in best:
        model.load_state_dict(best[sel][1])
        model.eval()
        out[sel] = (accuracy(model, x_iid, y_iid, device), accuracy(model, x_ood, y_ood, device))
    return out


def train_irm(splits0: dict[str, Split], splits1: dict[str, Split], seed: int, device: torch.device, epochs=40, patience=12, irm_lambda=1000.0, batch_size=128, learning_rate=1e-3, weight_decay=1e-4):
    # Table A6. IRMv1 on two Simple-OE environments (ρ=0.97 and ρ=0.85).
    both_train = np.concatenate([np.asarray(splits0["train"].mixed), np.asarray(splits1["train"].mixed)])
    mean, std = float(both_train.mean()), float(both_train.std()) or 1.0
    env_x, env_y = [], []
    for env in (splits0, splits1):
        x, y, _ = tensors(env["train"], mean, std)
        env_x.append(x)
        env_y.append(y)
    x_val, y_val, _ = tensors(splits0["val_iid"], mean, std)
    x_iid, y_iid, _ = tensors(splits0["iid_test"], mean, std)
    x_ood, y_ood, _ = tensors(splits0["ood_test"], mean, std)
    torch.manual_seed(seed * 31 + 14)
    model = build_model("sequence_cnn_gru", grid_size=splits0["train"].core_only.shape[-1], hidden_dim=64, input_channels=infer_input_channels(np.asarray(splits0["train"].mixed))).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    best_acc, best_state, bad = -1.0, None, 0
    for epoch in range(epochs):
        model.train()
        irm_penalty = 1.0 if epoch < 5 else irm_lambda
        perms = [torch.randperm(len(x)) for x in env_x]
        n_batches = min(len(x) for x in env_x) // batch_size
        for b in range(n_batches):
            optimizer.zero_grad(set_to_none=True)
            total = 0.0
            for e in range(2):
                idx = perms[e][b * batch_size : (b + 1) * batch_size]
                batch_x, batch_y = env_x[e][idx].to(device), env_y[e][idx].to(device)
                w = torch.ones(1, device=device, requires_grad=True)
                risk = F.cross_entropy(model(batch_x).logits * w, batch_y)
                grad = torch.autograd.grad(risk, w, create_graph=True)[0]
                total = total + risk + irm_penalty * (grad ** 2).sum()
            loss = total / 2
            if irm_penalty > 1:
                loss = loss / irm_penalty
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        acc = accuracy(model, x_val, y_val, device)
        if acc > best_acc:
            best_acc, bad = acc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return accuracy(model, x_iid, y_iid, device), accuracy(model, x_ood, y_ood, device), model
