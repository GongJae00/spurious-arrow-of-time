import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader


def as_tensor(a: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(a.astype(np.float32)))


@torch.no_grad()
def accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor, device: torch.device) -> float:
    preds = []
    for i in range(0, len(x), 512):
        preds.append(model(x[i : i + 512].to(device)).logits.argmax(1))
    return float((torch.cat(preds) == y.to(device)).float().mean().item())


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, has_counterfactual: bool = False) -> dict[str, float]:
    model.eval()
    total = 0
    loss_sum = 0.0
    correct = 0
    cf_correct = 0
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            y = batch[1].to(device)
            out = model(x)
            loss = F.cross_entropy(out.logits, y)
            batch_size = int(y.numel())
            total += batch_size
            loss_sum += float(loss.item()) * batch_size
            correct += int((out.logits.argmax(dim=1) == y).sum().item())
            if has_counterfactual and len(batch) >= 3:
                cf_correct += int((model(batch[2].to(device)).logits.argmax(dim=1) == y).sum().item())
    metrics = {"loss": loss_sum / max(total, 1), "accuracy": correct / max(total, 1)}
    if has_counterfactual:
        metrics["accuracy_on_x_cf"] = cf_correct / max(total, 1)
    return metrics


def aggregate(values) -> dict:
    a = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(a.mean()),
        "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
        "values": [round(float(v), 4) for v in a],
    }


def input_gradient_saliency(model: nn.Module, x: torch.Tensor, device: torch.device):
    model.train()
    xt = x.to(device).requires_grad_(True)
    logits = model(xt).logits
    logits.gather(1, logits.argmax(1, keepdim=True)).sum().backward()
    g = xt.grad.abs()
    core_g = g[:, :, 0].mean(dim=(0, 2, 3))
    nuis_g = g[:, :, 1].mean(dim=(0, 2, 3))
    share = float(nuis_g.sum() / (core_g.sum() + nuis_g.sum()))
    prof = (nuis_g / nuis_g.sum()).detach().cpu().numpy()
    return share, prof


def regime(iid: float, ood: float) -> str:
    if iid >= 0.8 and ood >= 0.8:
        return "core"
    if iid >= 0.8 and ood <= 0.2:
        return "collapse"
    return "chance"


def summarize_results(results: list[dict], primary_scenario: str) -> dict:
    primary = [r for r in results if str(r.get("scenario", primary_scenario)) == primary_scenario] or results
    by_method: dict[str, list] = {}
    for r in primary:
        by_method.setdefault(str(r["method"]), []).append(r)
    method_summary = {}
    for method, rows in sorted(by_method.items()):
        method_summary[method] = {}
        for key in ["val_iid_accuracy", "iid_test_accuracy", "ood_test_accuracy", "ood_gap"]:
            values = np.array([float(row[key]) for row in rows], dtype=float)
            method_summary[method][key] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "n": int(len(values)),
                "values": values.round(6).tolist(),
            }
    scenario_summary: dict = {}
    for r in results:
        scenario_summary.setdefault(str(r.get("scenario", primary_scenario)), {}).setdefault(str(r["method"]), []).append(r)
    compact = {}
    for scenario, methods in scenario_summary.items():
        compact[scenario] = {}
        for method, rows in methods.items():
            compact[scenario][method] = {}
            for key in ["iid_test_accuracy", "ood_test_accuracy", "ood_gap"]:
                values = np.array([float(row[key]) for row in rows], dtype=float)
                compact[scenario][method][key] = {
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    "n": int(len(values)),
                }
    return {"primary_scenario": primary_scenario, "methods": method_summary, "scenarios": compact}


@dataclass(frozen=True)
class Aggregate:
    mean: float
    std: float
    n: int


class MetricStore:
    def __init__(self, summary_path: Path, metrics_path: Path):
        self.summary = json.loads(summary_path.read_text(encoding="utf-8"))
        self.primary_scenario = str(self.summary["primary_scenario"])
        self.rows = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def aggregate(self, method: str, metric: str, scenario: str | None = None) -> Aggregate:
        scenario = scenario or self.primary_scenario
        if scenario == self.primary_scenario and method in self.summary["methods"] and metric in self.summary["methods"][method]:
            source = self.summary["methods"][method][metric]
            return Aggregate(mean=float(source["mean"]), std=float(source["std"]), n=int(source["n"]))
        source = self.summary["scenarios"][scenario][method][metric]
        return Aggregate(mean=float(source["mean"]), std=float(source["std"]), n=int(source["n"]))

    def values(self, method: str, metric: str, scenario: str | None = None) -> np.ndarray:
        scenario = scenario or self.primary_scenario
        matched = [row for row in self.rows if row.get("scenario") == scenario and row.get("method") == method and metric in row]
        if matched:
            matched = sorted(matched, key=lambda row: int(row["seed"]))
            return np.asarray([float(row[metric]) for row in matched], dtype=float)
        source = self.summary["methods"][method][metric]
        return np.asarray(source["values"], dtype=float)
