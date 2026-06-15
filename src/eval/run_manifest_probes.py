"""Run frozen-representation probes for runs listed in an experiment manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from src.eval.run_probing import run_probes
from src.train.common import save_json
from src.utils.hardware import get_device


def _load_manifest(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _select_manifest_runs(
    manifest: dict[str, Any],
    *,
    methods: set[str] | None = None,
    max_per_method: int = 1,
    condition_contains: str | None = None,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for run in manifest.get("runs", []):
        if run.get("status", "success") != "success":
            continue
        method = str(run.get("method", ""))
        run_dir = str(run.get("run_dir", ""))
        if methods is not None and method not in methods:
            continue
        if condition_contains is not None and condition_contains not in run_dir:
            continue
        if counts.get(method, 0) >= max_per_method:
            continue
        selected.append(run)
        counts[method] = counts.get(method, 0) + 1
    return selected


def _resolve_checkpoint(run_dir: Path, checkpoint: str) -> str:
    if checkpoint != "auto":
        return checkpoint
    for candidate in ("best.pt", "final.pt"):
        if (run_dir / candidate).exists():
            return candidate
    raise FileNotFoundError(f"no auto probe checkpoint found in {run_dir}")


def _resolve_manifest_run_dir(manifest_path: Path, run_dir: str) -> Path:
    path = Path(run_dir)
    if path.is_absolute():
        return path.resolve()
    manifest_root = manifest_path.parent.resolve()
    candidates = [
        (Path.cwd() / path).resolve(),
        (manifest_root / path).resolve(),
        (manifest_root.parent / path).resolve(),
    ]
    return next((candidate for candidate in candidates if candidate.exists()), candidates[0])


def run_manifest_probes(
    manifest_json: str | Path,
    output: str | Path | None = None,
    *,
    methods: list[str] | None = None,
    max_per_method: int = 1,
    checkpoint: str = "auto",
    probe_train_split: str = "val_iid",
    eval_splits: tuple[str, ...] = ("iid_test", "ood_test"),
    condition_contains: str | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    manifest_path = Path(manifest_json)
    manifest = _load_manifest(manifest_path)
    selected = _select_manifest_runs(
        manifest,
        methods=set(methods) if methods else None,
        max_per_method=max_per_method,
        condition_contains=condition_contains,
    )
    if not selected:
        raise ValueError("no manifest runs matched the requested probe selection")

    device = device or torch.device("cpu")
    rows = []
    for run in selected:
        method = str(run["method"])
        run_dir = _resolve_manifest_run_dir(manifest_path, str(run["run_dir"]))
        resolved_checkpoint = _resolve_checkpoint(run_dir, checkpoint)
        metrics = run_probes(
            run_dir,
            method,
            checkpoint=resolved_checkpoint,
            probe_train_split=probe_train_split,
            eval_splits=eval_splits,
            device=device,
        )
        rows.append(
            {
                "method": method,
                "run_dir": str(run_dir),
                "checkpoint": resolved_checkpoint,
                "probe_metrics_path": str(run_dir / "probe_metrics.json"),
                "metrics": metrics,
            }
        )

    summary = {
        "manifest": str(manifest_path),
        "checkpoint": checkpoint,
        "probe_train_split": probe_train_split,
        "eval_splits": list(eval_splits),
        "condition_contains": condition_contains,
        "max_per_method": max_per_method,
        "device": str(device),
        "n_runs": len(rows),
        "runs": rows,
    }
    output_path = Path(output) if output is not None else manifest_path.parent / "probe_summary.json"
    save_json(output_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run probes for runs in a manifest.")
    parser.add_argument("manifest_json")
    parser.add_argument("--output", default=None)
    parser.add_argument("--methods", nargs="*", default=None)
    parser.add_argument("--max-per-method", type=int, default=1)
    parser.add_argument("--checkpoint", default="auto")
    parser.add_argument("--probe-train-split", default="val_iid")
    parser.add_argument("--eval-splits", nargs="+", default=["iid_test", "ood_test"])
    parser.add_argument("--condition-contains", default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    device = get_device() if args.device == "auto" else torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is not available")
    summary = run_manifest_probes(
        args.manifest_json,
        args.output,
        methods=args.methods,
        max_per_method=args.max_per_method,
        checkpoint=args.checkpoint,
        probe_train_split=args.probe_train_split,
        eval_splits=tuple(args.eval_splits),
        condition_contains=args.condition_contains,
        device=device,
    )
    print(
        json.dumps(
            {
                "manifest": summary["manifest"],
                "output": str(Path(args.output) if args.output else Path(args.manifest_json).parent / "probe_summary.json"),
                "n_runs": summary["n_runs"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
