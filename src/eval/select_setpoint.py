"""Select SIB setpoints from fixed-grid runs using val_iid only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.train.common import save_json


def _candidate_from_run(run: dict[str, Any], metric: str) -> dict[str, Any] | None:
    if run.get("status", "success") != "success":
        return None
    sweep = run.get("sweep") or {}
    if sweep.get("type") != "setpoint":
        return None
    if sweep.get("oracle_assisted"):
        return None
    metrics = run.get("metrics") or {}
    if metric not in metrics:
        return None
    target = sweep.get("target")
    if target is None:
        return None
    if "multiplier" in sweep:
        condition = f"setpoint_{sweep['multiplier']}"
    else:
        condition = str(sweep.get("mode", "setpoint"))
    return {
        "method": run["method"],
        "run_dir": run["run_dir"],
        "target": float(target),
        "condition": condition,
        "selection_metric_value": float(metrics[metric]),
        "metrics": metrics,
        "sweep": sweep,
    }


def select_val_iid_setpoints(
    root: str | Path,
    output: str | Path | None = None,
    metric: str = "val_iid_accuracy",
) -> dict[str, Any]:
    """Choose setpoint targets using validation-IID metrics only.

    This intentionally ignores OOD metrics. It is a selector for already-run
    `fixed_grid` sweeps, not an OOD-tuned model picker.
    """

    root = Path(root)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    candidates_by_method: dict[str, list[dict[str, Any]]] = {}
    for run in manifest.get("runs", []):
        candidate = _candidate_from_run(run, metric)
        if candidate is None:
            continue
        candidates_by_method.setdefault(candidate["method"], []).append(candidate)

    selections: dict[str, Any] = {}
    for method, candidates in sorted(candidates_by_method.items()):
        selected = max(candidates, key=lambda row: row["selection_metric_value"])
        selections[method] = {
            "setpoint": {
                "mode": "val_iid_sweep",
                "selected_target": selected["target"],
                "selection_split": "val_iid",
                "selection_metric": metric,
            },
            "selected_target": selected["target"],
            "selected_condition": selected["condition"],
            "selected_run_dir": selected["run_dir"],
            "selection_metric_value": selected["selection_metric_value"],
            "n_candidates": len(candidates),
        }

    result = {
        "selection_split": "val_iid",
        "selection_metric": metric,
        "uses_ood_test": False,
        "methods": selections,
    }
    if output is not None:
        save_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Select fixed-grid setpoint by val_iid.")
    parser.add_argument("root")
    parser.add_argument("--output", default=None)
    parser.add_argument("--metric", default="val_iid_accuracy")
    args = parser.parse_args()
    result = select_val_iid_setpoints(args.root, args.output, args.metric)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
