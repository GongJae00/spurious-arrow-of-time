"""Aggregate final_metrics.json files across seeds/runs."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from src.train.common import save_json


def _numeric_items(d: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for k, v in d.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out[k] = float(v)
    return out


def _summarize_grouped(grouped: dict[str, list[dict[str, float]]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key, rows in grouped.items():
        metrics = sorted({metric for row in rows for metric in row})
        summary[key] = {"n_runs": len(rows)}
        for metric in metrics:
            values = [row[metric] for row in rows if metric in row]
            if not values:
                continue
            summary[key][f"{metric}_mean"] = mean(values)
            summary[key][f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0
    return summary


def _method_and_condition(
    root: Path,
    metrics_path: Path,
    manifest_method: str | None = None,
) -> tuple[str, str]:
    try:
        rel_parts = metrics_path.relative_to(root).parts
    except ValueError:
        return manifest_method or metrics_path.parent.parent.name, "."
    if len(rel_parts) < 3:
        return manifest_method or metrics_path.parent.parent.name, "."
    method = manifest_method or rel_parts[-3]
    condition_parts = rel_parts[:-3]
    condition = "/".join(condition_parts) if condition_parts else "."
    return method, condition


def _metric_paths_from_manifest(root: Path) -> list[tuple[Path, str | None]] | None:
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text())
    paths: list[tuple[Path, str | None]] = []
    for run in manifest.get("runs", []):
        if run.get("status", "success") != "success":
            continue
        run_dir = run.get("run_dir")
        if not run_dir:
            continue
        path = Path(run_dir)
        if not path.is_absolute():
            candidates = [
                (Path.cwd() / path).resolve(),
                (root / path).resolve(),
                (root.parent / path).resolve(),
            ]
            path = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
        metrics_path = path / "final_metrics.json"
        if metrics_path.exists():
            paths.append((metrics_path, run.get("method")))
    return paths


def aggregate_results(root: str | Path, output: str | Path | None = None) -> dict[str, Any]:
    root = Path(root).resolve()
    grouped: dict[str, list[dict[str, float]]] = defaultdict(list)
    grouped_by_condition: dict[str, dict[str, list[dict[str, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    skipped: list[dict[str, str]] = []
    manifest_paths = _metric_paths_from_manifest(root)
    paths = (
        manifest_paths
        if manifest_paths is not None
        else [(path, None) for path in root.rglob("final_metrics.json")]
    )
    for path, manifest_method in paths:
        try:
            method, condition = _method_and_condition(root, path, manifest_method)
            row = _numeric_items(json.loads(path.read_text()))
            grouped[method].append(row)
            grouped_by_condition[condition][method].append(row)
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            skipped.append(
                {
                    "path": str(path),
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
    summary = _summarize_grouped(grouped)
    summary["by_condition"] = {
        condition: _summarize_grouped(method_rows)
        for condition, method_rows in sorted(grouped_by_condition.items())
    }
    summary["n_skipped_metric_files"] = len(skipped)
    summary["skipped_metric_files"] = skipped
    if output is not None:
        save_json(output, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate final_metrics.json files.")
    parser.add_argument("root")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    summary = aggregate_results(args.root, args.output)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
