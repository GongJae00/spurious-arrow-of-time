"""Artifact-backed data access for manuscript figures."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Aggregate:
    mean: float
    std: float
    n: int


class MetricStore:
    """Read figure values from locked full-run artifacts."""

    def __init__(self, summary_path: Path, metrics_path: Path) -> None:
        self.summary_path = summary_path
        self.metrics_path = metrics_path
        self.summary: dict[str, Any] = json.loads(summary_path.read_text(encoding="utf-8"))
        self.primary_scenario = str(self.summary["primary_scenario"])
        self.rows: list[dict[str, Any]] = [
            json.loads(line)
            for line in metrics_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def aggregate(self, method: str, metric: str, scenario: str | None = None) -> Aggregate:
        """Return the logged aggregate for a method/metric/scenario."""

        scenario = scenario or self.primary_scenario
        if scenario == self.primary_scenario and method in self.summary["methods"]:
            source = self.summary["methods"][method].get(metric)
        else:
            source = self.summary["scenarios"].get(scenario, {}).get(method, {}).get(metric)
        if source is not None:
            return Aggregate(mean=float(source["mean"]), std=float(source["std"]), n=int(source["n"]))
        vals = self.values(method, metric, scenario)
        return Aggregate(mean=float(vals.mean()), std=float(vals.std()), n=int(vals.size))

    def values(self, method: str, metric: str, scenario: str | None = None) -> np.ndarray:
        """Return seed-level values reconstructed from metrics.jsonl."""

        scenario = scenario or self.primary_scenario
        matched = [
            row
            for row in self.rows
            if row.get("scenario") == scenario and row.get("method") == method and metric in row
        ]
        if not matched:
            source = None
            if scenario == self.primary_scenario and method in self.summary["methods"]:
                source = self.summary["methods"][method].get(metric)
            if source and source.get("values"):
                return np.asarray(source["values"], dtype=float)
            raise KeyError(f"No values for scenario={scenario!r}, method={method!r}, metric={metric!r}")
        matched = sorted(matched, key=lambda row: int(row["seed"]))
        return np.asarray([float(row[metric]) for row in matched], dtype=float)

    def paired_values(
        self,
        method_a: str,
        method_b: str,
        metric: str,
        scenario: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return seed-aligned metric values for two methods."""

        scenario = scenario or self.primary_scenario
        by_method: dict[str, dict[int, float]] = {method_a: {}, method_b: {}}
        for row in self.rows:
            if row.get("scenario") != scenario:
                continue
            method = str(row.get("method"))
            if method in by_method and metric in row:
                by_method[method][int(row["seed"])] = float(row[metric])
        seeds = sorted(set(by_method[method_a]) & set(by_method[method_b]))
        if not seeds:
            raise KeyError(f"No paired values for {method_a!r} and {method_b!r}")
        return (
            np.asarray(seeds, dtype=int),
            np.asarray([by_method[method_a][seed] for seed in seeds], dtype=float),
            np.asarray([by_method[method_b][seed] for seed in seeds], dtype=float),
        )
