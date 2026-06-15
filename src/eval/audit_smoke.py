"""Audit final smoke-suite outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.train.common import save_json


REQUIRED_METHODS = (
    "erm",
    "ib",
    "ep_min",
    "ep_max",
    "ocp_style",
    "lens_like_arrow_classifier",
    "sib",
    "sid",
    "itm",
)
EXPECTED_BENCHMARKS = {
    "sta": "sta_bench",
    "ink_advection_diffusion": "ink_advection_diffusion",
}


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _check(condition: bool, name: str, detail: str, checks: list[dict[str, Any]]) -> bool:
    checks.append({"name": name, "passed": bool(condition), "detail": detail})
    return bool(condition)


def audit_smoke(root: str | Path) -> dict[str, Any]:
    root = Path(root)
    checks: list[dict[str, Any]] = []
    run_count = 0
    for suite_name, benchmark_name in EXPECTED_BENCHMARKS.items():
        suite_root = root / suite_name
        manifest_path = suite_root / "manifest.json"
        sid_factor_path = suite_root / "sid_factor_audit.json"
        itm_mechanism_path = suite_root / "itm_mechanism_audit.json"
        _check(
            manifest_path.exists(),
            f"{suite_name}_manifest_exists",
            str(manifest_path),
            checks,
        )
        _check(
            sid_factor_path.exists(),
            f"{suite_name}_sid_factor_audit_exists",
            str(sid_factor_path),
            checks,
        )
        if sid_factor_path.exists():
            sid_factor = _load_json(sid_factor_path)
            _check(
                bool(sid_factor.get("passed", False)),
                f"{suite_name}_sid_factor_audit_passed",
                f"passed={sid_factor.get('passed')}, n_runs={sid_factor.get('n_runs')}",
                checks,
            )
        _check(
            itm_mechanism_path.exists(),
            f"{suite_name}_itm_mechanism_audit_exists",
            str(itm_mechanism_path),
            checks,
        )
        if itm_mechanism_path.exists():
            itm_mechanism = _load_json(itm_mechanism_path)
            _check(
                bool(itm_mechanism.get("passed", False)),
                f"{suite_name}_itm_mechanism_audit_passed",
                (
                    f"passed={itm_mechanism.get('passed')}, "
                    f"n_runs={itm_mechanism.get('n_runs')}"
                ),
                checks,
            )
        if not manifest_path.exists():
            continue
        manifest = _load_json(manifest_path)
        runs = manifest.get("runs", [])
        run_count += len(runs)
        methods = sorted({row.get("method") for row in runs})
        _check(
            methods == sorted(REQUIRED_METHODS),
            f"{suite_name}_required_methods_present",
            f"methods={methods}",
            checks,
        )
        failed = [row for row in runs if row.get("status") != "success"]
        _check(
            not failed,
            f"{suite_name}_no_failed_runs",
            f"failed={len(failed)}",
            checks,
        )
        for row in runs:
            method = str(row.get("method"))
            run_dir = Path(str(row.get("run_dir", "")))
            metadata_path = run_dir / "metadata.json"
            metrics_path = run_dir / "final_metrics.json"
            if not _check(
                metadata_path.exists(),
                f"{suite_name}_{method}_metadata_exists",
                str(metadata_path),
                checks,
            ):
                continue
            if not _check(
                metrics_path.exists(),
                f"{suite_name}_{method}_metrics_exists",
                str(metrics_path),
                checks,
            ):
                continue
            metadata = _load_json(metadata_path)
            metrics = _load_json(metrics_path)
            observed_benchmark = metadata.get("dataset_metadata", {}).get("train", {}).get(
                "benchmark_name"
            )
            _check(
                observed_benchmark == benchmark_name,
                f"{suite_name}_{method}_benchmark_metadata",
                f"observed={observed_benchmark}, expected={benchmark_name}",
                checks,
            )
            _check(
                metadata.get("method") == method,
                f"{suite_name}_{method}_method_metadata",
                f"metadata.method={metadata.get('method')}, manifest.method={method}",
                checks,
            )
            if method in {"ocp_style", "lens_like_arrow_classifier"}:
                metric_prefixes = ("frozen_encoder_", "fine_tuned_encoder_")
                has_iid = any(f"{prefix}iid_test_accuracy" in metrics for prefix in metric_prefixes)
                has_ood = any(f"{prefix}ood_test_accuracy" in metrics for prefix in metric_prefixes)
                has_gap = any(f"{prefix}ood_gap" in metrics for prefix in metric_prefixes)
            else:
                has_iid = "iid_test_accuracy" in metrics
                has_ood = "ood_test_accuracy" in metrics
                has_gap = "ood_gap" in metrics
            _check(
                has_iid and has_ood and has_gap,
                f"{suite_name}_{method}_final_eval_metrics",
                f"has_iid={has_iid}, has_ood={has_ood}, has_gap={has_gap}",
                checks,
            )
            if method in {"sib", "sid", "itm"}:
                has_cf = any(key.endswith("cf_prediction_consistency") for key in metrics)
                _check(
                    has_cf,
                    f"{suite_name}_{method}_counterfactual_metrics",
                    "requires cf_prediction_consistency metric",
                    checks,
                )
    passed = all(check["passed"] for check in checks)
    return {
        "passed": passed,
        "root": str(root),
        "required_methods": list(REQUIRED_METHODS),
        "expected_benchmarks": EXPECTED_BENCHMARKS,
        "run_count": run_count,
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit final smoke-suite outputs.")
    parser.add_argument("--root", default="results/smoke_run")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    report = audit_smoke(args.root)
    output = Path(args.output) if args.output else Path(args.root) / "smoke_audit.json"
    save_json(output, report)
    print(json.dumps({"passed": report["passed"], "run_count": report["run_count"]}, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
