"""Preflight checks before launching the full non-smoke full suite."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any

from src.eval.audit_smoke import EXPECTED_BENCHMARKS, REQUIRED_METHODS
from src.train.common import load_config


FINAL_CONFIGS = (
    "configs/sta_full.yaml",
    "configs/ink_advection_diffusion_full.yaml",
)
EXPECTED_FULL_SPLITS = {
    "train": 10_000,
    "val_iid": 2_000,
    "iid_test": 5_000,
    "ood_test": 5_000,
}


@dataclass(frozen=True)
class PreflightCheck:
    name: str
    passed: bool
    severity: str
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "severity": self.severity,
            "detail": self.detail,
        }


def _seed_count(seed_text: str) -> int:
    return len(seed_text.split())


def _is_smoke_like(value: str) -> bool:
    return "smoke" in value.lower()


def _which(name: str) -> str | None:
    return shutil.which(name)


def _disk_free_gb(path: Path) -> float:
    usage = shutil.disk_usage(path)
    return usage.free / (1024**3)


def _query_nvidia_smi() -> list[dict[str, float | str]]:
    nvidia_smi = _which("nvidia-smi")
    if nvidia_smi is None:
        return []
    command = [
        nvidia_smi,
        "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError:
        return []
    if result.returncode != 0:
        return []

    rows: list[dict[str, float | str]] = []
    for raw_line in result.stdout.splitlines():
        fields = [field.strip() for field in raw_line.split(",")]
        if len(fields) != 5:
            continue
        index, name, mem_used, mem_total, util = fields
        try:
            rows.append(
                {
                    "index": index,
                    "name": name,
                    "memory_used_mb": float(mem_used),
                    "memory_total_mb": float(mem_total),
                    "gpu_utilization_percent": float(util),
                }
            )
        except ValueError:
            continue
    return rows


def _torch_cuda_available() -> tuple[bool, str]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - defensive environment report
        return False, f"torch import failed: {type(exc).__name__}: {exc}"
    return bool(torch.cuda.is_available()), f"torch.cuda.is_available={torch.cuda.is_available()}"


def _sid_schedule_required_epochs(config: dict[str, Any]) -> int:
    schedule = config.get("sid_schedule", {})
    task_warmup = max(int(schedule.get("task_warmup_epochs", 5)), 0)
    dynamics_warmup = max(int(schedule.get("dynamics_warmup_epochs", 5)), 0)
    ramp = max(int(schedule.get("regularizer_ramp_epochs", 10)), 0)
    return max(task_warmup + dynamics_warmup + ramp, 1)


def _itm_schedule_required_epochs(config: dict[str, Any]) -> int:
    schedule = config.get("itm_schedule", {})
    task_warmup = max(int(schedule.get("task_warmup_epochs", 5)), 0)
    transition_warmup = max(int(schedule.get("transition_warmup_epochs", 5)), 0)
    ramp = max(int(schedule.get("regularizer_ramp_epochs", 10)), 0)
    return max(task_warmup + transition_warmup + ramp, 1)


def run_preflight(
    *,
    out: str = "results/full_run",
    device: str = "auto",
    seeds: str = "0 1 2 3 4",
    min_seeds: int | None = None,
    min_final_seeds_required: int = 5,
    epochs: int = 50,
    min_epochs_required: int = 25,
    methods: str = " ".join(REQUIRED_METHODS),
    clean_out: str = "1",
    run_tests: str = "1",
    finalize_paper: str = "1",
    require_cuda_for_full_run: str = "0",
    min_disk_free_gb: float = 10.0,
    max_gpu_utilization_percent: float = 30.0,
    min_gpu_free_memory_mb: float = 2048.0,
    require_canonical_paths: bool = True,
) -> dict[str, Any]:
    min_seeds = min_seeds if min_seeds is not None else _seed_count(seeds)
    method_ids = methods.split()
    checks: list[PreflightCheck] = []

    def add(name: str, passed: bool, detail: str, severity: str = "error") -> None:
        checks.append(PreflightCheck(name, bool(passed), severity, detail))

    add("out_not_smoke", not _is_smoke_like(out), out)
    add("run_tests_valid", run_tests in {"0", "1"}, run_tests)
    add("clean_out_valid", clean_out in {"0", "1"}, clean_out)
    add(
        "require_cuda_for_full_run_valid",
        require_cuda_for_full_run in {"0", "1"},
        require_cuda_for_full_run,
    )
    add(
        "finalize_paper_valid",
        finalize_paper in {"0", "1"},
        finalize_paper,
    )
    add(
        "final_seed_count",
        min_seeds >= min_final_seeds_required,
        f"min_seeds={min_seeds}, required={min_final_seeds_required}, seeds={seeds}",
    )
    add(
        "final_epoch_count",
        int(epochs) >= int(min_epochs_required),
        f"epochs={epochs}, required={min_epochs_required}",
    )
    add(
        "required_methods",
        sorted(method_ids) == sorted(REQUIRED_METHODS),
        f"methods={method_ids}, required={list(REQUIRED_METHODS)}",
    )

    if require_canonical_paths:
        add("out_canonical", out == "results/full_run", out)

    for path in [
        "experiments/full_suite.sh",
        "experiments/preflight.sh",
        "experiments/wait_for_preflight_and_run.sh",
    ]:
        add(f"path_exists:{path}", Path(path).exists(), path)

    for config_path in FINAL_CONFIGS:
        path = Path(config_path)
        if not path.exists():
            add(f"final_config_exists:{config_path}", False, config_path)
            continue
        add(f"final_config_exists:{config_path}", True, config_path)
        try:
            config = load_config(path)
        except Exception as exc:
            add(
                f"final_config_loads:{config_path}",
                False,
                f"{type(exc).__name__}: {exc}",
            )
            continue
        add(f"final_config_loads:{config_path}", True, config_path)
        experiment_name = str(config.get("experiment", ""))
        add(
            f"final_config_not_smoke:{config_path}",
            "smoke" not in experiment_name.lower() and "smoke" not in config_path.lower(),
            f"experiment={experiment_name}",
        )
        split_cfg = config.get("splits", {})
        split_sizes = {
            split_name: int(split_cfg.get(split_name, {}).get("n_sequences", 0))
            for split_name in EXPECTED_FULL_SPLITS
        }
        add(
            f"final_config_full_split_sizes:{config_path}",
            all(
                split_sizes[split_name] >= minimum
                for split_name, minimum in EXPECTED_FULL_SPLITS.items()
            ),
            f"observed={split_sizes}, required={EXPECTED_FULL_SPLITS}",
        )
        config_epochs = int(config.get("training", {}).get("epochs", 0))
        add(
            f"final_config_epoch_count:{config_path}",
            config_epochs >= int(min_epochs_required),
            f"epochs={config_epochs}, required={min_epochs_required}",
        )
        sid_required_epochs = _sid_schedule_required_epochs(config)
        itm_required_epochs = _itm_schedule_required_epochs(config)
        training_cfg = config.get("training", {})
        checkpoint_floor = int(training_cfg.get("min_epoch_for_checkpoint_selection", 0))
        early_stop_floor = int(training_cfg.get("min_epochs_before_early_stopping", 0))
        add(
            f"final_config_sid_checkpoint_floor:{config_path}",
            checkpoint_floor >= sid_required_epochs,
            (
                f"min_epoch_for_checkpoint_selection={checkpoint_floor}, "
                f"sid_schedule_required_epochs={sid_required_epochs}"
            ),
        )
        add(
            f"final_config_itm_checkpoint_floor:{config_path}",
            checkpoint_floor >= itm_required_epochs,
            (
                f"min_epoch_for_checkpoint_selection={checkpoint_floor}, "
                f"itm_schedule_required_epochs={itm_required_epochs}"
            ),
        )
        add(
            f"final_config_itm_early_stop_floor:{config_path}",
            early_stop_floor >= itm_required_epochs,
            (
                f"min_epochs_before_early_stopping={early_stop_floor}, "
                f"itm_schedule_required_epochs={itm_required_epochs}"
            ),
        )
        add(
            f"final_config_sid_early_stop_floor:{config_path}",
            early_stop_floor >= sid_required_epochs,
            (
                f"min_epochs_before_early_stopping={early_stop_floor}, "
                f"sid_schedule_required_epochs={sid_required_epochs}"
            ),
        )

    for suite_name, benchmark_name in EXPECTED_BENCHMARKS.items():
        add(
            f"expected_benchmark:{suite_name}",
            bool(benchmark_name),
            f"{suite_name} -> {benchmark_name}",
        )

    for check_name, binary in [
        ("python_available", "python"),
        ("bash_available", "bash"),
        ("git_available", "git"),
    ]:
        add(check_name, _which(binary) is not None, binary)
    if finalize_paper == "1":
        add("latexmk_available_for_final_paper", _which("latexmk") is not None, "latexmk")

    free_gb = _disk_free_gb(Path.cwd())
    add(
        "disk_free_space",
        free_gb >= min_disk_free_gb,
        f"free_gb={free_gb:.2f}, required_gb={min_disk_free_gb:.2f}",
    )

    device_lower = device.lower()
    needs_gpu_check = device_lower in {"auto", "cuda", "gpu"} or device_lower.startswith(
        "cuda"
    )
    if require_cuda_for_full_run == "1":
        add(
            "full_run_requires_cuda_capable_device",
            needs_gpu_check,
            (
                f"device={device}; final full suite must use DEVICE=cuda, "
                "DEVICE=gpu, DEVICE=cuda:N, or DEVICE=auto with CUDA available"
            ),
        )
        cuda_available, cuda_detail = _torch_cuda_available()
        add(
            "torch_cuda_available_for_full_suite",
            cuda_available,
            cuda_detail,
        )
    gpu_rows = _query_nvidia_smi()
    if needs_gpu_check and gpu_rows:
        available = [
            row
            for row in gpu_rows
            if float(row["gpu_utilization_percent"]) <= max_gpu_utilization_percent
            and (float(row["memory_total_mb"]) - float(row["memory_used_mb"]))
            >= min_gpu_free_memory_mb
        ]
        add(
            "gpu_available_for_full_suite",
            bool(available),
            (
                f"device={device}, gpus={gpu_rows}, "
                f"max_util={max_gpu_utilization_percent}, "
                f"min_free_mb={min_gpu_free_memory_mb}"
            ),
            severity="warning",
        )
    elif needs_gpu_check:
        add(
            "gpu_status_known",
            False,
            "nvidia-smi unavailable or returned no parseable GPUs; DEVICE=auto may fall back to CPU",
            severity="warning",
        )
    else:
        add("gpu_check_skipped_for_cpu", True, f"device={device}", severity="warning")

    n_errors_failed = sum(
        1 for check in checks if check.severity == "error" and not check.passed
    )
    n_warnings_failed = sum(
        1 for check in checks if check.severity == "warning" and not check.passed
    )
    launch_authorization = (
        "final_cuda_launch"
        if require_cuda_for_full_run == "1"
        else "maintenance_or_diagnostic"
    )
    return {
        "pass": n_errors_failed == 0 and n_warnings_failed == 0,
        "launch_recommended": n_errors_failed == 0 and n_warnings_failed == 0,
        "n_checks": len(checks),
        "n_errors_failed": n_errors_failed,
        "n_warnings_failed": n_warnings_failed,
        "checks": [check.to_dict() for check in checks],
        "environment": {
            "out": out,
            "device": device,
            "seeds": seeds,
            "min_seeds": min_seeds,
            "min_final_seeds_required": min_final_seeds_required,
            "epochs": int(epochs),
            "min_epochs_required": int(min_epochs_required),
            "methods": method_ids,
            "clean_out": clean_out,
            "run_tests": run_tests,
            "finalize_paper": finalize_paper,
            "require_cuda_for_full_run": require_cuda_for_full_run,
            "launch_authorization": launch_authorization,
        },
        "interpretation_lock": (
            "This preflight only decides whether the full suite should launch. "
            "It does not decide whether the hypothesis succeeds."
        ),
    }


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output")
    parser.add_argument("--out", default=os.environ.get("OUT", "results/full_run"))
    parser.add_argument("--device", default=os.environ.get("DEVICE", "auto"))
    parser.add_argument("--seeds", default=os.environ.get("SEEDS", "0 1 2 3 4"))
    parser.add_argument("--min-seeds", type=int, default=None)
    parser.add_argument(
        "--min-final-seeds-required",
        type=int,
        default=_env_int("MIN_FINAL_SEEDS_REQUIRED", 5),
    )
    parser.add_argument("--epochs", type=int, default=_env_int("EPOCHS", 50))
    parser.add_argument(
        "--min-epochs-required",
        type=int,
        default=_env_int("MIN_EPOCHS", 25),
    )
    parser.add_argument("--methods", default=os.environ.get("METHODS", " ".join(REQUIRED_METHODS)))
    parser.add_argument("--clean-out", default=os.environ.get("CLEAN_OUT", "1"))
    parser.add_argument("--run-tests", default=os.environ.get("RUN_TESTS", "1"))
    parser.add_argument(
        "--finalize-paper",
        default=os.environ.get("FINALIZE_PAPER", "1"),
    )
    parser.add_argument(
        "--require-cuda-for-full-run",
        default=os.environ.get("REQUIRE_CUDA_FOR_FULL_RUN", "0"),
    )
    parser.add_argument("--min-disk-free-gb", type=float, default=_env_float("MIN_DISK_FREE_GB", 10.0))
    parser.add_argument(
        "--max-gpu-utilization-percent",
        type=float,
        default=_env_float("MAX_GPU_UTILIZATION_PERCENT", 30.0),
    )
    parser.add_argument(
        "--min-gpu-free-memory-mb",
        type=float,
        default=_env_float("MIN_GPU_FREE_MEMORY_MB", 2048.0),
    )
    parser.add_argument("--no-require-canonical-paths", action="store_true")
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()

    result = run_preflight(
        out=args.out,
        device=args.device,
        seeds=args.seeds,
        min_seeds=args.min_seeds,
        min_final_seeds_required=args.min_final_seeds_required,
        epochs=args.epochs,
        min_epochs_required=args.min_epochs_required,
        methods=args.methods,
        clean_out=args.clean_out,
        run_tests=args.run_tests,
        finalize_paper=args.finalize_paper,
        require_cuda_for_full_run=args.require_cuda_for_full_run,
        min_disk_free_gb=args.min_disk_free_gb,
        max_gpu_utilization_percent=args.max_gpu_utilization_percent,
        min_gpu_free_memory_mb=args.min_gpu_free_memory_mb,
        require_canonical_paths=not args.no_require_canonical_paths,
    )
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                key: result[key]
                for key in [
                    "pass",
                    "launch_recommended",
                    "n_checks",
                    "n_errors_failed",
                    "n_warnings_failed",
                ]
            },
            indent=2,
            sort_keys=True,
        )
    )
    raise SystemExit(0 if result["launch_recommended"] else 1)


if __name__ == "__main__":
    main()
