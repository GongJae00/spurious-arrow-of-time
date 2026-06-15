"""Validate reuse of an full-run preflight artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


def _split_words(value: str) -> list[str]:
    return value.split()


def _path_matches(observed: Any, expected: str) -> bool:
    if observed is None:
        return False
    return Path(str(observed)).resolve() == Path(expected).resolve()


def validate_preflight_reuse(
    preflight_path: str | Path,
    *,
    expected_out: str,
    expected_device: str,
    expected_seeds: str,
    expected_min_seeds: int,
    expected_epochs: int,
    expected_min_epochs: int,
    expected_methods: str,
    expected_require_cuda_for_full_run: str,
) -> dict[str, Any]:
    """Return a validation report for a reusable preflight artifact."""

    path = Path(preflight_path)
    errors: list[str] = []
    try:
        payload = json.loads(path.read_text())
    except FileNotFoundError:
        return {
            "pass": False,
            "preflight_path": str(path),
            "errors": [f"missing reusable preflight artifact: {path}"],
        }
    except json.JSONDecodeError as exc:
        return {
            "pass": False,
            "preflight_path": str(path),
            "errors": [f"invalid reusable preflight artifact: {exc}"],
        }

    environment = payload.get("environment")
    if not isinstance(environment, dict):
        environment = {}
        errors.append("reusable preflight has no environment object")

    if payload.get("pass") is not True or payload.get("launch_recommended") is not True:
        errors.append(
            "pass=true and launch_recommended=true are required for preflight reuse"
        )
    if not _path_matches(environment.get("out"), expected_out):
        errors.append(
            f"preflight out={environment.get('out')!r} does not match OUT={expected_out!r}"
        )
    if str(environment.get("device")) != expected_device:
        errors.append(
            f"preflight device={environment.get('device')!r} does not match "
            f"DEVICE={expected_device!r}"
        )
    if str(environment.get("seeds")) != expected_seeds:
        errors.append(
            f"preflight seeds={environment.get('seeds')!r} does not match "
            f"SEEDS={expected_seeds!r}"
        )
    if int(environment.get("min_seeds", -1)) != int(expected_min_seeds):
        errors.append(
            f"preflight min_seeds={environment.get('min_seeds')!r} does not match "
            f"MIN_SEEDS={expected_min_seeds!r}"
        )
    if int(environment.get("epochs", -1)) != int(expected_epochs):
        errors.append(
            f"preflight epochs={environment.get('epochs')!r} does not match "
            f"EPOCHS={expected_epochs!r}"
        )
    if int(environment.get("min_epochs_required", -1)) != int(expected_min_epochs):
        errors.append(
            "preflight min_epochs_required="
            f"{environment.get('min_epochs_required')!r} does not match "
            f"MIN_EPOCHS={expected_min_epochs!r}"
        )
    if sorted(map(str, environment.get("methods", []))) != sorted(
        _split_words(expected_methods)
    ):
        errors.append(
            f"preflight methods={environment.get('methods')!r} does not match "
            f"METHODS={_split_words(expected_methods)!r}"
        )
    observed_require_cuda = str(environment.get("require_cuda_for_full_run"))
    if observed_require_cuda != expected_require_cuda_for_full_run:
        errors.append(
            "preflight require_cuda_for_full_run="
            f"{observed_require_cuda!r} does not match "
            "REQUIRE_CUDA_FOR_FULL_RUN="
            f"{expected_require_cuda_for_full_run!r}"
        )
    if (
        expected_require_cuda_for_full_run == "1"
        and environment.get("launch_authorization") != "final_cuda_launch"
    ):
        errors.append(
            "preflight launch_authorization must be final_cuda_launch when "
            "REQUIRE_CUDA_FOR_FULL_RUN=1"
        )

    return {
        "pass": not errors,
        "preflight_path": str(path),
        "errors": errors,
        "environment": environment,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-path", required=True)
    parser.add_argument("--expected-out", required=True)
    parser.add_argument("--expected-device", required=True)
    parser.add_argument("--expected-seeds", required=True)
    parser.add_argument("--expected-min-seeds", type=int, required=True)
    parser.add_argument("--expected-epochs", type=int, required=True)
    parser.add_argument("--expected-min-epochs", type=int, required=True)
    parser.add_argument("--expected-methods", required=True)
    parser.add_argument("--expected-require-cuda-for-full-run", required=True)
    args = parser.parse_args()

    report = validate_preflight_reuse(
        args.preflight_path,
        expected_out=args.expected_out,
        expected_device=args.expected_device,
        expected_seeds=args.expected_seeds,
        expected_min_seeds=args.expected_min_seeds,
        expected_epochs=args.expected_epochs,
        expected_min_epochs=args.expected_min_epochs,
        expected_methods=args.expected_methods,
        expected_require_cuda_for_full_run=(
            args.expected_require_cuda_for_full_run
        ),
    )
    if not report["pass"]:
        print("Refusing to reuse preflight artifact:", file=sys.stderr)
        for error in report["errors"]:
            print(f"- {error}", file=sys.stderr)
    raise SystemExit(0 if report["pass"] else 2)


if __name__ == "__main__":
    main()
