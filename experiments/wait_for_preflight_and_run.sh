#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
PREFLIGHT_OUTPUT="${PREFLIGHT_OUTPUT:-results/preflight.json}"
PREFLIGHT_WAIT_SECONDS="${PREFLIGHT_WAIT_SECONDS:-300}"
MAX_PREFLIGHT_ATTEMPTS="${MAX_PREFLIGHT_ATTEMPTS:-0}"
LAUNCH_AFTER_PREFLIGHT="${LAUNCH_AFTER_PREFLIGHT:-1}"
REQUIRE_CUDA_FOR_FULL_RUN="${REQUIRE_CUDA_FOR_FULL_RUN:-1}"

if [[ "${RUN_PREFLIGHT:-1}" != "1" ]]; then
  echo "This wait wrapper requires RUN_PREFLIGHT=1; it must not bypass preflight." >&2
  exit 2
fi

if [[ "${OUT:-results/full_run}" == *smoke* ]]; then
  echo "Refusing to wait-launch full suite with a smoke-like OUT." >&2
  exit 2
fi

if ! [[ "${PREFLIGHT_WAIT_SECONDS}" =~ ^[0-9]+$ ]]; then
  echo "PREFLIGHT_WAIT_SECONDS must be a non-negative integer." >&2
  exit 2
fi

if ! [[ "${MAX_PREFLIGHT_ATTEMPTS}" =~ ^[0-9]+$ ]]; then
  echo "MAX_PREFLIGHT_ATTEMPTS must be a non-negative integer, where 0 means unlimited." >&2
  exit 2
fi

if [[ "${LAUNCH_AFTER_PREFLIGHT}" != "0" && "${LAUNCH_AFTER_PREFLIGHT}" != "1" ]]; then
  echo "LAUNCH_AFTER_PREFLIGHT must be 0 or 1." >&2
  exit 2
fi

if [[ "${REQUIRE_CUDA_FOR_FULL_RUN}" != "0" && "${REQUIRE_CUDA_FOR_FULL_RUN}" != "1" ]]; then
  echo "REQUIRE_CUDA_FOR_FULL_RUN must be 0 or 1." >&2
  exit 2
fi

export RUN_PREFLIGHT=1
export REQUIRE_CUDA_FOR_FULL_RUN

attempt=1
while :; do
  echo "Preflight attempt ${attempt}..."
  ATTEMPT_PREFLIGHT_OUTPUT="${PREFLIGHT_OUTPUT}.attempt"
  if PREFLIGHT_OUTPUT="${ATTEMPT_PREFLIGHT_OUTPUT}" bash "${SCRIPT_DIR}/preflight.sh"; then
    if "${PYTHON_BIN}" - "${ATTEMPT_PREFLIGHT_OUTPUT}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text())
raise SystemExit(0 if payload.get("launch_recommended") is True else 1)
PY
    then
      mv -- "${ATTEMPT_PREFLIGHT_OUTPUT}" "${PREFLIGHT_OUTPUT}"
      echo "Preflight passed."
      if [[ "${LAUNCH_AFTER_PREFLIGHT}" == "0" ]]; then
        echo "LAUNCH_AFTER_PREFLIGHT=0; not launching full suite."
        exit 0
      fi
      RUN_PREFLIGHT=0 REUSE_PASSED_PREFLIGHT=1 \
        exec bash "${SCRIPT_DIR}/full_suite.sh"
    fi
  fi

  if (( MAX_PREFLIGHT_ATTEMPTS != 0 && attempt >= MAX_PREFLIGHT_ATTEMPTS )); then
    echo "Preflight did not pass after ${attempt} attempt(s)." >&2
    exit 1
  fi

  echo "Preflight not ready; sleeping ${PREFLIGHT_WAIT_SECONDS}s before retry."
  sleep "${PREFLIGHT_WAIT_SECONDS}"
  attempt=$((attempt + 1))
done
