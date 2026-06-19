#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${CONFIG:-configs/irreversible_source_smoke.yaml}"
OUT="${OUT:-results/smoke_benchmark}"
CLEAN_OUT="${CLEAN_OUT:-1}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"

if [[ "${CLEAN_OUT}" == "1" ]]; then
  rm -rf "${OUT}"
fi
mkdir -p "${OUT}"

"${PYTHON_BIN}" -m src.eval.benchmark_diagnostics --config "${CONFIG}" --out "${OUT}"
"${PYTHON_BIN}" -m src.eval.visualize_benchmark --config "${CONFIG}" --out "${OUT}"

echo "Smoke benchmark artifacts written to ${OUT}"
