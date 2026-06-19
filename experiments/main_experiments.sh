#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${CONFIG:-configs/irreversible_source_main.yaml}"
PROFILE="${PROFILE:-smoke}"
OUT="${OUT:-results/main_experiments/${PROFILE}}"
CLEAN_OUT="${CLEAN_OUT:-1}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"

if [[ "${CLEAN_OUT}" == "1" ]]; then
  rm -rf "${OUT}"
fi
mkdir -p "${OUT}"

"${PYTHON_BIN}" -m src.train.main_experiment \
  --config "${CONFIG}" \
  --profile "${PROFILE}" \
  --out "${OUT}"

"${PYTHON_BIN}" -m src.eval.main_results \
  --metrics "${OUT}/metrics.jsonl" \
  --summary "${OUT}/summary.json" \
  --out "${OUT}"

echo "Main experiment artifacts written to ${OUT}"
