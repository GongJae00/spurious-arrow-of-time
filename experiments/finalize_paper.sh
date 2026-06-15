#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
LATEXMK_BIN="${LATEXMK_BIN:-latexmk}"
RESULT_ROOT="${RESULT_ROOT:-results/full_run}"
PAPER_DIR="${PAPER_DIR:-paper}"
GENERATED_DIR="${GENERATED_DIR:-generated_irreversibility_trust}"
PAPER_ASSET_OUT="${PAPER_ASSET_OUT:-${PAPER_DIR}/${GENERATED_DIR}}"
RESULT_MAIN_FILE="${RESULT_MAIN_FILE:-main_irreversibility_trust.tex}"
PDF_PATH="${PDF_PATH:-${PAPER_DIR}/build/main_irreversibility_trust.pdf}"
MIN_SEEDS="${MIN_SEEDS:-5}"
MIN_EPOCHS="${MIN_EPOCHS:-25}"
MIN_FINAL_RUN_COUNT="${MIN_FINAL_RUN_COUNT:-80}"
PREFLIGHT_OUTPUT="${PREFLIGHT_OUTPUT:-${RESULT_ROOT}/preflight.json}"
RUN_QUALITY_CHECKS="${RUN_QUALITY_CHECKS:-1}"
ALLOW_SKIP_QUALITY_CHECKS="${ALLOW_SKIP_QUALITY_CHECKS:-0}"

QUALITY_CHECK_ENV=(
  env
  -u OUT
  -u RESULT_ROOT
  -u DEVICE
  -u REQUIRE_CUDA_FOR_FULL_RUN
  -u SEEDS
  -u MIN_SEEDS
  -u MIN_EPOCHS
  -u MIN_FINAL_RUN_COUNT
  -u EPOCHS
  -u RUN_TESTS
  -u CLEAN_OUT
  -u RUN_PREFLIGHT
  -u REUSE_PASSED_PREFLIGHT
  -u PREFLIGHT_OUTPUT
  -u PAPER_DIR
  -u GENERATED_DIR
  -u PAPER_ASSET_OUT
  -u RESULT_MAIN_FILE
  -u PDF_PATH
  -u RUN_QUALITY_CHECKS
  -u ALLOW_SKIP_QUALITY_CHECKS
)

case "${RESULT_ROOT}" in
  *smoke*|*SMOKE*)
    echo "Refusing to finalize final paper assets from smoke RESULT_ROOT='${RESULT_ROOT}'." >&2
    exit 2
    ;;
esac

if [[ "${RUN_QUALITY_CHECKS}" != "0" && "${RUN_QUALITY_CHECKS}" != "1" ]]; then
  echo "RUN_QUALITY_CHECKS must be 0 or 1." >&2
  exit 2
fi

if [[ "${ALLOW_SKIP_QUALITY_CHECKS}" != "0" && "${ALLOW_SKIP_QUALITY_CHECKS}" != "1" ]]; then
  echo "ALLOW_SKIP_QUALITY_CHECKS must be 0 or 1." >&2
  exit 2
fi

if [[ "${RUN_QUALITY_CHECKS}" == "0" && "${ALLOW_SKIP_QUALITY_CHECKS}" != "1" ]]; then
  echo "RUN_QUALITY_CHECKS=0 is diagnostic only. Set ALLOW_SKIP_QUALITY_CHECKS=1 to acknowledge final paper quality checks will be skipped." >&2
  exit 2
fi

if [[ ! -d "${RESULT_ROOT}" ]]; then
  echo "Missing RESULT_ROOT='${RESULT_ROOT}'. Run the full suite first." >&2
  exit 2
fi

EXPECTED_PAPER_ASSET_OUT="${PAPER_DIR%/}/${GENERATED_DIR}"
if [[ "$(readlink -m -- "${PAPER_ASSET_OUT}")" != "$(readlink -m -- "${EXPECTED_PAPER_ASSET_OUT}")" ]]; then
  echo "PAPER_ASSET_OUT must match PAPER_DIR/GENERATED_DIR for finalization." >&2
  echo "  PAPER_ASSET_OUT='${PAPER_ASSET_OUT}'" >&2
  echo "  expected='${EXPECTED_PAPER_ASSET_OUT}'" >&2
  exit 2
fi

if [[ "${RUN_QUALITY_CHECKS}" == "1" ]]; then
  "${QUALITY_CHECK_ENV[@]}" "${PYTHON_BIN}" -m pytest -q
  "${QUALITY_CHECK_ENV[@]}" "${PYTHON_BIN}" -m ruff check .
fi

"${PYTHON_BIN}" -m src.eval.prepare_paper_assets \
  --result-root "${RESULT_ROOT}" \
  --output-dir "${PAPER_ASSET_OUT}" \
  --min-seeds "${MIN_SEEDS}" \
  --min-epochs "${MIN_EPOCHS}" \
  --preflight-path "${PREFLIGHT_OUTPUT}"

"${LATEXMK_BIN}" -g -pdf -interaction=nonstopmode -halt-on-error \
  -outdir="${PAPER_DIR}/build" "${PAPER_DIR}/${RESULT_MAIN_FILE}"

if [[ ! -f "${PDF_PATH}" ]]; then
  echo "Expected PDF was not produced: ${PDF_PATH}" >&2
  exit 2
fi
