#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
OUT="${OUT:-results/full_run}"
DEVICE="${DEVICE:-auto}"
SEEDS="${SEEDS:-0 1 2 3 4}"
MIN_SEEDS="${MIN_SEEDS:-$(wc -w <<< "${SEEDS}")}"
MIN_FINAL_SEEDS_REQUIRED="${MIN_FINAL_SEEDS_REQUIRED:-5}"
EPOCHS="${EPOCHS:-50}"
MIN_EPOCHS="${MIN_EPOCHS:-25}"
METHODS="${METHODS:-erm ib ep_min ep_max ocp_style lens_like_arrow_classifier sib sid}"
CLEAN_OUT="${CLEAN_OUT:-1}"
RUN_TESTS="${RUN_TESTS:-1}"
FINALIZE_PAPER="${FINALIZE_PAPER:-1}"
PREFLIGHT_OUTPUT="${PREFLIGHT_OUTPUT:-results/preflight.json}"
REQUIRE_CUDA_FOR_FULL_RUN="${REQUIRE_CUDA_FOR_FULL_RUN:-0}"

"${PYTHON_BIN}" -m src.eval.preflight \
  --output "${PREFLIGHT_OUTPUT}" \
  --out "${OUT}" \
  --device "${DEVICE}" \
  --seeds "${SEEDS}" \
  --min-seeds "${MIN_SEEDS}" \
  --min-final-seeds-required "${MIN_FINAL_SEEDS_REQUIRED}" \
  --epochs "${EPOCHS}" \
  --min-epochs-required "${MIN_EPOCHS}" \
  --methods "${METHODS}" \
  --clean-out "${CLEAN_OUT}" \
  --run-tests "${RUN_TESTS}" \
  --finalize-paper "${FINALIZE_PAPER}" \
  --require-cuda-for-full-run "${REQUIRE_CUDA_FOR_FULL_RUN}"
