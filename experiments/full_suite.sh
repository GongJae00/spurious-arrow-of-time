#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
OUT="${OUT:-results/full_run}"
DEVICE="${DEVICE:-auto}"
REQUIRE_CUDA_FOR_FULL_RUN="${REQUIRE_CUDA_FOR_FULL_RUN:-1}"
RUN_TESTS="${RUN_TESTS:-1}"
CLEAN_OUT="${CLEAN_OUT:-1}"
RUN_PREFLIGHT="${RUN_PREFLIGHT:-1}"
REUSE_PASSED_PREFLIGHT="${REUSE_PASSED_PREFLIGHT:-0}"
PREFLIGHT_OUTPUT="${PREFLIGHT_OUTPUT:-results/preflight.json}"
FINALIZE_PAPER="${FINALIZE_PAPER:-1}"
ALLOW_UNFINALIZED_RUN="${ALLOW_UNFINALIZED_RUN:-0}"
EPOCHS="${EPOCHS:-50}"
SEEDS_STR="${SEEDS:-0 1 2 3 4}"
MIN_SEEDS="${MIN_SEEDS:-5}"
MIN_EPOCHS="${MIN_EPOCHS:-25}"
MIN_FINAL_SEEDS_REQUIRED="${MIN_FINAL_SEEDS_REQUIRED:-5}"
METHODS_STR="${METHODS:-erm ib ep_min ep_max ocp_style lens_like_arrow_classifier sib sid itm}"

read -r -a SEEDS_ARR <<< "${SEEDS_STR}"
read -r -a METHODS <<< "${METHODS_STR}"

if [[ "${RUN_PREFLIGHT}" != "0" && "${RUN_PREFLIGHT}" != "1" ]]; then
  echo "RUN_PREFLIGHT must be 0 or 1." >&2
  exit 2
fi

if [[ "${REQUIRE_CUDA_FOR_FULL_RUN}" != "0" && "${REQUIRE_CUDA_FOR_FULL_RUN}" != "1" ]]; then
  echo "REQUIRE_CUDA_FOR_FULL_RUN must be 0 or 1." >&2
  exit 2
fi

if [[ "${REUSE_PASSED_PREFLIGHT}" != "0" && "${REUSE_PASSED_PREFLIGHT}" != "1" ]]; then
  echo "REUSE_PASSED_PREFLIGHT must be 0 or 1." >&2
  exit 2
fi

if [[ "${FINALIZE_PAPER}" != "0" && "${FINALIZE_PAPER}" != "1" ]]; then
  echo "FINALIZE_PAPER must be 0 or 1." >&2
  exit 2
fi

if [[ "${ALLOW_UNFINALIZED_RUN}" != "0" && "${ALLOW_UNFINALIZED_RUN}" != "1" ]]; then
  echo "ALLOW_UNFINALIZED_RUN must be 0 or 1." >&2
  exit 2
fi

if [[ "${FINALIZE_PAPER}" == "0" && "${ALLOW_UNFINALIZED_RUN}" != "1" ]]; then
  echo "FINALIZE_PAPER=0 is diagnostic only. Set ALLOW_UNFINALIZED_RUN=1 to acknowledge final paper assets and PDF will not be refreshed." >&2
  exit 2
fi

if [[ "${RUN_PREFLIGHT}" == "0" && "${REUSE_PASSED_PREFLIGHT}" != "1" ]]; then
  echo "RUN_PREFLIGHT=0 is diagnostic only. Set REUSE_PASSED_PREFLIGHT=1 to reuse an existing passed preflight artifact." >&2
  exit 2
fi

if [[ "${RUN_PREFLIGHT}" == "1" ]]; then
  PYTHON_BIN="${PYTHON_BIN}" \
  OUT="${OUT}" \
  DEVICE="${DEVICE}" \
  EPOCHS="${EPOCHS}" \
  SEEDS="${SEEDS_STR}" \
  MIN_SEEDS="${MIN_SEEDS}" \
  MIN_EPOCHS="${MIN_EPOCHS}" \
  MIN_FINAL_SEEDS_REQUIRED="${MIN_FINAL_SEEDS_REQUIRED}" \
  METHODS="${METHODS_STR}" \
  CLEAN_OUT="${CLEAN_OUT}" \
  RUN_TESTS="${RUN_TESTS}" \
  FINALIZE_PAPER="${FINALIZE_PAPER}" \
  REQUIRE_CUDA_FOR_FULL_RUN="${REQUIRE_CUDA_FOR_FULL_RUN}" \
  PREFLIGHT_OUTPUT="${PREFLIGHT_OUTPUT}" \
    bash experiments/preflight.sh
fi

if [[ "${REUSE_PASSED_PREFLIGHT}" == "1" ]]; then
  "${PYTHON_BIN}" -m src.eval.validate_preflight_reuse \
    --preflight-path "${PREFLIGHT_OUTPUT}" \
    --expected-out "${OUT}" \
    --expected-device "${DEVICE}" \
    --expected-seeds "${SEEDS_STR}" \
    --expected-min-seeds "${MIN_SEEDS}" \
    --expected-epochs "${EPOCHS}" \
    --expected-min-epochs "${MIN_EPOCHS}" \
    --expected-methods "${METHODS_STR}" \
    --expected-require-cuda-for-full-run "${REQUIRE_CUDA_FOR_FULL_RUN}"
fi

if [[ "${CLEAN_OUT}" == "1" ]]; then
  case "${OUT}" in
    results/full_run|results/full_run/*|results/*full_run*)
      rm -rf -- "${OUT}"
      ;;
    *)
      echo "Refusing to clean OUT='${OUT}'. Use a path under results/*full_run* or set CLEAN_OUT=0." >&2
      exit 2
      ;;
  esac
fi

mkdir -p "${OUT}/diagnostics"
if [[ "${RUN_PREFLIGHT}" == "1" || "${REUSE_PASSED_PREFLIGHT}" == "1" ]]; then
  cp -- "${PREFLIGHT_OUTPUT}" "${OUT}/preflight.json"
fi

if [[ "${RUN_TESTS}" == "1" ]]; then
  env -u OUT -u DEVICE -u REQUIRE_CUDA_FOR_FULL_RUN -u RUN_TESTS -u CLEAN_OUT -u RUN_PREFLIGHT -u REUSE_PASSED_PREFLIGHT -u PREFLIGHT_OUTPUT -u FINALIZE_PAPER -u ALLOW_UNFINALIZED_RUN -u EPOCHS -u SEEDS -u MIN_SEEDS -u MIN_EPOCHS \
    "${PYTHON_BIN}" -m ruff check .
  env -u OUT -u DEVICE -u REQUIRE_CUDA_FOR_FULL_RUN -u RUN_TESTS -u CLEAN_OUT -u RUN_PREFLIGHT -u REUSE_PASSED_PREFLIGHT -u PREFLIGHT_OUTPUT -u FINALIZE_PAPER -u ALLOW_UNFINALIZED_RUN -u EPOCHS -u SEEDS -u MIN_SEEDS -u MIN_EPOCHS \
    "${PYTHON_BIN}" -m pytest -q
fi

"${PYTHON_BIN}" -m src.eval.sta_benchmark_diagnostics \
  --config configs/sta_full.yaml \
  --output "${OUT}/diagnostics/sta_benchmark_diagnostics.json" >/dev/null

"${PYTHON_BIN}" -m src.eval.ink_advection_diffusion_diagnostics \
  --config configs/ink_advection_diffusion_full.yaml \
  --output "${OUT}/diagnostics/ink_advection_diffusion_diagnostics.json" >/dev/null

"${PYTHON_BIN}" -m src.experiments.run_experiment \
  --config configs/sta_full.yaml \
  --output-dir "${OUT}/sta" \
  --device "${DEVICE}" \
  --epochs "${EPOCHS}" \
  --seeds "${SEEDS_ARR[@]}" \
  --methods "${METHODS[@]}" \
  --overwrite

"${PYTHON_BIN}" -m src.experiments.run_experiment \
  --config configs/ink_advection_diffusion_full.yaml \
  --output-dir "${OUT}/ink_advection_diffusion" \
  --device "${DEVICE}" \
  --epochs "${EPOCHS}" \
  --seeds "${SEEDS_ARR[@]}" \
  --methods "${METHODS[@]}" \
  --overwrite

"${PYTHON_BIN}" -m src.eval.sid_factor_audit \
  --manifest "${OUT}/sta/manifest.json" \
  --output "${OUT}/sta/sid_factor_audit.json" \
  --device "${DEVICE}"

"${PYTHON_BIN}" -m src.eval.sid_factor_audit \
  --manifest "${OUT}/ink_advection_diffusion/manifest.json" \
  --output "${OUT}/ink_advection_diffusion/sid_factor_audit.json" \
  --device "${DEVICE}"

"${PYTHON_BIN}" -m src.eval.itm_mechanism_audit \
  --manifest "${OUT}/sta/manifest.json" \
  --output "${OUT}/sta/itm_mechanism_audit.json" \
  --device "${DEVICE}"

"${PYTHON_BIN}" -m src.eval.itm_mechanism_audit \
  --manifest "${OUT}/ink_advection_diffusion/manifest.json" \
  --output "${OUT}/ink_advection_diffusion/itm_mechanism_audit.json" \
  --device "${DEVICE}"

"${PYTHON_BIN}" -m src.eval.audit_evidence \
  --root "${OUT}" \
  --output "${OUT}/evidence_audit.json" \
  --min-seeds "${MIN_SEEDS}" \
  --min-epochs "${MIN_EPOCHS}" \
  --preflight-path "${OUT}/preflight.json"

"${PYTHON_BIN}" -m src.eval.interpret_results \
  --root "${OUT}" \
  --output "${OUT}/result_interpretation.json" \
  --markdown-output "${OUT}/result_interpretation.md" \
  --min-seeds "${MIN_SEEDS}" \
  --min-epochs "${MIN_EPOCHS}" \
  --preflight-path "${OUT}/preflight.json"

if [[ "${FINALIZE_PAPER}" == "1" ]]; then
  PYTHON_BIN="${PYTHON_BIN}" \
  RESULT_ROOT="${OUT}" \
  MIN_SEEDS="${MIN_SEEDS}" \
  MIN_EPOCHS="${MIN_EPOCHS}" \
  PREFLIGHT_OUTPUT="${OUT}/preflight.json" \
    bash experiments/finalize_paper.sh
fi
