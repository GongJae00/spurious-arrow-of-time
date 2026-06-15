#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
OUT="${OUT:-results/closure_spurious_causality}"
DEVICE="${DEVICE:-auto}"
RUN_TESTS="${RUN_TESTS:-1}"
CLEAN_OUT="${CLEAN_OUT:-1}"
SMOKE="${SMOKE:-0}"
EPOCHS="${EPOCHS:-50}"
SEEDS_STR="${SEEDS:-0 1 2 3 4}"
METHODS_STR="${METHODS:-erm sib sid itm}"

read -r -a SEEDS_ARR <<< "${SEEDS_STR}"
read -r -a METHODS <<< "${METHODS_STR}"

if [[ "${CLEAN_OUT}" == "1" ]]; then
  case "${OUT}" in
    results/closure_spurious_causality|results/closure_spurious_causality/*|results/*closure_spurious_causality*)
      rm -rf -- "${OUT}"
      ;;
    *)
      echo "Refusing to clean OUT='${OUT}'. Use a path under results/*closure_spurious_causality* or set CLEAN_OUT=0." >&2
      exit 2
      ;;
  esac
fi

mkdir -p "${OUT}"

SMOKE_ARGS=()
if [[ "${SMOKE}" == "1" ]]; then
  SMOKE_ARGS=(--smoke)
elif [[ "${SMOKE}" != "0" ]]; then
  echo "SMOKE must be 0 or 1." >&2
  exit 2
fi

if [[ "${RUN_TESTS}" == "1" ]]; then
  env -u OUT -u DEVICE -u RUN_TESTS -u CLEAN_OUT -u SMOKE -u EPOCHS -u SEEDS -u METHODS \
    "${PYTHON_BIN}" -m ruff check .
  env -u OUT -u DEVICE -u RUN_TESTS -u CLEAN_OUT -u SMOKE -u EPOCHS -u SEEDS -u METHODS \
    "${PYTHON_BIN}" -m pytest \
      tests/test_experiment_runner.py \
      tests/test_closure_results.py \
      tests/test_sid_conditional_factor_audit.py \
      -q
fi

"${PYTHON_BIN}" -m src.experiments.run_experiment \
  --config configs/closure_sta.yaml \
  --output-dir "${OUT}/sta" \
  --device "${DEVICE}" \
  --epochs "${EPOCHS}" \
  --seeds "${SEEDS_ARR[@]}" \
  --methods "${METHODS[@]}" \
  "${SMOKE_ARGS[@]}" \
  --overwrite

"${PYTHON_BIN}" -m src.experiments.run_experiment \
  --config configs/closure_ink_advection_diffusion.yaml \
  --output-dir "${OUT}/ink_advection_diffusion" \
  --device "${DEVICE}" \
  --epochs "${EPOCHS}" \
  --seeds "${SEEDS_ARR[@]}" \
  --methods "${METHODS[@]}" \
  "${SMOKE_ARGS[@]}" \
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

"${PYTHON_BIN}" -m src.eval.sid_conditional_factor_audit \
  --result-root "${OUT}" \
  --output "${OUT}/sid_conditional_factor_audit_summary.json" \
  --checkpoint best.pt \
  --benchmark sta \
  --benchmark ink_advection_diffusion \
  --device "${DEVICE}"

"${PYTHON_BIN}" -m src.eval.interpret_closure_results \
  --root "${OUT}" \
  --output "${OUT}/closure_result_interpretation.json" \
  --markdown-output "${OUT}/closure_result_interpretation.md"
