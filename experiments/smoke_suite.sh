#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
OUT="${OUT:-results/smoke_run}"
DEVICE="${DEVICE:-cpu}"
RUN_TESTS="${RUN_TESTS:-1}"
CLEAN_OUT="${CLEAN_OUT:-1}"

METHODS_STR="${METHODS:-erm ib ep_min ep_max ocp_style lens_like_arrow_classifier sib sid itm}"
read -r -a METHODS <<< "${METHODS_STR}"

if [[ "${CLEAN_OUT}" == "1" ]]; then
  case "${OUT}" in
    results/smoke_run|results/smoke_run/*|results/*smoke_run*)
      rm -rf -- "${OUT}"
      ;;
    *)
      echo "Refusing to clean OUT='${OUT}'. Use a path under results/*smoke_run* or set CLEAN_OUT=0." >&2
      exit 2
      ;;
  esac
fi

mkdir -p "${OUT}/diagnostics"

if [[ "${RUN_TESTS}" == "1" ]]; then
  env -u OUT -u DEVICE -u RUN_TESTS -u CLEAN_OUT "${PYTHON_BIN}" -m ruff check .
  env -u OUT -u DEVICE -u RUN_TESTS -u CLEAN_OUT "${PYTHON_BIN}" -m pytest \
    tests/test_ink_advection_diffusion.py \
    tests/test_sid.py \
    tests/test_train_common.py \
    tests/test_experiment_runner.py \
    -q
fi

"${PYTHON_BIN}" -m src.eval.sta_benchmark_diagnostics \
  --config configs/sta_smoke.yaml \
  --output "${OUT}/diagnostics/sta_benchmark_diagnostics.json" >/dev/null

"${PYTHON_BIN}" -m src.eval.ink_advection_diffusion_diagnostics \
  --config configs/ink_advection_diffusion_smoke.yaml \
  --output "${OUT}/diagnostics/ink_advection_diffusion_diagnostics.json" >/dev/null

"${PYTHON_BIN}" -m src.experiments.run_experiment \
  --config configs/sta_smoke.yaml \
  --output-dir "${OUT}/sta" \
  --device "${DEVICE}" \
  --methods "${METHODS[@]}" \
  --overwrite

"${PYTHON_BIN}" -m src.experiments.run_experiment \
  --config configs/ink_advection_diffusion_smoke.yaml \
  --output-dir "${OUT}/ink_advection_diffusion" \
  --device "${DEVICE}" \
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

"${PYTHON_BIN}" -m src.eval.audit_smoke \
  --root "${OUT}" \
  --output "${OUT}/smoke_audit.json"

"${PYTHON_BIN}" -m src.eval.interpret_results \
  --root "${OUT}" \
  --output "${OUT}/result_interpretation.json" \
  --markdown-output "${OUT}/result_interpretation.md" \
  --min-seeds 1 \
  --min-epochs 1 \
  --allow-smoke
