# Experiment Runbook

This is the current public execution path.

## Benchmarks

```text
sta -> sta_bench
ink_advection_diffusion -> ink_advection_diffusion
```

No other benchmark is part of the public main path.

The current primary method is `itm`. SIB and SID remain diagnostic selective
baselines.

## Data Diagnostics

Run before model training:

```bash
python -m src.eval.sta_benchmark_diagnostics \
  --config configs/sta_smoke.yaml \
  --output results/diagnostics/sta_benchmark_diagnostics.json

python -m src.eval.ink_advection_diffusion_diagnostics \
  --config configs/ink_advection_diffusion_smoke.yaml \
  --output results/diagnostics/ink_advection_diffusion_diagnostics.json
```

Diagnostics must pass before interpreting model failures.

## Smoke Suite

```bash
PYTHON_BIN=python \
OUT=results/smoke_run \
DEVICE=cpu \
RUN_TESTS=1 \
CLEAN_OUT=1 \
bash experiments/smoke_suite.sh
```

Smoke output:

```text
results/smoke_run/sta/manifest.json
results/smoke_run/ink_advection_diffusion/manifest.json
results/smoke_run/smoke_audit.json
results/smoke_run/result_interpretation.json
```

Smoke success checks code plumbing only.

Expected smoke interpretation:

```text
smoke_audit.json passes
result_interpretation.json has positive_primary_claim_allowed=false
```

## Full Suite

Preflight:

```bash
PYTHON_BIN=python \
OUT=results/full_run \
DEVICE=cuda \
REQUIRE_CUDA_FOR_FULL_RUN=1 \
EPOCHS=50 \
SEEDS="0 1 2 3 4" \
MIN_SEEDS=5 \
MIN_EPOCHS=25 \
RUN_TESTS=1 \
CLEAN_OUT=1 \
bash experiments/preflight.sh
```

Full run:

```bash
PYTHON_BIN=python \
OUT=results/full_run \
DEVICE=cuda \
REQUIRE_CUDA_FOR_FULL_RUN=1 \
EPOCHS=50 \
SEEDS="0 1 2 3 4" \
MIN_SEEDS=5 \
MIN_EPOCHS=25 \
RUN_TESTS=1 \
CLEAN_OUT=1 \
bash experiments/full_suite.sh
```

Full output:

```text
results/full_run/preflight.json
results/full_run/diagnostics/sta_benchmark_diagnostics.json
results/full_run/diagnostics/ink_advection_diffusion_diagnostics.json
results/full_run/sta/aggregate.json
results/full_run/ink_advection_diffusion/aggregate.json
results/full_run/evidence_audit.json
results/full_run/result_interpretation.json
```

`evidence_audit.json` checks protocol validity.
`result_interpretation.json` controls claim language. Positive method
wording is allowed only if `positive_primary_claim_allowed=true`.

CPU maintenance preflight is allowed for source checks when GPU hardware is not
available, but final evidence runs should keep `REQUIRE_CUDA_FOR_FULL_RUN=1`.

When regenerating paper assets from a completed full run, use:

```bash
PREFLIGHT_OUTPUT=results/full_run/preflight.json
```

## Closure Suite

Purpose:

```text
Test whether OOD failure is tied to a dynamic spurious-arrow shift.
```

Run:

```bash
PYTHON_BIN=python \
OUT=results/closure_spurious_causality \
DEVICE=cuda \
RUN_TESTS=1 \
CLEAN_OUT=1 \
SMOKE=0 \
EPOCHS=50 \
SEEDS="0 1 2 3 4" \
METHODS="erm sib sid itm" \
bash experiments/21_closure_spurious_causality_suite.sh
```

Closure output:

```text
results/closure_spurious_causality/sta/aggregate.json
results/closure_spurious_causality/ink_advection_diffusion/aggregate.json
results/closure_spurious_causality/sid_conditional_factor_audit_summary.json
results/closure_spurious_causality/closure_result_interpretation.json
```

Closure evidence can support a failure-mode discussion. It cannot by itself
support a clean factorization claim.
