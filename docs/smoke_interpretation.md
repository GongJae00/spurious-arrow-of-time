# Smoke Interpretation

Smoke command:

```bash
PYENV_VERSION=rppg-310 PYTHON_BIN=python bash experiments/smoke_benchmark.sh
```

Artifact directory:

```text
results/smoke_benchmark/
```

## Current Decision

```text
benchmark gate passed: true
benchmark_variant: endpoint_matched
observation_layout: two_channel
```

This is a benchmark construction check. It is not a neural method result.

## Latest Smoke Gate

| Gate | Value | Interpretation |
|---|---:|---|
| final-frame core oracle accuracy | 0.695 | Final core frame has some signal, but is below the ambiguity threshold. |
| full-sequence core oracle accuracy | 1.000 | The core trajectory contains recoverable source information. |
| core-only OOD accuracy | 1.000 | The task-causal signal is stable under nuisance shift. |
| nuisance-only IID accuracy | 0.949 | The nuisance arrow is a strong IID shortcut. |
| nuisance-only OOD accuracy | 0.020 | The nuisance shortcut breaks under reversed OOD. |
| mixed feature-probe OOD gap | 0.930 | Diagnostic visible features are strongly misled by the nuisance arrow. |
| static feature accuracy | 0.547 | Static mixed features do not solve the task. |
| core forward/reverse arrow accuracy | 1.000 | The core process is temporally asymmetric. |
| nuisance forward/reverse arrow accuracy | 1.000 | The nuisance process has a detectable temporal arrow. |
| final nuisance frame IID accuracy | 0.445 | Endpoint-matched final nuisance frame is not label-predictive. |
| final nuisance frame OOD gap | -0.086 | Final nuisance frame does not explain the OOD gap. |

## What This Supports

The smoke run supports the benchmark-level statement:

```text
The generator creates a recoverable causal core, a stronger non-causal nuisance
arrow, and an endpoint-matched observation where final-frame nuisance leakage is
controlled.
```

The final-frame core oracle should be read as a non-trivial ambiguity check, not
as proof that the final frame contains no core information. The decisive static
leakage audit for the main claim is the endpoint-matched neural result:
`final_frame_mlp` stays near chance while `sequence_erm` collapses OOD.

## What This Does Not Support

Do not claim from smoke alone:

```text
neural ERM follows the nuisance arrow
any robustness method works
the final paper result is complete
learned scores are physical entropy production
```

Use the non-runtime-limited `main` or `full` profile for neural evidence; smoke
remains a benchmark-construction check.
