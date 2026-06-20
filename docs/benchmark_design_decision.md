# Benchmark Design Decision

## Chosen Main Smoke Benchmark

Use a grid-graph diffusion source inference benchmark.

Working name:

```text
irreversible_source_inference
```

## Reason

This choice is the cleanest reset target because it is:

```text
grounded in inverse diffusion and graph source localization
simple enough for exact data generation
fast enough for CPU smoke tests
easy to visualize
easy to audit for leakage
easy to construct core-only, nuisance-only, mixed, and counterfactual views
```

## Mathematical Specification

Core process:

```text
y in {0, 1}
source center c sampled uniformly on a 2D grid, independent of y
u_0 = a local source pattern centered at c
  y = 0: horizontal two-lobed pattern
  y = 1: vertical two-lobed pattern
u_{k+1} = (1 - alpha) u_k + alpha P u_k
observed core frames start after a diffusion delay and are sampled every
configured number of diffusion steps
```

`P` is a local grid transition operator. The process is irreversible as an
observed diffusion process: forward evolution spreads source information and the
late state loses sharp information about the initial source pattern.

Label:

```text
y = source pattern orientation
```

The label is always computed from the core source, never from the nuisance.

Nuisance process:

```text
a_s in {-1, +1}
train/IID: a_s correlated with y
OOD: a_s reversed or randomized
n_t = traveling pulse with a temporal trail and direction a_s
```

Observation:

```text
x_t has two channels in the active main benchmark:
  channel 0: core_scale * core_t + noise
  channel 1: nuisance_scale * nuisance_t + noise
```

The older additive observation is retained only as a diagnostic option. The
active two-channel layout avoids turning the problem into pixel-level occlusion:
when the nuisance is label-independent, a sequence model can learn the core.

Counterfactual:

```text
x_cf keeps y, source c, and core_t fixed
x_cf resamples or reverses nuisance_t
```

Active main variant:

```text
benchmark_variant: endpoint_matched
```

The nuisance keeps a directed trail over the sequence but its final endpoint
frame is label-independent. This makes the final-frame MLP audit meaningful:
if the sequence model collapses while the final-frame model stays near chance,
the shortcut is not explained by final endpoint residue alone.

## Required Diagnostics

The benchmark is not accepted until the smoke script writes:

```text
results/smoke_benchmark/benchmark_gate.json
results/smoke_benchmark/diagnostics.json
results/smoke_benchmark/candidate_trials.jsonl
results/smoke_benchmark/smoke_report.md
results/smoke_benchmark/component_visualization.png
results/smoke_benchmark/source_recoverability.png
```

## Explicit Non-Claims

This benchmark does not prove physical entropy production. It creates a
controlled irreversible inverse task and a controlled non-causal dynamic
shortcut.

Do not claim method success unless the benchmark gates and the mitigation-method
gates both pass.

Current full-run status:

```text
phenomenon gates: passed
counterfactual mitigation seed-stability gate: failed
```

Therefore the manuscript may claim the endpoint-matched spurious-arrow
phenomenon, but it must present simple counterfactual invariance as a partial
mitigation diagnostic rather than as a solved method. The synthetic
counterfactual assumption and seed-level variance must be reported.
