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
x_t = core_scale * core_t + nuisance_scale * nuisance_t + noise_t
```

Counterfactual:

```text
x_cf keeps y, source c, and core_t fixed
x_cf resamples or reverses nuisance_t
```

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

Do not claim method success until this benchmark passes its gates and later
full experiments are run.
