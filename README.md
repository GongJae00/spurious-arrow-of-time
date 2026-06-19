# Spurious Arrow of Time

This repository studies a narrow sequence-learning question:

```text
When the real task is to infer the hidden cause of an irreversible process,
can a model be misled by a stronger but non-causal arrow of time?
```

The project is organized around a clean benchmark and a small set of raw neural
sequence baselines. Older exploratory method stacks are not part of the active
public surface.

## Intuition

Ink spreads. Heat diffuses. A broken object does not usually reassemble itself.
Forward dynamics can be easy to recognize while the inverse question is hard:

```text
What was the hidden cause before the irreversible process washed information
away?
```

The active hypothesis is that a model can learn the wrong arrow when a separate
irreversible nuisance process is more visible and happens to correlate with the
label during training.

## Current Benchmark

The current smoke benchmark is:

```text
irreversible_source_inference
```

Each sample contains:

```text
core:
  a hidden horizontal or vertical source pattern on a grid
  isotropic diffusion that gradually erases source structure
  label y = source pattern type

nuisance:
  an independent moving pulse with a direction-sensitive arrow
  direction correlated with y in train/IID
  direction reversed in OOD

observation:
  core + nuisance + noise
```

The benchmark is accepted only if final-frame core recovery is weak, full-core
sequence recovery is strong, nuisance-only prediction fails OOD, and diagnostic
features show the shortcut is dynamic. Diagnostic features are not counted as
neural sequence-model results.

## Current Evidence

The latest non-runtime-limited main run uses five seeds in the reversed-OOD
setting.

```text
core_only_oracle:      IID 1.000, OOD 1.000
sequence_erm:          IID 0.968, OOD 0.032
nuisance_only_oracle:  IID 0.968, OOD 0.032
counterfactual control: IID 0.533, OOD 0.465
```

The main supported result is the benchmark phenomenon: a raw sequence model can
learn the non-causal nuisance arrow and fail almost completely when that arrow
reverses. The simple counterfactual control reduces the OOD gap by sacrificing
IID accuracy, so it is reported as a method failure, not as a solved robustness
method.

## Install

```bash
pip install -e ".[dev]"
```

## Smoke Benchmark

```bash
bash experiments/smoke_benchmark.sh
```

Outputs:

```text
results/smoke_benchmark/benchmark_gate.json
results/smoke_benchmark/diagnostics.json
results/smoke_benchmark/candidate_trials.jsonl
results/smoke_benchmark/smoke_report.md
results/smoke_benchmark/problem_schematic.png
results/smoke_benchmark/component_visualization.png
results/smoke_benchmark/source_recoverability.png
```

## Tests

```bash
python -m pytest -q
```

## Main Experiment

```bash
PROFILE=main OUT=results/main_experiments/main bash experiments/main_experiments.sh
```

Generated experiment artifacts live under ignored `results/`. Lightweight
interpretation documents live under `docs/`.

## Read First

```text
RESEARCH.md
goal.md
docs/literature_matrix.md
docs/novelty_gap.md
docs/benchmark_design_decision.md
docs/benchmark_acceptance_criteria.md
docs/main_result_interpretation.md
docs/static_leakage_audit.md
docs/paper_handoff.md
```

## Scientific Boundaries

This repository does not measure physical heat dissipation and does not claim
that learned scores are exact entropy production.

The current claim is benchmark-focused. Method success should not be claimed
unless a method improves OOD robustness without destroying IID performance.
