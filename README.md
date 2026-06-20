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
  two channels: diffused core channel and nuisance-arrow channel
  endpoint-matched main variant controls final-frame nuisance leakage
```

The benchmark is accepted only if the core sequence is learnable when the
nuisance is label-independent, the nuisance-only predictor fails OOD when the
nuisance arrow reverses, and the endpoint-matched main variant keeps final-frame
nuisance leakage near chance. Diagnostic features are not counted as neural
sequence-model results.

## Current Evidence

The latest non-runtime-limited full run uses ten seeds. The phenomenon gates
pass; the counterfactual mitigation gate does not pass the seed-stability
criterion.

```text
main_spurious_arrow, endpoint-matched:
  final_frame_mlp:       IID 0.501, OOD 0.500
  sequence_erm:          IID 0.971, OOD 0.124
  nuisance_only_oracle:  IID 0.969, OOD 0.031
  core_only_oracle:      IID 1.000, OOD 1.000
  counterfactual control: IID 0.943, OOD 0.756

no_spurious_correlation:
  sequence_erm:          IID 0.898, OOD 0.899
  seed success rate:     8/10

residue_visible_control:
  final_frame_mlp:       IID 0.969, OOD 0.031
```

This supports the central controlled-benchmark claim: the mixed observation is
learnable from the causal core when the nuisance is independent, but the same
sequence model follows the non-causal nuisance arrow when it is label-correlated.
The counterfactual result should be reported as a mitigation diagnostic, not a
primary solved method: it improves the mean OOD score but reaches the OOD
success threshold in only 7/10 full seeds and assumes valid generator-provided
nuisance counterfactuals.

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

Full ten-seed profile:

```bash
PROFILE=full OUT=results/main_experiments/full bash experiments/main_experiments.sh
```

Pre-flight gate pilot:

```bash
PROFILE=gate_pilot OUT=results/main_experiments/gate_pilot bash experiments/main_experiments.sh
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
