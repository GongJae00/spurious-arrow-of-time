# Main Result Interpretation

This document interprets the completed non-runtime-limited main run and the
runtime-limited sweep pilot. It separates benchmark evidence from method
success.

## Completed Main Run

```text
profile: main
scenario: main_reversed
seeds: 5
runtime_limited: false
source: results/main_experiments/main/
```

Main table:

```text
core_only_oracle:
  IID = 1.000
  OOD = 1.000
  gap = 0.000

sequence_erm:
  IID = 0.968
  OOD = 0.032
  gap = 0.936

nuisance_only_oracle:
  IID = 0.968
  OOD = 0.032
  gap = 0.936

final_frame_mlp:
  IID = 0.968
  OOD = 0.032
  gap = 0.936

time_reversed_sequence:
  IID = 0.968
  OOD = 0.032
  gap = 0.936

group_invariance_light:
  IID = 0.968
  OOD = 0.032
  gap = 0.936

counterfactual_invariance:
  IID = 0.533
  OOD = 0.465
  gap = 0.068
```

## Supported

The benchmark phenomenon is strongly observed in the main reversed-OOD setting:

```text
sequence_erm learns a high-IID predictor that collapses under OOD reversal.
nuisance_only_oracle matches the same IID/OOD pattern.
core_only_oracle remains high on both IID and OOD.
```

This supports the claim that a stronger non-causal irreversible nuisance process
can dominate a learned sequence predictor in this controlled setting.

## Not Supported

The simple counterfactual invariance control is not a successful robustness
method in this run:

```text
it reduces the OOD gap
but it also drops IID performance close to chance
```

This must be reported as a method failure or tradeoff, not as method success.

## High-Risk Observation

`final_frame_mlp` behaves like `sequence_erm` and `nuisance_only_oracle`.

Interpretation risk:

```text
The nuisance process may leave a visible endpoint or static residue strong
enough for a final-frame model to exploit.
```

This does not automatically invalidate the benchmark, but it weakens any claim
that the failure is purely due to sequence-temporal reasoning. The paper must
frame the shortcut as an irreversible nuisance process with visible trajectory
residue unless a stricter static-leak audit says otherwise.

## Sweep Pilot

```text
profile: sweep_pilot
seeds: 3
runtime_limited: true
source: results/main_experiments/sweep_pilot_goal/
```

The sweep pilot is diagnostic only. It suggests:

```text
reversed OOD produces the largest ERM gap
randomized OOD produces an intermediate gap
partial_shift produces an intermediate gap
low and mid nuisance-scale settings still produce large gaps in this pilot
easy and hard core-difficulty settings still leave sequence ERM dominated by
the nuisance shortcut
no_spurious_correlation removes the large OOD gap but raw mixed ERM does not
recover the core well in the pilot budget
core_only_no_nuisance improves sequence ERM but remains below the core oracle in
the pilot budget
```

These findings should guide discussion and further checks, but they should not
be presented as full-scale camera-ready sweep evidence unless rerun at full
scale.

The difficulty sweep is especially important for wording:

```text
Changing core difficulty alone does not rescue sequence ERM in the pilot.
This supports the benchmark stress-test interpretation, but it also warns that
the mixed observation is strongly shortcut-dominated.
```

## Claim Wording

Allowed:

```text
In a controlled irreversible inverse benchmark, standard raw sequence models can
prefer a non-causal nuisance arrow over the task-causal source process.
```

Not allowed:

```text
Counterfactual invariance solves the problem.
The model failure is proven to be purely temporal rather than endpoint leakage.
The result establishes a universal law of neural networks.
```
