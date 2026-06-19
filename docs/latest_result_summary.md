# Latest Result Summary

This tracked document records the current evidence state. Generated run
summaries under `results/` remain the raw artifacts.

## Primary Main Evidence

```text
profile: main
source: results/main_experiments/main/
scenario: main_reversed
seeds: 5
runtime_limited: false
```

| Method | IID Test | OOD Test | OOD Gap | Claim Role |
|---|---:|---:|---:|---|
| core_only_oracle | 1.000 | 1.000 | 0.000 | task-causal upper bound |
| sequence_erm | 0.968 | 0.032 | 0.936 | main neural shortcut evidence |
| nuisance_only_oracle | 0.968 | 0.032 | 0.936 | shortcut upper bound |
| final_frame_mlp | 0.968 | 0.032 | 0.936 | static/residue risk audit |
| time_reversed_sequence | 0.968 | 0.032 | 0.936 | temporal-direction diagnostic |
| group_invariance_light | 0.968 | 0.032 | 0.936 | comparison failure |
| counterfactual_invariance | 0.533 | 0.465 | 0.068 | method tradeoff/failure |

Claim status:

```text
Supported:
  Raw neural ERM can learn the non-causal nuisance arrow and collapse under
  reversed OOD shift in this controlled benchmark.

Not supported:
  Counterfactual invariance as implemented is a successful robust method.
```

## Sweep Diagnostic

```text
profile: sweep_pilot
source: results/main_experiments/sweep_pilot_goal/
seeds: 3
runtime_limited: true
```

This sweep includes:

```text
reversed OOD
randomized OOD
partial-shift OOD
nuisance-scale low/mid
core-difficulty easy/hard
no-spurious-correlation
core-only-no-nuisance
```

The sweep is diagnostic only. It shows that OOD gap size changes with OOD shift
mode, while the benchmark remains strongly shortcut-dominated across the pilot
nuisance and core-difficulty settings.

Generated sweep figures:

```text
results/main_experiments/sweep_pilot_goal/scenario_ood_gap_heatmap.png
results/main_experiments/sweep_pilot_goal/scenario_ood_accuracy_heatmap.png
```

## Validation

```text
pytest: 11 passed, 1 warning
smoke benchmark: passed
main experiment smoke: passed
main profile: passed
```

Use `docs/main_result_interpretation.md`, `docs/static_leakage_audit.md`, and
`docs/paper_handoff.md` before writing claims.
