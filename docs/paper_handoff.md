# Paper Handoff

This is the current manuscript handoff after the benchmark redesign and
ten-seed full run.

## Working Title

```text
Spurious Arrows of Time in Irreversible Inverse Inference
```

## Core Question

```text
When the true task is to infer the hidden cause of an irreversible process,
what happens if a stronger but non-causal irreversible process is also visible?
```

## Active Experimental Claim

```text
A sequence model can rely on a non-causal nuisance arrow instead of a
recoverable task-causal core when that nuisance arrow is correlated with the
label.
```

## Main Evidence

The non-runtime-limited `full` profile passes the phenomenon evidence gates.
The counterfactual mitigation gate fails seed stability and should not be
described as a solved method:

```text
profile: full
seeds: 10
runtime_limited: false
source: results/main_experiments/full/
```

Main endpoint-matched result:

```text
final_frame_mlp:          IID 0.501, OOD 0.500
sequence_erm:             IID 0.971, OOD 0.124
nuisance_only_oracle:     IID 0.969, OOD 0.031
core_only_oracle:         IID 1.000, OOD 1.000
counterfactual_invariance: IID 0.943, OOD 0.756
counterfactual OOD-threshold seeds: 7/10
```

Required controls:

```text
no_spurious_correlation:
  sequence_erm: IID 0.898, OOD 0.899, 8/10 seed-level success

residue_visible_control:
  final_frame_mlp: IID 0.969, OOD 0.031
```

## Manuscript Contributions

```text
1. A controlled irreversible inverse inference benchmark grounded in diffusion
   and source inference.
2. A spurious-arrow setting where the causal core is learnable without spurious
   correlation, but a standard sequence model follows the nuisance arrow when it
   is correlated.
3. An endpoint-matched audit separating sequence-level spurious-arrow reliance
   from final-frame nuisance leakage.
4. A residue-visible control showing why endpoint control matters.
5. A partial synthetic counterfactual mitigation result: it improves mean OOD
   robustness but fails the seed-stability gate, so it should not be presented
   as a solved method.
```

## Result Artifacts

Use:

```text
results/main_experiments/full/final_gate_audit.md
results/main_experiments/full/method_table.md
results/main_experiments/full/scenario_table.md
results/main_experiments/full/main_results_bars.png
results/main_experiments/full/scenario_ood_gap_heatmap.png
```

## Non-Claims

Do not claim:

```text
physical entropy production is measured
all static cues are impossible
the counterfactual result directly transfers to real data
the result is universal across all sequence architectures
```
