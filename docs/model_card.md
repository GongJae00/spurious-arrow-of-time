# Model Card

## final_frame_mlp

Input:

```text
raw mixed final frame only
```

Role:

```text
tests whether the final observation alone leaks enough information
```

## sequence_erm

Input:

```text
raw mixed sequence
```

Role:

```text
ordinary learned sequence predictor
```

## core_only_oracle

Input:

```text
raw core_only sequence
```

Role:

```text
task-causal upper bound under the same neural architecture family
```

## nuisance_only_oracle

Input:

```text
raw nuisance_only sequence
```

Role:

```text
shortcut upper bound
```

## time_reversed_sequence

Input:

```text
raw mixed sequence reversed along the temporal axis
```

Role:

```text
tests dependence on temporal direction
```

## counterfactual_invariance

Input:

```text
raw mixed sequence and raw counterfactual sequence
```

Objective:

```text
CE(model(x), y)
+ CE(model(x_cf), y)
+ prediction-consistency(model(x), model(x_cf))
```

Role:

```text
tests whether nuisance-arrow sensitivity can be reduced
```

## group_invariance_light

Input:

```text
raw mixed sequence
```

Training signal:

```text
train-only nuisance-direction group is used to balance the loss across groups
```

Role:

```text
standard invariant-learning control; not a novelty claim
```

## Forbidden Inputs

```text
source_center
source_index
source_orientation
nuisance_direction
metadata-derived features
diagnostic feature functions
```
