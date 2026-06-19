# Table Plan

## Table 1: Benchmark Gates

Source:

```text
diagnostics.json
```

Rows:

```text
final-frame core oracle
full-sequence core oracle
core-only OOD
nuisance-only IID/OOD
mixed feature-probe IID/OOD
static feature accuracy
arrow probe accuracies
counterfactual residual
```

## Table 2: Main Neural Results

Source:

```text
metrics.jsonl
summary.json
```

Columns:

```text
method
val_iid_accuracy
iid_test_accuracy
ood_test_accuracy
ood_gap
number_of_seeds
```

Feature-probe rows must not appear in this table.

## Table 3: Sweep Summary

Rows:

```text
OOD mode
nuisance scale
core difficulty
negative controls
```

Report actual measured correlations, not only configured values.

Minimum scenario groups:

```text
main_reversed
ood_randomized
ood_partial_shift
nuisance_scale_low
nuisance_scale_mid
core_difficulty_easy
core_difficulty_hard
no_spurious_correlation
core_only_no_nuisance
random_labels if full profile is run
```
