# Benchmark Acceptance Criteria

The benchmark is accepted only if all gates can be evaluated from generated
artifacts. The gates are not method claims.

## Required Splits

```text
train: nuisance arrow correlated with y
val_iid: same nuisance rule as train, for smoke diagnostics
iid_test: same nuisance rule as train, final IID report
ood_test: nuisance arrow reversed or randomized
```

No OOD metric may be used to choose model hyperparameters in later full runs.

## Required Fields

Every split must expose:

```text
core_only
nuisance_only
mixed
counterfactual
y
source_index
source_center
source_orientation
nuisance_direction
split
metadata
```

`counterfactual` must preserve the source and label while changing the nuisance
direction or instance.

## Gate Metrics

| Metric | Initial smoke target | Reason |
|---|---:|---|
| final_frame_core_oracle_accuracy | <= 0.65 | The final diffused state alone should not trivially reveal the source label. |
| full_sequence_core_oracle_accuracy | >= 0.80 | The core sequence should still contain recoverable source information. |
| core_only_ood_accuracy | >= 0.80 | The task-causal process must remain valid OOD. |
| nuisance_only_iid_accuracy | >= 0.80 | The nuisance arrow must be a tempting shortcut in IID. |
| nuisance_only_ood_accuracy | <= 0.40 for reversed OOD | The nuisance shortcut must break OOD. |
| mixed_feature_probe_ood_gap | >= 0.20 | A mixed diagnostic probe using visible dynamic features should be measurably misled. This is not yet a neural ERM result. |
| abs(corr_y_nuisance_arrow_train) | >= 0.70 | Train nuisance direction must be label-correlated. |
| abs(corr_y_nuisance_arrow_ood) | >= 0.70 with opposite sign, or near 0 if randomized | OOD must reverse or remove shortcut. |
| static_feature_accuracy | <= 0.65 | Static leaks should not solve the task. |
| core_forward_reverse_arrow_accuracy | >= 0.80 | The task-causal core process should contain detectable temporal irreversibility. |
| nuisance_forward_reverse_arrow_accuracy | >= 0.80 | The nuisance must be a real spurious arrow, not merely a left/right moving object. |
| counterfactual_preserves_core | true | Counterfactual must not change the true task. |
| counterfactual_changes_nuisance | true | Counterfactual must actually break the nuisance instance. |

## Stop Rules

Stop before method experiments if:

```text
final-frame core oracle is high
full-sequence core oracle is low
nuisance-only IID is weak
nuisance-only OOD remains high under reversed shift
mixed feature probe does not show an OOD gap
visualization cannot explain the task
```

Do not rename a failed benchmark as a successful method result.
