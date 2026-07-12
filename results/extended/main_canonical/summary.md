# Latest Main Result Summary

Profile: `main_canonical`
Primary scenario: `main_spurious_arrow`
Config: `configs\irreversible_source_extended.yaml`
Command: `python -m src.train.main_experiment --config configs\irreversible_source_extended.yaml --out results\extended\main_canonical --profile main_canonical`
Device: `cuda`
Runtime-limited: `False`

## Method Table

| Method | Val IID | IID Test | OOD Test | OOD Gap | Seeds |
|---|---:|---:|---:|---:|---:|
| core_only_oracle | 1.000 | 1.000 | 1.000 | 0.000 | 30 |
| counterfactual_invariance | 0.980 | 0.980 | 0.886 | 0.094 | 30 |
| final_frame_mlp | 0.515 | 0.502 | 0.501 | 0.001 | 30 |
| group_invariance_light | 0.971 | 0.971 | 0.063 | 0.908 | 30 |
| nuisance_channel_dropout | 0.989 | 0.988 | 0.640 | 0.349 | 30 |
| nuisance_only_oracle | 0.970 | 0.970 | 0.031 | 0.939 | 30 |
| sequence_erm | 0.971 | 0.971 | 0.061 | 0.910 | 30 |

## Main Gap Reduction

Mean ERM-minus-counterfactual OOD-gap reduction: `0.8161295572916667`
Consistent positive across common seeds: `False`

## Claim Status

Supported in this controlled benchmark: ERM shows an OOD gap and counterfactual invariance reduces it with seed-level stability.

This summary separates neural model evidence from diagnostic feature probes.
