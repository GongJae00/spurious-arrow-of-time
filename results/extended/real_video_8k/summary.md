# Latest Main Result Summary

Profile: `real_video_8k`
Primary scenario: `rv_main`
Config: `configs\irreversible_source_extended.yaml`
Command: `python -m src.train.main_experiment --config configs\irreversible_source_extended.yaml --out results\extended\real_video_8k --profile real_video_8k`
Device: `cuda`
Runtime-limited: `False`

## Method Table

| Method | Val IID | IID Test | OOD Test | OOD Gap | Seeds |
|---|---:|---:|---:|---:|---:|
| core_only_oracle | 1.000 | 1.000 | 1.000 | 0.000 | 5 |
| counterfactual_invariance | 0.807 | 0.802 | 0.793 | 0.009 | 5 |
| final_frame_mlp | 0.562 | 0.536 | 0.457 | 0.079 | 5 |
| nuisance_only_oracle | 0.768 | 0.772 | 0.234 | 0.538 | 5 |
| sequence_erm | 0.900 | 0.896 | 0.697 | 0.199 | 5 |

## Main Gap Reduction

Mean ERM-minus-counterfactual OOD-gap reduction: `0.18994140625`
Consistent positive across common seeds: `False`

## Claim Status

Partially supported: neural ERM did not show the target OOD gap.

This summary separates neural model evidence from diagnostic feature probes.
