# Latest Main Result Summary

Profile: `real_video_4k`
Primary scenario: `rv_main`
Config: `configs\irreversible_source_extended.yaml`
Command: `python -m src.train.main_experiment --config configs\irreversible_source_extended.yaml --out results\extended\real_video_4k --profile real_video_4k`
Device: `cuda`
Runtime-limited: `False`

## Method Table

| Method | Val IID | IID Test | OOD Test | OOD Gap | Seeds |
|---|---:|---:|---:|---:|---:|
| core_only_oracle | 1.000 | 1.000 | 1.000 | 0.000 | 5 |
| counterfactual_invariance | 0.549 | 0.541 | 0.448 | 0.093 | 5 |
| final_frame_mlp | 0.562 | 0.535 | 0.465 | 0.070 | 5 |
| nuisance_only_oracle | 0.703 | 0.681 | 0.315 | 0.366 | 5 |
| sequence_erm | 0.689 | 0.664 | 0.333 | 0.330 | 5 |

## Main Gap Reduction

Mean ERM-minus-counterfactual OOD-gap reduction: `0.23681640625`
Consistent positive across common seeds: `True`

## Claim Status

Partially supported: ERM shows an OOD gap, but counterfactual training reduces the gap by sacrificing IID accuracy.

This summary separates neural model evidence from diagnostic feature probes.
