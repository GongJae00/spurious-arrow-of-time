# Latest Main Result Summary

Profile: `nuisance_scale_sweep`
Primary scenario: `scale_2p8`
Config: `configs\irreversible_source_extended.yaml`
Command: `python -m src.train.main_experiment --config configs\irreversible_source_extended.yaml --out results\extended\nuisance_scale_sweep --profile nuisance_scale_sweep`
Device: `cuda`
Runtime-limited: `False`

## Method Table

| Method | Val IID | IID Test | OOD Test | OOD Gap | Seeds |
|---|---:|---:|---:|---:|---:|
| core_only_oracle | 1.000 | 1.000 | 1.000 | 0.000 | 10 |
| counterfactual_invariance | 0.978 | 0.978 | 0.321 | 0.656 | 10 |
| final_frame_mlp | 0.514 | 0.501 | 0.504 | -0.002 | 10 |
| nuisance_only_oracle | 0.969 | 0.969 | 0.031 | 0.938 | 10 |
| sequence_erm | 0.969 | 0.969 | 0.031 | 0.938 | 10 |

## Main Gap Reduction

Mean ERM-minus-counterfactual OOD-gap reduction: `0.281396484375`
Consistent positive across common seeds: `False`

## Claim Status

Phenomenon supported: ERM shows an OOD gap. Counterfactual invariance improves mean OOD but is not seed-stable enough for a primary method-success claim.

This summary separates neural model evidence from diagnostic feature probes.
