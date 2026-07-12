# Latest Main Result Summary

Profile: `corr_sweep`
Primary scenario: `corr_0p97`
Config: `configs\irreversible_source_extended.yaml`
Command: `python -m src.train.main_experiment --config configs\irreversible_source_extended.yaml --out results\extended\corr_sweep --profile corr_sweep`
Device: `cuda`
Runtime-limited: `False`

## Method Table

| Method | Val IID | IID Test | OOD Test | OOD Gap | Seeds |
|---|---:|---:|---:|---:|---:|
| core_only_oracle | 1.000 | 1.000 | 1.000 | 0.000 | 10 |
| counterfactual_invariance | 0.986 | 0.985 | 0.715 | 0.270 | 10 |
| final_frame_mlp | 0.520 | 0.501 | 0.507 | -0.006 | 10 |
| nuisance_only_oracle | 0.969 | 0.969 | 0.031 | 0.938 | 10 |
| sequence_erm | 0.969 | 0.969 | 0.031 | 0.938 | 10 |

## Main Gap Reduction

Mean ERM-minus-counterfactual OOD-gap reduction: `0.66748046875`
Consistent positive across common seeds: `False`

## Claim Status

Phenomenon supported: ERM shows an OOD gap. Counterfactual invariance improves mean OOD but is not seed-stable enough for a primary method-success claim.

This summary separates neural model evidence from diagnostic feature probes.
