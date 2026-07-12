# Latest Main Result Summary

Profile: `scenario_audit`
Primary scenario: `main_spurious_arrow`
Config: `configs\irreversible_source_extended.yaml`
Command: `python -m src.train.main_experiment --config configs\irreversible_source_extended.yaml --out results\extended\scenario_audit --profile scenario_audit`
Device: `cuda`
Runtime-limited: `False`

## Method Table

| Method | Val IID | IID Test | OOD Test | OOD Gap | Seeds |
|---|---:|---:|---:|---:|---:|
| counterfactual_invariance | 0.943 | 0.943 | 0.757 | 0.186 | 10 |
| final_frame_mlp | 0.516 | 0.504 | 0.501 | 0.003 | 10 |
| nuisance_only_oracle | 0.969 | 0.969 | 0.031 | 0.938 | 10 |
| sequence_erm | 0.969 | 0.969 | 0.031 | 0.938 | 10 |

## Main Gap Reduction

Mean ERM-minus-counterfactual OOD-gap reduction: `0.7515869140625`
Consistent positive across common seeds: `False`

## Claim Status

Phenomenon supported: ERM shows an OOD gap. Counterfactual invariance improves mean OOD but is not seed-stable enough for a primary method-success claim.

This summary separates neural model evidence from diagnostic feature probes.
