# Latest Main Result Summary

Profile: `channel_mixing`
Primary scenario: `additive_single_channel`
Config: `configs\irreversible_source_extended.yaml`
Command: `python -m src.train.main_experiment --config configs\irreversible_source_extended.yaml --out results\extended\channel_mixing --profile channel_mixing`
Device: `cuda`
Runtime-limited: `False`

## Method Table

| Method | Val IID | IID Test | OOD Test | OOD Gap | Seeds |
|---|---:|---:|---:|---:|---:|
| core_only_oracle | 1.000 | 1.000 | 1.000 | 0.000 | 10 |
| counterfactual_invariance | 0.959 | 0.960 | 0.640 | 0.320 | 10 |
| final_frame_mlp | 0.515 | 0.501 | 0.504 | -0.003 | 10 |
| nuisance_only_oracle | 0.969 | 0.969 | 0.031 | 0.938 | 10 |
| sequence_erm | 0.969 | 0.969 | 0.031 | 0.938 | 10 |

## Main Gap Reduction

Mean ERM-minus-counterfactual OOD-gap reduction: `0.6176513671875`
Consistent positive across common seeds: `False`

## Claim Status

Phenomenon supported: ERM shows an OOD gap. Counterfactual invariance improves mean OOD but is not seed-stable enough for a primary method-success claim.

This summary separates neural model evidence from diagnostic feature probes.
