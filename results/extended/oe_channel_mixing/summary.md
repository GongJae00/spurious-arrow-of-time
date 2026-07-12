# Latest Main Result Summary

Profile: `oe_channel_mixing`
Primary scenario: `oe_additive`
Config: `configs\irreversible_source_extended.yaml`
Command: `python -m src.train.main_experiment --config configs\irreversible_source_extended.yaml --out results\extended\oe_channel_mixing --profile oe_channel_mixing`
Device: `cuda`
Runtime-limited: `False`

## Method Table

| Method | Val IID | IID Test | OOD Test | OOD Gap | Seeds |
|---|---:|---:|---:|---:|---:|
| counterfactual_invariance | 0.772 | 0.772 | 0.229 | 0.543 | 10 |
| final_frame_mlp | 0.515 | 0.501 | 0.499 | 0.002 | 10 |
| sequence_erm | 0.969 | 0.969 | 0.031 | 0.938 | 10 |

## Main Gap Reduction

Mean ERM-minus-counterfactual OOD-gap reduction: `0.3947509765625`
Consistent positive across common seeds: `False`

## Claim Status

Partially supported: ERM shows an OOD gap, but counterfactual training reduces the gap by sacrificing IID accuracy.

This summary separates neural model evidence from diagnostic feature probes.
