# Latest Main Result Summary

Profile: `order_encoded`
Primary scenario: `oe_main`
Config: `configs\irreversible_source_extended.yaml`
Command: `python -m src.train.main_experiment --config configs\irreversible_source_extended.yaml --out results\extended\order_encoded --profile order_encoded`
Device: `cuda`
Runtime-limited: `False`

## Method Table

| Method | Val IID | IID Test | OOD Test | OOD Gap | Seeds |
|---|---:|---:|---:|---:|---:|
| core_only_oracle | 1.000 | 1.000 | 1.000 | 0.000 | 10 |
| counterfactual_invariance | 0.896 | 0.896 | 0.504 | 0.392 | 10 |
| final_frame_mlp | 0.517 | 0.499 | 0.502 | -0.003 | 10 |
| nuisance_only_oracle | 0.969 | 0.969 | 0.031 | 0.938 | 10 |
| sequence_erm | 0.969 | 0.969 | 0.031 | 0.938 | 10 |
| sequence_erm_temporal_pool | 1.000 | 0.999 | 1.000 | -0.000 | 10 |

## Main Gap Reduction

Mean ERM-minus-counterfactual OOD-gap reduction: `0.54619140625`
Consistent positive across common seeds: `False`

## Claim Status

Phenomenon supported: ERM shows an OOD gap. Counterfactual invariance improves mean OOD but is not seed-stable enough for a primary method-success claim.

This summary separates neural model evidence from diagnostic feature probes.
