# Latest Main Result Summary

Profile: `oe_canonical`
Primary scenario: `oe_main`
Config: `configs\irreversible_source_extended.yaml`
Command: `python -m src.train.main_experiment --config configs\irreversible_source_extended.yaml --out results\extended\oe_canonical --profile oe_canonical`
Device: `cuda`
Runtime-limited: `False`

## Method Table

| Method | Val IID | IID Test | OOD Test | OOD Gap | Seeds |
|---|---:|---:|---:|---:|---:|
| core_only_oracle | 1.000 | 1.000 | 1.000 | 0.000 | 30 |
| final_frame_mlp | 0.517 | 0.500 | 0.501 | -0.000 | 30 |
| nuisance_only_oracle | 0.970 | 0.970 | 0.031 | 0.939 | 30 |
| sequence_erm | 0.970 | 0.970 | 0.031 | 0.939 | 30 |

## Claim Status

Partially supported: ERM gap exists, but counterfactual result is missing.

This summary separates neural model evidence from diagnostic feature probes.
