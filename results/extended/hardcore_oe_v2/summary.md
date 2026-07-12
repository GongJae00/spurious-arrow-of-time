# Latest Main Result Summary

Profile: `hardcore_oe`
Primary scenario: `hc_main`
Config: `configs\irreversible_source_extended.yaml`
Command: `python -m src.train.main_experiment --config configs\irreversible_source_extended.yaml --out results\extended\hardcore_oe_v2 --profile hardcore_oe`
Device: `cuda`
Runtime-limited: `False`

## Method Table

| Method | Val IID | IID Test | OOD Test | OOD Gap | Seeds |
|---|---:|---:|---:|---:|---:|
| core_only_oracle | 0.960 | 0.958 | 0.958 | 0.000 | 10 |
| final_frame_mlp | 0.520 | 0.496 | 0.503 | -0.007 | 10 |
| nuisance_only_oracle | 0.969 | 0.969 | 0.031 | 0.938 | 10 |
| sequence_erm | 0.969 | 0.969 | 0.031 | 0.938 | 10 |
| sequence_erm_temporal_pool | 0.931 | 0.926 | 0.924 | 0.001 | 10 |

## Claim Status

Partially supported: ERM gap exists, but counterfactual result is missing.

This summary separates neural model evidence from diagnostic feature probes.
