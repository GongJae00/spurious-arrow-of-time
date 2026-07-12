# Latest Main Result Summary

Profile: `complexity`
Primary scenario: `cx_grid32_clutter`
Config: `configs\irreversible_source_extended.yaml`
Command: `python -m src.train.main_experiment --config configs\irreversible_source_extended.yaml --out results\extended\complexity --profile complexity`
Device: `cuda`
Runtime-limited: `False`

## Method Table

| Method | Val IID | IID Test | OOD Test | OOD Gap | Seeds |
|---|---:|---:|---:|---:|---:|
| core_only_oracle | 1.000 | 1.000 | 1.000 | 0.000 | 5 |
| final_frame_mlp | 0.527 | 0.502 | 0.503 | -0.001 | 5 |
| nuisance_only_oracle | 0.973 | 0.969 | 0.031 | 0.938 | 5 |
| sequence_erm | 0.973 | 0.969 | 0.032 | 0.937 | 5 |

## Claim Status

Partially supported: ERM gap exists, but counterfactual result is missing.

This summary separates neural model evidence from diagnostic feature probes.
