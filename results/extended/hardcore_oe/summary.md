# Latest Main Result Summary

Profile: `hardcore_oe`
Primary scenario: `hc_main`
Config: `configs\irreversible_source_extended.yaml`
Command: `python -m src.train.main_experiment --config configs\irreversible_source_extended.yaml --out results\extended\hardcore_oe --profile hardcore_oe`
Device: `cuda`
Runtime-limited: `False`

## Method Table

| Method | Val IID | IID Test | OOD Test | OOD Gap | Seeds |
|---|---:|---:|---:|---:|---:|
| core_only_oracle | 0.930 | 0.930 | 0.928 | 0.002 | 10 |
| final_frame_mlp | 0.519 | 0.499 | 0.503 | -0.004 | 10 |
| nuisance_only_oracle | 0.969 | 0.969 | 0.031 | 0.938 | 10 |
| sequence_erm | 0.969 | 0.969 | 0.031 | 0.938 | 10 |
| sequence_erm_temporal_pool | 0.691 | 0.686 | 0.684 | 0.001 | 10 |

## Claim Status

Partially supported: ERM gap exists, but counterfactual result is missing.

This summary separates neural model evidence from diagnostic feature probes.
