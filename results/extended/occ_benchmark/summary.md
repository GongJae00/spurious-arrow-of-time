# Latest Main Result Summary

Profile: `occ_benchmark`
Primary scenario: `occ_main`
Config: `configs\irreversible_source_extended.yaml`
Command: `python -m src.train.main_experiment --config configs\irreversible_source_extended.yaml --out results\extended\occ_benchmark --profile occ_benchmark`
Device: `cuda`
Runtime-limited: `False`

## Method Table

| Method | Val IID | IID Test | OOD Test | OOD Gap | Seeds |
|---|---:|---:|---:|---:|---:|
| core_only_oracle | 1.000 | 1.000 | 1.000 | 0.000 | 10 |
| final_frame_mlp | 0.512 | 0.501 | 0.499 | 0.001 | 10 |
| nuisance_only_oracle | 0.969 | 0.970 | 0.031 | 0.939 | 10 |
| sequence_erm | 1.000 | 1.000 | 1.000 | 0.000 | 10 |
| sequence_erm_temporal_pool | 0.515 | 0.499 | 0.500 | -0.002 | 10 |

## Claim Status

Partially supported: neural ERM did not show the target OOD gap.

This summary separates neural model evidence from diagnostic feature probes.
