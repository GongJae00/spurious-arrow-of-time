# Latest Main Result Summary

Profile: `model_family`
Primary scenario: `model_family_main`
Config: `configs\irreversible_source_extended.yaml`
Command: `python -m src.train.main_experiment --config configs\irreversible_source_extended.yaml --out results\extended\model_family --profile model_family`
Device: `cuda`
Runtime-limited: `False`

## Method Table

| Method | Val IID | IID Test | OOD Test | OOD Gap | Seeds |
|---|---:|---:|---:|---:|---:|
| core_only_oracle | 1.000 | 1.000 | 1.000 | 0.000 | 10 |
| final_frame_mlp | 0.519 | 0.501 | 0.505 | -0.004 | 10 |
| nuisance_only_oracle | 0.969 | 0.969 | 0.031 | 0.938 | 10 |
| sequence_erm | 0.972 | 0.972 | 0.127 | 0.845 | 10 |
| sequence_erm_lstm | 0.972 | 0.972 | 0.126 | 0.845 | 10 |
| sequence_erm_tcn | 0.969 | 0.969 | 0.031 | 0.938 | 10 |
| sequence_erm_temporal_pool | 0.972 | 0.972 | 0.122 | 0.850 | 10 |
| sequence_erm_transformer | 0.987 | 0.987 | 0.596 | 0.390 | 10 |

## Claim Status

Partially supported: ERM gap exists, but counterfactual result is missing.

This summary separates neural model evidence from diagnostic feature probes.
