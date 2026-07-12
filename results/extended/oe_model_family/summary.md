# Latest Main Result Summary

Profile: `oe_model_family`
Primary scenario: `oe_family_main`
Config: `configs\irreversible_source_extended.yaml`
Command: `python -m src.train.main_experiment --config configs\irreversible_source_extended.yaml --out results\extended\oe_model_family --profile oe_model_family`
Device: `cuda`
Runtime-limited: `False`

## Method Table

| Method | Val IID | IID Test | OOD Test | OOD Gap | Seeds |
|---|---:|---:|---:|---:|---:|
| sequence_erm | 0.969 | 0.969 | 0.031 | 0.938 | 10 |
| sequence_erm_lstm | 0.969 | 0.969 | 0.031 | 0.938 | 10 |
| sequence_erm_tcn | 0.969 | 0.969 | 0.031 | 0.938 | 10 |
| sequence_erm_temporal_pool | 1.000 | 1.000 | 0.999 | 0.000 | 10 |
| sequence_erm_transformer | 1.000 | 1.000 | 0.999 | 0.000 | 10 |

## Claim Status

Partially supported: ERM gap exists, but counterfactual result is missing.

This summary separates neural model evidence from diagnostic feature probes.
