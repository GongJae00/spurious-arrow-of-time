# Latest Main Result Summary

Profile: `oe_ext_budget`
Primary scenario: `oe_main`
Config: `configs\irreversible_source_extended.yaml`
Command: `python -m src.train.main_experiment --config configs\irreversible_source_extended.yaml --out results\extended\oe_ext_budget --profile oe_ext_budget`
Device: `cuda`
Runtime-limited: `False`

## Method Table

| Method | Val IID | IID Test | OOD Test | OOD Gap | Seeds |
|---|---:|---:|---:|---:|---:|
| sequence_erm | 0.969 | 0.969 | 0.031 | 0.938 | 10 |

## Claim Status

Partially supported: ERM gap exists, but counterfactual result is missing.

This summary separates neural model evidence from diagnostic feature probes.
