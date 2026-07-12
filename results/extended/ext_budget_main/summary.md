# Latest Main Result Summary

Profile: `ext_budget_main`
Primary scenario: `main_spurious_arrow`
Config: `configs\irreversible_source_extended.yaml`
Command: `python -m src.train.main_experiment --config configs\irreversible_source_extended.yaml --out results\extended\ext_budget_main --profile ext_budget_main`
Device: `cuda`
Runtime-limited: `False`

## Method Table

| Method | Val IID | IID Test | OOD Test | OOD Gap | Seeds |
|---|---:|---:|---:|---:|---:|
| sequence_erm | 0.981 | 0.980 | 0.380 | 0.601 | 30 |

## Claim Status

Partially supported: ERM gap exists, but counterfactual result is missing.

This summary separates neural model evidence from diagnostic feature probes.
